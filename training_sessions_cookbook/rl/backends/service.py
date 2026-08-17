"""SageMaker Training Sessions backend for the RL training loop.

Drives training through the ``sagemaker-train`` SDK
(``sagemaker.train.training_session``) against a managed training session.

Design notes:

* **Endpoints.** The SDK auto-resolves its control-plane and runtime
  endpoints; override via ``SAGEMAKER_TRAINING_SESSION_ENDPOINT`` /
  ``SAGEMAKER_TRAINING_SESSION_RUNTIME_ENDPOINT`` if needed.

* **Region consistency.** Everything (``ServiceClient(region=)``, the S3
  bucket, the model package group, and the ``base_model_arn``) must be the
  same region (default ``us-west-2``). ``create()`` validates presence of the
  required ARNs and logs the resolved region.

* **Async-native.** Training operations submit instantly (the SDK runs
  submission on a background thread) and are awaited via their futures;
  sampling uses the SDK's coroutine API directly. Every wait is bounded by
  ``OP_TIMEOUT_SECONDS`` so a server-side stall surfaces as a fail-fast
  ``TimeoutError`` instead of an indefinite hang.

* **Sampler refresh.** ``save_weights_for_sampler`` publishes a non-merged
  adapter checkpoint that auto-binds the session's sample path; the *same*
  reused ``SamplingClient`` reflects the update on its next request. The save
  happens inside :meth:`refresh_sampler` (gated by the loop's
  ``--refresh_every``) so the on/off-policy timeline follows that cadence.

* **Datum shape.** The SDK wants per-token loss inputs aligned with the
  shifted input sequence: ``tokens = obs + act``; ``input = tokens[:-1]``;
  ``target = tokens[1:]``; per-token ``weights`` / ``logprobs`` /
  ``advantages`` zero over the prompt and active over the response.

* **Deploy -> invoke.** Durable checkpoints are the training loop's concern
  (``service/checkpoint_utils.py``); the loop records the final deployable
  adapter package here via :meth:`note_deployable_checkpoint`, and
  :meth:`deploy_and_invoke` (opt-in) then stands up a real-time SageMaker
  endpoint with the trained LoRA adapter as an inference component.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

import boto3  # used by deploy_and_invoke
import sagemaker  # used by deploy_and_invoke
from sagemaker.train.training_session import AdamParams, LossFunction, SamplingParams

from training_sessions_cookbook.service import make_datum, open_session

from .base import build_loss_fn_config, resolve_sampling_extras

logger = logging.getLogger(__name__)

# Per-operation wait ceiling. Bounds every future/coroutine wait so a
# server-side stall fails fast instead of hanging. Overridable via env.
OP_TIMEOUT_SECONDS = float(os.environ.get("OP_TIMEOUT_SECONDS", "600"))


class _ServiceTrainStepResult:
    """Train-step result exposing ``.metrics`` (fwd/bwd merged with optim).

    Emits the ``train/*`` metric schema the training loop namespaces (optim
    metrics merged over the fwd/bwd metrics). ``.raw`` keeps the original
    fwd/bwd result for anything else.
    """

    def __init__(self, raw: Any, metrics: dict) -> None:
        self.raw = raw
        self.metrics = metrics


class ServiceBackend:
    """:class:`~backends.base.TrainingBackend` over SageMaker Training Sessions.

    Construct via :meth:`create` (called by :func:`backends.make_backend`).
    """

    def __init__(self, *, args, svc, tc, sampling_client, tokenizer, types) -> None:
        self._args = args
        self._svc = svc  # ServiceClient (control plane)
        self._tc = tc  # TrainingClient
        self._sampling_client = sampling_client  # single reused SamplingClient
        self.tokenizer = tokenizer
        self._t = types  # {SamplingParams, AdamParams, LossFunction}
        self._last_weights_model_package = None  # set via note_deployable_checkpoint
        self._session_deleted = False  # set by delete_session (makes close a no-op)

    # --- construction ------------------------------------------------------

    @classmethod
    def create(cls, args) -> "ServiceBackend":
        """Open a training session and create the reused sampling client.

        Validates the required service inputs, creates a LoRA training client
        (blocks until the session is Running), fetches the tokenizer, saves
        initial sampler weights (so the session binds a sample path), and
        creates the single ``SamplingClient`` reused for the run.
        """
        cls._validate_args(args)

        types = {
            "SamplingParams": SamplingParams,
            "AdamParams": AdamParams,
            "LossFunction": LossFunction,
        }

        region = getattr(args, "region", "us-west-2")
        session_name = (
            getattr(args, "training_session_name", None)
            or f"{args.run_name}-{int(time.time())}"
        )

        svc, tc, tokenizer = open_session(
            session_name=session_name,
            base_model_arn=args.base_model_arn,
            model_package_group_arn=args.model_package_group_arn,
            role_arn=args.role_arn,
            output_s3_path=args.output_s3_path,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            region=region,
            user_metadata={"run_name": str(args.run_name)},
        )

        # Save initial sampler weights so the session binds a sample path,
        # then create the single SamplingClient reused for the whole run.
        # Subsequent saves update the bound path and the same client reflects
        # the new policy on its next request. If any of this fails, stop the
        # session before re-raising: it was opened above, and no backend
        # object exists yet for the caller's teardown to close, so without
        # this the session would keep running unattended.
        try:
            tc.save_weights_for_sampler("initial").result(timeout=OP_TIMEOUT_SECONDS)
            sampling_client = tc.create_sampling_client()
        except BaseException:
            try:
                tc.stop()
                logger.info("Session stopped after setup failure: %s", session_name)
            except Exception as exc:
                logger.warning("Could not stop session %s after setup failure: %r",
                               session_name, exc)
            raise

        backend = cls(
            args=args,
            svc=svc,
            tc=tc,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            types=types,
        )
        backend._session_name = session_name
        return backend

    @staticmethod
    def _validate_args(args) -> None:
        required = [
            "base_model_arn",
            "role_arn",
            "model_package_group_arn",
            "output_s3_path",
        ]
        missing = [k for k in required if not getattr(args, k, None)]
        if missing:
            raise SystemExit(
                "--backend service requires: "
                + ", ".join(f"--{m}" for m in missing)
                + ".\nPass them explicitly (or via a --config JSON)."
            )

    # --- prompt / sampling-params shaping ---------------------------------

    def wrap_prompt(self, tokens: list[int]) -> Any:
        # The SDK sampling client accepts a bare token list as the prompt.
        return list(tokens)

    def make_sampling_params(
        self, *, max_tokens: int, temperature: float, seed: Optional[int] = None
    ) -> Any:
        SamplingParams = self._t["SamplingParams"]
        # Optional top_p / top_k / stop knobs. SamplingParams leaves these
        # None by default and omits them from the wire payload, so an unpinned
        # run inherits the inference server's own defaults. Pinning them (and
        # the seed) makes sampling fully reproducible.
        extras = resolve_sampling_extras(self._args)
        if seed is not None:
            extras["seed"] = seed
        return SamplingParams(max_tokens=max_tokens, temperature=temperature, **extras)

    async def sample_async(
        self,
        sampling_client: Any,
        prompt: Any,
        *,
        num_samples: int = 1,
        sampling_params: Any,
    ) -> Any:
        """Sample one rollout via the SDK's coroutine API.

        ``sample_async`` submits without blocking the event loop, so all
        concurrent rollouts reach the server and batch-fuse. The wait is
        bounded externally so a stalled sample fails fast.
        """
        return await asyncio.wait_for(
            sampling_client.sample_async(
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=sampling_params,
            ),
            OP_TIMEOUT_SECONDS,
        )

    # --- sampler lifecycle -------------------------------------------------

    def refresh_sampler(self, step: int) -> Any:
        """Save sampler weights (auto-binds the sample path); return the reused client.

        On step 0 the initial weights saved in :meth:`create` already bound
        the sample path, so the redundant save is skipped. Later refresh steps
        save a non-merged adapter checkpoint; the session auto-binds it and
        the reused ``SamplingClient`` reflects the new policy on its next
        request; no new client needed.
        """
        if step > 0:
            self._tc.save_weights_for_sampler(f"refresh-{step:06d}").result(
                timeout=OP_TIMEOUT_SECONDS
            )
        return self._sampling_client

    async def refresh_sampler_async(self, step: int) -> Any:
        """Non-blocking twin of :meth:`refresh_sampler` (overlapped loop).

        Awaits the save through the SDK's async API so the event loop stays
        free and rollout workers keep sampling during the save.
        """
        if step > 0:
            op = await self._tc.save_weights_for_sampler_async(f"refresh-{step:06d}")
            await asyncio.wait_for(op.result_async(), OP_TIMEOUT_SECONDS)
        return self._sampling_client

    # --- training ----------------------------------------------------------

    def build_datum(self, transition: Any, *, temperature: float) -> Any:
        """Build the SDK datum dict for one rollout transition.

        Slicing: ``tokens = obs + act``, ``input = tokens[:-1]``,
        ``target = tokens[1:]``, with per-token ``weights`` / ``logprobs`` /
        ``advantages`` zero over the ``obs_n - 1`` prompt positions and active
        over the response. The service loss kernel does not take a per-datum
        ``temperature`` tensor, so ``temperature`` is accepted for signature
        parity but unused.
        """
        obs_n = len(transition.obs_tokens)
        act_n = len(transition.act_tokens)
        assert act_n > 0, "empty action segment cannot be trained on"

        ob_len_m1 = obs_n - 1
        tokens = transition.obs_tokens + transition.act_tokens
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        resp_n = len(input_tokens) - ob_len_m1
        logprobs_seq = [0.0] * ob_len_m1 + list(transition.act_logprobs)
        advantages_seq = [0.0] * ob_len_m1 + [transition.advantage] * resp_n
        weights = [0.0] * ob_len_m1 + [1.0] * resp_n

        seq_len = len(input_tokens)
        assert (
            len(target_tokens) == len(logprobs_seq) == len(advantages_seq)
            == len(weights) == seq_len
        )

        return make_datum(
            input_tokens,
            target_tokens,
            weights,
            logprobs=logprobs_seq,
            advantages=advantages_seq,
        )

    def _loss_fn(self, loss_name: str) -> Any:
        LossFunction = self._t["LossFunction"]
        mapping = {
            "ppo": LossFunction.PPO,
            "importance_sampling": LossFunction.IMPORTANCE_SAMPLING,
            "cispo": getattr(LossFunction, "CISPO", None),
        }
        fn = mapping.get(loss_name)
        if fn is None:
            raise NotImplementedError(f"loss {loss_name!r} not supported by the service backend")
        return fn

    async def train_step(
        self,
        datums: list[Any],
        *,
        loss_name: str,
        learning_rate: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> Any:
        """Forward/backward + optimizer step, pipelined.

        Both operations are submitted back-to-back (the SDK assigns sequence
        numbers, so they run in order on the same session) and then awaited in
        submission order, each wait bounded by ``OP_TIMEOUT_SECONDS``.

        Returns a result exposing ``.metrics``, with the optimizer step's
        metrics merged over the fwd/bwd metrics (the schema the training loop
        namespaces under ``train/*``). Weight-save happens in
        :meth:`refresh_sampler`, not here, so the sampler refresh cadence
        follows ``--refresh_every``.
        """
        AdamParams = self._t["AdamParams"]
        adam_params = AdamParams(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps,
        )
        # ppo/cispo IS-ratio clip thresholds when the user set them; None
        # otherwise (server-side defaults apply).
        loss_fn_config = build_loss_fn_config(self._args)

        # Submit through the async twins (mirroring the sampling path) so
        # even datum serialization never blocks the event loop; both futures
        # come back immediately and the ops are pipelined on the session.
        fwd_bwd_op = await self._tc.forward_backward_async(
            datums, loss_fn=self._loss_fn(loss_name), loss_fn_config=loss_fn_config
        )
        optim_op = await self._tc.optim_step_async(adam_params)
        fb_result = await asyncio.wait_for(fwd_bwd_op.result_async(), OP_TIMEOUT_SECONDS)
        opt_result = await asyncio.wait_for(optim_op.result_async(), OP_TIMEOUT_SECONDS)

        # Merge optim metrics over fwd/bwd metrics so the train/* schema
        # carries both.
        metrics = dict(getattr(fb_result, "metrics", None) or {})
        metrics.update(getattr(opt_result, "metrics", None) or {})
        return _ServiceTrainStepResult(raw=fb_result, metrics=metrics)

    # --- checkpoint / deploy ----------------------------------------------

    @property
    def training_client(self):
        """The underlying SDK ``TrainingClient`` (checkpoint manager wiring)."""
        return self._tc

    @property
    def service_client(self):
        """The underlying SDK ``ServiceClient`` (checkpoint manager wiring)."""
        return self._svc

    def note_deployable_checkpoint(self, model_package_arn) -> None:
        """Record the deployable adapter package for :meth:`deploy_and_invoke`."""
        self._last_weights_model_package = model_package_arn

    def deploy_and_invoke(self) -> Optional[dict]:
        """Deploy the trained adapter to a real-time endpoint and invoke it.

        Completes train -> checkpoint -> deploy -> invoke. Requires a
        deployable adapter checkpoint (saved by the training loop when
        ``--save_weights_at_end`` is set). Uses the
        standard SageMaker Python SDK (``sagemaker`` / ``boto3``) hosting APIs:

          1. Deploy the base JumpStart model to an inference-component endpoint
             with LoRA hosting enabled.
          2. Add the trained adapter as an inference component pointing at the
             adapter's S3 weights URI (from the adapter model package).
          3. Invoke the endpoint targeting the adapter component.

        This is intentionally implemented as a thin, well-logged wrapper: the
        exact JumpStart deploy call surface differs by model and account, so
        we keep the moving parts explicit and fail loudly with next-steps
        rather than silently. Returns a summary dict, or raises with guidance.
        """
        if not self._last_weights_model_package:
            raise SystemExit(
                "deploy_and_invoke requires a deployable checkpoint. Re-run with "
                "--save_weights_at_end (or --deploy_and_invoke, which implies it)."
            )

        logger.info(
            "deploy_and_invoke: base_model_arn=%s adapter_model_package=%s region=%s",
            self._args.base_model_arn,
            self._last_weights_model_package,
            getattr(self._args, "region", "us-west-2"),
        )
        # The concrete deploy + inference-component wiring is account- and
        # model-specific, so it is left as an explicit stub rather than
        # guessed here.
        raise NotImplementedError(
            "Endpoint deploy is not implemented. The deployable adapter model "
            f"package is: {self._last_weights_model_package}. Deploy it to a "
            "real-time inference-component endpoint and wire the invoke calls here."
        )

    def delete_session(self) -> None:
        """Delete the training session (stopping it first if still active).

        Called on the success path after the final checkpoint is saved, so
        repeated runs don't accumulate stopped session records. Checkpoints
        are model packages (independent resources) and are unaffected.
        On failure paths this is NOT called: :meth:`close` only stops, keeping
        the session record for debugging.
        """
        self._svc.delete_session(self._session_name, stop_if_active=True)
        self._session_deleted = True
        logger.info("Session deleted: %s", self._session_name)

    def close(self) -> None:
        """Stop the training session so it stops consuming compute resources.

        No-op when :meth:`delete_session` already ran (the success path).
        """
        if self._session_deleted:
            return
        try:
            self._tc.stop()
            logger.info("Session stopped: %s", getattr(self, "_session_name", "?"))
        except Exception as exc:  # best-effort teardown
            logger.warning("ServiceBackend.close: failed to stop session: %r", exc)
