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
  sampling uses the SDK's coroutine API directly. Waits are unbounded by
  default: the service bounds and retries the work itself, so a client-side
  ceiling would end a session that is merely queued. Set
  ``OP_TIMEOUT_SECONDS`` to restore one.

* **Sampler refresh.** ``save_weights_and_get_sampling_client`` publishes a
  non-merged adapter checkpoint, explicitly binds it to the session's sampling
  path, and returns an ARN-bound client; the loop swaps to the returned client
  each refresh (an ARN-less request samples the BASE model). The save
  happens inside :meth:`refresh_sampler` (gated by the loop's
  ``--refresh_every``) so the on/off-policy timeline follows that cadence.

* **Datum shape.** The SDK wants per-token loss inputs aligned with the
  shifted input sequence: ``tokens = obs + act``; ``input = tokens[:-1]``;
  ``target = tokens[1:]``; per-token ``weights`` / ``logprobs`` /
  ``advantages`` zero over the prompt and active over the response.

* **Deploy -> invoke.** Durable checkpoints are the training loop's concern
  (``service/checkpoint_utils.py``); the loop records the final deployable
  adapter package here via :meth:`note_deployable_checkpoint` for
  the run log records the final deployable adapter package for SageMaker
  endpoint with the trained LoRA adapter as an inference component.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

from sagemaker.train.training_session import AdamParams, LossFunction, SamplingParams

from training_sessions_cookbook.service import make_datum, open_session

from .base import build_loss_fn_config, resolve_sampling_extras
from training_sessions_cookbook.service.session_watchdog import SessionWatchdog, result_with_liveness
from ..config import cfg

logger = logging.getLogger(__name__)

def _resolve_op_timeout(env_var: str = "OP_TIMEOUT_SECONDS") -> float | None:
    """Read the per-operation wait ceiling from the environment.

    ``None`` (the default) means wait for the service. That is deliberate: the
    service already bounds the work, and a client ceiling below the service's own
    envelope turns *slow* into *dead* without reclaiming anything.

    * A training operation is bounded per attempt by the session manager
      (10 minutes for ``forward_backward``, 5 for ``optim_step`` and
      ``save_weights``) and retried a few times, so a genuinely stuck operation
      comes back ``Failed`` on its own. Giving up sooner is strictly worse:
      operations execute in sequence order, so the only thing a client-side
      expiry can do is end the session, and the abandoned operation keeps
      running on the engine regardless.
    * A sample request has no per-request service ceiling; the session's own
      lifetime is the outer bound. Waiting therefore trades a fast local signal
      for not discarding a rollout that is merely queued behind other tenants.

    Set ``OP_TIMEOUT_SECONDS`` to a positive number of seconds to restore a
    ceiling (useful when a run should fail rather than sit in a queue); ``0``,
    an empty value, or ``none`` keep the wait unbounded.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw or raw.lower() == "none":
        return None
    value = float(raw)
    return value if value > 0 else None


def _resolve_class_timeout(env_var: str) -> float | None:
    """Per-class ceiling: the class-specific env var wins, else the shared one.

    Training operations and sample requests have different waiting economics
    (see :func:`_resolve_op_timeout`): training ops are bounded per attempt by
    the session manager so a client ceiling mostly turns *slow* into *dead*,
    while sample requests can legitimately queue behind other tenants for a
    long time. ``TRAIN_OP_TIMEOUT_SECONDS`` / ``SAMPLE_OP_TIMEOUT_SECONDS``
    override per class; ``OP_TIMEOUT_SECONDS`` is the shared fallback for both.

    Called at BACKEND CONSTRUCTION, not at import: resolving env at import
    froze the values for the process lifetime (an env change after import
    silently did nothing) and forced tests to reload the module to exercise
    different settings.
    """
    if os.environ.get(env_var, "").strip():
        return _resolve_op_timeout(env_var)
    return _resolve_op_timeout("OP_TIMEOUT_SECONDS")


#: Sentinel distinguishing "not passed" from an explicit ``None``. ``None`` is
#: a meaningful timeout value (unbounded), so it cannot double as "use the env
#: default": a caller must be able to request unbounded waits even when
#: ``OP_TIMEOUT_SECONDS`` is set in the environment.
_UNSET: object = object()


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

    def __init__(
        self,
        *,
        args,
        svc,
        tc,
        sampling_client,
        tokenizer,
        types,
        train_op_timeout: "float | None | object" = _UNSET,
        sample_op_timeout: "float | None | object" = _UNSET,
    ) -> None:
        # Per-op-class wait ceilings, injectable for programmatic callers and
        # tests. Left unset, they resolve from the environment NOW (not at
        # import); an explicit ``None`` means unbounded regardless of the env.
        self._train_op_timeout = (
            _resolve_class_timeout("TRAIN_OP_TIMEOUT_SECONDS")
            if train_op_timeout is _UNSET
            else train_op_timeout
        )
        self._sample_op_timeout = (
            _resolve_class_timeout("SAMPLE_OP_TIMEOUT_SECONDS")
            if sample_op_timeout is _UNSET
            else sample_op_timeout
        )
        self._args = args
        self._svc = svc  # ServiceClient (control plane)
        self._tc = tc  # TrainingClient
        self._sampling_client = sampling_client  # single reused SamplingClient
        self.tokenizer = tokenizer
        self._t = types  # {SamplingParams, AdamParams, LossFunction}
        self._last_weights_model_package = None  # set via note_deployable_checkpoint
        self._session_deleted = False  # set by delete_session (makes close a no-op)
        # session-liveness watchdog (constructed lazily; see service/session_watchdog.py)
        self._watchdog = None

    # --- construction ------------------------------------------------------

    @classmethod
    def create(
        cls,
        args,
        *,
        train_op_timeout: "float | None | object" = _UNSET,
        sample_op_timeout: "float | None | object" = _UNSET,
    ) -> "ServiceBackend":
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

        region = cfg(args, "region")
        session_name = (
            cfg(args, "training_session_name")
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
        # Resolve the train-class ceiling once, before the setup call, so the
        # initial save honours the same value (injected or env) the instance
        # will use for every later training operation.
        resolved_train_timeout = (
            _resolve_class_timeout("TRAIN_OP_TIMEOUT_SECONDS")
            if train_op_timeout is _UNSET
            else train_op_timeout
        )
        try:
            # Liveness-aware wait: a session that dies during the initial save
            # would otherwise hang create() forever WHILE BILLING (no exception
            # means the stop-on-failure below never fires). No event loop exists
            # yet, so this polls the future in slices and describes in between.
            result_with_liveness(
                tc.save_weights_for_sampler("initial"),
                svc,
                session_name,
                timeout=resolved_train_timeout,
            )
            # ARN-less client = base model, which IS the correct step-0 policy
            # (LoRA starts at zero delta). Every later refresh returns a NEW
            # client bound to the fresh checkpoint (see refresh_sampler); this
            # initial client is only ever used for step-0 rollouts and the
            # optional val-at-start baseline.
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
            train_op_timeout=resolved_train_timeout,
            sample_op_timeout=sample_op_timeout,
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
        concurrent rollouts reach the server and batch-fuse. The wait honours
        the sample-class ceiling (``SAMPLE_OP_TIMEOUT_SECONDS`` / ``OP_TIMEOUT_SECONDS``), unbounded by default: a sample request
        has no per-request service ceiling, so a rollout queued behind other
        tenants is waited out rather than discarded. With a ceiling configured,
        expiry raises ``TimeoutError``, which the caller treats as a failed
        rollout and drops.
        """
        return await self._await_guarded(
            asyncio.wait_for(
                sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=num_samples,
                    sampling_params=sampling_params,
                ),
                self._sample_op_timeout,
            ),
            what="sample request",
        )

    # --- session liveness watchdog ------------------------------------------
    #
    # Shared implementation in service/session_watchdog.py: polls
    # DescribeTrainingSession and fails every guarded wait with the service's
    # FailureReason when the session leaves the live states, instead of
    # hanging forever on results that can never arrive.

    @property
    def watchdog(self):
        """The session-liveness watchdog (created lazily; None before first use
        or on backends without a session name, e.g. direct construction)."""
        self._ensure_watchdog()
        return self._watchdog

    def _ensure_watchdog(self) -> None:
        if self._watchdog is None:
            # Only backends opened via create() know their session name; a
            # directly-constructed backend (tests, programmatic embedding)
            # cannot poll and degrades to unguarded waits rather than crashing.
            name = getattr(self, "_session_name", None)
            if name:
                self._watchdog = SessionWatchdog(self._svc, name)

    async def _await_guarded(self, awaitable: Any, what: str) -> Any:
        self._ensure_watchdog()
        if self._watchdog is None:
            return await awaitable
        return await self._watchdog.guard(awaitable, what=what)

    # --- sampler lifecycle -------------------------------------------------

    def refresh_sampler(self, step: int) -> Any:
        """Save weights, BIND them to the sampling path, and return a bound client.

        The bind is explicit and load-bearing. ``save_weights_for_sampler``
        alone only registers a checkpoint; it does not touch the sampling
        path, and a sample request without a ``modelPackageArn`` samples the
        BASE model (the SDK's documented contract). The previous
        implementation relied on the service auto-binding each save to the
        session's sample path - the pre-GA behavior - so after the service
        change every rollout sampled frozen base weights for the whole run:
        reward pinned at the base model's level with PPO clipping against a
        never-moving behavior policy (measured: 40 flat steps on prod,
        2026-08-24). ``save_weights_and_get_sampling_client`` performs the
        save, issues the explicit ``loadWeightsForSampling`` bind, and returns
        a client whose requests carry the checkpoint ARN.

        On step 0 the initial client from :meth:`create` is returned as-is
        (sampling the base model IS the correct step-0 policy).
        """
        if step > 0:
            self._sampling_client = self._tc.save_weights_and_get_sampling_client(
                f"refresh-{step:06d}"
            )
        return self._sampling_client

    async def refresh_sampler_async(self, step: int) -> Any:
        """Non-blocking twin of :meth:`refresh_sampler` (overlapped loop).

        Same explicit save + bind + bound-client sequence (see
        :meth:`refresh_sampler` for why the bind is load-bearing); the SDK's
        async twin runs the blocking sequence on a worker thread so the event
        loop stays free and rollout workers keep sampling during the save.
        """
        if step > 0:
            self._sampling_client = await self._await_guarded(
                self._tc.save_weights_and_get_sampling_client_async(
                    f"refresh-{step:06d}"
                ),
                what=f"sampler refresh (step {step})",
            )
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
        submission order, each wait honouring the train-class ceiling (unbounded
        by default; the service bounds and retries these operations itself, and
        a client-side expiry can only end the session because operations execute
        in sequence order).

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
        fb_result = await self._await_guarded(
            asyncio.wait_for(fwd_bwd_op.result_async(), self._train_op_timeout),
            what="forward_backward",
        )
        opt_result = await self._await_guarded(
            asyncio.wait_for(optim_op.result_async(), self._train_op_timeout),
            what="optim_step",
        )

        # Merge optim metrics over fwd/bwd metrics so the train/* schema
        # carries both.
        metrics = dict(getattr(fb_result, "metrics", None) or {})
        metrics.update(getattr(opt_result, "metrics", None) or {})
        return _ServiceTrainStepResult(raw=fb_result, metrics=metrics)

    # --- streamed training (--fb_stream_chunks) -----------------------------
    #
    # The streaming protocol: the loop submits forward_backward sub-batches as
    # rollout groups complete (fire-and-forget), then settles everything once
    # via finish_train_step. Fire-and-forget matters: awaiting each sub-batch
    # serially re-pays the dispatch round-trip per op (measured 4x256 datums:
    # 47s awaited serially vs 23s submitted back-to-back), and would stall the
    # rollout collector. Optional protocol - loops feature-detect via hasattr,
    # so custom backends without it fall back to the monolithic train_step.

    async def submit_forward_backward(self, datums: list[Any], *, loss_name: str) -> Any:
        """Submit one forward_backward sub-batch; return its op WITHOUT waiting.

        Gradients accumulate server-side across consecutive forward_backward
        ops until optim_step applies them - the same mechanism the SDK's own
        payload chunking relies on, and verified weight-equivalent to a single
        monolithic call (chunked-vs-monolithic max weight delta 7.0e-5 vs a
        6.7e-5 rerun-to-rerun noise floor).
        """
        loss_fn_config = build_loss_fn_config(self._args)
        return await self._tc.forward_backward_async(
            datums, loss_fn=self._loss_fn(loss_name), loss_fn_config=loss_fn_config
        )

    async def finish_train_step(
        self,
        fb_ops: list[Any],
        *,
        learning_rate: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> Any:
        """Submit optim_step after all streamed sub-batches, then await everything.

        The single settle point of a streamed step: optim is sequenced AFTER
        every submitted forward_backward (SDK sequence numbers preserve
        submission order on the session), so the weight update sees the full
        accumulated gradient - the on-policy barrier is unchanged.

        Returns the same ``.metrics``-bearing result shape as
        :meth:`train_step`; ``global_token_num`` is summed across sub-batches
        (each op reports only its own tokens), other fb metrics are taken from
        the last sub-batch, and optim metrics are merged over the top, exactly
        as in the monolithic path.
        """
        assert fb_ops, "finish_train_step requires at least one submitted forward_backward"
        AdamParams = self._t["AdamParams"]
        adam_params = AdamParams(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps,
        )
        optim_op = await self._tc.optim_step_async(adam_params)

        fb_results = []
        for i, op in enumerate(fb_ops):
            fb_results.append(await self._await_guarded(
                asyncio.wait_for(op.result_async(), self._train_op_timeout),
                what=f"forward_backward (streamed {i + 1}/{len(fb_ops)})",
            ))
        opt_result = await self._await_guarded(
            asyncio.wait_for(optim_op.result_async(), self._train_op_timeout),
            what="optim_step",
        )

        metrics = dict(getattr(fb_results[-1], "metrics", None) or {})
        token_sum = 0.0
        saw_tokens = False
        for r in fb_results:
            m = getattr(r, "metrics", None) or {}
            if "global_token_num" in m:
                token_sum += float(m["global_token_num"])
                saw_tokens = True
        if saw_tokens:
            metrics["global_token_num"] = token_sum
        metrics.update(getattr(opt_result, "metrics", None) or {})
        return _ServiceTrainStepResult(raw=fb_results[-1], metrics=metrics)

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
        """Record the final deployable adapter model package for the run log."""
        self._last_weights_model_package = model_package_arn

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
        watchdog = getattr(self, "_watchdog", None)
        if watchdog is not None:
            watchdog.stop()
        if self._session_deleted:
            return
        try:
            self._tc.stop()
            logger.info("Session stopped: %s", getattr(self, "_session_name", "?"))
        except Exception as exc:  # best-effort teardown
            logger.warning("ServiceBackend.close: failed to stop session: %r", exc)
