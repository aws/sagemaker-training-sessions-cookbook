"""The :class:`TrainingBackend` contract behind the service backend.

The training loop (``run_training`` in ``rl/train.py``) is written once against
this protocol so it stays ignorant of the concrete SDK. Everything specific to
the "SageMaker Training Sessions service" is expressed as a method here:

* **prompt / params shaping**: ``wrap_prompt`` and ``make_sampling_params``
  turn plain ``list[int]`` tokens and scalar knobs into whatever the backend's
  sampling client expects. The loop's rollout coroutines stay ignorant of the
  concrete SDK types.
* **sampler refresh**: ``refresh_sampler(step)`` snapshots the current policy
  weights and returns the ``sampling_client`` rollouts should use. The service
  saves a (non-merged) checkpoint that auto-binds and hands back the *same*
  reused client.
* **datum construction**: ``build_datum`` produces the backend-native training
  datum from a rollout ``Transition``.
* **optimization**: ``train_step`` runs the pipelined forward/backward +
  optimizer step and returns the trainer metrics for the step.
* **checkpoint clients + deploy**: ``training_client`` / ``service_client`` feed the loop's checkpoint manager;
  ``deploy_and_invoke`` stands up an endpoint and invokes it.

The sampling-result contract the loop relies on: the loop calls
``await backend.sample_async(sampling_client, prompt, num_samples=, sampling_params=)``
and the returned result must expose ``.samples[i].tokens`` and
``.samples[i].logprobs``. ``sample_async`` submits without blocking the event
loop and awaits the result off-loop so many rollouts stay in-flight at once
(the service backend uses the SDK's ``sample_async`` submit/await split).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


def resolve_sampling_extras(args: Any) -> dict:
    """Resolve the optional top_p / top_k / stop sampling knobs off ``args``.

    Returns a dict with only the keys the user explicitly set (``--top_p``,
    ``--top_k``, ``--stop``); an unset flag is omitted entirely so the backend
    keeps its historical behavior byte-for-byte (service omits them and inherits
    the server default). ``--stop`` is a comma-separated string on the CLI and
    is split into a ``list[str]`` here; empty segments are dropped.
    """
    extras: dict = {}
    top_p = getattr(args, "top_p", None)
    if top_p is not None:
        extras["top_p"] = float(top_p)
    top_k = getattr(args, "top_k", None)
    if top_k is not None:
        extras["top_k"] = int(top_k)
    stop = getattr(args, "stop", None)
    if stop is not None:
        seqs = [s for s in str(stop).split(",") if s]
        if seqs:
            extras["stop"] = seqs
    return extras


def build_loss_fn_config(args: Any) -> Optional[dict]:
    """Build the SDK ``loss_fn_config`` dict from CLI clip-threshold args.

    Reads ``--clip_low_threshold`` / ``--clip_high_threshold`` off ``args`` and
    returns the dict the SDK loss expects (keys ``clip_low_threshold`` /
    ``clip_high_threshold``, absolute IS ratio bounds converted to verl
    epsilons server-side). Returns ``None`` when NEITHER is set, so the backend
    passes ``loss_fn_config=None`` and reproduces the historical default-clip
    behavior byte-for-byte, the plumbing is inert unless a threshold is
    explicitly provided. Only the keys that are set are included, so you can pin
    just one bound. Applies to ``ppo`` and ``cispo``; ignored by
    ``importance_sampling`` (unclipped) on the SDK side.
    """
    low = getattr(args, "clip_low_threshold", None)
    high = getattr(args, "clip_high_threshold", None)
    if low is None and high is None:
        return None
    config: dict = {}
    if low is not None:
        config["clip_low_threshold"] = float(low)
    if high is not None:
        config["clip_high_threshold"] = float(high)
    return config


@runtime_checkable
class TrainingBackend(Protocol):
    """Backend-specific operations behind the shared RL training loop.

    Concrete implementation: :class:`backends.service.ServiceBackend`. Construct
    via :func:`backends.make_backend`, which calls the class's ``create(args)``
    factory.
    """

    #: The tokenizer fetched from the training client. Must expose
    #: ``.encode(str) -> list[int]`` and ``.decode(list[int]) -> str``.
    tokenizer: Any

    # --- prompt / sampling-params shaping ---------------------------------

    def wrap_prompt(self, tokens: list[int]) -> Any:
        """Wrap raw token IDs into the sampling client's prompt type.

        Service: the token list is passed through (the PySDK sampling wrapper
        accepts a bare list).
        """
        ...

    def make_sampling_params(
        self, *, max_tokens: int, temperature: float, seed: Optional[int] = None
    ) -> Any:
        """Build the backend-native ``SamplingParams`` for a rollout.

        ``seed`` (when not None) requests reproducible generation for this
        request; the wire API documents it as ``samplingParams.seed``. Callers that
        want distinct-but-deterministic rollouts (e.g. GRPO's K responses per
        prompt) must pass a DIFFERENT seed per rollout; a shared seed would
        collapse a group to K identical responses. Backends pass the field
        defensively (dropped with a warning on SDK builds that lack it).
        """
        ...

    async def sample_async(
        self,
        sampling_client: Any,
        prompt: Any,
        *,
        num_samples: int = 1,
        sampling_params: Any,
    ) -> Any:
        """Sample from ``sampling_client`` without blocking the event loop.

        The backend implements this with its most concurrent-friendly path so
        ``--num_rollout_workers`` requests can be genuinely in-flight at once.
        The service backend submits via the SDK's ``sample_async`` (fires the
        request on the loop, cheaply) then awaits the operation's blocking
        The returned result must expose ``.samples[i].tokens`` and
        ``.samples[i].logprobs``. Implementations must not block the event
        loop: submission and waiting go through the SDK's async API, with the
        wait bounded so a stalled sample fails fast.
        """
        ...

    # --- sampler lifecycle -------------------------------------------------

    def refresh_sampler(self, step: int) -> Any:
        """Snapshot current weights and return the sampling client to use now.

        Called by the loop on every ``--refresh_every`` step (and at
        ``--val_at_start``). The returned object is what rollout coroutines
        call ``.sample(...)`` on. Service returns the single reused client after
        a weight save that auto-binds the sample path.
        """
        ...

    async def refresh_sampler_async(self, step: int) -> Any:
        """Non-blocking twin of :meth:`refresh_sampler` for the off-by-k loop.

        Identical result to :meth:`refresh_sampler` (snapshot the current
        weights, return the sampling client to roll out against), but the
        blocking work, the sampler weight-save, goes through the SDK's async
        API so the event loop stays free and concurrent rollout workers keep
        generating during a refresh. The strict serial loop
        (``run_training``) still calls the sync :meth:`refresh_sampler`; only
        the overlapped ``run_training_async`` (``--max_steps_off_policy > 0``)
        awaits this. The default Protocol-level implementation just offloads the
        sync method, which is correct for any backend whose refresh is
        thread-safe to run off-loop; backends may override for a finer split.
        """
        ...

    # --- training ----------------------------------------------------------

    def build_datum(self, transition: Any, *, temperature: float) -> Any:
        """Build one backend-native training datum from a rollout transition."""
        ...

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
        """Run the forward/backward + optimizer step for one training step.

        Coroutine (``await backend.train_step(...)``). Both operations are
        submitted back-to-back (pipelined on the session) and awaited in
        submission order without blocking the event loop; every wait is
        bounded by ``OP_TIMEOUT_SECONDS``.

        Returns an object exposing ``.metrics`` (a ``dict[str, float]``, with
        the optim step's metrics merged over the fwd/bwd metrics), matching what
        the loop namespaces under ``train/``. Weight saves happen in
        :meth:`refresh_sampler`, never here.
        """
        ...

    # --- checkpoint / deploy ----------------------------------------------
    # Durable checkpoints are the training loop's concern (it drives a
    # ``service.checkpoint_utils.CheckpointManager``); the backend only
    # exposes the SDK clients the manager needs and records the deployable
    # package for ``deploy_and_invoke``.

    @property
    def training_client(self):
        """The SDK ``TrainingClient`` behind this backend."""
        ...

    @property
    def service_client(self):
        """The SDK ``ServiceClient`` behind this backend."""
        ...

    def note_deployable_checkpoint(self, model_package_arn) -> None:
        """Record the final deployable adapter package (deploy_and_invoke)."""
        ...

    def deploy_and_invoke(self) -> Optional[dict]:
        """Deploy the trained adapter and invoke it.

        Returns ``None`` when endpoint deployment is not requested.
        """
        ...

    def delete_session(self) -> None:
        """Delete the session (success-path teardown; stops it first if active)."""
        ...

    def close(self) -> None:
        """Release backend resources (stop/delete the session, etc.)."""
        ...
