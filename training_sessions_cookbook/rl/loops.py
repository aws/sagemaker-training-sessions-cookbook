"""The two RL training loops.

* :func:`run_training` - the on-policy loop: refresh sampler, collect a batch,
  compute advantages, one ``forward_backward`` + ``optim_step`` per step.
* :func:`run_training_async` - the off-by-K (overlapped, bounded-staleness)
  loop: rollout workers stream GRPO groups through a staleness-aware batcher
  while training proceeds, overlapping sampling with optimization.

Split out of ``train.py`` so the orchestration layer (parser, ``run``/
``run_async`` lifecycle, teardown) and the per-step training logic can be read
and changed independently. ``train.py`` re-exports both loops, so existing
imports keep working.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import json
import logging
import os
import time

import psutil

from training_sessions_cookbook.service import learning_rate as _learning_rate

from .backends import TrainingBackend  # noqa: F401  # type hints
from .advantages import compute_grouped_advantages, compute_monte_carlo_advantages
from .config import cfg
from .dataset import _build_val_dataset, make_epoch_iter, next_batch
from .metrics import MetricsLogger, MetricsTablePrinter, capture_rollouts
from .rollouts import (
    WrappedGroup,
    collect_batch_async,
    collect_grouped_batch_async,
    collect_grouped_batch_streamed_async,
    collect_one_group_async,
)
from .runstate import run_artifacts
from .staleness import _AsyncLoopState, _PROMPT_SENTINEL, _StalenessBatcher, _WORKER_DONE
from .validation import ValidationRunner
from training_sessions_cookbook.service.session_watchdog import SessionDeadError


def compute_learning_rate(step: int, base_lr: float, warmup_steps: int) -> float:
    """Effective LR for ``step`` with optional linear warmup (no decay)."""
    return _learning_rate(base_lr, step, warmup_steps=warmup_steps)


# === Sampler refresh (lives in the backend) ===
#
class _FbStreamer:
    """Accumulate completed rollout groups and stream fb sub-batches (sync loop).

    Count-based trigger: after every ``ceil(n_groups / chunks)`` completed
    groups, the accumulated transitions become one ``forward_backward``
    submission (fire-and-forget - awaiting per sub-batch would re-pay the
    dispatch round-trip serially and stall the collector). Advantages are
    computed per group at completion, which is exactly equal to computing
    them over the whole batch (grouped GRPO advantages are per-group by
    construction). ``optim_step`` is NOT submitted here: the loop settles
    everything once via ``backend.finish_train_step`` - the on-policy barrier.
    """

    def __init__(self, backend, args, n_groups: int, chunks: int):
        self._backend = backend
        self._args = args
        self._chunk_groups = max(1, -(-n_groups // max(1, chunks)))
        self._pending: list = []
        self._groups_since_submit = 0
        self.fb_ops: list = []
        self.transitions: list = []

    async def on_group(self, group) -> None:
        transitions = compute_grouped_advantages([group])
        self.transitions.extend(transitions)
        self._pending.extend(transitions)
        self._groups_since_submit += 1
        if self._groups_since_submit >= self._chunk_groups:
            await self._submit()

    async def _submit(self) -> None:
        self._groups_since_submit = 0
        if not self._pending:
            return  # every episode in the accumulated groups failed/was empty
        datums = [
            self._backend.build_datum(t, temperature=self._args.temperature)
            for t in self._pending
        ]
        self._pending = []
        self.fb_ops.append(
            await self._backend.submit_forward_backward(
                datums, loss_name=self._args.loss
            )
        )

    async def flush(self) -> None:
        """Submit whatever remains (the last, possibly short, sub-batch)."""
        await self._submit()


# The "save weights then (re)bind a sampling client" logic is backend-specific: ServiceBackend saves a non-merged checkpoint that auto-binds the sample path and reuses a *single* SamplingClient (see `backends/service.py`). The loop calls `backend.refresh_sampler(step)` and gets back the sampling client to roll out against, staying ignorant of the mechanism.


# === Training loop body ===
#
# `run_training` composes the helpers above into the per-step training sequence. The signature is extended with keyword-only `tokenizer`, `env_factory`, and `template_fn` parameters because the rollout helpers need them; `main()` wires these in so this function stays pure.
#
# Per-step sequence:
#
#   1. If `step % args.refresh_every == 0`, snapshot the LoRA weights and rebind `sampling_client` to the fresh snapshot via the sampler-refresh helper. `step == 0` always satisfies this gate, so the very first step bootstraps the sampler from the current weights.
#
#   2. Pull `args.batch_size` prompts from the persistent `epoch_iter`. An empty batch means the dataset is exhausted (modulo `--drop_last`); we break out of the loop in that case so we don't log a no-op step.
#
#   3. Branch on `args.advantage_mode`:
#        - "grouped" -> K rollouts per prompt + GRPO group centering.
#        - "monte_carlo" -> one rollout per prompt + discounted-return recurrence + optional ReB normalization.
#      We retain an `episodes` view alongside the flat `transitions` view so we can compute episode-level metrics (mean return, mean turns, count).
#
#   4. Subsample/shuffle `transitions` down to `args.batch_size` honoring `--drop_last`. On `len < batch_size` with drop_last we `continue` rather than train on a short batch, so the `forward_backward` payload size stays constant across steps.
#
#   5. Build training datums via `build_training_datum(t, args.temperature)`.
#
#   6. Pipeline submission: we submit `forward_backward` and `optim_step` back-to-back, then await `fb_future.result()` followed by `opt_future.result()`. The service schedules the optimizer kernel as soon as gradients land instead of round-tripping through Python first.
#
#   7. Append one JSON object per step to `{run_folder}/metrics.jsonl`. The run folder is created by `main()` before calling `run_training`.


async def run_training(
    args,
    backend: "TrainingBackend",
    dataset,
    *,
    tokenizer,
    env_factory,
    template_fn,
    validation_runner=None,
    checkpoint_manager=None,
) -> int:
    """Drive the per-step RL training loop against the backend.

    Sampler refresh, datum construction, and the forward/backward + optimizer step all go through `backend`, so the loop stays independent of the concrete SDK and emits a stable `metrics.jsonl` schema.

    Args:
        args: Parsed argparse namespace from the CLI.
        backend: The `TrainingBackend` constructed in `main()`. Owns the training/sampling clients; the loop only calls `refresh_sampler`, `build_datum`, `train_step`, and the prompt/param shaping helpers.
        dataset: List of prompt dicts produced by `load_dataset`.
        tokenizer: Tokenizer obtained from the backend's training client; consumed by `rollout_one` for `.encode` / `.decode`.
        env_factory: Zero-arg callable returning a fresh `GSM8KMultiTurnEnv`. One env is constructed per concurrent rollout because the env is stateful.
        template_fn: Unary `(observation: str) -> str` template function; `main()` resolves `--template` against `TEMPLATE_FACTORY` and binds `apply_auto` against `args.base_model` before passing it through.
    """
    # Enlarge the event loop's default thread pool so blocking
    # `sampling_client.sample()` calls (dispatched via `asyncio.to_thread`) can
    # actually run `--num_rollout_workers`-wide. The implicit default executor
    # caps at `min(32, os.cpu_count() + 4)`, which silently throttles in-flight
    # sampling below the rollout semaphore on smaller boxes (e.g. ~12 on an
    # 8-vCPU host). Passing an explicit `max_workers` bypasses that formula
    # entirely. Threads are cheap here because `sample()` blocks on network I/O
    # and releases the GIL while waiting. Sized off the semaphore with a small
    # margin for the loop's other `to_thread` work (e.g. GSM8K subprocess grading).
    _sample_workers = cfg(args, "num_rollout_workers") + 8
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=_sample_workers,
            thread_name_prefix="rollout-sample",
        )
    )
    logging.getLogger(__name__).info(
        "rollout thread pool: max_workers=%d (num_rollout_workers=%d)",
        _sample_workers,
        cfg(args, "num_rollout_workers"),
    )

    # Optimizer params: rebuilt each step with warmup LR. The effective learning rate is computed per step via `compute_learning_rate` (linear warmup then constant) and passed to `backend.train_step`.
    adam_beta1 = cfg(args, "adam_beta1")
    adam_beta2 = cfg(args, "adam_beta2")
    adam_epsilon = cfg(args, "adam_epsilon")
    warmup_steps = cfg(args, "warmup_steps")

    # Metrics path: use the run folder path set up in main(). Falls back to the log_dir path if _metrics_path was not set.
    metrics_path = run_artifacts(args).metrics_path
    if metrics_path is None:
        os.makedirs(args.log_dir, exist_ok=True)
        metrics_path = os.path.join(args.log_dir, f"metrics_{args.run_name}.jsonl")

    # MetricsLogger handles all per-step metric computation and JSONL writing.
    metrics_logger = MetricsLogger(metrics_path, publisher=run_artifacts(args).mlflow)
    table = MetricsTablePrinter()

    # Periodic validation. Build a ValidationRunner only when a validation set
    # is configured; otherwise validation is skipped entirely. For code_rl,
    # --val_code_split loads the REAL DeepCoder split (graded via SandboxFusion,
    # exactly like training) and takes precedence over --val_dataset_path. The
    # run folder (used to write validation_examples.jsonl) comes from main();
    # fall back to the metrics file's directory for backward compatibility.
    if validation_runner is None:
        val_dataset = _build_val_dataset(args)
        if val_dataset is not None:
            run_folder = run_artifacts(args).run_folder or os.path.dirname(metrics_path)
            validation_runner = ValidationRunner(
                val_dataset,
                args.val_examples,
                run_folder,
                args.env_type,
            )

    # Persistent dataset cursor. `next_batch` reads/advances state on this object so batches are sequential and round-robin across `--epochs`.
    epoch_iter = make_epoch_iter(dataset, args.epochs, args.drop_last)

    sampling_client = None

    # Pre-training baseline validation (--val_at_start). Snapshot the initial LoRA weights, build a sampling client, and run one validation pass at logical step=0 BEFORE any rollouts or gradient updates. The val_metrics row goes straight to metrics.jsonl as a baseline record; we also keep a handle so the in-loop step=0 val is skipped to avoid a duplicate.
    sampling_client = None
    skip_step0_inloop_val = False
    if (
        validation_runner is not None
        and cfg(args, "val_at_start")
    ):
        # async twin: runs the blocking save+bind off-loop and is guarded by
        # the session-liveness watchdog (the sync form would block the event
        # loop, which also prevents the watchdog from detecting a dead session).
        sampling_client = await backend.refresh_sampler_async(0)
        baseline_val_metrics = await validation_runner.run_validation(
            0,
            sampling_client,
            tokenizer,
            template_fn,
            args,
            backend,
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "step": 0,
                "phase": "pre_training_baseline",
                **baseline_val_metrics,
            }) + "\n")
        if run_artifacts(args).mlflow is not None:
            run_artifacts(args).mlflow.log_metrics(baseline_val_metrics, step=0)
        skip_step0_inloop_val = True

    trained_steps = 0
    last_trained_step = 0
    for step in range(args.max_steps):
        # 1. Sampler refresh: snapshot weights and rebind the sampler.
        if step % args.refresh_every == 0:
            # If --val_at_start already refreshed the sampler at step 0 with the initial weights, reuse it instead of re-refreshing (the trainer weights haven't changed yet).
            if not (step == 0 and sampling_client is not None):
                sampling_client = await backend.refresh_sampler_async(step)

        # 2. Pull next batch of prompts. Empty -> dataset exhausted.
        step_start = time.time()
        prompts = next_batch(dataset, args.batch_size, epoch_iter)
        if not prompts:
            break

        # 3. Async rollouts + advantage computation. We keep the `episodes` view (a flat list of episodes) alongside the `transitions` view so episode-level metrics are computed on the right object.
        # --fb_stream_chunks: overlap fb with sampling (grouped mode + a
        # backend with the streaming protocol only; anything else falls back).
        fb_streamer = None
        _stream_n = int(cfg(args, "fb_stream_chunks") or 0)
        if _stream_n > 0 and args.advantage_mode == "grouped":
            if hasattr(backend, "submit_forward_backward") and hasattr(backend, "finish_train_step"):
                fb_streamer = _FbStreamer(backend, args, len(prompts), _stream_n)
            elif step == 0:
                logging.getLogger(__name__).warning(
                    "--fb_stream_chunks=%d ignored: backend %s has no streaming "
                    "protocol (submit_forward_backward/finish_train_step); "
                    "using the monolithic train_step.",
                    _stream_n, type(backend).__name__)
        elif _stream_n > 0 and step == 0:
            logging.getLogger(__name__).warning(
                "--fb_stream_chunks=%d ignored: requires --advantage_mode "
                "grouped (monte_carlo/rebn normalizes across the whole batch).",
                _stream_n)

        if args.advantage_mode == "grouped":
            if fb_streamer is not None:
                groups = await collect_grouped_batch_streamed_async(
                    prompts,
                    K=args.responses_per_prompt,
                    sampling_client=sampling_client,
                    args=args,
                    tokenizer=tokenizer,
                    env_factory=env_factory,
                    template_fn=template_fn,
                    backend=backend,
                    on_group_complete=fb_streamer.on_group,
                )
                await fb_streamer.flush()
                episodes = [episode for group in groups for episode in group]
                # Advantages were already computed per group at completion
                # (bitwise-identical to the batch computation); rebuild the
                # prompt-ordered flat view for metrics/capture parity.
                transitions = [t for g in groups for ep in g for t in ep]
            else:
                groups = await collect_grouped_batch_async(
                    prompts,
                    K=args.responses_per_prompt,
                    sampling_client=sampling_client,
                    args=args,
                    tokenizer=tokenizer,
                    env_factory=env_factory,
                    template_fn=template_fn,
                    backend=backend,
                )
                episodes = [episode for group in groups for episode in group]
                transitions = compute_grouped_advantages(groups)
        else:  # monte_carlo
            episodes = await collect_batch_async(
                prompts,
                sampling_client=sampling_client,
                args=args,
                tokenizer=tokenizer,
                env_factory=env_factory,
                template_fn=template_fn,
                backend=backend,
            )
            transitions = compute_monte_carlo_advantages(
                episodes,
                gamma=args.gamma,
                use_rebn=args.use_rebn,
            )
        sample_end = time.time()
        sample_seconds = sample_end - step_start

        # 4. Train on ALL transitions collected this step. This mirrors the SageMaker reference (gsm8k_rl_training.py): `--batch_size` only sets the number of prompts pulled per step, and every `batch_size x responses_per_prompt` episode's transitions are fed to forward_backward: there is NO subsampling/capping of the training set. Prompt-level `--drop_last` is already honored by `next_batch` via the epoch iterator, so no transition-level drop is needed here. `all_transitions` is the full set, used for both training and metrics.
        all_transitions = transitions

        if not transitions:
            # Every rollout failed or produced zero turns: skip the update
            # (loudly; an engine-side sampling outage looks exactly like
            # this) rather than train on an empty batch.
            logging.getLogger(__name__).warning(
                "step %d: no usable transitions (all %d rollouts failed or "
                "were empty); skipping training update",
                step, args.batch_size * args.responses_per_prompt)
            continue
        trained_steps += 1
        last_trained_step = step

        # Rollout capture: trajectories.jsonl and/or MLflow traces (opt-in).
        capture_rollouts(args, step, episodes, tokenizer, metrics_path)

        # 5+6. Train. Streamed path: the sub-batch forward_backward ops were
        # already submitted while rollouts were generating; settle them and the
        # optimizer here (train_seconds then measures only the EXPOSED tail).
        # Monolithic path: build all datums and run the pipelined
        # fb+optim train_step exactly as before.
        train_start = time.time()
        # Compute effective LR with linear warmup.
        effective_lr = compute_learning_rate(step, args.learning_rate, warmup_steps)
        if fb_streamer is not None and fb_streamer.fb_ops:
            fb_result = await backend.finish_train_step(
                fb_streamer.fb_ops,
                learning_rate=effective_lr,
                beta1=adam_beta1,
                beta2=adam_beta2,
                eps=adam_epsilon,
            )
        else:
            datums = [
                backend.build_datum(t, temperature=args.temperature)
                for t in transitions
            ]
            fb_result = await backend.train_step(
                datums,
                loss_name=args.loss,
                learning_rate=effective_lr,
                beta1=adam_beta1,
                beta2=adam_beta2,
                eps=adam_epsilon,
            )
        train_end = time.time()
        train_seconds = train_end - train_start
        step_total_seconds = train_end - step_start

        # 7. Periodic validation. Runs every --val_every steps when a validation dataset was provided. Only fires when a sampling_client exists (created on refresh steps). Collected metrics are merged into this step's JSONL record below.
        val_metrics = None
        if (
            validation_runner is not None
            and sampling_client is not None
            and args.val_every > 0
            and step % args.val_every == 0
            and not (step == 0 and skip_step0_inloop_val)
        ):
            val_metrics = await validation_runner.run_validation(
                step,
                sampling_client,
                tokenizer,
                template_fn,
                args,
                backend,
            )

        # 8. Log metrics via MetricsLogger. MetricsLogger computes all rollout stats, namespaces training metrics under train/, records timing, and appends a JSONL line. Validation metrics (when present) are merged into the same record.
        metrics_record = metrics_logger.log_step(
            step=step,
            episodes=episodes,
            transitions=all_transitions,
            fb_result=fb_result,
            sample_seconds=sample_seconds,
            train_seconds=train_seconds,
            step_total_seconds=step_total_seconds,
            effective_lr=effective_lr,
            kl_coef=getattr(args, 'kl_coef', 0.0),
            extra_metrics=val_metrics,
        )

        # 9. Console reporting: the compact row is the every-step heartbeat;
        # the full metric table (everything in metrics.jsonl for this step)
        # prints on the --log_every cadence.
        table.print_row(metrics_record, step)
        # Detail-table cadence runs on the 1-based displayed step
        # (--log_every 2 prints steps 2, 4, ...); the last step
        # always prints so no run ends without its final table.
        log_every = cfg(args, "log_every")
        display_step = step + 1
        if log_every > 0 and (display_step % log_every == 0
                              or display_step == args.max_steps):
            table.print_details(metrics_record, step)

        # Periodic durable checkpoint on the --save_every cadence (never
        # fires with the default save_every=max_steps; the final save in
        # run_async covers the end of training).
        if checkpoint_manager is not None:
            await checkpoint_manager.maybe_save_async(step, {"step": step})

    if trained_steps == 0:
        # Nothing trained: every step was skipped (all rollouts failed,
        # typically an engine-side sampling outage). Fail loudly instead of
        # ending "successfully" with an untrained final checkpoint.
        raise RuntimeError(
            f"no training occurred: all {args.max_steps} step(s) were "
            "skipped because no rollout produced usable transitions; "
            "check sampling health (see rollout_one warnings above)")
    return last_trained_step


async def run_training_async(
    args,
    backend: "TrainingBackend",
    dataset,
    *,
    tokenizer,
    env_factory,
    template_fn,
    validation_runner=None,
    checkpoint_manager=None,
) -> int:
    """Off-by-K (overlapped, bounded-staleness) RL training loop.

    Drop-in alternative to :func:`run_training`, selected by
    ``--max_steps_off_policy > 0``. Emits the identical ``metrics.jsonl`` schema
    (plus ``sampler/step_*`` staleness keys) so the same plotters and readers
    work. Only ``--advantage_mode grouped`` is supported here, the loop's unit
    of staleness is one prompt's K-rollout GRPO group.

    Args mirror :func:`run_training`.
    """
    log = logging.getLogger(__name__)

    max_steps_off_policy = int(cfg(args, "max_steps_off_policy"))
    assert max_steps_off_policy > 0, (
        "run_training_async requires --max_steps_off_policy > 0 "
        "(the sync run_training handles the on-policy case)"
    )
    if cfg(args, "advantage_mode") != "grouped":
        raise SystemExit(
            "--max_steps_off_policy > 0 (off-by-k loop) requires "
            "--advantage_mode grouped: the loop's staleness unit is one "
            "prompt's K-rollout group. Use the sync loop for monte_carlo."
        )

    # Same executor enlargement as the sync loop: blocking sample()/save calls
    # dispatched via asyncio.to_thread need enough worker threads to run
    # num_rollout_workers-wide (the default min(32, cpu+4) throttles below the
    # worker count on small boxes). Add headroom for the off-thread refresh.
    _sample_workers = cfg(args, "num_rollout_workers") + 8
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=_sample_workers,
            thread_name_prefix="rollout-sample",
        )
    )
    if int(cfg(args, "fb_stream_chunks") or 0) > 0:
        # Default-on knob, so being >0 here is normal operation, not a
        # misconfiguration: debug, not info.
        log.debug(
            "fb_stream_chunks=%s is a no-op in the off-by-k loop: rollout "
            "collection already overlaps training here.",
            cfg(args, "fb_stream_chunks"))
    log.info(
        "off-by-k loop: max_steps_off_policy=%d num_rollout_workers=%d "
        "batch_size(groups)=%d responses_per_prompt(K)=%d max_inflight_rollouts=%d",
        max_steps_off_policy,
        cfg(args, "num_rollout_workers"),
        args.batch_size,
        args.responses_per_prompt,
        cfg(args, "max_inflight_rollouts") or cfg(args, "num_rollout_workers"),
    )

    adam_beta1 = cfg(args, "adam_beta1")
    adam_beta2 = cfg(args, "adam_beta2")
    adam_epsilon = cfg(args, "adam_epsilon")
    warmup_steps = cfg(args, "warmup_steps")

    # Metrics path (identical to run_training).
    metrics_path = run_artifacts(args).metrics_path
    if metrics_path is None:
        os.makedirs(args.log_dir, exist_ok=True)
        metrics_path = os.path.join(args.log_dir, f"metrics_{args.run_name}.jsonl")
    metrics_logger = MetricsLogger(metrics_path, publisher=run_artifacts(args).mlflow)
    table = MetricsTablePrinter()

    # Validation (identical to run_training).
    if validation_runner is None:
        val_dataset = _build_val_dataset(args)
        if val_dataset is not None:
            run_folder = run_artifacts(args).run_folder or os.path.dirname(metrics_path)
            validation_runner = ValidationRunner(
                val_dataset, args.val_examples, run_folder, args.env_type,
            )

    epoch_iter = make_epoch_iter(dataset, args.epochs, args.drop_last)

    state = _AsyncLoopState()
    num_workers = cfg(args, "num_rollout_workers")
    K = args.responses_per_prompt

    # R9 / R10: client-side cap on live collect_one_group_async awaits, shared
    # across all rollout workers. Constructed unconditionally so the acquire
    # path in _worker is uniform (when the cap >= num_workers it never blocks,
    # which is a no-op observationally but still runs the same code: matches
    # R9.4). Scoped to collect_one_group_async only; workers still race freely
    # on prompt_queue.get and results_queue.put, so no double-gate deadlock
    # (R10.5). psutil sampler seeds first RSS row for R8.7 / R12.3.
    # `or num_workers` also covers a present-but-None attribute (the CLI
    # default), which plain getattr would have passed to Semaphore(None).
    max_inflight = cfg(args, "max_inflight_rollouts") or num_workers
    rollout_sem = asyncio.Semaphore(max_inflight)
    _process = psutil.Process()

    # Bootstrap the initial sampler (step 0) BEFORE starting workers so the very
    # first rollouts have a client. Optional --val_at_start baseline reuses it.
    state.sampling_client = await backend.refresh_sampler_async(0)
    skip_step0_inloop_val = False
    if validation_runner is not None and cfg(args, "val_at_start"):
        baseline_val_metrics = await validation_runner.run_validation(
            0, state.sampling_client, tokenizer, template_fn, args, backend,
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "step": 0,
                "phase": "pre_training_baseline",
                **baseline_val_metrics,
            }) + "\n")
        if run_artifacts(args).mlflow is not None:
            run_artifacts(args).mlflow.log_metrics(baseline_val_metrics, step=0)
        skip_step0_inloop_val = True

    # Bounded queues give natural backpressure: if the trainer is slow,
    # results_queue fills and workers block on put; if workers are slow,
    # prompt_queue fills and the dataloader blocks. Neither side runs away.
    prompt_queue: "asyncio.Queue" = asyncio.Queue(maxsize=max(2, num_workers * 2))
    results_queue: "asyncio.Queue" = asyncio.Queue(
        maxsize=max(2, args.batch_size + num_workers)
    )

    # Per-prompt seed base: distinct per group (keeps GRPO groups diverse). Not
    # fully reproducible across async runs (scheduling/dropping is nondeterministic),
    # so the A/B timing sweeps stay on the sync loop.
    _group_counter = {"n": 0}

    async def _dataloader() -> None:
        """Feed prompts one at a time until the epoch iterator is exhausted."""
        while True:
            prompts = next_batch(dataset, 1, epoch_iter)
            if not prompts:
                break
            await prompt_queue.put(prompts[0])
        # Cascade shutdown: one sentinel per worker.
        for _ in range(num_workers):
            await prompt_queue.put(_PROMPT_SENTINEL)

    async def _worker() -> None:
        """Pull a prompt, run K rollouts vs the current sampler, push a group.

        The whole loop is wrapped so the ``_WORKER_DONE`` sentinel reaches the
        consumer even if this worker dies with an unexpected exception
        (rollout failures are already best-effort inside
        ``collect_one_group_async``; this guards everything else). Without the
        guarantee, a crashed worker would silently reduce the producer count
        and the consumer's ``results_queue.get()`` could wait forever.
        """
        try:
            await _worker_loop()
        except asyncio.CancelledError:
            raise
        except SessionDeadError:
            # The session is terminal server-side: every worker would die the
            # same way one by one. Propagate so the loop fails once, with the
            # service's reason, instead of 80 stack traces and a puzzling
            # wind-down.
            raise
        except Exception:
            log.exception("off-by-k rollout worker died; sending its sentinel")
        finally:
            await results_queue.put(_WORKER_DONE)

    async def _worker_loop() -> None:
        while True:
            prompt = await prompt_queue.get()
            try:
                if prompt is _PROMPT_SENTINEL:
                    break
                # Read (step, client) atomically: no await between the two
                # reads, so we always get a consistent pair (see _AsyncLoopState).
                birth_step = state.step
                client = state.sampling_client
                seed_base = (
                    args.sampling_seed + _group_counter["n"] * 100003
                    if cfg(args, "sampling_seed") is not None
                    else None
                )
                _group_counter["n"] += 1
                # R10.2/R10.3: acquire AFTER reading birth_step/client, release
                # AFTER collect_one_group_async returns and BEFORE results_queue.put.
                # `async with` guarantees release even on exception (R10.3) and does
                # NOT wrap prompt_queue.get / results_queue.put: that would create
                # the double-gate deadlock R10.5 rules out.
                async with rollout_sem:
                    episodes = await collect_one_group_async(
                        prompt,
                        K=K,
                        sampling_client=client,
                        args=args,
                        tokenizer=tokenizer,
                        env_factory=env_factory,
                        template_fn=template_fn,
                        backend=backend,
                        seed_base=seed_base,
                    )
                # Stash the prompt in the wrapped group so the consumer's
                # staleness batcher can hand it back for a requeue if the group
                # is too stale to train (see `--stale_policy requeue`). Drop
                # policy discards this field.
                await results_queue.put(
                    WrappedGroup(
                        episodes=episodes,
                        birth_step=birth_step,
                        prompt=prompt,
                    )
                )
            finally:
                prompt_queue.task_done()

    dataloader_task = asyncio.create_task(_dataloader(), name="dataloader")
    worker_tasks = [
        asyncio.create_task(_worker(), name=f"rollout-worker-{i}")
        for i in range(num_workers)
    ]

    _requeue_tasks: set[asyncio.Task] = set()
    stale_policy = cfg(args, "stale_policy")
    batcher = _StalenessBatcher(
        args.batch_size, max_steps_off_policy, stale_policy=stale_policy,
    )
    workers_done = 0
    batch_wait_start = time.time()

    try:
        while state.step < args.max_steps:
            item = await results_queue.get()
            try:
                if item is _WORKER_DONE:
                    workers_done += 1
                    if workers_done >= num_workers:
                        # All producers stopped (dataset exhausted): no more data
                        # will arrive. Drop any partial batch to keep batch size
                        # constant, exactly like the sync loop's drop_last.
                        if batcher.pending:
                            log.info(
                                "off-by-k: dataset exhausted with %d/%d groups "
                                "accumulated; dropping partial batch and stopping "
                                "at step %d.",
                                len(batcher.pending), args.batch_size, state.step,
                            )
                        break
                    continue

                # Staleness filter + accumulate-to-batch_size. Returns a
                # 2-tuple (batch_or_None, stale_prompt_or_None). ``batch`` is
                # a full batch when batch_size fresh-enough groups have
                # accumulated. ``stale_prompt`` is populated only under
                # ``--stale_policy requeue`` when the just-offered group was
                # over-stale: we push that prompt back onto prompt_queue so a
                # worker can re-sample it fresh against the current sampler.
                # Skip the requeue after the dataloader is done: adding more
                # prompts once the shutdown cascade is running would either
                # deadlock on the bounded queue or race the sentinel drain.
                batch, stale_prompt = batcher.offer(item, state.step)
                if stale_prompt is not None and not dataloader_task.done():
                    # Requeue as a task so this consumer coroutine doesn't
                    # block on the (bounded) prompt_queue while workers wait
                    # for it to drain. Hold a reference until it completes:
                    # an unreferenced task may be garbage-collected before it
                    # runs, silently losing the prompt.
                    requeue_task = asyncio.create_task(
                        prompt_queue.put(stale_prompt),
                        name="requeue_stale_prompt_task",
                    )
                    _requeue_tasks.add(requeue_task)
                    requeue_task.add_done_callback(_requeue_tasks.discard)
                if batch is None:
                    continue
            finally:
                results_queue.task_done()

            # --- Train once on `batch_size` fresh-enough groups --------------
            step = state.step
            groups = [wg.episodes for wg in batch]
            episodes = [ep for group in groups for ep in group]
            transitions = compute_grouped_advantages(groups)

            # Staleness spread across the trained batch.
            staleness_vals = [step - wg.birth_step for wg in batch]

            if not transitions:
                # Defensive: every rollout in the batch failed. Skip training
                # (no step bump), like the sync loop's `if not transitions`.
                log.warning(
                    "off-by-k: batch at step %d had zero trainable transitions; "
                    "skipping train.", step,
                )
                # R8.4: still GC on the zero-transitions branch: the failed
                # batch's `groups`/`episodes`/`transitions` and the WrappedGroup
                # payloads referenced by `batch` are dead but only reachable via
                # asyncio Future cycles that generational GC may not sweep in
                # time.
                del batch, groups, episodes, transitions, staleness_vals
                gc.collect()
                batch_wait_start = time.time()
                continue

            sample_seconds = time.time() - batch_wait_start

            # Rollout capture: trajectories.jsonl and/or MLflow traces (opt-in).
            capture_rollouts(args, step, episodes, tokenizer, metrics_path)

            datums = [
                backend.build_datum(t, temperature=args.temperature)
                for t in transitions
            ]

            train_start = time.time()
            effective_lr = compute_learning_rate(step, args.learning_rate, warmup_steps)
            fb_result = await backend.train_step(
                datums,
                loss_name=args.loss,
                learning_rate=effective_lr,
                beta1=adam_beta1,
                beta2=adam_beta2,
                eps=adam_epsilon,
            )
            train_end = time.time()
            train_seconds = train_end - train_start
            step_total_seconds = train_end - batch_wait_start

            # --- Bump step + refresh sampler (gated by --refresh_every) ------
            # Do the slow refresh await BEFORE flipping shared state, then flip
            # (client, step) with no await between so workers never see a new
            # step paired with an old client. During the await, workers keep
            # sampling the old client (that lag is what the cap bounds).
            refresh_start = time.time()
            new_step = step + 1
            if new_step % args.refresh_every == 0:
                new_client = await backend.refresh_sampler_async(new_step)
            else:
                new_client = state.sampling_client
            refresh_seconds = time.time() - refresh_start
            state.sampling_client = new_client
            state.step = new_step

            # --- Periodic validation (mirrors run_training) ------------------
            val_metrics = None
            if (
                validation_runner is not None
                and args.val_every > 0
                and step % args.val_every == 0
                and not (step == 0 and skip_step0_inloop_val)
            ):
                val_metrics = await validation_runner.run_validation(
                    step, state.sampling_client, tokenizer, template_fn, args, backend,
                )

            # --- Log metrics (identical schema + staleness keys) -------------
            # R8.7 / R12.3: launcher RSS sampled inline (psutil, not an external
            # sampler) at every step so the leak-detection budgets (R8.7 step-5
            # <= 1.3x step-1; R12.5 final <= 1.5x post-warmup baseline) have
            # data. Sampled BEFORE del/gc so the value reflects state at the
            # moment the trainer body finished the iteration.
            timing_extras = {
                "time/refresh_seconds": refresh_seconds,
                "sampler/step_min": min(staleness_vals),
                "sampler/step_max": max(staleness_vals),
                "sampler/step_mean": sum(staleness_vals) / len(staleness_vals),
                "sampler/max_steps_off_policy": max_steps_off_policy,
                "rollout/num_stale_dropped": batcher.num_dropped,
                "rollout/num_stale_requeued": batcher.num_requeued,
                "sampler/stale_policy": stale_policy,
                "launcher/rss_bytes": _process.memory_info().rss,
            }
            if val_metrics:
                timing_extras.update(val_metrics)
            metrics_record = metrics_logger.log_step(
                step=step,
                episodes=episodes,
                transitions=transitions,
                fb_result=fb_result,
                sample_seconds=sample_seconds,
                train_seconds=train_seconds,
                step_total_seconds=step_total_seconds,
                effective_lr=effective_lr,
                kl_coef=cfg(args, "kl_coef"),
                extra_metrics=timing_extras,
                groups=groups,
            )

            table.print_row(metrics_record, step)
            # Detail-table cadence runs on the 1-based displayed step
            # (--log_every 2 prints steps 2, 4, ...); the last step
            # always prints so no run ends without its final table.
            log_every = cfg(args, "log_every")
            display_step = step + 1
            if log_every > 0 and (display_step % log_every == 0
                                  or display_step == args.max_steps):
                table.print_details(metrics_record, step)

            # Periodic durable checkpoint on the --save_every cadence (see
            # run_training; defaults to final-save-only).
            if checkpoint_manager is not None:
                await checkpoint_manager.maybe_save_async(step, {"step": step})

            # R8.1 / R8.2 / R8.3: explicit del + gc.collect() AFTER metrics have
            # been logged/printed (so nothing dereferences the deleted names)
            # and BEFORE we restart the batch-wait clock for the next iteration.
            # Targets the six names the OOM doc fingers as held across the loop
            # by asyncio Future cycles the generational GC can't sweep in time.
            del datums, transitions, groups, episodes, batch, fb_result
            gc.collect()

            # The batcher already reset its accumulator when it released this
            # batch; just restart the wall-clock timer for the next one.
            batch_wait_start = time.time()
    finally:
        # Shutdown: cancel producers (they may be blocked on queue put/get) and
        # await them so no task is left pending. return_exceptions swallows the
        # CancelledError we just raised in each.
        dataloader_task.cancel()
        for t in worker_tasks:
            t.cancel()
        await asyncio.gather(dataloader_task, *worker_tasks, return_exceptions=True)

    if state.step == 0:
        # Nothing trained: no batch ever yielded usable transitions (all
        # rollouts failed, typically an engine-side sampling outage). Fail
        # loudly instead of ending "successfully" with an untrained final
        # checkpoint.
        raise RuntimeError(
            "no training occurred: the off-by-k loop exited without a single "
            "training update; check sampling health (see rollout warnings "
            "above)")
    return state.step - 1
