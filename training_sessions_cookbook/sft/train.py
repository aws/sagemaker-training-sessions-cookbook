from __future__ import annotations

import argparse
import asyncio
import logging
import time

import numpy as np
from sagemaker.train.training_session import AdamParams, LossFunction

from training_sessions_cookbook.service import checkpoint_utils
from training_sessions_cookbook.service.console import (
    format_metric_values,
    print_kv_table,
    print_run_banner,
    setup_console_logging,
)
from training_sessions_cookbook.service import learning_rate, open_session
from training_sessions_cookbook.service.config_file import apply_config_defaults, parse_known_config_path
from training_sessions_cookbook.service.session_watchdog import NullWatchdog, SessionWatchdog
from training_sessions_cookbook.service.mlflow_log import add_mlflow_args
from training_sessions_cookbook.service.runlog import RunLog
from training_sessions_cookbook.sft.data import build_sft_batches

logger = logging.getLogger(__name__)


# === Arg parsing ===
#
# All run knobs are command-line flags: the SageMaker session inputs, the LoRA adapter shape, dataset selection, and the training/eval schedule.



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT training on a SageMaker Training Session")

    # SageMaker required
    parser.add_argument("--training_session_name", type=str, default=None)
    parser.add_argument("--base_model_arn", type=str, required=True)
    parser.add_argument("--model_package_group_arn", type=str, required=True)
    parser.add_argument("--role_arn", type=str, required=True)
    parser.add_argument("--output_s3_path", type=str, required=True)
    parser.add_argument("--region", type=str, default="us-west-2")

    # Model
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)

    # Dataset
    parser.add_argument("--dataset_path", type=str, default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--train_split", type=str, default=None, help="Explicit HF split name for training (e.g. train_sft)")
    parser.add_argument("--eval_split", type=str, default=None, help="Explicit HF split name for eval (e.g. test_sft)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--max_conversations", type=int, default=None)
    parser.add_argument("--eval_size", type=int, default=None, help="Fixed eval set size (overrides eval_ratio)")
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Fraction of data for eval if no dedicated split")

    # Training
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--rolling_save_every", type=int, default=0,
                        help="Cheap resume-only checkpoints every N steps (0 = off). "
                             "Each save deletes the previous rolling model package.")
    parser.add_argument("--checkpoint_ttl_seconds", type=int, default=None,
                        help="Optional retention TTL for periodic checkpoints "
                             "(default: keep indefinitely). The final checkpoint "
                             "never expires regardless.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the newest checkpoint recorded in "
                             "{log_path}/checkpoints.jsonl (requires --log_path of "
                             "the interrupted run).")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print the full metric table every N steps (0 = "
                             "never). The one-line step summary prints every "
                             "step regardless.")
    parser.add_argument("--lr_schedule", type=str, default="linear", choices=["linear", "cosine", "constant"])

    # Adam
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)

    # Logging
    parser.add_argument("--log_path", type=str, default="results/sft_service")

    # MLflow (optional)
    add_mlflow_args(parser)

    # A JSON file whose keys seed the argparse DEFAULTS, so any flag passed
    # explicitly still wins over the file. A finished run's config.json is
    # exactly this shape, which makes runs directly re-runnable:
    #   --config results/<run>/config.json
    parser.add_argument(
        "--config", default=None,
        help="Path to a JSON config (dump_config/config.json shape) whose keys "
             "become defaults; explicit CLI flags override them.",
    )

    return parser


def parse_args(argv=None):
    """Parse ``argv`` (default: ``sys.argv``) with ``--config`` file support.

    Pass a list when calling from a notebook or another program so the host
    process's own argv is ignored. Two-pass parse: sniff ``--config`` first
    (argparse cannot satisfy required args from a file it has not read yet),
    fold its keys in as defaults, then parse for real so explicit CLI flags
    still override the file.
    """
    parser = build_parser()
    config_path = parse_known_config_path(argv)
    if config_path is not None:
        apply_config_defaults(parser, config_path)
    return parser.parse_args(argv)


# === Metrics and evaluation ===
#
# The service returns a weighted ``actor/loss``; these helpers rescale it to a per-response-token NLL and run a forward-only pass over the eval batches.


def compute_nll_from_actor_loss(actor_loss: float, batch) -> float:
    """Rescale the service's actor/loss to per-datum weighted NLL.

    ``weights`` are normalized to sum to 1 per datum, so the service's ``actor/loss`` is a weighted mean over normalized weights. Multiplying by the response-token count and dividing by the weight sum recovers the mean NLL per response token.
    """
    total_response_tokens = 0
    total_weight_sum = 0.0
    for datum in batch:
        lfi = datum.get("lossFnInputs", {})
        w = lfi.get("weights")
        if w is not None:
            w_data = w["data"] if isinstance(w, dict) else w
            w_arr = np.array(w_data, dtype=np.float32)
            total_response_tokens += int(np.count_nonzero(w_arr))
            total_weight_sum += float(w_arr.sum())
    if total_weight_sum > 0:
        return actor_loss * total_response_tokens / total_weight_sum
    return actor_loss


def _actor_loss(result):
    """Pull actor/loss out of a training/forward result (outputs or metrics).

    ``loss_fn_outputs`` is a per-datum list (the SDK normalizes older
    single-mapping responses into a one-entry list).
    """
    lfo = result.loss_fn_outputs
    if lfo:
        entries = lfo if isinstance(lfo, list) else [lfo]
        first = entries[0]
        if isinstance(first, dict) and "actor/loss" in first:
            value = first["actor/loss"]
            return value["data"][0] if isinstance(value, dict) else value
    if result.metrics and "actor/loss" in result.metrics:
        return result.metrics["actor/loss"]
    return None


async def evaluate(training_client, eval_batches, loss_fn, *, watchdog=None) -> dict[str, float]:
    """Run a forward-only pass on eval data and return ``eval/mean_nll``.

    ``watchdog`` (optional) fails the eval fast if the session dies
    server-side mid-pass, instead of hanging on results that can never arrive.
    """
    total_weighted = 0.0
    total_weights = 0.0

    for batch in eval_batches:
        fwd_future = await training_client.forward_async(batch, loss_fn=loss_fn)
        wait = fwd_future.result_async()
        if watchdog is not None:
            fwd_result = await watchdog.guard(wait, what="eval forward")
        else:
            fwd_result = await wait
        actor_loss = _actor_loss(fwd_result)
        if actor_loss is not None:
            batch_nll = compute_nll_from_actor_loss(actor_loss, batch)
            batch_weight = sum(
                sum(d.get("lossFnInputs", {}).get("weights", {}).get("data", []))
                for d in batch
            )
            total_weighted += batch_nll * batch_weight
            total_weights += batch_weight

    if total_weights == 0:
        logger.warning("  [eval] total_weights=0, returning NaN")
        return {"eval/mean_nll": float("nan")}
    return {"eval/mean_nll": total_weighted / total_weights}


# === Training loop body ===
#
# Open the session, build the train/eval batches from the service tokenizer, then step through forward_backward + optim_step, logging metrics and saving checkpoints on the configured cadence.


def _model_label(base_model_arn) -> str:
    """A short model name for display: the hub content name inside the ARN
    (``.../Model/<name>/<version>``), or the full value when the shape is
    anything else."""
    text = str(base_model_arn)
    parts = text.split("/")
    return parts[-2] if len(parts) >= 2 else text


async def run_async(args, *, train_batches=None, eval_batches=None) -> None:
    """Drive one full SFT run from a parsed/constructed ``args`` object.

    The programmatic entry point: build ``args`` via
    ``build_parser().parse_args(...)`` (or construct any object with the same
    attributes), override fields, and ``await run_async(args)`` from async
    code (e.g. a notebook cell). Synchronous scripts use :func:`run`.

    ``train_batches`` / ``eval_batches`` bypass the built-in dataset loading
    (``--dataset_path`` et al.). Each accepts either a ready list of datum
    batches or a callable ``(tokenizer) -> batches``, the callable form
    exists because the service-resolved tokenizer is only available once the
    session is open.
    """
    # Console logging through the shared console (service/console.py); see
    # rl/train.py for the rationale. No-op when the caller configured logging.
    setup_console_logging()

    session_name = args.training_session_name or f"sft-{int(time.time())}"

    # --- Run logging ------------------------------------------------------
    # RunLog owns the run folder, config.json, metrics.jsonl, and: when
    # --mlflow_tracking_uri is set: the MLflow run lifecycle: params at
    # start, artifacts on exit, closed FINISHED on clean exit / FAILED on
    # exception. MLflow publishing is best-effort and can never fail training.
    with RunLog.open(session_name, config=vars(args), output_dir=args.log_path,
                    file_logging=True) as runlog:
        print_run_banner("SFT training run", {
            "training_session_name": session_name,
            "model": _model_label(args.base_model_arn),
            "dataset_path": args.dataset_path,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps if args.max_steps is not None else "one epoch",
            "run_folder": runlog.run_folder,
        })
        # --- Session ------------------------------------------------------
        # Open the SageMaker Training Session, which builds the training client and resolves the tokenizer (shared wiring with the RL path in `service/`). The heavy SDK is imported lazily inside `open_session`.
        resume_record = None
        if args.resume:
            if not args.log_path:
                raise SystemExit("--resume requires --log_path of the interrupted run")
            resume_record = checkpoint_utils.get_last_checkpoint(runlog.run_folder)
            if resume_record is None:
                logger.info("--resume: no checkpoint found; starting fresh")
        svc, training_client, tokenizer = await asyncio.to_thread(open_session,
            session_name=session_name,
            base_model_arn=args.base_model_arn,
            model_package_group_arn=args.model_package_group_arn,
            role_arn=args.role_arn,
            output_s3_path=args.output_s3_path,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            region=args.region,
            load_state_arn=resume_record.state_path if resume_record else None,
        )
        runlog.set_tags({
            "training_session_name": training_client.training_session_name,
        })
        # Fail fast if the session dies server-side mid-run (engine fault
        # etc.); without this the result_async() awaits in the loop would hang
        # forever on results that can never arrive. Created here (not inside
        # _train) so the finally below stops the poller on EVERY exit path -
        # success, exception, or cancellation.
        watchdog = SessionWatchdog(svc, training_client.training_session_name)
        try:
            await _train(args, session_name, runlog, svc, training_client,
                         tokenizer, resume_record=resume_record,
                         train_batches=train_batches, eval_batches=eval_batches,
                         watchdog=watchdog)
        except BaseException:
            # Stop (not delete) the session so a mid-training crash doesn't
            # leave billed GPU resources running. Stopping preserves the
            # session record for debugging. Best-effort: a teardown error
            # must not mask the original exception.
            try:
                training_client.stop()
                logger.info("Session stopped after failure: %s", session_name)
            except Exception as exc:
                logger.warning("Failed to stop session %s: %r", session_name, exc)
            raise
        finally:
            watchdog.stop()


async def _train(args, session_name, runlog, svc, training_client, tokenizer,
                 resume_record=None, train_batches=None, eval_batches=None,
                 watchdog=None):
    """The training loop body (async, mirroring the RL path's call pattern).

    ``watchdog`` is the caller-owned :class:`SessionWatchdog`; the caller
    stops it in a ``finally`` so the poller never outlives the run on any
    exit path. ``None`` (programmatic callers) degrades to unguarded waits.
    """
    if watchdog is None:
        watchdog = NullWatchdog()

    # --- Dataset --------------------------------------------------------
    # Injected batches (list, or callable(tokenizer) -> batches) win; otherwise
    # build them from the service tokenizer (same tokenizer the RL path uses).
    # `build_sft_batches` masks loss to assistant-response tokens and
    # normalizes per-datum weights.
    if callable(train_batches):
        train_batches = train_batches(tokenizer)
    if callable(eval_batches):
        eval_batches = eval_batches(tokenizer)
    if train_batches is not None:
        logger.info("Using injected training batches")
    else:
        logger.info("Loading dataset from %s", args.dataset_path)
        train_batches = build_sft_batches(
            dataset_path=args.dataset_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            split="train",
            eval_size=args.eval_size,
            eval_ratio=args.eval_ratio,
            max_conversations=args.max_conversations,
            hf_split_name=args.train_split,
        )
    n_batches = len(train_batches)
    logger.info("Dataset: %d batches of %d", n_batches, args.batch_size)
    if n_batches == 0:
        raise SystemExit(
            "No training batches were produced. Common causes: the dataset is "
            "smaller than --eval_size (the eval split is carved out first), "
            "every conversation exceeded --max_length, or the dataset path is "
            "empty. Check the dataset flags and rerun."
        )

    # --- Eval dataset ---------------------------------------------------
    # Only built when --eval_every > 0; uses the held-out split (or a slice carved out by eval_size/eval_ratio when no dedicated split exists).
    if eval_batches is None and args.eval_every > 0:
        eval_batches = build_sft_batches(
            dataset_path=args.dataset_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            split="eval",
            eval_size=args.eval_size,
            eval_ratio=args.eval_ratio,
            hf_split_name=args.eval_split,
        )
        logger.info("Eval dataset: %d batches", len(eval_batches))

    # --- Step budget ----------------------------------------------------
    # One step per batch across all epochs, capped by --max_steps when set.
    total_steps = n_batches * args.num_epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    logger.info(
        "Training for %d steps (lr=%s, schedule=%s)",
        total_steps, args.learning_rate, args.lr_schedule,
    )

    # --- Checkpointing ----------------------------------------------------
    # Periodic full checkpoints (state + sampler weights) on --save_every,
    # optional rolling resume-only checkpoints, and the permanent final save.
    # Records land in {run_folder}/checkpoints.jsonl (see service/checkpoint_utils.py).
    ckpt = checkpoint_utils.CheckpointManager(
        training_client,
        svc,
        runlog.run_folder,
        watchdog=watchdog,
        save_every=args.save_every,
        ttl_seconds=args.checkpoint_ttl_seconds,
        rolling_save_every=args.rolling_save_every,
    )

    metrics = {}
    step = 0
    if resume_record is not None and resume_record.step is not None:
        step = int(resume_record.step) + 1
        logger.info("Resumed from checkpoint %r; continuing at step %d",
                    resume_record.name, step)
    elif resume_record is not None:
        logger.warning(
            "Resume checkpoint %r has no 'step' field; weights were loaded "
            "but training restarts at step 0", resume_record.name)
    for epoch in range(args.num_epochs):
        for batch_idx in range(n_batches):
            # One step trains one batch, in order, so the global batch index
            # equals the step counter. On resume, skip the batches the
            # checkpointed run already trained; without this the resumed run
            # would retrain the head of the dataset under a continued step
            # counter and stop before ever reaching the untrained batches.
            if epoch * n_batches + batch_idx < step:
                continue
            if step >= total_steps:
                break

            # 1. Effective LR for this step (warmup/decay per --lr_schedule).
            t0 = time.time()
            lr = learning_rate(
                args.learning_rate, step,
                total_steps=total_steps, schedule=args.lr_schedule,
            )
            batch = train_batches[batch_idx]

            # 2. Forward/backward + optimizer step on the service.
            # `forward_backward` accumulates gradients; `optim_step` applies the
            # Adam update with the effective LR. Both are submitted back-to-back
            # through the async twins (pipelined on the session) and their
            # futures awaited in submission order.
            fb_future = await training_client.forward_backward_async(
                batch, loss_fn=LossFunction.CROSS_ENTROPY
            )
            opt_future = await training_client.optim_step_async(AdamParams(
                learning_rate=lr,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                eps=args.adam_eps,
            ))
            fb_result = await watchdog.guard(
                fb_future.result_async(), what="forward_backward"
            )
            opt_result = await watchdog.guard(
                opt_future.result_async(), what="optim_step"
            )
            elapsed = time.time() - t0

            # 3. Calculate the per-token NLL from the returned actor/loss.
            actor_loss = _actor_loss(fb_result)
            train_nll = compute_nll_from_actor_loss(actor_loss, batch) if actor_loss is not None else None

            # 4. Assemble the per-step metrics record. Keys are grouped by namespace (`train/`, `time/`, `config/`, `eval/`), with `step` / `epoch` / `progress` left flat as run coordinates. Service-reported metrics go under `train/`, then the noisy/internal keys are dropped so the JSONL schema stays stable for the analyze/plot scripts.
            metrics = {
                "step": step,
                "epoch": epoch,
                "progress": (step + 1) / total_steps,
                "config/learning_rate": lr,
                "train/mean_nll": train_nll,
                "train/num_sequences": len(batch),
                "train/num_tokens": sum(len(d["modelInput"]["chunks"][0]["tokens"]) for d in batch),
                "time/step_total_seconds": round(elapsed, 2),
            }
            for result in (fb_result, opt_result):
                if result.metrics:
                    metrics.update({f"train/{k}": v for k, v in result.metrics.items()})
            for k in ("train/estimated_flops", "train/promised_flops", "train/mfu", "train/actor/loss", "train/actor/grad_norm"):
                metrics.pop(k, None)

            # 5. Print the one-line progress row.
            nll_str = f"{train_nll:.4f}" if train_nll else "?"
            logger.info(
                "step=%d/%d epoch=%d train/mean_nll=%s lr=%.2e time=%.1fs",
                # 1-based for display ("3/3" on the last step); the step field
                # in metrics.jsonl and checkpoints.jsonl stays 0-based.
                step + 1, total_steps, epoch, nll_str, lr, elapsed,
            )

            # 6. Periodic eval. Runs a forward-only pass over the held-out batches every --eval_every steps; the eval NLL is merged into this step's record.
            if eval_batches and args.eval_every > 0 and step % args.eval_every == 0:
                eval_metrics = await evaluate(training_client, eval_batches, LossFunction.CROSS_ENTROPY, watchdog=watchdog)
                metrics.update(eval_metrics)
                logger.info("  eval/mean_nll=%.4f", eval_metrics.get("eval/mean_nll", float("nan")))

            # 7. Append the step record to metrics.jsonl (and mirror it to
            # MLflow when configured; non-numeric values are skipped there).
            runlog.log_metrics(metrics, step=step)

            # Full metric table (everything in metrics.jsonl for this step)
            # on the --log_every cadence. The cadence runs on the same
            # 1-based step number the summary line and the table title show
            # (so --log_every 2 prints steps 2, 4, ...), and the last step
            # always prints so a run never ends without its final table.
            display_step = step + 1
            if args.log_every > 0 and (
                display_step % args.log_every == 0
                or display_step == total_steps
            ):
                print_kv_table(f"Step {display_step} metrics",
                               format_metric_values(metrics))

            # 8. Checkpoints on cadence: a full periodic save (state + sampler
            # weights, resumable) every --save_every steps, plus optional rolling
            # saves. Records go to checkpoints.jsonl; the sampler ARN is also
            # tagged on the MLflow run when publishing is enabled.
            saved = await ckpt.maybe_save_async(step, {"step": step, "epoch": epoch})
            if saved:
                runlog.log_checkpoint(step, saved.get("sampler_path"))

            step += 1

    # --- Final save -----------------------------------------------------
    # The canonical post-training checkpoint: permanent (no TTL), recorded
    # under the "final" name; also cleans up any leftover rolling checkpoint.
    logger.info("Saving final checkpoint...")
    final_paths = await ckpt.save_final_async({"step": max(step - 1, 0)})
    runlog.log_checkpoint("final", final_paths.get("sampler_path"))
    logger.info("Final checkpoint: %s", final_paths)
    # --- Teardown -------------------------------------------------------
    # Delete the training session (stopping it first if still active) so no billed resources are left running.
    svc.delete_session(session_name, stop_if_active=True)
    logger.info("Training complete. %d steps. Metrics: %s", step, runlog.metrics_path)


def run(args, *, train_batches=None, eval_batches=None) -> None:
    """Synchronous wrapper around :func:`run_async` for scripts and the CLI.

    Raises a self-explanatory error when called from async code (e.g. a
    Jupyter cell, where an event loop is already running), use
    ``await run_async(...)`` there.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop, the normal script/CLI case
    else:
        raise RuntimeError(
            "run() cannot be called from a running event loop (e.g. a Jupyter "
            "notebook). Use `await run_async(...)` instead."
        )
    asyncio.run(run_async(args, train_batches=train_batches, eval_batches=eval_batches))


def main():
    run(parse_args())

if __name__ == "__main__":
    main()
