"""KTO preference training over a SageMaker Training Session.

KTO's loss is non-linear, but the model enters it only through the per-sequence
completion log-probability, so we train it with the stock service API - no
custom server-side loss kernel - using the surrogate-gradient trick (see
:mod:`training_sessions_cookbook.preference.kto` for the derivation). Each step:

1. **forward** (no grad) on the matched ``(x_i, y_i)`` batch -> per-token policy
   log-probs -> ``log pi_theta(y_i)``.
2. **forward** (no grad) on the mismatched KL batch ``(x_i, y_{p(i)})`` ->
   ``z_ref = max(0, mean(log pi_theta - log pi_ref))`` (``--kl_estimator none``
   skips this pass and sets ``z_ref = 0``).
3. Per example, compute the implicit reward ``r_i`` and the per-sequence
   gradient scalar ``s_i``; set the completion-token advantage to ``-s_i``.
4. **forward_backward** on the matched batch with the importance-sampling loss
   (old log-probs = the step-1 log-probs, so the ratio is 1 and the surrogate
   gradient equals the exact KTO gradient), then **optim_step**.

The reference model is the frozen base model (a LoRA adapter starts at zero
delta), so the reference log-probs are computed **once before the first
optimizer step** and cached to ``{run_folder}/kto_reference.json`` for resume.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any

from training_sessions_cookbook.service import checkpoint_utils, learning_rate, open_session
from training_sessions_cookbook.service.config_file import apply_config_defaults, parse_known_config_path
from training_sessions_cookbook.service.console import (
    format_metric_values,
    print_kv_table,
    print_run_banner,
    setup_console_logging,
)
from training_sessions_cookbook.service.mlflow_log import add_mlflow_args
from training_sessions_cookbook.service.runlog import RunLog
from training_sessions_cookbook.service.session_watchdog import NullWatchdog, SessionWatchdog

from .data import (
    KtoExample,
    batch_kl_partners,
    build_kto_batches,
    forward_datum,
    kl_slice,
    matched_slice,
    pair_slice,
    train_datum,
)
from .kto import advantage_from_grad, kl_reference_point, kto_term, sum_completion_logprob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Sdk:
    """The SDK loss/optimizer types the loop needs, resolved lazily.

    Bundling them lets the module import (and its pure helpers be tested)
    without the ``sagemaker-train`` wheel; :func:`_resolve_sdk` fills it in once
    a run actually starts. ``fwd_loss`` is used for the no-grad forward passes
    (any loss works - we read per-token log-probs, not the loss value -
    cross-entropy needs only the target_tokens + weights every datum carries);
    ``is_loss`` is the importance-sampling loss that carries the surrogate
    gradient.
    """

    fwd_loss: Any
    is_loss: Any
    AdamParams: Any


def _resolve_sdk() -> _Sdk:
    from sagemaker.train.training_session import AdamParams, LossFunction

    return _Sdk(
        fwd_loss=LossFunction.CROSS_ENTROPY,
        is_loss=LossFunction.IMPORTANCE_SAMPLING,
        AdamParams=AdamParams,
    )


# === Arg parsing ===


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KTO preference training on a SageMaker Training Session")

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
    parser.add_argument("--dataset_path", type=str, default="trl-lib/kto-mix-14k")
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--eval_size", type=int, default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.1)

    # KTO objective
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KTO temperature on the implicit reward (higher = sharper).")
    parser.add_argument("--desirable_weight", type=float, default=1.0,
                        help="Loss weight lambda_D for desirable examples.")
    parser.add_argument("--undesirable_weight", type=float, default=1.0,
                        help="Loss weight lambda_U for undesirable examples.")
    parser.add_argument("--kl_estimator", type=str, default="shuffled",
                        choices=["shuffled", "none"],
                        help="'shuffled': estimate z_ref from mismatched pairs "
                             "(one extra forward per step); 'none': z_ref = 0.")
    parser.add_argument("--reference_source", type=str, default="precompute",
                        choices=["precompute", "compute_logprobs"],
                        help="Where reference (base-model) log-probs come from. "
                             "'precompute': one training-forward pass at step 0 "
                             "(LoRA delta still zero), cached; KL pairing fixed. "
                             "'compute_logprobs': score the frozen base on demand "
                             "via an ARN-less sampling client, re-randomizing the "
                             "KL pairing every step (TRL-style).")

    # Training
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--rolling_save_every", type=int, default=0)
    parser.add_argument("--checkpoint_ttl_seconds", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the newest checkpoint in "
                             "{log_path}/checkpoints.jsonl (reuses the cached "
                             "reference log-probs; requires --log_path).")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=50,
                        help="Run a held-out eval pass every N steps (0 = off). "
                             "Uses the eval split carved out by --eval_size / "
                             "--eval_ratio.")
    parser.add_argument("--lr_schedule", type=str, default="linear",
                        choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_steps", type=int, default=0)

    # Adam
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)

    # Logging
    parser.add_argument("--log_path", type=str, default="results/kto_service")
    add_mlflow_args(parser)
    parser.add_argument("--config", default=None,
                        help="Path to a JSON config whose keys become argparse "
                             "defaults; explicit CLI flags override them.")
    return parser


def parse_args(argv=None):
    """Parse ``argv`` with ``--config`` file support (two-pass, like the SFT path)."""
    parser = build_parser()
    config_path = parse_known_config_path(argv)
    if config_path is not None:
        apply_config_defaults(parser, config_path)
    return parser.parse_args(argv)


# === Service helpers ===


def _datum_input_len(datum: dict) -> int:
    """Token count of a datum's model input (== len(targets) == len(mask))."""
    return sum(len(c["tokens"]) for c in datum["modelInput"]["chunks"])


def _entry_logprobs(entry) -> list[float]:
    """The 'logprobs' tensor of one loss_fn_outputs entry, as a flat float list."""
    lp = entry.get("logprobs") if isinstance(entry, dict) else None
    if lp is None:
        raise RuntimeError(
            "forward result has no per-token 'logprobs'. KTO needs the "
            "service's per-token log-prob output; update sagemaker-train to a "
            "version whose forward returns 'logprobs'."
        )
    data = lp["data"] if isinstance(lp, dict) else lp
    return [float(x) for x in data]


def _extract_logprobs(result, datums: list[dict]) -> list[list[float]]:
    """Pull per-token log-probs out of a forward result, one list per datum.

    The engine returns ``loss_fn_outputs`` as a single flat entry today: one
    ``logprobs`` tensor with all sequences concatenated, ``len(input) + 1``
    values per datum (the datum adapter appends the final target token), no
    padding. This splits it back per datum by input length and trims each
    slice to ``len(targets)``, front-aligned. If the engine returns one entry
    per datum (the documented future shape), that is used directly.
    """
    lfo = getattr(result, "loss_fn_outputs", None)
    entries = lfo if isinstance(lfo, list) else ([lfo] if lfo else [])
    lens = [_datum_input_len(d) for d in datums]

    if len(entries) == len(datums):
        # Per-datum shape (future engines) — trim any appended extra position.
        return [_entry_logprobs(e)[:n] for e, n in zip(entries, lens)]

    if len(entries) == 1:
        flat = _entry_logprobs(entries[0])
        # Current engines: len(input)+1 logprobs per datum. Accept len(input)
        # too in case the appended position is ever dropped.
        for extra in (1, 0):
            if sum(n + extra for n in lens) == len(flat):
                out, offset = [], 0
                for n in lens:
                    out.append(flat[offset:offset + n])
                    offset += n + extra
                return out
        raise RuntimeError(
            f"forward returned one flat loss_fn_outputs entry with {len(flat)} "
            f"logprobs, which does not partition into {len(datums)} datums of "
            f"input lengths {lens} (+0 or +1 each)."
        )

    raise RuntimeError(
        f"forward returned {len(entries)} loss_fn_outputs for {len(datums)} "
        "datums; expected one flat entry or one per datum with per-token "
        "'logprobs'."
    )


async def _forward_logprobs(training_client, datums, *, loss_fn, watchdog, what):
    """Run a no-grad forward and return per-datum per-token log-probs."""
    fwd_op = await training_client.forward_async(datums, loss_fn=loss_fn)
    result = await watchdog.guard(fwd_op.result_async(), what=what)
    return _extract_logprobs(result, datums)


# === Reference scoring via an ARN-less base sampling client ==========
#
# CAVEAT (engine mismatch): the ARN-less sampling client runs on the INFERENCE
# engine, while the policy matched log-probs come from the TRAINING forward. So
# the reward r = log pi_theta(train) - log pi_ref(inference) crosses two engines
# - the rollout-vs-training log-prob gap. For KTO's z_ref (a detached, clamped
# baseline) that is fine. Do NOT use these inference-path log-probs for the
# POLICY side of the reward, where the engine gap and the sampler checkpoint lag
# would bias the actual gradient. The policy side always uses the training
# forward (`_forward_logprobs`).


async def _score_reference_sums(ref_client, pairs, *, watchdog, what):
    """Sum frozen-base completion log-probs for ``(prompt, completion)`` pairs.

    Uses the ARN-less sampling client's ``compute_logprobs_async`` (which serves
    the base model), which returns one log-prob per token with the first element
    ``None``. For ``full = prompt + completion``, the completion log-probs are
    the entries at absolute positions ``[len(prompt) : len(prompt) + len(completion)]``,
    so their sum is ``log pi_ref(completion | prompt)``. Requests are fired
    concurrently so the inference server can batch-fuse them.
    """
    async def _one(prompt_tokens, completion_tokens):
        full = list(prompt_tokens) + list(completion_tokens)
        p, c = len(prompt_tokens), len(completion_tokens)
        lps = await watchdog.guard(
            ref_client.compute_logprobs_async(prompt=full), what=what)
        return sum(x for x in lps[p : p + c] if x is not None)

    return list(await asyncio.gather(*(_one(p, c) for p, c in pairs)))


# === Reference log-probs (computed once, cached) ===


def _reference_cache_path(run_folder: str) -> str:
    return os.path.join(run_folder, "kto_reference.json")


async def _compute_reference_logprobs(training_client, batches, *, use_kl, sdk, watchdog):
    """Sum reference (base-model) completion log-probs for every example.

    Must run BEFORE the first optim_step, while the LoRA delta is still zero so
    the training client scores the base model. Returns ``(ref_matched,
    ref_kl)`` as lists of per-batch lists of per-example scalar sums; ``ref_kl``
    is empty when the KL term is disabled.
    """
    ref_matched: list[list[float]] = []
    ref_kl: list[list[float]] = []
    for bi, batch in enumerate(batches):
        m_datums, m_masks = [], []
        for ex in batch:
            inp, tgt, mask = matched_slice(ex)
            m_datums.append(forward_datum(inp, tgt, mask))
            m_masks.append(mask)
        m_lps = await _forward_logprobs(training_client, m_datums, loss_fn=sdk.fwd_loss,
                                        watchdog=watchdog,
                                        what=f"reference forward (matched {bi + 1}/{len(batches)})")
        ref_matched.append([sum_completion_logprob(lp, mask) for lp, mask in zip(m_lps, m_masks)])

        if use_kl:
            k_datums, k_masks = [], []
            for ex in batch:
                inp, tgt, mask = kl_slice(ex)
                k_datums.append(forward_datum(inp, tgt, mask))
                k_masks.append(mask)
            k_lps = await _forward_logprobs(training_client, k_datums, loss_fn=sdk.fwd_loss,
                                            watchdog=watchdog,
                                            what=f"reference forward (kl {bi + 1}/{len(batches)})")
            ref_kl.append([sum_completion_logprob(lp, mask) for lp, mask in zip(k_lps, k_masks)])
    return ref_matched, ref_kl


async def _matched_reference_sums(batches, *, source, training_client, ref_client,
                                  sdk, watchdog, label):
    """Reference completion log-prob sums for the matched pairs of each batch.

    Works for both reference sources: ``precompute`` scores via a training
    forward (valid only before the first optim_step, when the LoRA delta is
    zero); ``compute_logprobs`` scores the frozen base on demand via the
    ARN-less client. Returns a per-batch list of per-example scalar sums.
    """
    out: list[list[float]] = []
    for bi, batch in enumerate(batches):
        what = f"reference {label} ({bi + 1}/{len(batches)})"
        if source == "compute_logprobs":
            sums = await _score_reference_sums(
                ref_client, [(e.prompt_tokens, e.completion_tokens) for e in batch],
                watchdog=watchdog, what=what)
        else:
            slices = [matched_slice(e) for e in batch]
            datums = [forward_datum(inp, tgt, mask) for inp, tgt, mask in slices]
            lps = await _forward_logprobs(training_client, datums, loss_fn=sdk.fwd_loss,
                                          watchdog=watchdog, what=what)
            sums = [sum_completion_logprob(lp, mask) for lp, (_, _, mask) in zip(lps, slices)]
        out.append(sums)
    return out


async def _evaluate_kto(training_client, eval_batches, eval_ref_matched, *,
                        z_ref, sdk, watchdog, step=None, run_folder=None,
                        tokenizer=None, examples_per_eval=8) -> dict:
    """Held-out KTO diagnostics: reward margin, accuracy, and per-class log-probs.

    Forward-only (no gradient). Rewards are ``logπ_θ − logπ_ref`` on the eval
    completions; ``accuracy`` uses the current training ``z_ref`` as the
    threshold (desirable above it, undesirable below). ``reward_margin`` needs
    no threshold and is the headline held-out number.

    When ``run_folder`` and ``tokenizer`` are given, the first
    ``examples_per_eval`` examples of each pass are appended to
    ``validation_examples.jsonl`` with their decoded text and rewards.
    """
    rewards_d, rewards_u, logps_d, logps_u, correct, total = [], [], [], [], 0, 0
    records = []
    for bi, (batch, ref_sums) in enumerate(zip(eval_batches, eval_ref_matched)):
        slices = [matched_slice(ex) for ex in batch]
        datums = [forward_datum(inp, tgt, mask) for inp, tgt, mask in slices]
        lps = await _forward_logprobs(training_client, datums, loss_fn=sdk.fwd_loss,
                                      watchdog=watchdog,
                                      what=f"eval forward ({bi + 1}/{len(eval_batches)})")
        for ex, (_, _, mask), lp, ref_sum in zip(batch, slices, lps, ref_sums):
            policy_sum = sum_completion_logprob(lp, mask)
            reward = policy_sum - ref_sum
            (rewards_d if ex.label else rewards_u).append(reward)
            (logps_d if ex.label else logps_u).append(policy_sum)
            if (ex.label and reward > z_ref) or (not ex.label and reward < z_ref):
                correct += 1
            total += 1
            if tokenizer is not None and len(records) < examples_per_eval:
                records.append({
                    "step": step,
                    "prompt": tokenizer.decode(ex.prompt_tokens),
                    "completion": tokenizer.decode(ex.completion_tokens),
                    "label": "desirable" if ex.label else "undesirable",
                    "logp_policy": round(policy_sum, 3),
                    "logp_ref": round(ref_sum, 3),
                    "reward": round(reward, 3),
                })
    if run_folder is not None and records:
        with open(os.path.join(run_folder, "validation_examples.jsonl"), "a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    return {
        "kto/eval_reward_desirable": mean(rewards_d),
        "kto/eval_reward_undesirable": mean(rewards_u),
        "kto/eval_reward_margin": (mean(rewards_d) - mean(rewards_u))
        if rewards_d and rewards_u else float("nan"),
        "kto/eval_logps_desirable": mean(logps_d),
        "kto/eval_logps_undesirable": mean(logps_u),
        "kto/eval_accuracy": correct / total if total else float("nan"),
        "kto/eval_num_examples": total,
    }


def _load_reference_cache(run_folder: str):
    path = _reference_cache_path(run_folder)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cached = json.load(f)
    return cached.get("matched", []), cached.get("kl", []), cached.get("eval_matched", [])


def _save_reference_cache(run_folder: str, ref_matched, ref_kl, eval_ref_matched) -> None:
    with open(_reference_cache_path(run_folder), "w") as f:
        json.dump({"matched": ref_matched, "kl": ref_kl,
                   "eval_matched": eval_ref_matched}, f)


# === Run ===


def _model_label(base_model_arn) -> str:
    parts = str(base_model_arn).split("/")
    return parts[-2] if len(parts) >= 2 else str(base_model_arn)


async def run_async(args, *, train_batches=None) -> None:
    """Drive one full KTO run from a parsed/constructed ``args`` object.

    ``train_batches`` bypasses dataset loading: a ready list of
    :class:`~training_sessions_cookbook.preference.data.KtoExample` batches, or a
    callable ``(tokenizer) -> batches`` (the service tokenizer only exists once
    the session opens).
    """
    setup_console_logging()
    session_name = args.training_session_name or f"kto-{int(time.time())}"

    with RunLog.open(session_name, config=vars(args), output_dir=args.log_path,
                     file_logging=True) as runlog:
        print_run_banner("KTO preference training run", {
            "training_session_name": session_name,
            "model": _model_label(args.base_model_arn),
            "dataset_path": args.dataset_path,
            "batch_size": args.batch_size,
            "beta": args.beta,
            "kl_estimator": args.kl_estimator,
            "reference_source": args.reference_source,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps if args.max_steps is not None else "one epoch",
            "run_folder": runlog.run_folder,
        })

        resume_record = None
        if args.resume:
            if not args.log_path:
                raise SystemExit("--resume requires --log_path of the interrupted run")
            resume_record = checkpoint_utils.get_last_checkpoint(runlog.run_folder)
            if resume_record is None:
                logger.info("--resume: no checkpoint found; starting fresh")

        svc, training_client, tokenizer = await asyncio.to_thread(
            open_session,
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
        runlog.set_tags({"training_session_name": training_client.training_session_name})
        watchdog = SessionWatchdog(svc, training_client.training_session_name)
        try:
            await _train(args, session_name, runlog, svc, training_client, tokenizer,
                         resume_record=resume_record, train_batches=train_batches,
                         watchdog=watchdog)
        except BaseException:
            try:
                training_client.stop()
                logger.info("Session stopped after failure: %s", session_name)
            except Exception as exc:
                logger.warning("Failed to stop session %s: %r", session_name, exc)
            raise
        finally:
            watchdog.stop()


async def _train(args, session_name, runlog, svc, training_client, tokenizer,
                 resume_record=None, train_batches=None, watchdog=None):
    if watchdog is None:
        watchdog = NullWatchdog()
    use_kl = args.kl_estimator == "shuffled"
    sdk = _resolve_sdk()

    # --- Dataset --------------------------------------------------------
    if callable(train_batches):
        train_batches = train_batches(tokenizer)
    if train_batches is None:
        logger.info("Loading preference dataset from %s", args.dataset_path)
        train_batches = build_kto_batches(
            dataset_path=args.dataset_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            split="train",
            eval_size=args.eval_size,
            eval_ratio=args.eval_ratio,
            max_rows=args.max_rows,
            hf_split_name=args.train_split,
        )
    n_batches = len(train_batches)
    if n_batches == 0:
        raise SystemExit(
            "No training batches were produced. Check --dataset_path, the "
            "prompt/completion/label columns, and --max_length."
        )
    logger.info("Dataset: %d batches of up to %d", n_batches, args.batch_size)

    # --- Eval dataset ---------------------------------------------------
    # Held-out split (carved from train by --eval_size / --eval_ratio) for the
    # periodic eval pass. Built only when --eval_every > 0; empty disables eval.
    eval_batches = []
    if args.eval_every > 0:
        eval_batches = build_kto_batches(
            dataset_path=args.dataset_path, tokenizer=tokenizer,
            batch_size=args.batch_size, max_length=args.max_length, split="eval",
            eval_size=args.eval_size, eval_ratio=args.eval_ratio,
            max_rows=args.max_rows, hf_split_name=args.eval_split,
        )
        logger.info("Eval dataset: %d batches", len(eval_batches))

    # --- Reference client (compute_logprobs source) ---------------------
    # An ARN-less sampling client serves the frozen BASE model = the KTO
    # reference. Created here, before any optim_step, so it unambiguously binds
    # base weights. See the engine-mismatch CAVEAT above _score_reference_sums:
    # these inference-path log-probs feed only the detached z_ref baseline and
    # the reference side of the reward, never the policy gradient.
    ref_client = None
    if args.reference_source == "compute_logprobs":
        ref_client = training_client.create_sampling_client()

    # --- Reference log-probs (frozen base model) ------------------------
    # Matched references (train + eval) are frozen; the KL references depend on
    # the source (precompute: fixed pairing, cached; compute_logprobs: scored
    # per step). Everything here runs before the first optim_step.
    if args.reference_source == "compute_logprobs":
        # Score the base model on demand (recomputed cheaply on --resume, so no
        # disk cache is needed). KL references are re-randomized per step.
        logger.info("Scoring reference matched log-probs via compute_logprobs "
                    "(%d train + %d eval batches)", n_batches, len(eval_batches))
        ref_matched = await _matched_reference_sums(
            train_batches, source="compute_logprobs", training_client=training_client,
            ref_client=ref_client, sdk=sdk, watchdog=watchdog, label="matched (train)")
        eval_ref_matched = await _matched_reference_sums(
            eval_batches, source="compute_logprobs", training_client=training_client,
            ref_client=ref_client, sdk=sdk, watchdog=watchdog, label="matched (eval)")
        ref_kl = []  # unused: KL references are scored per step
    else:
        # precompute source: training-forward passes at step 0 (LoRA delta is
        # zero -> base model), cached so a --resume run (adapter no longer zero)
        # can reload them; the KL pairing is fixed on the example.
        cached = _load_reference_cache(runlog.run_folder) if resume_record is not None else None
        if cached is not None:
            ref_matched, ref_kl, eval_ref_matched = cached
            logger.info("Loaded cached reference log-probs for %d batches", len(ref_matched))
        else:
            logger.info("Computing reference (base-model) log-probs for %d train batches%s "
                        "+ %d eval batches", n_batches,
                        " (+KL pairs)" if use_kl else "", len(eval_batches))
            ref_matched, ref_kl = await _compute_reference_logprobs(
                training_client, train_batches, use_kl=use_kl, sdk=sdk, watchdog=watchdog)
            eval_ref_matched = await _matched_reference_sums(
                eval_batches, source="precompute", training_client=training_client,
                ref_client=None, sdk=sdk, watchdog=watchdog, label="matched (eval)")
            _save_reference_cache(runlog.run_folder, ref_matched, ref_kl, eval_ref_matched)

    # --- Step budget ----------------------------------------------------
    total_steps = n_batches * args.num_epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    logger.info("Training for %d steps (lr=%s, schedule=%s, beta=%s)",
                total_steps, args.learning_rate, args.lr_schedule, args.beta)

    ckpt = checkpoint_utils.CheckpointManager(
        training_client, svc, runlog.run_folder, watchdog=watchdog,
        save_every=args.save_every, ttl_seconds=args.checkpoint_ttl_seconds,
        rolling_save_every=args.rolling_save_every,
    )

    step = 0
    if resume_record is not None and resume_record.step is not None:
        step = int(resume_record.step) + 1
        logger.info("Resumed from %r; continuing at step %d", resume_record.name, step)

    for epoch in range(args.num_epochs):
        for batch_idx in range(n_batches):
            if epoch * n_batches + batch_idx < step:
                continue
            if step >= total_steps:
                break

            t0 = time.time()
            lr = learning_rate(args.learning_rate, step, total_steps=total_steps,
                               schedule=args.lr_schedule, warmup_steps=args.warmup_steps)
            batch: list[KtoExample] = train_batches[batch_idx]

            step_metrics = await _kto_step(
                args, training_client, batch,
                ref_matched[batch_idx],
                ref_kl[batch_idx] if (use_kl and args.reference_source == "precompute") else None,
                lr=lr, use_kl=use_kl, sdk=sdk, watchdog=watchdog,
                reference_source=args.reference_source, ref_client=ref_client,
                step_seed=step,
            )

            elapsed = time.time() - t0
            # Identity keys first so metrics.jsonl records read step-first.
            metrics = {
                "step": step,
                "epoch": epoch,
                "progress": (step + 1) / total_steps,
                **step_metrics,
                "config/learning_rate": lr,
                "train/num_examples": len(batch),
                "time/step_total_seconds": round(elapsed, 2),
            }

            logger.info(
                "step=%d/%d epoch=%d kto/loss=%.4f reward_margin=%.4f z_ref=%.4f lr=%.2e time=%.1fs",
                step + 1, total_steps, epoch,
                metrics.get("kto/loss", float("nan")),
                metrics.get("kto/reward_margin", float("nan")),
                metrics.get("kto/z_ref", float("nan")), lr, elapsed,
            )

            # Held-out eval on cadence; merged into this step's record. Uses the
            # step's training z_ref as the accuracy threshold (reward_margin
            # needs none). Forward-only, so it never perturbs the weights.
            if eval_batches and args.eval_every > 0 and step % args.eval_every == 0:
                eval_metrics = await _evaluate_kto(
                    training_client, eval_batches, eval_ref_matched,
                    z_ref=metrics.get("kto/z_ref", 0.0), sdk=sdk, watchdog=watchdog,
                    step=step, run_folder=runlog.run_folder, tokenizer=tokenizer)
                metrics.update(eval_metrics)
                logger.info("  eval reward_margin=%.4f accuracy=%.4f",
                            eval_metrics.get("kto/eval_reward_margin", float("nan")),
                            eval_metrics.get("kto/eval_accuracy", float("nan")))

            runlog.log_metrics(metrics, step=step)
            display_step = step + 1
            if args.log_every > 0 and (display_step % args.log_every == 0 or display_step == total_steps):
                print_kv_table(f"Step {display_step} metrics", format_metric_values(metrics))

            saved = await ckpt.maybe_save_async(step, {"step": step, "epoch": epoch})
            if saved:
                runlog.log_checkpoint(step, saved.get("sampler_path"))

            step += 1

    logger.info("Saving final checkpoint...")
    final_paths = await ckpt.save_final_async({"step": max(step - 1, 0)})
    runlog.log_checkpoint("final", final_paths.get("sampler_path"))
    logger.info("Final checkpoint: %s", final_paths)
    svc.delete_session(session_name, stop_if_active=True)
    logger.info("Training complete. %d steps. Metrics: %s", step, runlog.metrics_path)


async def _kto_step(args, training_client, batch, ref_matched_sums, ref_kl_sums,
                    *, lr, use_kl, sdk, watchdog,
                    reference_source="precompute", ref_client=None, step_seed=0) -> dict:
    """One KTO step: two no-grad forwards -> advantages -> forward_backward + optim.

    Returns the ``kto/*`` metrics for the step. ``ref_matched_sums`` are the
    precomputed reference completion log-prob sums for this batch. For the KL
    term: with ``reference_source='precompute'`` the pairing is fixed and
    ``ref_kl_sums`` supplies the reference side; with ``'compute_logprobs'`` the
    pairing is re-randomized here and both sides are scored on the fly (reference
    via the ARN-less ``ref_client`` - see the engine-mismatch CAVEAT above
    ``_score_reference_sums``).
    """
    # 1. Policy log-probs on the matched pairs (per-token, for old_log_probs).
    #    ALWAYS the training forward - never the inference-path ref_client -
    #    because these drive the actual gradient (see CAVEAT).
    m_slices = [matched_slice(ex) for ex in batch]
    m_datums = [forward_datum(inp, tgt, mask) for inp, tgt, mask in m_slices]
    m_lps = await _forward_logprobs(training_client, m_datums, loss_fn=sdk.fwd_loss,
                                    watchdog=watchdog, what="policy forward (matched)")
    policy_matched_sums = [
        sum_completion_logprob(lp, mask) for lp, (_, _, mask) in zip(m_lps, m_slices)
    ]

    # 2. KL reference point z_ref from the mismatched pairs (optional).
    if not use_kl:
        z_ref = 0.0
    elif reference_source == "compute_logprobs":
        # Re-randomize the KL pairing this step, then score the policy (training
        # forward) and the reference (base client) on the SAME mismatched pairs.
        partners = batch_kl_partners(batch, random.Random(step_seed))
        k_slices = [pair_slice(ex.prompt_tokens, comp) for ex, comp in zip(batch, partners)]
        k_datums = [forward_datum(inp, tgt, mask) for inp, tgt, mask in k_slices]
        k_lps = await _forward_logprobs(training_client, k_datums, loss_fn=sdk.fwd_loss,
                                        watchdog=watchdog, what="policy forward (kl)")
        policy_kl_sums = [
            sum_completion_logprob(lp, mask) for lp, (_, _, mask) in zip(k_lps, k_slices)
        ]
        ref_kl_step = await _score_reference_sums(
            ref_client, [(ex.prompt_tokens, comp) for ex, comp in zip(batch, partners)],
            watchdog=watchdog, what="reference scoring (kl)")
        z_ref = kl_reference_point(policy_kl_sums, ref_kl_step)
    else:
        # precompute source: fixed pairing on the example; reference precomputed.
        k_slices = [kl_slice(ex) for ex in batch]
        k_datums = [forward_datum(inp, tgt, mask) for inp, tgt, mask in k_slices]
        k_lps = await _forward_logprobs(training_client, k_datums, loss_fn=sdk.fwd_loss,
                                        watchdog=watchdog, what="policy forward (kl)")
        policy_kl_sums = [
            sum_completion_logprob(lp, mask) for lp, (_, _, mask) in zip(k_lps, k_slices)
        ]
        z_ref = kl_reference_point(policy_kl_sums, ref_kl_sums)

    # 3. Per-example KTO term -> per-token advantage (-s_i) -> train datum.
    train_datums = []
    losses, rewards_d, rewards_u, logps_d, logps_u, correct = [], [], [], [], [], 0
    for ex, (inp, tgt, mask), policy_sum, ref_sum, per_token in zip(
        batch, m_slices, policy_matched_sums, ref_matched_sums, m_lps
    ):
        term = kto_term(
            policy_sum, ref_sum, desirable=ex.label, z_ref=z_ref, beta=args.beta,
            desirable_weight=args.desirable_weight,
            undesirable_weight=args.undesirable_weight,
        )
        advantage = advantage_from_grad(term.grad_wrt_logp)
        train_datums.append(train_datum(inp, tgt, mask, advantage=advantage,
                                        old_logprobs=per_token))
        losses.append(term.loss)
        (rewards_d if ex.label else rewards_u).append(term.reward)
        # Mean completion log-prob by class (TRL's logps/chosen, logps/rejected):
        # a crashing value here is the earliest sign the policy is collapsing.
        (logps_d if ex.label else logps_u).append(policy_sum)
        # "Correct" = desirable reward above z_ref, undesirable below it.
        if (ex.label and term.reward > z_ref) or (not ex.label and term.reward < z_ref):
            correct += 1

    # 4. Surrogate forward_backward + optimizer step (ratio 1 -> exact gradient).
    fb_op = await training_client.forward_backward_async(
        train_datums, loss_fn=sdk.is_loss)
    opt_op = await training_client.optim_step_async(sdk.AdamParams(
        learning_rate=lr, beta1=args.adam_beta1, beta2=args.adam_beta2, eps=args.adam_eps))
    fb_result = await watchdog.guard(fb_op.result_async(), what="forward_backward")
    opt_result = await watchdog.guard(opt_op.result_async(), what="optim_step")

    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    metrics = {
        "kto/loss": mean(losses),
        "kto/z_ref": z_ref,
        "kto/reward_desirable": mean(rewards_d),
        "kto/reward_undesirable": mean(rewards_u),
        "kto/reward_margin": mean(rewards_d) - mean(rewards_u) if rewards_d and rewards_u else float("nan"),
        "kto/logps_desirable": mean(logps_d),
        "kto/logps_undesirable": mean(logps_u),
        "kto/accuracy": correct / len(batch),
        "kto/frac_desirable": len(rewards_d) / len(batch),
    }
    for result in (fb_result, opt_result):
        if getattr(result, "metrics", None):
            metrics.update({f"train/{k}": v for k, v in result.metrics.items()})
    return metrics


def run(args, *, train_batches=None) -> None:
    """Synchronous wrapper around :func:`run_async` for scripts and the CLI."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run() cannot be called from a running event loop (e.g. a Jupyter "
            "notebook). Use `await run_async(...)` instead."
        )
    asyncio.run(run_async(args, train_batches=train_batches))


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
