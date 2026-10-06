"""On-policy distillation (OPD / OPSD / SDPO) over the RL training loop.

Distillation reuses the RL loop end to end: the student samples rollouts,
``forward_backward(importance_sampling)`` + ``optim_step`` update it, and
checkpoints, validation and metrics work unchanged. The one difference is the
advantage stage: instead of GRPO's one scalar per episode, every sampled token
gets ``A_t = log p_teacher(y_t) - log q_student(y_t)`` from a teacher that
re-reads the rollout (see :mod:`.losses` for why this is the reverse-KL
gradient). The recipes pick the teacher and what it reads:

=========  ======================  =================================  ===========
method     default ``--teacher``   teacher context                    recipe
=========  ======================  =================================  ===========
``opd``    ``session``             the student's prompt               opd_math
``opsd``   ``base``                prompt + reference solution        opsd_math
``sdpo``   ``base``                prompt + feedback / sibling success sdpo_code
=========  ======================  =================================  ===========

Programmatic use mirrors the RL recipes::

    from training_sessions_cookbook.distill.train import build_parser, parse_with_config, run_async

    args = parse_with_config(build_parser("opd", envs=("gsm8k",)), argv=[...])
    await run_async(args)
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from training_sessions_cookbook.rl import train as rl_train
from training_sessions_cookbook.rl.config import cfg

from .advantages import DistillAdvantageFn
from .context import (
    OPSD_TEMPLATE,
    SDPO_TEMPLATE,
    FeedbackContext,
    PrivilegedContext,
    SameContext,
    code_solution,
    sandbox_feedback,
    text_solution,
)
from .teacher import TEACHER_SOURCES, make_teacher

logger = logging.getLogger(__name__)

METHODS = ("opd", "opsd", "sdpo")
_DEFAULT_TEACHER = {"opd": "session", "opsd": "base", "sdpo": "base"}


def add_distill_args(parser: argparse.ArgumentParser, method: str) -> None:
    """Add the ``distillation`` flag group, defaulted for ``method``."""
    if method not in METHODS:
        raise ValueError(f"unknown distillation method {method!r}; expected one of {METHODS}")
    g = parser.add_argument_group("distillation")
    g.add_argument("--distill_method", choices=METHODS, default=method,
                   help="Selects the teacher context: 'opd' = the student's prompt; "
                        "'opsd' = prompt + privileged reference solution; 'sdpo' = "
                        "prompt + feedback on the attempt / a successful sibling.")
    g.add_argument("--teacher", choices=TEACHER_SOURCES, default=_DEFAULT_TEACHER[method],
                   help="Where teacher log-probs come from: 'base' = the student's "
                        "frozen base (same session); 'checkpoint' = a saved adapter "
                        "of the student's base (--teacher_model_package_arn); "
                        "'lagged' = the student, re-snapshotted every "
                        "--teacher_refresh_every steps; 'session' = a different base "
                        "model (--teacher_base_model_arn). All but 'base' open a "
                        "second, sampling-only session.")
    g.add_argument("--teacher_base_model_arn", default=None,
                   help="--teacher session: hub-content ARN of the teacher's base "
                        "model. Must share the student's tokenizer (same family).")
    g.add_argument("--teacher_model_package_arn", default=None,
                   help="--teacher checkpoint: model package ARN of a saved adapter "
                        "of the student's base model.")
    g.add_argument("--teacher_lora_rank", type=int, default=8,
                   help="LoRA rank to open a teacher session with (it is never "
                        "trained). Ignored for --teacher base.")
    g.add_argument("--teacher_refresh_every", type=int, default=10,
                   help="--teacher lagged: steps between teacher snapshots of the "
                        "live policy (1 = the current policy, which SDPO reports "
                        "diverges).")
    g.add_argument("--teacher_max_concurrency", type=int, default=32,
                   help="Max in-flight teacher compute_logprobs requests.")
    g.add_argument("--allow_tokenizer_mismatch", action="store_true",
                   help="--teacher session: proceed even if the teacher's tokenizer "
                        "differs from the student's (the signal is then meaningless "
                        "for any token the two encode differently).")
    g.add_argument("--adv_clip", type=float, default=None,
                   help="Clip each per-token advantage to [-c, c]. Unset = unclipped.")
    g.add_argument("--teacher_temperature", type=float, default=1.0,
                   help="Grade against the teacher tempered to T (its log-probs "
                        "scaled by 1/T; the normalizer is a per-position baseline "
                        "and drops out of the expected gradient). T < 1 sharpens "
                        "the teacher, the only signal when the teacher is the "
                        "student's own model on the same prompt. Must be > 0: "
                        "T -> 0 gives -inf to every non-argmax token.")
    g.add_argument("--alpha_task", type=float, default=0.0,
                   help="Weight of the GRPO task-reward advantage added to every "
                        "token's distillation advantage (0 = pure distillation).")
    g.add_argument("--logprob_check_samples", type=int, default=8,
                   help="On the first step, re-score this many rollouts with the "
                        "student's own sampler via compute_logprobs and report "
                        "distill/logprob_check_abs_diff (should be ~0). 0 disables.")
    # OPSD
    g.add_argument("--privileged_field", default="answer",
                   help="opsd: dataset-row field holding the privileged context "
                        "(GSM8K: 'answer', the worked solution).")
    g.add_argument("--privileged_mode", choices=["solution", "final_answer"],
                   default="solution",
                   help="opsd: pass the full field ('solution') or only the text "
                        "after '####' ('final_answer').")
    # SDPO
    g.add_argument("--sdpo_sibling_solutions", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="sdpo: show the teacher a successful sibling rollout's "
                        "solution (needs --responses_per_prompt > 1).")
    g.add_argument("--sdpo_env_feedback", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="sdpo: show the teacher the environment's feedback on a "
                        "failed attempt (code_rl: sandbox test output).")
    g.add_argument("--feedback_max_chars", type=int, default=2000,
                   help="sdpo: truncate environment feedback to this many characters.")
    g.add_argument("--teacher_context_template", default=None,
                   help="Override the teacher prompt template (opsd placeholders: "
                        "{observation}, {privileged}; sdpo: {observation}, "
                        "{solution_section}, {feedback_section}).")


def build_parser(method: str, envs: tuple[str, ...] | None = None) -> argparse.ArgumentParser:
    """The RL parser plus the ``distillation`` group, defaulted for ``method``."""
    parser = rl_train.build_parser(envs=envs)
    parser.description = f"On-policy distillation ({method.upper()}) on SageMaker Training Sessions."
    add_distill_args(parser, method)
    return parser


def parse_with_config(parser: argparse.ArgumentParser, argv=None):
    """Parse with ``--config`` support, then validate the distillation flags."""
    args = rl_train.parse_with_config(parser, argv)
    validate_distill_args(args, parser)
    return args


def validate_distill_args(args, parser: argparse.ArgumentParser | None = None) -> None:
    """Reject incompatible flag combinations and warn on biased settings."""
    def fail(msg):
        if parser is not None:
            parser.error(msg)
        raise SystemExit(msg)

    if args.teacher == "session" and not args.teacher_base_model_arn:
        fail("--teacher session requires --teacher_base_model_arn")
    if args.teacher == "checkpoint" and not args.teacher_model_package_arn:
        fail("--teacher checkpoint requires --teacher_model_package_arn")
    if args.teacher_temperature <= 0:
        fail("--teacher_temperature must be > 0 (T -> 0 assigns -inf to every "
             "token but the teacher's argmax); use e.g. 0.5 to sharpen")
    if (args.distill_method == "opd" and args.teacher in ("base", "lagged")
            and args.teacher_temperature == 1.0 and args.alpha_task == 0.0):
        logger.warning(
            "OPD with --teacher %s at --teacher_temperature 1.0: the teacher is "
            "the student's own model on the same prompt, so the advantage is ~0 "
            "(no signal beyond a pull back toward the base). Set "
            "--teacher_temperature < 1, --alpha_task > 0, or use a stronger "
            "--teacher checkpoint.", args.teacher)
    if (args.teacher == "session" and args.teacher_base_model_arn
            and args.teacher_base_model_arn == getattr(args, "base_model_arn", None)):
        logger.warning("--teacher session on the student's own base model opens a "
                       "second session for weights --teacher base already serves; "
                       "use --teacher base.")
    if args.advantage_mode != "grouped":
        fail("distillation requires --advantage_mode grouped")
    if (args.distill_method == "sdpo" and args.sdpo_sibling_solutions
            and not args.sdpo_env_feedback and args.responses_per_prompt < 2):
        fail("sdpo with only sibling solutions needs --responses_per_prompt >= 2")
    if args.temperature != 1.0 or cfg(args, "top_p") is not None or cfg(args, "top_k") is not None:
        logger.warning(
            "Distilling with --temperature %s / --top_p %s / --top_k %s: the "
            "advantage uses the sampler's log-probs, which are then those of the "
            "processed (tempered/truncated) distribution, not the student's. "
            "Use --temperature 1.0 with --top_p/--top_k unset for an unbiased "
            "reverse-KL signal.", args.temperature, cfg(args, "top_p"), cfg(args, "top_k"))
    if args.distill_method == "opsd" and args.teacher in ("session", "checkpoint"):
        logger.info("OPSD with --teacher %s: privileged context is shown to a "
                    "non-self teacher (a hybrid of OPD and OPSD).", args.teacher)


def build_context(args, *, tokenizer, template_fn):
    """The teacher context for ``args.distill_method``."""
    method = args.distill_method
    template = args.teacher_context_template
    if method == "opd":
        return SameContext(tokenizer=tokenizer, template_fn=template_fn)
    if method == "opsd":
        return PrivilegedContext(
            tokenizer=tokenizer, template_fn=template_fn,
            field=args.privileged_field, mode=args.privileged_mode,
            template=template or OPSD_TEMPLATE,
        )
    is_code = args.env_type == "code_rl"
    max_chars = args.feedback_max_chars
    return FeedbackContext(
        tokenizer=tokenizer, template_fn=template_fn,
        feedback_fn=lambda info: sandbox_feedback(info, max_chars=max_chars),
        solution_fn=code_solution if is_code else text_solution,
        use_sibling_solutions=args.sdpo_sibling_solutions,
        use_env_feedback=args.sdpo_env_feedback,
        template=template or SDPO_TEMPLATE,
    )


async def make_advantage_fn(*, args, backend, tokenizer, template_fn) -> DistillAdvantageFn:
    """``advantage_fn_factory`` for :func:`rl.train.run_async`: open the teacher, wire the context."""
    if args.distill_method == "opsd" and args.env_type == "gsm8k" and not cfg(args, "dataset_path"):
        logger.warning("OPSD on the builtin GSM8K subset: its 'answer' field is the "
                       "bare number, so the privileged context is only the final "
                       "answer. Use data/gsm8k_train.jsonl (tools/download_gsm8k.py) "
                       "for worked solutions.")
    teacher = await make_teacher(args, backend, tokenizer)
    try:
        context = build_context(args, tokenizer=tokenizer, template_fn=template_fn)
    except BaseException:
        teacher.close()
        raise
    logger.info("Distillation: method=%s teacher=%s teacher_temperature=%s "
                "adv_clip=%s alpha_task=%s", args.distill_method, teacher.name,
                args.teacher_temperature, args.adv_clip, args.alpha_task)
    return DistillAdvantageFn(
        teacher=teacher, context=context, adv_clip=args.adv_clip,
        alpha_task=args.alpha_task, teacher_temperature=args.teacher_temperature,
        logprob_check_samples=args.logprob_check_samples,
    )


async def run_async(args, *, env_factory=None, validation_runner=None) -> None:
    """Drive one distillation run (the RL loop with the teacher-scoring stage)."""
    await rl_train.run_async(
        args, env_factory=env_factory, validation_runner=validation_runner,
        advantage_fn_factory=make_advantage_fn,
    )


def run(args, *, env_factory=None, validation_runner=None) -> None:
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
    asyncio.run(run_async(args, env_factory=env_factory, validation_runner=validation_runner))


def main(method: str, envs: tuple[str, ...] | None = None) -> None:
    run(parse_with_config(build_parser(method, envs=envs)))
