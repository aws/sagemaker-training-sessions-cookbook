from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import gc
import json
import logging
import os
import time
import warnings
import psutil

from training_sessions_cookbook.service import create_run_folder, dump_config, setup_file_logging  # noqa: F401  # re-exported
from training_sessions_cookbook.service import git_hash as _get_git_hash  # noqa: F401  # re-exported
from training_sessions_cookbook.service import learning_rate as _learning_rate
from training_sessions_cookbook.service import checkpoint_utils
from training_sessions_cookbook.service.console import (
    console_print,
    print_run_banner,
    setup_console_logging,
)
from training_sessions_cookbook.service.mlflow_log import add_mlflow_args
from training_sessions_cookbook.service.runlog import RunLog

from .envs import env_cli_args, get_env_factory, registered_envs
from .backends import make_backend, TrainingBackend  # TrainingBackend: type hints

# The concerns below were extracted into sibling modules; re-import every
# moved name so the loop bodies here (and existing imports of rl.train)
# keep working unchanged.
from .metrics import MetricsLogger, _METRICS_TABLE_HEADER, capture_rollouts, compute_advantage_stats, compute_episode_length_mean, compute_group_health_stats, compute_reward_stats, count_episodes, episodes_to_rollout_dicts, format_metrics_table, MetricsTablePrinter, record_trajectories  # noqa: F401
from .validation import ValidationExample, ValidationRunner, compute_validation_accuracy  # noqa: F401
from .dataset import _EpochIter, _build_val_dataset, load_dataset, make_epoch_iter, next_batch  # noqa: F401
from .advantages import _ADV_EPS, compute_grouped_advantages, compute_monte_carlo_advantages  # noqa: F401
from .prompts import TEMPLATE_FACTORY, _apply_chat_template_progressive, _factory_needs_tokenizer, _normalize_chat_template_output, _reasoning_to_kwargs, apply_auto, apply_gemma, apply_gpt_oss, apply_hf, apply_no, apply_qwen3_game, apply_qwen3_general, encode_prompt  # noqa: F401
from .rollouts import Transition, WrappedGroup, collect_batch_async, collect_grouped_batch_async, collect_one_group_async, rollout_one, rollout_one_code  # noqa: F401
from .staleness import _AsyncLoopState, _PROMPT_SENTINEL, _StalenessBatcher, _WORKER_DONE  # noqa: F401


# === Run folder management ===
#
# Run folder, config dump, and file logging are shared with the SFT path in ``service/runlog.py``; the aliases below preserve the names this module and its tests reference.


def compute_learning_rate(step: int, base_lr: float, warmup_steps: int) -> float:
    """Effective LR for ``step`` with optional linear warmup (no decay)."""
    return _learning_rate(base_lr, step, warmup_steps=warmup_steps)




def _apply_config_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Seed ``parser`` defaults from a JSON config so CLI flags still override.

    Reads the JSON at ``config_path`` (the shape ``dump_config`` writes) and
    for every key that matches a known argparse destination, calls
    ``parser.set_defaults(dest=value)``. Because argparse applies an explicitly
    supplied flag *over* its default, any flag on the command line still wins, the config only fills in the rest. Two consequences handled here:

    * **Required args become satisfiable from the file.** ``--base_model`` is
      ``required=True``; once the config supplies it, we clear ``required`` on
      that action so a config-only invocation (no ``--base_model`` on the CLI)
      does not error. If neither the config nor the CLI provides it, argparse
      still errors as before.
    * **Unknown keys are ignored** (with one warning listing them), so a real
      ``config.json``, which carries ``git_hash`` and any future non-arg keys, loads cleanly.

    ``--config`` itself and the meta key ``git_hash`` are never treated as args.
    """
    log = logging.getLogger(__name__)
    with open(config_path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"--config {config_path!r} must be a JSON object")

    # Map every known destination -> its Action, so we can both set defaults and
    # relax `required` for keys the config provides.
    dest_to_action = {a.dest: a for a in parser._actions}
    ignored = {"config", "git_hash"}

    applied, unknown = {}, []
    for key, value in cfg.items():
        if key in ignored:
            continue
        if key in dest_to_action:
            applied[key] = value
            action = dest_to_action[key]
            if getattr(action, "required", False):
                action.required = False
        else:
            unknown.append(key)

    if applied:
        parser.set_defaults(**applied)
    if unknown:
        log.warning(
            "--config %s: ignoring %d key(s) with no matching CLI flag: %s",
            config_path, len(unknown), ", ".join(sorted(unknown)),
        )


def build_parser(
    default_env: str | None = None,
    envs: tuple[str, ...] | None = None,
) -> argparse.ArgumentParser:
    """Build the training CLI parser.

    ``envs`` selects which registered environments' flags to include --
    recipe entry points pass their own env so ``--help`` shows only relevant
    flags; ``None`` includes every registered environment (the ``rl.train``
    default, and how custom envs added via ``envs.register_env`` surface
    their flags). ``default_env`` presets ``--env_type`` (an explicit flag
    still wins); when ``envs`` names exactly one environment it defaults to
    that one, so single-env recipes only pass ``envs``.

    Mutually-exclusive boolean pairs (`--use_rebn`/`--no_rebn` and `--drop_last`/`--no_drop_last`) use `argparse.BooleanOptionalAction` so the negated forms come for free.
    """
    if default_env is None and envs is not None and len(envs) == 1:
        default_env = envs[0]
    parser = argparse.ArgumentParser(
        description="RL training cookbook (GSM8K / Code RL on SageMaker Training Sessions).",
    )

    # --- Config file -----------------------------------------------------
    # A JSON file (the exact shape dump_config writes to a run's config.json)
    # whose keys seed the argparse DEFAULTS. Any flag passed explicitly on the
    # command line still wins over the file, so `--config foo.json
    # --max_steps_off_policy 2` reuses foo.json's hyperparameters while
    # overriding just that one. Unknown keys (e.g. git_hash) are ignored with
    # a warning. Applied in a two-pass parse below (see _apply_config_defaults).
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON config (dump_config/config.json shape) whose keys "
             "become argparse defaults; explicit CLI flags override it.",
    )

    # --- Backend group ---------------------------------------------------
    # The training loop drives every backend through the TrainingBackend
    # protocol (see rl/backends/base.py); the cookbook ships the `service`
    # backend, which runs on the SageMaker Training Sessions PySDK and reads
    # --role_arn / --model_package_group_arn / --output_s3_path / --base_model_arn.
    bkd = parser.add_argument_group("backend")
    bkd.add_argument(
        "--backend",
        choices=["service"],
        default="service",
        help="Training backend. `service` uses the SageMaker Training Sessions "
             "PySDK; reads --role_arn / --model_package_group_arn / "
             "--output_s3_path / --base_model_arn.",
    )

    # --- Model group -----------------------------------------------------
    mdl = parser.add_argument_group("model")
    mdl.add_argument("--base_model", required=True)
    mdl.add_argument("--lora_rank", type=int, default=32)
    mdl.add_argument("--lora_alpha", type=int, default=64)

    # --- Service group ---------------------------------------------------
    # These mirror the SageMaker Training Sessions PySDK inputs. ServiceBackend.create() validates that the required ones are present.
    svc = parser.add_argument_group("service")
    svc.add_argument("--region", default="us-west-2",
                     help="AWS region for the service backend (default us-west-2).")
    svc.add_argument("--base_model_arn", default=None,
                     help="JumpStart hub-content ARN of the base model (service backend).")
    svc.add_argument("--role_arn", default=None,
                     help="IAM execution role the training session assumes (service backend).")
    svc.add_argument("--model_package_group_arn", default=None,
                     help="Model package group for checkpoints (service backend).")
    svc.add_argument("--output_s3_path", default=None,
                     help="S3 path for model/checkpoint outputs (service backend).")
    svc.add_argument("--training_session_name", default=None,
                     help="Custom training session name (service backend; default auto-generated).")
    svc.add_argument(
        "--save_weights_at_end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Service backend: also save an inference-deployable (merged LoRA) "
             "checkpoint at the end so it can be deployed to a SageMaker endpoint.",
    )
    svc.add_argument(
        "--deploy_and_invoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Service backend: after training, deploy the trained adapter to a "
             "real-time SageMaker endpoint and invoke it on a few held-out "
             "prompts (train -> checkpoint -> deploy -> invoke). Implies "
             "--save_weights_at_end.",
    )
    svc.add_argument("--inference_role_arn", default=None,
                     help="IAM role for the inference endpoint (service --deploy_and_invoke).")

    # --- Training group --------------------------------------------------
    trn = parser.add_argument_group("training")
    trn.add_argument("--batch_size", type=int, default=8)
    trn.add_argument("--learning_rate", type=float, default=4e-5)
    trn.add_argument("--max_tokens", type=int, default=2048,
                     help="Max sampled tokens per response. The service "
                          "requires at least 512; smaller values fail "
                          "request validation on every sample.")
    trn.add_argument("--max_steps", type=int, default=100)
    trn.add_argument("--save_every", type=int, default=None,
                     help="Save a full checkpoint (state + sampler weights) "
                          "every N steps, recorded in checkpoints.jsonl. "
                          "Default: max_steps, i.e. only the final save.")
    trn.add_argument("--gamma", type=float, default=0.9)
    trn.add_argument(
        "--use_rebn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply ReB normalization to advantages (use --no_rebn to disable).",
    )
    trn.add_argument(
        "--loss",
        choices=["ppo", "importance_sampling", "cispo"],
        default="importance_sampling",
    )
    trn.add_argument("--refresh_every", type=int, default=1)
    trn.add_argument("--temperature", type=float, default=1.0)

    # IS-ratio clip thresholds for ppo/cispo, expressed as ABSOLUTE ratio
    # bounds (clamp the IS ratio π_current/π_behavior into
    # [clip_low_threshold, clip_high_threshold]). Both default to None → the
    # SDK loss uses its own built-in clip (behavior byte-identical to before).
    # The service converts these to verl epsilons internally
    # (clip_ratio_low = 1 - low, clip_ratio_high = high - 1), so e.g.
    # --clip_low_threshold 1.0 --clip_high_threshold 6.0 clamps the ratio to
    # [1, 6] (a wide asymmetric upper clip, useful for cispo under off-policy
    # staleness). Only forwarded when set; ppo also honors them.
    trn.add_argument(
        "--clip_low_threshold",
        type=float,
        default=None,
        help="Lower absolute bound on the IS ratio for ppo/cispo "
             "(maps to clip_ratio_low = 1 - this). Unset = SDK loss default.",
    )
    trn.add_argument(
        "--clip_high_threshold",
        type=float,
        default=None,
        help="Upper absolute bound on the IS ratio for ppo/cispo "
             "(maps to clip_ratio_high = this - 1). Unset = SDK loss default.",
    )
    # --- sampling tail / stop knobs ---------------------------------------
    # By default a rollout sends ONLY max_tokens + temperature, so the
    # inference server falls back to its own server-side defaults for nucleus /
    # top-k / stop. These flags let a run pin all three explicitly so sampling
    # is fully reproducible. Unset (None) = the server's historical default,
    # byte-for-byte (the service client leaves top_p/top_k/stop=None so they are
    # omitted from the wire payload).
    trn.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling probability. Unset = server default (the service "
             "omits it and inherits the server default). Set explicitly "
             "(e.g. 1.0) to pin it.",
    )
    trn.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling cutoff (-1 = no limit). Unset = server default "
             "(the service omits it). Set explicitly to pin it.",
    )
    trn.add_argument(
        "--stop",
        type=str,
        default=None,
        help="Comma-separated stop strings passed to the sampler. Unset = "
             "server default (the server recognizes gpt-oss harmony end tokens "
             "natively, or not). Set (e.g. '<|return|>,<|call|>,<|end|>') "
             "to pin explicit stop sequences.",
    )
    #   --sampling_seed: base seed for reproducible generation (the wire API's
    #   samplingParams.seed). Distinct per rollout/step (GRPO groups stay
    #   diverse) but identical across runs sharing the flag. Default None
    #   preserves historical unseeded sampling. Dropped with a warning on SDK
    #   builds whose SamplingParams lacks seed.
    trn.add_argument(
        "--sampling_seed",
        type=int,
        default=None,
        help="Base seed for reproducible rollout generation (see file header).",
    )
    #   --max_steps_off_policy: the off-by-k switch. 0 (default) = the strict
    #   serial run_training loop, byte-for-byte unchanged. >0 = the overlapped
    #   run_training_async loop where rollout workers generate continuously
    #   while the trainer trains; a rollout group started against sampler
    #   version t may be consumed up to this many gradient steps later (t+k),
    #   its distribution shift corrected by the loss's IS ratio. Groups staler
    #   than the cap are dropped. Requires --advantage_mode grouped.
    trn.add_argument(
        "--max_steps_off_policy",
        type=int,
        default=0,
        help="0 (default) runs the strict serial loop unchanged. >0 enables "
             "the off-by-k async loop with this staleness cap (see file "
             "header).",
    )
    trn.add_argument(
        "--stale_policy",
        choices=["drop", "requeue"],
        default="drop",
        help="How to handle groups whose staleness exceeds "
             "--max_steps_off_policy. 'drop' (default) discards the group "
             "(prompt is lost from epoch coverage). 'requeue' pushes the "
             "prompt back onto the prompt queue for a fresh resample, "
             "preserving epoch coverage at the cost of extra rollout compute. "
             "Silently accepted on on-policy runs (--max_steps_off_policy 0).",
    )

    # --- Rollout group ---------------------------------------------------
    rol = parser.add_argument_group("rollout")
    rol.add_argument("--num_rollout_workers", type=int, default=4)
    rol.add_argument("--max_turns", type=int, default=5)
    rol.add_argument(
        "--template",
        choices=["qwen3_general", "qwen3_game", "gpt_oss", "gemma", "hf", "auto", "no"],
        default="qwen3_general",
        help="Chat-template selector. `auto` resolves by substring match on "
             "--base_model (qwen→qwen3_general, gpt-oss→gpt_oss, gemma→gemma, "
             "else→qwen3_general). `hf` forces tokenizer.apply_chat_template() "
             "with the model-shipped jinja regardless of model, useful for "
             "running gpt-oss / gemma / qwen with their official prompt "
             "format including system framing.",
    )
    rol.add_argument(
        "--advantage_mode",
        choices=["grouped", "monte_carlo"],
        default="grouped",
    )
    rol.add_argument("--responses_per_prompt", type=int, default=4)
    # R9: client-side cap on in-flight collect_one_group_async awaits. See
    # Off-by-k with the default flag layout can leave up to `(k+1)*workers*K`
    # rollout futures alive in Python at once, which can drive the launcher
    # OOM. Default of `num_rollout_workers` is a semantic no-op (semaphore
    # never blocks) but keeps the acquire/release code path uniform.
    rol.add_argument(
        "--max_inflight_rollouts",
        type=int,
        default=None,
        help="Client-side cap on live collect_one_group_async awaits inside "
             "the off-by-k trainer's rollout workers. Default: value of "
             "--num_rollout_workers (a no-op cap that only serves to keep "
             "the semaphore code path uniform). Must be a positive integer. "
             "Silently ignored on on-policy runs (--max_steps_off_policy 0).",
    )
    rol.add_argument(
        "--reasoning",
        choices=["off", "low", "medium", "high", "auto"],
        default="auto",
        help="Reasoning / thinking knob for chat templates that support it. "
             "`off`→no thinking (gemma: enable_thinking=False; gpt-oss: "
             "Reasoning=low). `low|medium|high` set the corresponding level. "
             "`auto` keeps each template's own default (gpt-oss legacy: high; "
             "gpt-oss hf: medium; gemma: thinking on). Silently ignored for "
             "templates without a reasoning concept (qwen3, qwen3_game, no).",
    )

    # --- Dataset group ---------------------------------------------------
    dsg = parser.add_argument_group("dataset")
    dsg.add_argument("--dataset_path", default=None)
    dsg.add_argument("--max_prompts", type=int, default=None)
    dsg.add_argument("--epochs", type=int, default=1)
    dsg.add_argument(
        "--drop_last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the trailing partial batch (use --no_drop_last to keep it).",
    )

    # --- Logging group ---------------------------------------------------
    lgg = parser.add_argument_group("logging")
    lgg.add_argument("--log_dir", default="./logs")
    lgg.add_argument(
        "--run_name",
        default=None,
        help="Run identifier used for the metrics filename and snapshot path. "
             "Defaults to rl-train-{int(time.time())} when unset.",
    )


    # --- Hyperparameters group -------------------------------------------
    hyp = parser.add_argument_group("hyperparameters")
    hyp.add_argument("--kl_coef", type=float, default=0.0)
    # DEPRECATED / NO-OP: never read by the training path. Use
    # --clip_low_threshold / --clip_high_threshold (in the training group) to
    # set the IS-ratio clip for ppo/cispo. Kept so old configs/scripts that
    # pass it still parse.
    hyp.add_argument("--clip_range", type=float, default=0.2,
                     help="DEPRECATED no-op; use --clip_low_threshold / "
                          "--clip_high_threshold instead.")
    hyp.add_argument("--warmup_steps", type=int, default=0)
    hyp.add_argument("--entropy_coef", type=float, default=0.0)
    hyp.add_argument("--max_grad_norm", type=float, default=1.0)
    hyp.add_argument("--adam_beta1", type=float, default=0.9)
    hyp.add_argument("--adam_beta2", type=float, default=0.95)
    hyp.add_argument("--adam_epsilon", type=float, default=1e-8)

    # --- Environment group -----------------------------------------------
    # Each environment owns its CLI flags (envs/__init__.py registry); only
    # the selected environments' flags are attached, so a recipe entry's
    # --help shows exactly its surface.
    selected = tuple(envs) if envs is not None else registered_envs()
    env = parser.add_argument_group("environment")
    env.add_argument("--env_type", choices=list(selected),
                     default=default_env or selected[0])
    for name in selected:
        hook = env_cli_args(name)
        if hook is not None:
            hook(parser)

    # --- Output group ----------------------------------------------------
    out = parser.add_argument_group("output")
    out.add_argument("--output_dir", default=None,
                     help="Override auto-generated run folder path")
    out.add_argument("--log_every", type=int, default=10,
                     help="Print the full metric table every N steps (0 = "
                          "never). The compact one-line row prints every "
                          "step regardless.")
    out.add_argument(
        "--record_trajectories",
        action="store_true",
        help="Append every training rollout (decoded prompt/response, reward, "
             "advantage, per-turn) to {run_folder}/trajectories.jsonl, one JSON "
             "object per transition per step. Off by default (keeps metrics.jsonl "
             "schema stable and avoids large files on long runs).",
    )

    # --- MLflow group ------------------------------------------------------
    # Optional publishing of per-step metrics to MLflow (a SageMaker managed
    # MLflow App by ARN, or any MLflow tracking URI). Best-effort: never
    # fails the run; metrics.jsonl is unaffected.
    mlf = parser.add_argument_group("mlflow")
    add_mlflow_args(mlf)

    # --- Validation group ------------------------------------------------
    val = parser.add_argument_group("validation")
    val.add_argument("--val_dataset_path", default=None,
                     help="Path to a JSONL or Parquet validation dataset")
    val.add_argument(
        "--val_code_split",
        choices=["none", "test", "train"],
        default="none",
        help="Code RL only: validate on the REAL DeepCoder split graded with "
             "the same SandboxFusion harness as training. 'test' uses the "
             "held-out codeforces+lcbv5 split (recommended); 'train' re-scores "
             "training tasks. 'none' (default) keeps the --val_dataset_path "
             "file path. Takes precedence over --val_dataset_path for code_rl. "
             "Cap the number of problems with --val_max_prompts.",
    )
    val.add_argument("--val_every", type=int, default=50,
                     help="Run validation every N training steps")
    val.add_argument("--val_examples", type=int, default=5,
                     help="Number of qualitative examples to print per validation")
    val.add_argument("--val_max_prompts", type=int, default=None,
                     help="Cap the validation dataset to the first N prompts "
                          "(deterministic head-slice). Defaults to using the "
                          "full file. Use e.g. 50 to keep periodic validation "
                          "fast on a large held-out set.")
    val.add_argument(
        "--val_at_start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one validation pass on the INITIAL (untrained) weights "
             "before the first training step, to record a baseline. Use "
             "--no-val_at_start to skip. Only fires when a validation set is "
             "configured (--val_dataset_path or --val_code_split).",
    )

    # Two-pass parse for --config: read the config path with a minimal side
    # parser FIRST (a plain parse_known_args() on the full parser would trip its
    return parser


def parse_with_config(parser: argparse.ArgumentParser, argv=None):
    """Parse ``argv`` (default: ``sys.argv``) with ``--config`` file support,
    then apply the post-parse defaulting the loops rely on. Returns ``args``
    ready for :func:`run`.
    """
    # Two-pass parse: sniff --config first (argparse cannot know about
    # required=True args before we can satisfy them from the file), fold the
    # config's keys into `parser` as defaults, then do the real parse so
    # explicit CLI flags still override the file. When --config is absent this
    # is just a normal parse_args().
    config_sniffer = argparse.ArgumentParser(add_help=False)
    config_sniffer.add_argument("--config", default=None)
    pre_args, _ = config_sniffer.parse_known_args(argv)
    if pre_args.config is not None:
        _apply_config_defaults(parser, pre_args.config)

    args = parser.parse_args(argv)

    # --- Post-parse defaulting ------------------------------------------
    # Auto-name the run when --run_name was not provided so the metrics file and snapshot path stay unique across back-to-back invocations.
    if args.run_name is None:
        args.run_name = f"rl-train-{int(time.time())}"
    # save_every defaults to max_steps so the only save is the final one below; users can override to checkpoint mid-run.
    if args.save_every is None:
        args.save_every = args.max_steps

    # R9: resolve --max_inflight_rollouts default (num_rollout_workers) and
    # validate positivity. Silently accepted on on-policy runs: run_training
    # never reads it.
    if getattr(args, "max_inflight_rollouts", None) is None:
        args.max_inflight_rollouts = args.num_rollout_workers
    if args.max_inflight_rollouts <= 0:
        parser.error(
            "--max_inflight_rollouts must be a positive integer "
            f"(got {args.max_inflight_rollouts})"
        )
    return args


async def run_async(args, *, env_factory=None, validation_runner=None) -> None:
    """Drive one full training run from a parsed/constructed ``args`` object.

    This is the programmatic entry point: build ``args`` via
    ``parse_with_config(build_parser(...))`` (or construct any object with
    the same attributes), override fields, and ``await run_async(args)`` from
    async code (a notebook cell, an ``asyncio.gather`` sweep). Synchronous
    scripts use :func:`run`, which is ``asyncio.run(run_async(...))``.

    Optional seams (``None`` selects the flag-driven default):

    - ``env_factory``: a zero-arg callable returning a fresh ``Env`` --
      bypasses the ``--env_type`` registry lookup entirely.
    - ``validation_runner``: any object with ``async run_validation(step,
      sampling_client, tokenizer, template_fn, args, backend) -> dict``
      (see :class:`~.validation.ValidationRunner`); its metrics merge into
      each validation step's record.

    The concrete clients are created by the service backend via
    ``make_backend(...)``, which opens the SageMaker Training Session and
    fetches the tokenizer from the training client.
    """
    # Console logging through the shared console (service/console.py): log
    # records, the run banner, metric rows, and validation examples all share
    # one stdout stream, so output stays ordered even when redirected to a
    # file. {run_folder}/training.log additionally gets every log record via
    # setup_file_logging. No-op when the caller already configured logging.
    setup_console_logging()

    # --- Run logging -----------------------------------------------------
    # RunLog owns the run folder, config.json, training.log, metrics.jsonl,
    # and: when --mlflow_tracking_uri is set, the MLflow run lifecycle
    # (params at start, artifacts + FINISHED/FAILED status at close). MLflow
    # publishing is best-effort and can never fail the training run; a
    # malformed App ARN, however, errors out right here, before any session
    # exists.
    runlog = RunLog.open(
        args.run_name, config=vars(args),
        output_dir=args.output_dir, file_logging=True,
    )
    run_folder = runlog.run_folder
    metrics_path = runlog.metrics_path
    # Attach to args so downstream code (the training loops) can access them.
    # The loops publish through the raw publisher (MetricsLogger, baseline
    # validation, rollout traces); it is None when publishing is off.
    args._run_folder = run_folder
    args._metrics_path = metrics_path
    args._mlflow = runlog.publisher

    print_run_banner("RL training run", {
        "run_name": args.run_name,
        "model": args.base_model,
        "env_type": args.env_type,
        "dataset_path": getattr(args, "dataset_path", None) or "builtin",
        "loss": args.loss,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "responses_per_prompt": args.responses_per_prompt,
        "learning_rate": args.learning_rate,
        "run_folder": run_folder,
    })

    # --- Backend -------------------------------------------------------
    # Construct the backend. This opens the SageMaker training session and fetches the tokenizer from the training client. The heavy SDK is imported lazily inside the backend's create(), so `--help` and pure imports never require sagemaker.
    # Session creation blocks for minutes; keep it off the event loop so
    # concurrent run_async calls (e.g. a gather'd sweep) don't serialize.
    backend = await asyncio.to_thread(make_backend, args.backend, args)
    tokenizer = backend.tokenizer

    # Durable checkpoints (periodic on --save_every, final at the end), all
    # recorded in {run_folder}/checkpoints.jsonl. Distinct from the backend's
    # sampler-refresh saves, which are policy-serving snapshots on the
    # --refresh_every cadence and are deliberately not recorded. save_every
    # defaults to max_steps, which never fires mid-run: final-save-only.
    checkpoint_manager = checkpoint_utils.CheckpointManager(
        backend.training_client,
        backend.service_client,
        run_folder,
        save_every=args.save_every,
    )

    # --- Dataset ---------------------------------------------------------
    dataset = load_dataset(args.dataset_path, args.max_prompts)

    # --- Template resolution --------------------------------------------
    # Most templates are unary `(observation: str) -> str`. Two exceptions need the tokenizer baked in so the call sites stay tokenizer-agnostic:
    #   - `auto` may resolve to `apply_gemma` or `apply_hf` (their special tokens can't round-trip through `tokenizer.encode`).
    #   - explicit `--template gemma` / `--template hf` always emit tokens.
    # In all cases the `apply_*` may return either `str` (text to encode at the call site) or `list[int]` (already-tokenized prompt). Downstream call sites pass the result through `encode_prompt(result, tokenizer)` which handles both shapes.
    #
    # `--reasoning` threads into whichever path supports it; non-thinking templates silently ignore it. `--template hf` forces the model-shipped jinja regardless of model family: useful for running gpt-oss with its canonical ChatGPT+date framing or for trying gemma without the cookbook's hand-rolled fallback.
    _reasoning = args.reasoning
    if args.template == "auto":
        _auto_base_model = args.base_model
        _auto_tokenizer = tokenizer

        def template_fn(observation: str):
            return apply_auto(
                observation, _auto_base_model, _auto_tokenizer, reasoning=_reasoning
            )
    elif args.template == "hf":
        _hf_tokenizer = tokenizer

        def template_fn(observation: str) -> list[int]:
            return apply_hf(observation, _hf_tokenizer, reasoning=_reasoning)
    elif args.template == "gemma":
        _gemma_tokenizer = tokenizer

        def template_fn(observation: str) -> list[int]:
            return apply_gemma(observation, _gemma_tokenizer, reasoning=_reasoning)
    elif args.template == "gpt_oss":
        # Bake the reasoning level into the hand-rolled gpt-oss template.
        def template_fn(observation: str) -> str:
            return apply_gpt_oss(observation, reasoning=_reasoning)
    else:
        # qwen3_general, qwen3_game, no: `--reasoning` is N/A for these.
        template_fn = TEMPLATE_FACTORY[args.template]

    # --- Env factory -----------------------------------------------------
    # Env selection goes through the registry: the training loop never imports concrete env classes directly. Each concurrent rollout gets its own fresh env instance (envs are stateful) because the returned factory is zero-arg and constructs a new env per call.
    #
    # The GSM8K dataset is built here via `load_dataset` (above) and attached to `args` so the registry's gsm8k branch can read it via `getattr(args, "_dataset", None)` and construct `GSM8KMultiTurnEnv(max_turns=args.max_turns, dataset=args._dataset)` on the same dataset object.
    #
    # For `code_rl`, the registry calls `build_code_rl_env_factory(args)`, which builds the DeepCoder dataset itself from `args` (no `_dataset`).
    args._dataset = dataset
    if env_factory is None:
        env_factory = get_env_factory(args.env_type, args)
    # The code_rl factory replaces `args._dataset` with the DeepCoder tasks so the central sampler / next_batch iterates code tasks. For GSM8K the factory leaves `args._dataset` untouched, so this is a no-op there.
    dataset = args._dataset

    # --- Drive the training loop ----------------------------------------
    # --max_steps_off_policy > 0 selects the overlapped off-by-k loop
    # (run_training_async); 0 (default) keeps the strict serial run_training so
    # existing gsm8k/code_rl parity runs are byte-for-byte unchanged.
    _loop_fn = (
        run_training_async
        if int(getattr(args, "max_steps_off_policy", 0)) > 0
        else run_training
    )
    _mlflow_status = "FAILED"
    try:
        await _loop_fn(
            args,
            backend,
            dataset,
            tokenizer=tokenizer,
            env_factory=env_factory,
            template_fn=template_fn,
            validation_runner=validation_runner,
            checkpoint_manager=checkpoint_manager,
        )

        # --- Final save --------------------------------------------------
        # The canonical post-training checkpoint, recorded in
        # checkpoints.jsonl. Parity with the original backend behavior: a
        # resumable save_state always; the deployable adapter package only
        # when --save_weights_at_end (implied by --deploy_and_invoke).
        want_weights = getattr(args, "save_weights_at_end", False) or getattr(
            args, "deploy_and_invoke", False)
        final_paths = await checkpoint_utils.save_checkpoint_async(
            backend.training_client,
            name="final",
            log_path=run_folder,
            loop_state={"step": max(args.max_steps - 1, 0), "final": True},
            kind="both" if want_weights else "state",
        )
        await checkpoint_manager.finalize_async()
        if final_paths.get("sampler_path"):
            backend.note_deployable_checkpoint(final_paths["sampler_path"])

        # Tag the final checkpoint model packages on the MLflow run.
        runlog.set_tags({
            "checkpoint_final_save_state": final_paths.get("state_path"),
            "checkpoint_final_save_weights": final_paths.get("sampler_path"),
        })

        # Training + final checkpoint succeeded: record that outcome now.
        # deploy_and_invoke below is an optional post-training step; its
        # failure must not mislabel a successful training run as FAILED.
        _mlflow_status = "FINISHED"

        # --- Deploy -> invoke (opt-in) -----------------------------------
        # Completes the train -> checkpoint -> deploy -> invoke loop. Gated on the flag so plain training runs don't stand up an endpoint.
        if getattr(args, "deploy_and_invoke", False):
            result = backend.deploy_and_invoke()
            if result is not None:
                logging.getLogger(__name__).info("deploy_and_invoke result: %s", result)

        # --- Teardown (success) -------------------------------------------
        # Delete the session so repeated runs don't accumulate stopped session
        # records (mirrors the SFT path). On failure the finally-block close()
        # only stops it, keeping the record for debugging.
        await asyncio.to_thread(backend.delete_session)
    finally:
        await asyncio.to_thread(backend.close)
        # Uploads run-folder artifacts (metrics/config/validation/trajectories
        # jsonl) and closes the MLflow run with the right status.
        runlog.close(_mlflow_status)


def run(args, *, env_factory=None, validation_runner=None) -> None:
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
    asyncio.run(run_async(
        args, env_factory=env_factory, validation_runner=validation_runner,
    ))


def main(default_env: str | None = None, envs: tuple[str, ...] | None = None) -> None:
    """CLI entry point: ``run(parse_with_config(build_parser(...)))``.

    ``default_env`` / ``envs`` let a recipe entry point preset ``--env_type``
    and restrict the flag surface to its own environment(s).
    """
    run(parse_with_config(build_parser(default_env=default_env, envs=envs)))






# === Sampler refresh (lives in the backend) ===
#
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
) -> None:
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
    _sample_workers = getattr(args, "num_rollout_workers", 8) + 8
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=_sample_workers,
            thread_name_prefix="rollout-sample",
        )
    )
    logging.getLogger(__name__).info(
        "rollout thread pool: max_workers=%d (num_rollout_workers=%d)",
        _sample_workers,
        getattr(args, "num_rollout_workers", 8),
    )

    # Optimizer params: rebuilt each step with warmup LR. The effective learning rate is computed per step via `compute_learning_rate` (linear warmup then constant) and passed to `backend.train_step`.
    adam_beta1 = getattr(args, "adam_beta1", 0.9)
    adam_beta2 = getattr(args, "adam_beta2", 0.95)
    adam_epsilon = getattr(args, "adam_epsilon", 1e-8)
    warmup_steps = getattr(args, "warmup_steps", 0)

    # Metrics path: use the run folder path set up in main(). Falls back to the log_dir path if _metrics_path was not set.
    metrics_path = getattr(args, "_metrics_path", None)
    if metrics_path is None:
        os.makedirs(args.log_dir, exist_ok=True)
        metrics_path = os.path.join(args.log_dir, f"metrics_{args.run_name}.jsonl")

    # MetricsLogger handles all per-step metric computation and JSONL writing.
    metrics_logger = MetricsLogger(metrics_path, publisher=getattr(args, "_mlflow", None))
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
            run_folder = getattr(args, "_run_folder", None) or os.path.dirname(metrics_path)
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
        and getattr(args, "val_at_start", False)
    ):
        sampling_client = backend.refresh_sampler(0)
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
        if getattr(args, "_mlflow", None) is not None:
            args._mlflow.log_metrics(baseline_val_metrics, step=0)
        skip_step0_inloop_val = True

    trained_steps = 0
    for step in range(args.max_steps):
        # 1. Sampler refresh: snapshot weights and rebind the sampler.
        if step % args.refresh_every == 0:
            # If --val_at_start already refreshed the sampler at step 0 with the initial weights, reuse it instead of re-refreshing (the trainer weights haven't changed yet).
            if not (step == 0 and sampling_client is not None):
                sampling_client = backend.refresh_sampler(step)

        # 2. Pull next batch of prompts. Empty -> dataset exhausted.
        step_start = time.time()
        prompts = next_batch(dataset, args.batch_size, epoch_iter)
        if not prompts:
            break

        # 3. Async rollouts + advantage computation. We keep the `episodes` view (a flat list of episodes) alongside the `transitions` view so episode-level metrics are computed on the right object.
        if args.advantage_mode == "grouped":
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

        # Rollout capture: trajectories.jsonl and/or MLflow traces (opt-in).
        capture_rollouts(args, step, episodes, tokenizer, metrics_path)

        # 5. Build datums via the backend (a PySDK datum dict). The rollout `Transition` is the shared interchange type.
        datums = [
            backend.build_datum(t, temperature=args.temperature)
            for t in transitions
        ]

        # 6. Pipelined forward/backward + optimizer step, delegated to the backend. It submits both ops back-to-back and awaits in submission order so the optimizer kernel is scheduled as soon as gradients land (the service backend additionally saves weights here so the reused sampling client reflects the update). Returns fb_result (exposes `.metrics`).
        train_start = time.time()
        # Compute effective LR with linear warmup.
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
        log_every = getattr(args, "log_every", 10)
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
) -> None:
    """Off-by-K (overlapped, bounded-staleness) RL training loop.

    Drop-in alternative to :func:`run_training`, selected by
    ``--max_steps_off_policy > 0``. Emits the identical ``metrics.jsonl`` schema
    (plus ``sampler/step_*`` staleness keys) so the same plotters and readers
    work. Only ``--advantage_mode grouped`` is supported here, the loop's unit
    of staleness is one prompt's K-rollout GRPO group.

    Args mirror :func:`run_training`.
    """
    log = logging.getLogger(__name__)

    max_steps_off_policy = int(getattr(args, "max_steps_off_policy", 0))
    assert max_steps_off_policy > 0, (
        "run_training_async requires --max_steps_off_policy > 0 "
        "(the sync run_training handles the on-policy case)"
    )
    if getattr(args, "advantage_mode", "grouped") != "grouped":
        raise SystemExit(
            "--max_steps_off_policy > 0 (off-by-k loop) requires "
            "--advantage_mode grouped: the loop's staleness unit is one "
            "prompt's K-rollout group. Use the sync loop for monte_carlo."
        )

    # Same executor enlargement as the sync loop: blocking sample()/save calls
    # dispatched via asyncio.to_thread need enough worker threads to run
    # num_rollout_workers-wide (the default min(32, cpu+4) throttles below the
    # worker count on small boxes). Add headroom for the off-thread refresh.
    _sample_workers = getattr(args, "num_rollout_workers", 8) + 8
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=_sample_workers,
            thread_name_prefix="rollout-sample",
        )
    )
    log.info(
        "off-by-k loop: max_steps_off_policy=%d num_rollout_workers=%d "
        "batch_size(groups)=%d responses_per_prompt(K)=%d max_inflight_rollouts=%d",
        max_steps_off_policy,
        getattr(args, "num_rollout_workers", 8),
        args.batch_size,
        args.responses_per_prompt,
        getattr(args, "max_inflight_rollouts", getattr(args, "num_rollout_workers", 8)),
    )

    adam_beta1 = getattr(args, "adam_beta1", 0.9)
    adam_beta2 = getattr(args, "adam_beta2", 0.95)
    adam_epsilon = getattr(args, "adam_epsilon", 1e-8)
    warmup_steps = getattr(args, "warmup_steps", 0)

    # Metrics path (identical to run_training).
    metrics_path = getattr(args, "_metrics_path", None)
    if metrics_path is None:
        os.makedirs(args.log_dir, exist_ok=True)
        metrics_path = os.path.join(args.log_dir, f"metrics_{args.run_name}.jsonl")
    metrics_logger = MetricsLogger(metrics_path, publisher=getattr(args, "_mlflow", None))
    table = MetricsTablePrinter()

    # Validation (identical to run_training).
    if validation_runner is None:
        val_dataset = _build_val_dataset(args)
        if val_dataset is not None:
            run_folder = getattr(args, "_run_folder", None) or os.path.dirname(metrics_path)
            validation_runner = ValidationRunner(
                val_dataset, args.val_examples, run_folder, args.env_type,
            )

    epoch_iter = make_epoch_iter(dataset, args.epochs, args.drop_last)

    state = _AsyncLoopState()
    num_workers = getattr(args, "num_rollout_workers", 4)
    K = args.responses_per_prompt

    # R9 / R10: client-side cap on live collect_one_group_async awaits, shared
    # across all rollout workers. Constructed unconditionally so the acquire
    # path in _worker is uniform (when the cap >= num_workers it never blocks,
    # which is a no-op observationally but still runs the same code: matches
    # R9.4). Scoped to collect_one_group_async only; workers still race freely
    # on prompt_queue.get and results_queue.put, so no double-gate deadlock
    # (R10.5). psutil sampler seeds first RSS row for R8.7 / R12.3.
    max_inflight = getattr(args, "max_inflight_rollouts", num_workers)
    rollout_sem = asyncio.Semaphore(max_inflight)
    _process = psutil.Process()

    # Bootstrap the initial sampler (step 0) BEFORE starting workers so the very
    # first rollouts have a client. Optional --val_at_start baseline reuses it.
    state.sampling_client = await backend.refresh_sampler_async(0)
    skip_step0_inloop_val = False
    if validation_runner is not None and getattr(args, "val_at_start", False):
        baseline_val_metrics = await validation_runner.run_validation(
            0, state.sampling_client, tokenizer, template_fn, args, backend,
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "step": 0,
                "phase": "pre_training_baseline",
                **baseline_val_metrics,
            }) + "\n")
        if getattr(args, "_mlflow", None) is not None:
            args._mlflow.log_metrics(baseline_val_metrics, step=0)
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
        """Pull a prompt, run K rollouts vs the current sampler, push a group."""
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
                    if getattr(args, "sampling_seed", None) is not None
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
        await results_queue.put(_WORKER_DONE)

    dataloader_task = asyncio.create_task(_dataloader(), name="dataloader")
    worker_tasks = [
        asyncio.create_task(_worker(), name=f"rollout-worker-{i}")
        for i in range(num_workers)
    ]

    stale_policy = getattr(args, "stale_policy", "drop")
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
                    # Fire-and-forget requeue task; must be a task so this
                    # consumer coroutine doesn't block on the (bounded)
                    # prompt_queue while workers wait for it to drain.
                    asyncio.create_task(
                        prompt_queue.put(stale_prompt),
                        name="requeue_stale_prompt_task",
                    )
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
                kl_coef=getattr(args, "kl_coef", 0.0),
                extra_metrics=timing_extras,
                groups=groups,
            )

            table.print_row(metrics_record, step)
            # Detail-table cadence runs on the 1-based displayed step
            # (--log_every 2 prints steps 2, 4, ...); the last step
            # always prints so no run ends without its final table.
            log_every = getattr(args, "log_every", 10)
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


# Module entry point. Must stay at the very end of the file: `main()` calls names (`load_dataset`, `run_training`, `Transition`, etc.) defined in the sections above. Python executes top-level statements sequentially, so the guard must run after every `def` and `class` has been bound.
if __name__ == "__main__":
    # Suppress shutdown noise from the InternalClientHolder background reader loop (no close() method): the warnings filter swallows the "coroutine ... was never awaited" RuntimeWarning, and raising the asyncio logger to CRITICAL hides the "Task was destroyed but it is pending!" message (logged via the asyncio logger, not the warnings module).
    warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    main()
