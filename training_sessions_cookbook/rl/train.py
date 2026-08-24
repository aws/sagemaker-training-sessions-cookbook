from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import warnings

from training_sessions_cookbook.service import checkpoint_utils
from training_sessions_cookbook.service.console import (
    print_run_banner,
    setup_console_logging,
)
from training_sessions_cookbook.service.mlflow_log import add_mlflow_args
from training_sessions_cookbook.service.runlog import RunLog

from .envs import env_cli_args, get_env_factory, registered_envs
from .backends import make_backend
from .config import cfg, default as default_knob
from .dataset import load_dataset
from .prompts import (
    TEMPLATE_FACTORY,
    apply_auto,
    apply_gemma,
    apply_gpt_oss,
    apply_hf,
)
from .runstate import attach_run_artifacts, set_shared_dataset, shared_dataset
from .loops import run_training, run_training_async

# --- Public API / compatibility re-exports ---------------------------------
#
# The names below are not used in this module; they are re-exported because
# this module is the package's documented entry point (recipes and the README
# say ``from training_sessions_cookbook.rl.train import ...``) and because the
# concerns they belong to were extracted into sibling modules over time. Each
# name lives in exactly one implementing module; import from there for new
# code. This list is deliberately explicit - a name must earn its place here
# by being part of the documented surface, not by having once been defined in
# this file.
from training_sessions_cookbook.service import (  # noqa: F401
    create_run_folder,
    dump_config,
    setup_file_logging,
)
from training_sessions_cookbook.service import git_hash as _get_git_hash  # noqa: F401
from .loops import compute_learning_rate  # noqa: F401
from .metrics import (  # noqa: F401
    MetricsLogger,
    MetricsTablePrinter,
    _METRICS_TABLE_HEADER,
    compute_advantage_stats,
    compute_episode_length_mean,
    compute_reward_stats,
    count_episodes,
    episodes_to_rollout_dicts,
    format_metrics_table,
)
from .prompts import apply_qwen3_general  # noqa: F401
from .rollouts import Transition, WrappedGroup  # noqa: F401
from .staleness import _StalenessBatcher  # noqa: F401
from .validation import (  # noqa: F401
    ValidationExample,
    ValidationRunner,
    compute_validation_accuracy,
)


# === Run folder management ===
#
# Run folder, config dump, and file logging are shared with the SFT path in ``service/runlog.py``; the aliases below preserve the names this module and its tests reference.

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
    trn.add_argument("--rolling_save_every", type=int,
                     default=default_knob("rolling_save_every"),
                     help="Keep a rolling crash-recovery checkpoint every N "
                          "steps (state-only, saved in the background, each "
                          "save deletes the previous rolling package so they "
                          "never accumulate). 0 disables. Without it, a run "
                          "that dies at step 90/100 loses everything.")
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
    dsg.add_argument("--task_offset", type=int, default=0,
                     help="code_rl only: skip this many tasks of the seeded "
                          "DeepCoder shuffle before taking --max_prompts. "
                          "Disjoint windows let concurrent sessions avoid "
                          "prompt overlap (KV-cache isolation).")
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
    if cfg(args, "max_inflight_rollouts") is None:
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
    attach_run_artifacts(
        args,
        run_folder=run_folder,
        metrics_path=metrics_path,
        mlflow=runlog.publisher,
    )

    print_run_banner("RL training run", {
        "run_name": args.run_name,
        "model": args.base_model,
        "env_type": args.env_type,
        "dataset_path": cfg(args, "dataset_path") or "builtin",
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
        # Crash recovery: state-only background saves that self-prune (each
        # deletes the previous rolling package). Before this default, a run
        # with the default --save_every (final-only) that died mid-run lost
        # all progress.
        rolling_save_every=cfg(args, "rolling_save_every"),
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

        def template_fn(observation: str, tools=None):
            return apply_auto(
                observation, _auto_base_model, _auto_tokenizer, reasoning=_reasoning, tools=tools
            )
    elif args.template == "hf":
        _hf_tokenizer = tokenizer

        def template_fn(observation: str, tools=None) -> list[int]:
            return apply_hf(observation, _hf_tokenizer, reasoning=_reasoning, tools=tools)
    elif args.template == "gemma":
        _gemma_tokenizer = tokenizer

        def template_fn(observation: str, tools=None) -> list[int]:
            return apply_gemma(observation, _gemma_tokenizer, reasoning=_reasoning, tools=tools)
    elif args.template == "gpt_oss":
        # Bake the reasoning level into the hand-rolled gpt-oss template.
        def template_fn(observation: str, tools=None) -> str:
            return apply_gpt_oss(observation, reasoning=_reasoning, tools=tools)
    else:
        # qwen3_general, qwen3_game, no: `--reasoning` is N/A for these.
        # `apply_qwen3_general` accepts `tools`; the others do not take tools
        # and multi-turn code RL rejects them via the wrapper below.
        _plain_fn = TEMPLATE_FACTORY[args.template]
        _plain_name = args.template

        def template_fn(observation: str, tools=None):
            if tools:
                if _plain_name == "qwen3_general":
                    return _plain_fn(observation, tools=tools)
                raise ValueError(
                    f"--template {_plain_name} does not support tool declarations; "
                    "multi-turn code RL needs a tool-aware template"
                )
            return _plain_fn(observation)

    # --- Env factory -----------------------------------------------------
    # Env selection goes through the registry: the training loop never imports concrete env classes directly. Each concurrent rollout gets its own fresh env instance (envs are stateful) because the returned factory is zero-arg and constructs a new env per call.
    #
    # The GSM8K dataset is built here via `load_dataset` (above) and published via `set_shared_dataset` so the registry's gsm8k branch can read it back through `shared_dataset(args)` and construct `GSM8KMultiTurnEnv` on the same dataset object (rl.runstate owns this contract).
    #
    # For `code_rl`, the registry calls `build_code_rl_env_factory(args)`, which builds the DeepCoder dataset itself from `args` (no `_dataset`).
    set_shared_dataset(args, dataset)
    if env_factory is None:
        env_factory = get_env_factory(args.env_type, args)
    # The code_rl factory replaces the shared dataset with the DeepCoder tasks so the central sampler / next_batch iterates code tasks. For GSM8K the factory leaves it untouched, so this is a no-op there.
    dataset = shared_dataset(args)

    # --- Drive the training loop ----------------------------------------
    # --max_steps_off_policy > 0 selects the overlapped off-by-k loop
    # (run_training_async); 0 (default) keeps the strict serial run_training so
    # existing gsm8k/code_rl parity runs are byte-for-byte unchanged.
    _loop_fn = (
        run_training_async
        if int(cfg(args, "max_steps_off_policy")) > 0
        else run_training
    )
    _mlflow_status = "FAILED"
    try:
        last_step = await _loop_fn(
            args,
            backend,
            dataset,
            tokenizer=tokenizer,
            env_factory=env_factory,
            template_fn=template_fn,
            validation_runner=validation_runner,
            checkpoint_manager=checkpoint_manager,
        )
        if last_step is None:  # a custom loop that returns nothing
            last_step = max(args.max_steps - 1, 0)

        # --- Final save --------------------------------------------------
        # The canonical post-training checkpoint, recorded in
        # checkpoints.jsonl. Parity with the original backend behavior: a
        # resumable save_state always; the deployable adapter package only
        # when --save_weights_at_end.
        want_weights = cfg(args, "save_weights_at_end")
        final_paths = await checkpoint_utils.save_checkpoint_async(
            backend.training_client,
            name="final",
            log_path=run_folder,
            # Label with the step that actually trained last - the dataset can
            # exhaust before max_steps, and a checkpoint claiming an untrained
            # step is misleading metadata.
            loop_state={"step": last_step, "final": True},
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
        _mlflow_status = "FINISHED"

        # --- Teardown (success) -------------------------------------------
        # Delete the session so repeated runs don't accumulate stopped session
        # records (mirrors the SFT path). On failure the finally-block close()
        # only stops it, keeping the record for debugging.
        await asyncio.to_thread(backend.delete_session)
    finally:
        # SHIELDED teardown: this block must survive cancellation. On Ctrl-C,
        # asyncio.run() cancels this task, and any `await` inside a finally is
        # itself a cancellation point - the previous `await
        # asyncio.to_thread(backend.close)` could be killed mid-flight, leaving
        # a BILLED training session running with nothing left to stop it.
        # Synchronous calls cannot be interrupted by task cancellation, so the
        # session stop and the artifact upload both run to completion (a second
        # SIGINT still aborts the process; that trade-off is deliberate: one
        # Ctrl-C means "stop cleanly", two means "I accept the leak").
        backend.close()
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


# Module entry point. Must stay at the very end of the file: `main()` calls names (`load_dataset`, `run_training`, `Transition`, etc.) defined in the sections above. Python executes top-level statements sequentially, so the guard must run after every `def` and `class` has been bound.
if __name__ == "__main__":
    # Suppress shutdown noise from the InternalClientHolder background reader loop (no close() method): the warnings filter swallows the "coroutine ... was never awaited" RuntimeWarning, and raising the asyncio logger to CRITICAL hides the "Task was destroyed but it is pending!" message (logged via the asyncio logger, not the warnings module).
    warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    main()
