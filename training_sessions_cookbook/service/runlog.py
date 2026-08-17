"""Per-run output: run folder, config snapshot, file logging, metrics log.

Every training run writes to one directory: ``config.json`` (the CLI args plus the git hash), ``training.log`` (INFO+ from every logger), and ``metrics.jsonl`` (one JSON record per step). Both the RL and SFT loops use these helpers so the on-disk layout is identical.
"""

from __future__ import annotations

import json
import logging

from training_sessions_cookbook.service import console
import os
import subprocess
import time

logger = logging.getLogger(__name__)


def git_hash() -> str:
    """Return the current git HEAD hash, or ``"unknown"`` outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def create_run_folder(run_name: str, output_dir: str | None = None) -> str:
    """Create and return the absolute path of the run folder.

    Uses ``output_dir`` verbatim when given; otherwise ``runs/{run_name}_{YYYYMMDD_HHMMSS}/``.
    """
    if output_dir is not None:
        folder = output_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        folder = os.path.join("runs", f"{run_name}_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    return os.path.abspath(folder)


def dump_config(args, run_folder: str) -> None:
    """Write ``config.json`` with every CLI arg plus ``git_hash``."""
    config = vars(args).copy()
    config["git_hash"] = git_hash()
    with open(os.path.join(run_folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)


def setup_file_logging(run_folder: str) -> logging.FileHandler:
    """Attach a ``training.log`` FileHandler (INFO+) to the root logger.

    The handler is attached directly (not via ``logging.basicConfig``) so it composes with whatever console handler the caller configured. Repeat calls for the same file are ignored.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    log_path = os.path.join(run_folder, "training.log")
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(
            h, "baseFilename", None
        ) == os.path.abspath(log_path):
            return h

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return file_handler


def append_metrics(metrics_path: str, record: dict) -> None:
    """Append one JSON record as a line to ``metrics_path``."""
    with open(metrics_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _mlflow_sink(run_name: str, config: dict):
    """Built-in MLflow sink factory; ``None`` when not configured/available."""
    from .mlflow_log import MlflowPublisher
    return MlflowPublisher.create(
        config.get("mlflow_tracking_uri"),
        experiment_name=config.get("mlflow_experiment_name"),
        run_name=config.get("mlflow_run_name") or run_name,
        config=config,
    )


#: Sink factories tried at ``RunLog.open()``: ``(run_name, config) -> sink | None``.
#: To add a tracker (e.g. W&B), implement a sink class honoring the contract
#: documented on :class:`RunLog` and append its factory here.
_SINK_FACTORIES = (_mlflow_sink,)


class RunLog:
    """One object that owns a training run's logging: run folder, config
    snapshot, ``metrics.jsonl``, and any configured tracker sinks.

    This is the surface a recipe touches, deriving a recipe means porting
    three lines, not the lifecycle plumbing::

        with RunLog.open(run_name, config=vars(args), output_dir=args.log_path) as runlog:
            ...
            runlog.log_metrics(record, step=step)
            runlog.log_checkpoint("final", model_package_arn)

    Local files (``config.json``, ``metrics.jsonl``, ``checkpoints.jsonl``)
    are the always-on ground truth, written by RunLog itself. Trackers are
    **sinks**: objects RunLog fans every call out to. MLflow is the built-in
    sink, enabled iff ``config["mlflow_tracking_uri"]`` is set (put
    ``add_mlflow_args(parser)`` in the recipe's parser and pass ``vars(args)``).

    **Sink contract** (duck-typed; all methods optional, RunLog forwards only
    what a sink implements, this is how future trackers like W&B plug in):

    - ``log_metrics(record: dict, step: int)``: one training-step record
    - ``set_tags(tags: dict)``: checkpoint ARNs, session names
    - ``log_rollout_traces(step, rollouts)``: decoded RL rollouts
    - ``log_artifacts(paths: list)``: run-folder files at close
    - ``end(status: 'FINISHED'|'FAILED')``: terminate the sink's run

    A new tracker = one sink class + one entry in ``_SINK_FACTORIES`` above;
    recipes are untouched. Sink calls are exception-guarded in ``_fanout``,
    so a buggy sink can never raise into the training loop (sinks like
    ``MlflowPublisher`` additionally handle their own failures).

    On context exit the run's known output files are uploaded to every sink
    and the sinks are closed ``FINISHED`` (clean exit) / ``FAILED`` (exception).
    """

    #: Files in the run folder uploaded as artifacts at close, when present.
    ARTIFACT_FILES = (
        "config.json", "metrics.jsonl", "checkpoints.jsonl",
        "validation_examples.jsonl", "trajectories.jsonl",
    )

    def __init__(self, run_folder: str, sinks: list | None = None):
        self.run_folder = run_folder
        self.metrics_path = os.path.join(run_folder, "metrics.jsonl")
        self.sinks = list(sinks or [])
        self._closed = False
        self._file_handler = None  # set when open(file_logging=True)
        self._echo_path = None     # set when open(file_logging=True)

    @property
    def publisher(self):
        """The MLflow sink, or ``None``, kept for callers that publish
        MLflow-only data directly (e.g. the RL loops' MetricsLogger)."""
        from .mlflow_log import MlflowPublisher
        for sink in self.sinks:
            if isinstance(sink, MlflowPublisher):
                return sink
        return None

    # --- construction ---------------------------------------------------------

    @classmethod
    def open(
        cls,
        run_name: str,
        config: dict,
        output_dir: str | None = None,
        file_logging: bool = False,
    ) -> "RunLog":
        """Create the run folder, dump the config, and wire the sinks.

        ``output_dir`` is used verbatim when given (mirrors ``--log_path`` /
        ``--output_dir``); otherwise ``runs/{run_name}_{timestamp}/``.
        ``file_logging=True`` also attaches ``training.log`` to the root
        logger.
        """
        folder = create_run_folder(run_name, output_dir=output_dir)
        config = dict(config)
        config.setdefault("git_hash", git_hash())
        with open(os.path.join(folder, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)
        sinks = [s for f in _SINK_FACTORIES if (s := f(run_name, config))]
        runlog = cls(folder, sinks=sinks)
        if file_logging:
            runlog._file_handler = setup_file_logging(folder)
            # Mirror the console layer (banner, rules, metric rows/tables)
            # into the same training.log, so the file is a complete plain
            # record of the run, not just the log records. Both the handler
            # and the mirror are detached in close(), so repeated
            # programmatic runs in one process do not accumulate handlers or
            # cross-write each other's files.
            runlog._echo_path = os.path.join(folder, "training.log")
            console.add_console_echo_file(runlog._echo_path)
        return runlog

    # --- logging -------------------------------------------------------------

    def _fanout(self, method: str, *a, **kw) -> None:
        """Forward a call to every sink that implements ``method``.

        Each call is exception-guarded so the sink contract ("never raise
        into training") is enforced here, not just documented, a buggy sink
        degrades to a warning and the remaining sinks still receive the call.
        """
        for sink in self.sinks:
            fn = getattr(sink, method, None)
            if fn is None:
                continue
            try:
                fn(*a, **kw)
            except Exception as exc:
                logger.warning("Sink %s.%s failed: %r",
                               type(sink).__name__, method, exc)

    def log_metrics(self, record: dict, step: int | None = None) -> None:
        """Append ``record`` to ``metrics.jsonl`` and mirror it to all sinks."""
        append_metrics(self.metrics_path, record)
        self._fanout("log_metrics", record,
                     step=step if step is not None else int(record.get("step", 0)))

    def log_checkpoint(self, name, model_package_arn: str | None) -> None:
        """Tag a checkpoint's model-package ARN on the sinks (e.g. MLflow).

        ``name`` is a step number or a label like ``"final"``; the tag
        becomes ``checkpoint_step_N`` / ``checkpoint_<name>``. The
        ``checkpoints.jsonl`` record itself is written by
        ``service.checkpoint_utils`` (the single writer of that file).
        """
        if not model_package_arn:
            return
        tag = f"checkpoint_step_{name}" if isinstance(name, int) else f"checkpoint_{name}"
        self._fanout("set_tags", {tag: model_package_arn})

    def set_tags(self, tags: dict) -> None:
        """Set run tags on all sinks (no-op when no sink is configured)."""
        self._fanout("set_tags", tags)

    def log_rollout_traces(self, step: int, rollouts: list) -> None:
        """Publish decoded rollouts to sinks that support traces."""
        self._fanout("log_rollout_traces", step, rollouts)

    # --- lifecycle -----------------------------------------------------------

    def close(self, status: str = "FINISHED") -> None:
        """Upload run-folder artifacts to all sinks and close them. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Detach this run's logging BEFORE uploading training.log, so the
        # uploaded artifact is complete and no handler/mirror outlives the
        # run (repeated programmatic runs stay independent).
        if self._echo_path is not None:
            console.remove_console_echo_file(self._echo_path)
        if self._file_handler is not None:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()
        paths = [os.path.join(self.run_folder, name) for name in self.ARTIFACT_FILES]
        self._fanout("log_artifacts", paths)
        self._fanout("end", status)

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close("FAILED" if exc_type is not None else "FINISHED")
