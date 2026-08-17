"""Optional MLflow metric publishing for the RL and SFT training paths.

Publishes each training step's metrics record to an MLflow tracking server.
One flag points at the destination, ``--mlflow_tracking_uri``, and accepts
either (both feed ``mlflow.set_tracking_uri``):

* the **ARN of a SageMaker managed MLflow App**
  (``arn:aws:sagemaker:...:mlflow-app/...``). The ``sagemaker-mlflow`` plugin
  resolves the ARN and signs requests with the caller's AWS credentials.
  ARN-shaped values are validated up front (see :func:`validate_tracking_uri`)
  so a typo'd ARN fails fast at startup instead of surfacing as a cryptic
  plugin error mid-run.
* any **MLflow-compatible tracking URI**: a local or self-hosted server
  (``http://localhost:5000``), a direct database store
  (``sqlite:///mlflow.db``), etc. No plugin needed.

Design rules:

* **Opt-in.** ``MlflowPublisher.create`` returns ``None`` when no destination
  is given; call sites hold an ``Optional[MlflowPublisher]`` and skip on None.
* **Soft dependency.** ``mlflow`` is imported lazily inside ``create``. When
  the ``mlflow`` extra is not installed we warn and return ``None``, the run
  continues and metrics still land in ``metrics.jsonl``.
* **Never fail the run.** Every publish call is wrapped; a broken tracking
  server degrades to warnings, and after ``_MAX_CONSECUTIVE_FAILURES``
  consecutive failures the publisher disables itself to stop the spam. The
  single exception is the up-front ARN shape validation: that fires before
  the training session exists, where a loud early error is strictly better.
* **Metrics are floats.** Non-numeric values in a record (e.g. the RL
  baseline's ``phase`` string) are skipped for ``log_metric``, strings belong
  in params/tags, which is where the config and checkpoint ARNs go.
"""

from __future__ import annotations

import logging
import numbers
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# MLflow caps param values at 500 chars (server-side); truncate client-side so
# a long CLI value can never fail the whole log_params batch.
_MAX_PARAM_VALUE_LEN = 500

# After this many consecutive publish failures the publisher disables itself.
_MAX_CONSECUTIVE_FAILURES = 5

# Shape of a SageMaker managed MLflow destination ARN. Covers both resource
# types: the original mlflow-tracking-server and the newer mlflow-app.
_SAGEMAKER_MLFLOW_ARN_RE = re.compile(
    r"^arn:aws[a-z0-9-]*:sagemaker:[a-z0-9-]+:\d{12}:"
    r"(mlflow-tracking-server|mlflow-app)/.+$"
)


def validate_tracking_uri(tracking_uri: str) -> str:
    """Validate an ``--mlflow_tracking_uri`` value; return it unchanged.

    Values starting with ``arn:`` must match the SageMaker managed MLflow ARN
    shape (``mlflow-tracking-server/...`` or ``mlflow-app/...``), anything
    else raises ``ValueError`` immediately, because the user explicitly asked
    for publishing and a typo'd ARN would otherwise surface as an opaque
    plugin error. Non-ARN values (http://, sqlite:///, ...) pass through
    untouched; MLflow itself validates those.
    """
    if tracking_uri.startswith("arn:") and not _SAGEMAKER_MLFLOW_ARN_RE.match(
        tracking_uri
    ):
        raise ValueError(
            f"--mlflow_tracking_uri looks like an ARN but is not a valid "
            f"SageMaker MLflow tracking server / MLflow App ARN: "
            f"{tracking_uri!r}. Expected "
            f"arn:aws:sagemaker:<region>:<account-id>:mlflow-app/<name> or "
            f"...:mlflow-tracking-server/<name>."
        )
    return tracking_uri


class MlflowPublisher:
    """Best-effort publisher of per-step metrics to a managed MLflow app.

    Construct via :meth:`create` (never directly): it owns the soft-import of
    ``mlflow``, tracking-URI/experiment/run setup, and the initial params dump.
    """

    def __init__(self, mlflow_module, run_id: str):
        self._mlflow = mlflow_module
        self.run_id = run_id
        self._consecutive_failures = 0
        self._ended = False

    # --- construction ------------------------------------------------------

    @classmethod
    def create(
        cls,
        tracking_uri: Optional[str],
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> Optional["MlflowPublisher"]:
        """Start an MLflow run against ``tracking_uri``; return the publisher.

        ``tracking_uri`` may be a SageMaker MLflow tracking server / App ARN
        (resolved by the ``sagemaker-mlflow`` plugin) or any MLflow-compatible
        tracking URI (``http://localhost:5000``, ``sqlite:///mlflow.db``, ...).
        ARN-shaped values are validated with :func:`validate_tracking_uri`,
        which raises ``ValueError`` on a malformed ARN (fail fast, nothing
        has been provisioned yet). Otherwise returns ``None`` (with a warning
        where appropriate) when publishing is not requested (no URI), the
        ``mlflow`` extra is not installed, or the tracking server cannot be
        reached. ``config``, when given, is logged as run params (values
        stringified and truncated to the MLflow limit).
        """
        if not tracking_uri:
            return None
        validate_tracking_uri(tracking_uri)

        try:
            import mlflow  # noqa: PLC0415, soft dependency, see module docstring
        except ImportError:
            logger.warning(
                "mlflow is not installed; skipping MLflow publishing. "
                "Install the extra: pip install -e '.[mlflow]' "
                "(metrics still go to metrics.jsonl)."
            )
            return None

        # With sagemaker-mlflow installed, its request-header provider tries
        # to parse every tracking URI as an ARN and logs a noisy (harmless)
        # "list index out of range" warning per request for plain http/sqlite
        # URIs. Silence just that registry logger for non-ARN destinations.
        if not tracking_uri.startswith("arn:"):
            logging.getLogger(
                "mlflow.tracking.request_header.registry"
            ).setLevel(logging.ERROR)

        try:
            mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            run = mlflow.start_run(run_name=run_name)
        except Exception as exc:
            hint = (
                "is the 'sagemaker-mlflow' plugin installed and do your AWS "
                "credentials allow access to the app?"
                if tracking_uri.startswith("arn:")
                else "is the tracking server reachable?"
            )
            logger.warning(
                "Failed to start MLflow run against %s: %r, continuing "
                "without MLflow (%s).",
                tracking_uri, exc, hint,
            )
            return None

        # Mark the experiment as model training (best-effort). Without the
        # `mlflow.experimentKind` tag, the MLflow 3.x UI infers a GenAI-style
        # experiment and lists training runs under "Evaluation runs". Only set
        # the tag when absent so a customer's own kind is never overwritten.
        try:
            client = mlflow.MlflowClient()
            exp_id = run.info.experiment_id
            if "mlflow.experimentKind" not in (client.get_experiment(exp_id).tags or {}):
                client.set_experiment_tag(
                    exp_id, "mlflow.experimentKind", "custom_model_development"
                )
        except Exception as exc:
            logger.warning("Could not set MLflow experiment kind tag: %r", exc)

        publisher = cls(mlflow, run.info.run_id)
        logger.info(
            "MLflow publishing enabled: tracking_uri=%s experiment=%s run_name=%s run_id=%s",
            tracking_uri, experiment_name, run_name, publisher.run_id,
        )
        if config:
            publisher.log_params(config)
        return publisher

    # --- publishing --------------------------------------------------------

    def log_params(self, config: dict) -> None:
        """Log a config dict as run params (stringified, truncated)."""
        params = {
            str(k): str(v)[:_MAX_PARAM_VALUE_LEN]
            for k, v in config.items()
            if not str(k).startswith("_")
        }
        self._call("log_params", lambda m: m.log_params(params))

    def log_metrics(self, record: dict, step: int) -> None:
        """Publish the numeric entries of one step record.

        Non-numeric values (strings such as the baseline record's ``phase``)
        and the redundant ``step`` key are skipped; the metric names keep the
        record's ``rollout/`` / ``train/`` / ``eval/`` namespacing verbatim so
        the MLflow view stays 1:1 with ``metrics.jsonl``.
        """
        metrics = {
            k: float(v)
            for k, v in record.items()
            if k != "step"
            and isinstance(v, numbers.Real)
            and not isinstance(v, bool)
        }
        if not metrics:
            return
        self._call("log_metrics", lambda m: m.log_metrics(metrics, step=step))

    def set_tags(self, tags: dict) -> None:
        """Set run tags (e.g. checkpoint model-package ARNs)."""
        clean = {str(k): str(v) for k, v in tags.items() if v is not None}
        if not clean:
            return
        self._call("set_tags", lambda m: m.set_tags(clean))

    def log_artifacts(self, paths: list) -> None:
        """Upload each existing file in ``paths`` as a run artifact."""
        for path in paths:
            if path and os.path.isfile(str(path)):
                p = str(path)
                self._call("log_artifact", lambda m, _p=p: m.log_artifact(_p))

    def log_rollout_traces(self, step: int, rollouts: list) -> None:
        """Publish decoded rollout episodes as MLflow traces (one per episode).

        Each item in ``rollouts`` is a plain dict:
        ``{"episode": int, "turns": [{"prompt", "response", "reward",
        "advantage", "done"}, ...]}``, already decoded to text by the caller
        (the publisher stays tokenizer-free). Each episode becomes one trace:
        a root ``AGENT`` span carrying step/episode/reward attributes, with
        one child ``LLM`` span per turn holding the prompt as input and the
        response as output. Traces are linked to this run automatically by
        MLflow (the trace processor records the active run as the source run)
        and appear in the experiment's Traces view.
        """
        for rollout in rollouts:
            self._call(
                "log_rollout_trace",
                lambda m, _r=rollout: self._emit_rollout_trace(m, step, _r),
            )

    @staticmethod
    def _emit_rollout_trace(mlflow_module, step: int, rollout: dict) -> None:
        """Emit one episode as a trace: root AGENT span + one LLM span per turn."""
        turns = rollout.get("turns") or []
        episode = rollout.get("episode", 0)
        total_reward = sum(float(t.get("reward") or 0.0) for t in turns)
        with mlflow_module.start_span(
            name=f"rollout_step{step}_ep{episode}",
            span_type="AGENT",
            attributes={
                "step": step,
                "episode": episode,
                "num_turns": len(turns),
                "reward": total_reward,
            },
        ) as root:
            root.set_inputs({"prompt": turns[0].get("prompt") if turns else None})
            for i, turn in enumerate(turns):
                with mlflow_module.start_span(
                    name=f"turn_{i}",
                    span_type="LLM",
                    attributes={
                        "reward": float(turn.get("reward") or 0.0),
                        "advantage": float(turn.get("advantage") or 0.0),
                        "done": bool(turn.get("done", False)),
                    },
                ) as span:
                    span.set_inputs({"prompt": turn.get("prompt")})
                    span.set_outputs({"response": turn.get("response")})
            root.set_outputs({
                "response": turns[-1].get("response") if turns else None,
                "reward": total_reward,
            })

    def end(self, status: str = "FINISHED") -> None:
        """Terminate the MLflow run (``FINISHED``/``FAILED``). Idempotent.

        Flushes async trace export first (best-effort): traces are exported on
        a background thread, so without the flush the final step's rollout
        traces could be dropped when the training process exits.
        """
        if self._ended:
            return
        self._ended = True
        try:
            self._mlflow.flush_trace_async_logging()
        except Exception:
            pass  # older mlflow builds may not have it; never block run end
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:
            logger.warning("Failed to end MLflow run: %r", exc)

    # --- internals ----------------------------------------------------------

    def _call(self, what: str, fn) -> None:
        """Run one mlflow call; warn on failure, self-disable after repeats."""
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            return
        try:
            fn(self._mlflow)
            self._consecutive_failures = 0
        except Exception as exc:
            self._consecutive_failures += 1
            logger.warning("MLflow %s failed (%d/%d): %r", what,
                           self._consecutive_failures,
                           _MAX_CONSECUTIVE_FAILURES, exc)
            if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Disabling MLflow publishing after %d consecutive "
                    "failures; the run continues and metrics.jsonl is "
                    "unaffected.", _MAX_CONSECUTIVE_FAILURES,
                )


def add_mlflow_args(parser_or_group) -> None:
    """Attach the shared MLflow CLI flags."""
    parser_or_group.add_argument(
        "--mlflow_tracking_uri", default=None,
        help="Where to publish training metrics. Accepts a SageMaker managed "
             "MLflow App ARN "
             "(arn:aws:sagemaker:...:mlflow-app/<name>; requires the "
             "sagemaker-mlflow plugin and AWS credentials with access) OR any "
             "MLflow-compatible tracking URI (http://localhost:5000, "
             "sqlite:///mlflow.db, ...). Requires: pip install -e '.[mlflow]'. "
             "Publishing is best-effort and never fails the training run.",
    )
    parser_or_group.add_argument(
        "--mlflow_experiment_name", default=None,
        help="MLflow experiment to log under (default: the MLflow default "
             "experiment).",
    )
    parser_or_group.add_argument(
        "--mlflow_run_name", default=None,
        help="MLflow run display name (default: the training session / run "
             "name).",
    )
    parser_or_group.add_argument(
        "--mlflow_log_rollouts", type=int, default=0,
        help="RL only: publish rollout episodes as MLflow traces (decoded "
             "prompt/response per turn, with reward and advantage), shown in "
             "the experiment's Traces view. N>0 = first N episodes per "
             "training step; -1 = ALL episodes (beware volume: batch_size x "
             "responses_per_prompt traces per step); 0 (default) = off.",
    )
