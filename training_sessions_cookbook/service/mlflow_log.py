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
import concurrent.futures
import numbers
import time
import os
import re
from typing import Optional

#: Attempts per assessment, and the linear backoff between them. Trace
#: ingestion is eventually consistent: a "trace not found" is usually "not yet".
_ASSESSMENT_ATTEMPTS = 3

#: Worker threads attaching assessments off the training thread.
_ASSESSMENT_WORKERS = 8

#: Upper bound on how long run teardown waits for outstanding assessments.
_ASSESSMENT_CLOSE_TIMEOUT_SECONDS = 30.0
_ASSESSMENT_RETRY_BACKOFF_SECONDS = 2.0

#: How long to wait for the backend to ingest the final step's traces before
#: attaching their assessments at run end. Mid-run batches need no wait: they
#: are drained a full training step later.
_ASSESSMENT_INGEST_GRACE_SECONDS = 5.0

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
        # (trace_id, reward) awaiting assessment; see _drain_assessments.
        # (trace_id, reward, step, episode) awaiting attachment; deferred by a
        # step so the backend has ingested the trace (see _drain_assessments).
        self._pending_assessments: "list[tuple[str, float, int, int]]" = []
        # Assessments are attached off the training thread: one HTTP call per
        # trace, no batch API, so a slow backend must not lengthen a step.
        self._assessment_pool = None
        self._assessment_futures: list = []
        self._assessments_failed = 0
        self._assessments_total = 0
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

        Each trace is named ``rollout_step30_ep7`` and appears in the Traces
        table's default-visible Trace name column; the step and episode are
        also tags, shown as discrete fields once the (opt-in) Tags column is
        enabled. The reward is attached as a feedback *assessment* (shown
        in the Assessments column, filterable by exact value:
        ``feedback.reward = '0.0'`` pulls up the rollouts that scored nothing),
        and the same coordinates go out as tags for precise queries, e.g.
        ``tags.step = '30' AND tags.reward = '0.0000'``.

        Traces are a *sampling* tool. MLflow exports them on a background
        queue holding 1000 by default and DISCARDS the overflow, so a step that
        publishes more than that loses traces (raise
        ``MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE`` if you really need it).
        Use ``--record_trajectories`` when every episode must be captured.

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
        # Assessments attach only to an ALREADY-INGESTED trace, and ingestion is
        # asynchronous end to end: flushing the client queue does not mean the
        # backend has committed the traces. Assessing immediately worked for a
        # handful of traces and failed for a thousand
        # ("RESOURCE_DOES_NOT_EXIST: Trace ... not found") - a race that only
        # appears at scale. So assessments for THIS step are queued and attached
        # at the start of the next step: a whole training step of delay, at no
        # cost to the hot path and with no retry loop or sleep.
        self._drain_assessments()

        emitted: "list" = []
        for rollout in rollouts:
            self._call(
                "log_rollout_trace",
                lambda m, _r=rollout: emitted.append(
                    self._emit_rollout_trace(m, step, _r)
                ),
            )
        self._pending_assessments.extend(
            (tid, reward, step, ep) for (tid, reward, ep) in (e for e in emitted if e)
        )

    _SOURCE_UNSET = object()
    _assessment_source_cache = _SOURCE_UNSET

    @classmethod
    def _assessment_source(cls):
        """``AssessmentSource`` identifying these as programmatic (CODE) feedback.

        MLflow's guidance is a specific, traceable ``source_id`` rather than a
        generic label, so the reward's provenance is unambiguous when a trace
        also carries human or LLM-judge feedback. Returns ``None`` on older
        mlflow builds without the entities, where feedback is logged unsourced.
        """
        if cls._assessment_source_cache is not cls._SOURCE_UNSET:
            return cls._assessment_source_cache
        try:
            from mlflow.entities import AssessmentSource
            from mlflow.entities.assessment_source import AssessmentSourceType

            source = AssessmentSource(
                source_type=AssessmentSourceType.CODE,
                source_id="training_sessions_cookbook.rl.reward",
            )
        except Exception:
            source = None
        cls._assessment_source_cache = source
        return source

    def _drain_assessments(self) -> None:
        """Hand queued reward assessments (from an earlier step) to a worker pool.

        Fire-and-forget: observability must not lengthen a training step, and
        MLflow offers no batch feedback API - each assessment is its own HTTP
        round-trip that may retry. So the training thread only submits, and
        :meth:`end` settles the stragglers within a bounded wait.

        Failures are counted and reported once rather than per trace, and they
        deliberately bypass :meth:`_call`: assessments are a convenience, and
        must never trip the publisher-wide disable that protects metric
        logging (a burst of them once silenced an entire run's metrics).
        """
        pending, self._pending_assessments = self._pending_assessments, []
        if not pending:
            return
        log_feedback = getattr(self._mlflow, "log_feedback", None)
        if log_feedback is None:  # older mlflow without the assessments API
            return

        def _one(item):
            # Resolved inside the worker: building the source imports
            # mlflow.entities, and that import must not land on the training
            # thread (measured ~2s on first touch).
            source = self._assessment_source()
            trace_id, reward, step, episode = item
            kwargs = {
                "trace_id": trace_id,
                "name": "reward",
                "value": float(reward),
                "metadata": {"step": str(step), "episode": str(episode)},
            }
            if source is not None:
                kwargs["source"] = source
            # Ingestion is eventually consistent, so a miss is usually "not
            # yet" rather than "never": retry briefly instead of relying on
            # the one-step deferral alone (back-to-back steps, or a fast
            # recipe, can close that gap to nothing).
            for attempt in range(_ASSESSMENT_ATTEMPTS):
                try:
                    log_feedback(**kwargs)
                    return True
                except Exception:
                    if attempt == _ASSESSMENT_ATTEMPTS - 1:
                        return False
                    time.sleep(_ASSESSMENT_RETRY_BACKOFF_SECONDS * (attempt + 1))
            return False

        if self._assessment_pool is None:
            self._assessment_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=_ASSESSMENT_WORKERS, thread_name_prefix="mlflow-assess",
            )
        self._assessment_futures.extend(
            self._assessment_pool.submit(_one, item) for item in pending
        )
        self._tally_assessments()

    def _tally_assessments(self) -> None:
        """Fold finished assessment futures into the run totals.

        Called wherever futures are added or awaited, so the pending list stays
        flat instead of growing with every published trace.
        """
        still_running = []
        for fut in self._assessment_futures:
            if not fut.done():
                still_running.append(fut)
                continue
            self._assessments_total += 1
            # Cancelled futures are counted, never unwrapped: CancelledError
            # derives from BaseException, so result() would escape the guard.
            if fut.cancelled():
                self._assessments_failed += 1
                continue
            try:
                if not fut.result():
                    self._assessments_failed += 1
            except Exception:
                self._assessments_failed += 1
        self._assessment_futures = still_running

    def _close_assessments(self) -> None:
        """Settle outstanding assessments within a bounded wait, then report.

        The wait is bounded because this runs in the run's teardown path (a
        shielded ``finally``, so it also runs on Ctrl-C): a slow or unreachable
        backend must not hold the process open. Anything unfinished is
        cancelled and counted.
        """
        if self._assessment_pool is None:
            return
        concurrent.futures.wait(
            self._assessment_futures, timeout=_ASSESSMENT_CLOSE_TIMEOUT_SECONDS,
        )
        self._tally_assessments()
        abandoned = len(self._assessment_futures)
        self._assessment_pool.shutdown(wait=False, cancel_futures=True)
        self._assessment_pool = None
        self._assessment_futures = []
        unattached = self._assessments_failed + abandoned
        if unattached:
            logger.warning(
                "MLflow: %d of %d rollout reward assessments could not be "
                "attached (traces may not have been ingested yet); traces, "
                "metrics and the reward tag are unaffected.",
                unattached, self._assessments_total + abandoned,
            )

    @staticmethod
    def _emit_rollout_trace(mlflow_module, step: int, rollout: dict):
        """Emit one episode as a trace: root AGENT span + one LLM span per turn.

        Returns ``(trace_id, total_reward, episode)`` so the caller can queue
        the reward assessment for attachment once the trace is ingested (see
        :meth:`_drain_assessments`).
        """
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
            # Trace TAGS (not span attributes) are what the Traces view can
            # filter on, so the same coordinates go out both ways: attributes
            # for reading a trace, tags for finding one. ``step`` is
            # zero-padded because tag values are strings and sort
            # lexicographically ("0009" before "0010"). Verified against
            # mlflow 3.14: `search_traces(filter_string="tags.step = '0030'")`
            # and compound filters (step AND reward) both match.
            update = getattr(mlflow_module, "update_current_trace", None)
            if update is not None:  # older mlflow: attributes only
                # Deliberately NOT grouped into an MLflow *session*: a session
                # is a conversation, and its view numbers members "Turn 0,
                # Turn 1, ...". Rollouts in a step are independent episodes,
                # and multi-turn RL has real turns of its own (a tool call and
                # its result), so that labelling would directly contradict the
                # domain concept. Real turns are the child spans of one trace.
                #
                # The Request/Response previews are left alone: they exist to
                # show the actual prompt and response, and prefixing them with
                # coordinates would pollute the content. The step and episode
                # are already in the trace NAME (a default-visible column) and
                # in tags; the reward is a default-visible assessment.
                update(tags={
                    "step": str(step),
                    "episode": str(episode),
                    "reward": f"{total_reward:.4f}",
                    "num_turns": str(len(turns)),
                })
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
            trace_id = getattr(root, "trace_id", None)
        return (trace_id, total_reward, episode) if trace_id else None

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
            # The last step's assessments are still queued: the traces were
            # just flushed, so give the backend a moment to ingest them before
            # attaching (best-effort - a miss costs a badge, not data).
            if self._pending_assessments:
                time.sleep(_ASSESSMENT_INGEST_GRACE_SECONDS)
                self._drain_assessments()
        except Exception:
            pass  # older mlflow builds may not have it; never block run end
        # Settle whatever the worker pool still owes, bounded, and report once.
        try:
            self._close_assessments()
        except Exception:
            pass
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
        "--mlflow_traces_per_step", type=int, default=0,
        help="RL only: how many rollout episodes to publish as MLflow traces "
             "per training step (decoded prompt/response per turn, with the "
             "reward as a filterable assessment). This is a SAMPLING knob: "
             "small values (4-32) are what a human reads, and large ones "
             "overflow MLflow's async export queue (1000 traces by default), "
             "which then discards them. Use --record_trajectories to "
             "capture every episode losslessly to the run folder instead. "
             "N>0 = the first N episodes of each step; -1 = ALL episodes; "
             "0 (default) = off.",
    )
