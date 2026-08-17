"""Tests for service.runlog.RunLog, the single-object recipe logging facade.

Same no-network approach as test_mlflow_log: a mlflow_mock ``mlflow`` module is
injected into ``sys.modules`` so ``MlflowPublisher.create``'s lazy import
picks it up. Pins the contract recipes rely on:

* no mlflow flags in config -> pure-local mode (jsonl only), no error
* with a tracking URI -> dual-sink logging, config.json + git_hash written
* checkpoints -> MLflow tag (step vs name forms); the checkpoints.jsonl
  record itself is written by service.checkpoint_utils, not RunLog
* context manager -> FINISHED on clean exit, FAILED on exception,
  artifacts uploaded, close idempotent
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from training_sessions_cookbook.service.runlog import RunLog
from tests.mock_mlflow import FAKE_URI, make_mock_mlflow


def _open(tmpdir, config=None, mlflow_mock=None):
    cfg = {"learning_rate": 1e-4}
    cfg.update(config or {})
    ctx = mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}) if mlflow_mock else None
    if ctx:
        with ctx:
            return RunLog.open("t", config=cfg, output_dir=tmpdir)
    return RunLog.open("t", config=cfg, output_dir=tmpdir)


class RunLogLocalOnlyTest(unittest.TestCase):
    def test_no_mlflow_flags_is_pure_local(self):
        with tempfile.TemporaryDirectory() as d:
            runlog = _open(d)
            self.assertIsNone(runlog.publisher)
            runlog.log_metrics({"step": 0, "train/loss": 1.0}, step=0)
            runlog.log_checkpoint("final", "arn:aws:...:model-package/x/1")
            runlog.close()
            # config.json with git_hash; metrics written. checkpoints.jsonl is
            # owned by service.checkpoint_utils, so log_checkpoint (tags-only,
            # and there is no sink here) writes no local file.
            cfg = json.load(open(os.path.join(d, "config.json")))
            self.assertIn("git_hash", cfg)
            lines = open(os.path.join(d, "metrics.jsonl")).read().splitlines()
            self.assertEqual(json.loads(lines[0])["train/loss"], 1.0)
            self.assertFalse(os.path.exists(os.path.join(d, "checkpoints.jsonl")))


class RunLogMlflowTest(unittest.TestCase):
    def _make(self, d):
        mlflow_mock = make_mock_mlflow()
        runlog = _open(d, config={"mlflow_tracking_uri": FAKE_URI,
                                  "mlflow_experiment_name": "exp"}, mlflow_mock=mlflow_mock)
        return mlflow_mock, runlog

    def test_dual_sink_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            mlflow_mock, runlog = self._make(d)
            self.assertIsNotNone(runlog.publisher)
            runlog.log_metrics({"step": 2, "train/loss": 0.5}, step=2)
            # jsonl side
            rec = json.loads(open(os.path.join(d, "metrics.jsonl")).read())
            self.assertEqual(rec["step"], 2)
            # mlflow side
            mcall = [c for c in mlflow_mock.calls if c[0] == "log_metrics"][0]
            self.assertEqual(mcall[1][0], {"train/loss": 0.5})
            self.assertEqual(mcall[2], {"step": 2})

    def test_step_defaults_from_record(self):
        with tempfile.TemporaryDirectory() as d:
            mlflow_mock, runlog = self._make(d)
            runlog.log_metrics({"step": 7, "train/loss": 0.5})  # no step kwarg
            mcall = [c for c in mlflow_mock.calls if c[0] == "log_metrics"][0]
            self.assertEqual(mcall[2], {"step": 7})

    def test_checkpoint_step_and_name_tags(self):
        with tempfile.TemporaryDirectory() as d:
            mlflow_mock, runlog = self._make(d)
            runlog.log_checkpoint(3, "arn:x/3")
            runlog.log_checkpoint("final", "arn:x/9")
            runlog.log_checkpoint("skipped", None)  # no-op
            tags = [c[1][0] for c in mlflow_mock.calls if c[0] == "set_tags"]
            self.assertEqual(tags, [{"checkpoint_step_3": "arn:x/3"},
                                    {"checkpoint_final": "arn:x/9"}])
            # checkpoints.jsonl is written by service.checkpoint_utils, not here.
            self.assertFalse(os.path.exists(os.path.join(d, "checkpoints.jsonl")))

    def test_context_manager_finished_and_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            mlflow_mock, runlog = self._make(d)
            with runlog:
                runlog.log_metrics({"step": 0, "train/loss": 1.0}, step=0)
            ends = [c for c in mlflow_mock.calls if c[0] == "end_run"]
            self.assertEqual(ends[-1][2], {"status": "FINISHED"})
            arts = [os.path.basename(c[1][0]) for c in mlflow_mock.calls if c[0] == "log_artifact"]
            self.assertEqual(sorted(arts), ["config.json", "metrics.jsonl"])

    def test_context_manager_failed_on_exception(self):
        with tempfile.TemporaryDirectory() as d:
            mlflow_mock, runlog = self._make(d)
            with self.assertRaises(RuntimeError):
                with runlog:
                    raise RuntimeError("training exploded")
            ends = [c for c in mlflow_mock.calls if c[0] == "end_run"]
            self.assertEqual(ends[-1][2], {"status": "FAILED"})

    def test_close_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            mlflow_mock, runlog = self._make(d)
            runlog.close()
            runlog.close("FAILED")  # ignored, already closed
            ends = [c for c in mlflow_mock.calls if c[0] == "end_run"]
            self.assertEqual(len(ends), 1)
            self.assertEqual(ends[0][2], {"status": "FINISHED"})


class RunLogSinkExtensibilityTest(unittest.TestCase):
    """A new tracker = one sink class; RunLog fans out to all sinks.

    This pins the extension contract a future WandbSink would rely on:
    duck-typed optional methods, forwarded only when implemented, alongside
    the built-in MLflow sink and the always-on local files.
    """

    class RecorderSink:
        """Minimal sink implementing a subset of the contract."""
        def __init__(self):
            self.calls = []

        def log_metrics(self, record, step):
            self.calls.append(("log_metrics", record, step))

        def set_tags(self, tags):
            self.calls.append(("set_tags", tags))

        def end(self, status):
            self.calls.append(("end", status))
        # note: no log_rollout_traces / log_artifacts; both must be skipped safely

    def test_fanout_to_custom_sink_alongside_mlflow(self):
        mlflow_mock = make_mock_mlflow()
        recorder = self.RecorderSink()
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}):
                runlog = RunLog.open("t", config={"mlflow_tracking_uri": FAKE_URI},
                                     output_dir=d)
            runlog.sinks.append(recorder)
            self.assertEqual(len(runlog.sinks), 2)
            self.assertIsNotNone(runlog.publisher)  # mlflow sink still found

            with runlog:
                runlog.log_metrics({"step": 0, "train/loss": 1.0}, step=0)
                runlog.log_checkpoint("final", "arn:x/1")
                runlog.log_rollout_traces(0, [])  # recorder lacks it, no error

            # Custom sink saw metrics, checkpoint tag, and FINISHED close.
            names = [c[0] for c in recorder.calls]
            self.assertEqual(names, ["log_metrics", "set_tags", "end"])
            self.assertEqual(recorder.calls[-1], ("end", "FINISHED"))
            # MLflow sink saw the same events.
            self.assertTrue(any(c[0] == "log_metrics" for c in mlflow_mock.calls))
            self.assertTrue(any(c[0] == "end_run" for c in mlflow_mock.calls))
            # Local ground truth written regardless of sinks.
            self.assertTrue(os.path.isfile(os.path.join(d, "metrics.jsonl")))

    def test_raising_sink_is_isolated(self):
        # The sink contract is ENFORCED by _fanout, not just documented: a
        # buggy sink that raises must not propagate into the training loop,
        # and the remaining sinks must still receive the call.
        class BuggySink:
            def log_metrics(self, record, step):
                raise RuntimeError("buggy sink exploded")

            def end(self, status):
                raise RuntimeError("even close explodes")

        recorder = self.RecorderSink()
        with tempfile.TemporaryDirectory() as d:
            runlog = RunLog.open("t", config={}, output_dir=d)
            runlog.sinks.extend([BuggySink(), recorder])

            with runlog:  # must not raise despite BuggySink
                runlog.log_metrics({"step": 0, "train/loss": 1.0}, step=0)

            # Healthy sink got everything; local file written.
            self.assertEqual([c[0] for c in recorder.calls],
                             ["log_metrics", "end"])
            self.assertEqual(recorder.calls[-1], ("end", "FINISHED"))
            self.assertTrue(os.path.isfile(os.path.join(d, "metrics.jsonl")))

    def test_pure_local_when_no_sink_configured(self):
        with tempfile.TemporaryDirectory() as d:
            runlog = RunLog.open("t", config={}, output_dir=d)
            self.assertEqual(runlog.sinks, [])
            self.assertIsNone(runlog.publisher)
            runlog.log_metrics({"step": 0, "train/loss": 1.0}, step=0)
            runlog.close()  # no sinks, still fine


if __name__ == "__main__":
    unittest.main()
