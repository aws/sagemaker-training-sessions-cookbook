"""Tests for service.mlflow_log, MLflow publishing (mlflow_mock mlflow module).

No network, no real mlflow: a mlflow_mock ``mlflow`` module is injected into
``sys.modules`` so ``MlflowPublisher.create``'s lazy import picks it up. The
tests pin the contract the trainers rely on:

* opt-in (None ARN -> None publisher) and soft-dependency (ImportError -> None)
* numeric-only metric filtering, ``step`` key exclusion, bool exclusion
* params stringified + truncated; private ``_``-prefixed keys dropped
* exception isolation: a broken tracking server never raises into the
  training loop, and repeated failures self-disable the publisher
* run end status is forwarded and ``end`` is idempotent
"""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from training_sessions_cookbook.service.mlflow_log import (
    _MAX_CONSECUTIVE_FAILURES,
    _MAX_PARAM_VALUE_LEN,
    MlflowPublisher,
    validate_tracking_uri,
)

from tests.mock_mlflow import FAKE_ARN, FAKE_URI, make_mock_mlflow


class MlflowPublisherCreateTest(unittest.TestCase):
    def test_none_arn_returns_none(self):
        self.assertIsNone(MlflowPublisher.create(None))
        self.assertIsNone(MlflowPublisher.create(""))

    def test_missing_mlflow_returns_none(self):
        with mock.patch.dict(sys.modules, {"mlflow": None}):
            # import mlflow raises ImportError when sys.modules entry is None
            self.assertIsNone(MlflowPublisher.create(FAKE_ARN))

    def test_create_sets_uri_experiment_run_and_params(self):
        mlflow_mock = make_mock_mlflow()
        with mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}):
            pub = MlflowPublisher.create(
                FAKE_ARN,
                experiment_name="exp",
                run_name="my-run",
                config={"learning_rate": 4e-5, "_private": "skip me"},
            )
        self.assertIsNotNone(pub)
        self.assertEqual(pub.run_id, "run-123")
        names = [c[0] for c in mlflow_mock.calls]
        self.assertEqual(
            names,
            ["set_tracking_uri", "set_experiment", "start_run",
             "get_experiment", "set_experiment_tag", "log_params"],
        )
        self.assertEqual(mlflow_mock.calls[0][1], (FAKE_ARN,))
        # Private keys are dropped from params; values are stringified.
        params = [c for c in mlflow_mock.calls if c[0] == "log_params"][0][1][0]
        self.assertEqual(params, {"learning_rate": "4e-05"})

    def test_create_tags_experiment_kind_when_absent(self):
        # Fresh experiment (no tags): the publisher marks it as model
        # training so the MLflow 3.x UI shows runs under "Runs", not
        # "Evaluation runs".
        mlflow_mock = make_mock_mlflow()
        with mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}):
            MlflowPublisher.create(FAKE_ARN, experiment_name="exp")
        tag_calls = [c for c in mlflow_mock.calls if c[0] == "set_experiment_tag"]
        self.assertEqual(
            tag_calls,
            [("set_experiment_tag",
              ("exp-1", "mlflow.experimentKind", "custom_model_development"),
              {})],
        )

    def test_create_preserves_existing_experiment_kind(self):
        # Customer already chose a kind: never overwrite it.
        mlflow_mock = make_mock_mlflow(
            experiment_tags={"mlflow.experimentKind": "genai_development"})
        with mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}):
            MlflowPublisher.create(FAKE_ARN, experiment_name="exp")
        self.assertEqual(
            [c for c in mlflow_mock.calls if c[0] == "set_experiment_tag"], [])
        self.assertEqual(
            mlflow_mock.experiment_tags["mlflow.experimentKind"], "genai_development")

    def test_create_survives_experiment_tagging_failure(self):
        # Tagging is best-effort: a broken client must not kill create().
        mlflow_mock = make_mock_mlflow()

        class BoomClient:
            def __init__(self):
                raise RuntimeError("no client for you")

        mlflow_mock.MlflowClient = BoomClient
        with mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}):
            pub = MlflowPublisher.create(FAKE_ARN, experiment_name="exp")
        self.assertIsNotNone(pub)

    def test_create_failure_returns_none(self):
        mlflow_mock = make_mock_mlflow()

        def boom(*a, **kw):
            raise RuntimeError("no credentials")

        mlflow_mock.set_tracking_uri = boom
        with mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}):
            self.assertIsNone(MlflowPublisher.create(FAKE_ARN))

    def test_param_values_truncated(self):
        mlflow_mock = make_mock_mlflow()
        with mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}):
            MlflowPublisher.create(FAKE_ARN, config={"long": "x" * 2000})
        params = [c for c in mlflow_mock.calls if c[0] == "log_params"][0][1][0]
        self.assertEqual(len(params["long"]), _MAX_PARAM_VALUE_LEN)


class MlflowPublisherLoggingTest(unittest.TestCase):
    def setUp(self):
        self.mlflow_mock = make_mock_mlflow()
        self.pub = MlflowPublisher(self.mlflow_mock, "run-123")

    def _metric_calls(self):
        return [c for c in self.mlflow_mock.calls if c[0] == "log_metrics"]

    def test_log_metrics_filters_non_numeric_and_step(self):
        # Mirrors a real RL baseline record: `phase` is a string, `step` is
        # the record's own coordinate: neither is an MLflow metric.
        record = {
            "step": 7,
            "phase": "pre_training_baseline",
            "rollout/reward_mean": 0.5,
            "rollout/num_episodes": 64,
            "train/loss": None,
            "flag": True,
        }
        self.pub.log_metrics(record, step=7)
        calls = self._metric_calls()
        self.assertEqual(len(calls), 1)
        metrics, kwargs = calls[0][1][0], calls[0][2]
        self.assertEqual(
            metrics,
            {"rollout/reward_mean": 0.5, "rollout/num_episodes": 64.0},
        )
        self.assertEqual(kwargs, {"step": 7})

    def test_log_metrics_all_non_numeric_is_noop(self):
        self.pub.log_metrics({"step": 1, "phase": "x"}, step=1)
        self.assertEqual(self._metric_calls(), [])

    def test_set_tags_drops_none_values(self):
        self.pub.set_tags({"checkpoint_final": "arn:...:model-package/x/1",
                           "missing": None})
        calls = [c for c in self.mlflow_mock.calls if c[0] == "set_tags"]
        self.assertEqual(calls[0][1][0],
                         {"checkpoint_final": "arn:...:model-package/x/1"})

    def test_log_artifacts_only_existing_files(self):
        import tempfile
        with tempfile.NamedTemporaryFile() as f:
            self.pub.log_artifacts([f.name, "/nonexistent/nope.jsonl", None])
        calls = [c for c in self.mlflow_mock.calls if c[0] == "log_artifact"]
        self.assertEqual(len(calls), 1)

    def test_end_forwards_status_and_is_idempotent(self):
        self.pub.end("FAILED")
        self.pub.end("FINISHED")
        calls = [c for c in self.mlflow_mock.calls if c[0] == "end_run"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], {"status": "FAILED"})


class MlflowPublisherFailureIsolationTest(unittest.TestCase):
    def test_publish_failure_never_raises_and_self_disables(self):
        mlflow_mock = make_mock_mlflow()

        def boom(*a, **kw):
            raise RuntimeError("server down")

        mlflow_mock.log_metrics = boom
        pub = MlflowPublisher(mlflow_mock, "run-123")

        # Never raises into the training loop, no matter how often it fails.
        for step in range(_MAX_CONSECUTIVE_FAILURES + 3):
            pub.log_metrics({"train/loss": 1.0}, step=step)

        # After self-disable, other publish calls are skipped too.
        pub.set_tags({"k": "v"})
        self.assertEqual([c for c in mlflow_mock.calls if c[0] == "set_tags"], [])

    def test_success_resets_failure_counter(self):
        mlflow_mock = make_mock_mlflow()
        real_log_metrics = mlflow_mock.log_metrics
        state = {"fail": True}

        def flaky(*a, **kw):
            if state["fail"]:
                raise RuntimeError("blip")
            return real_log_metrics(*a, **kw)

        mlflow_mock.log_metrics = flaky
        pub = MlflowPublisher(mlflow_mock, "run-123")

        for step in range(_MAX_CONSECUTIVE_FAILURES - 1):
            pub.log_metrics({"train/loss": 1.0}, step=step)
        state["fail"] = False
        pub.log_metrics({"train/loss": 1.0}, step=99)  # success resets counter
        state["fail"] = True
        pub.log_metrics({"train/loss": 1.0}, step=100)  # 1 failure, not disabled
        state["fail"] = False
        pub.log_metrics({"train/loss": 1.0}, step=101)

        successes = [c for c in mlflow_mock.calls if c[0] == "log_metrics"]
        self.assertEqual(len(successes), 2)


class ValidateTrackingUriTest(unittest.TestCase):
    """ARN-shaped values must match the SageMaker MLflow ARN pattern."""

    def test_valid_mlflow_app_arn(self):
        self.assertEqual(validate_tracking_uri(FAKE_ARN), FAKE_ARN)

    def test_valid_tracking_server_arn(self):
        arn = ("arn:aws:sagemaker:us-east-1:123456789012:"
               "mlflow-tracking-server/my-server")
        self.assertEqual(validate_tracking_uri(arn), arn)

    def test_non_arn_uris_pass_through(self):
        for uri in (FAKE_URI, "https://mlflow.corp:8443", "sqlite:///m.db",
                    "file:///tmp/mlruns"):
            self.assertEqual(validate_tracking_uri(uri), uri)

    def test_malformed_arns_raise(self):
        bad = [
            "arn:aws:sagemaker:us-west-2:123:mlflow-app/x", # short acct
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/x/1",
            "arn:aws:s3:::bucket/key",
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/", # no name
        ]
        for arn in bad:
            with self.assertRaises(ValueError, msg=arn):
                validate_tracking_uri(arn)

    def test_create_raises_on_malformed_arn(self):
        # Fail-fast happens before the mlflow import is even needed.
        with self.assertRaises(ValueError):
            MlflowPublisher.create("arn:aws:iam::123456789012:role/NotMlflow")


class GenericTrackingUriTest(unittest.TestCase):
    """create() accepts any MLflow tracking URI, not just SageMaker ARNs."""

    def test_create_with_http_uri(self):
        mlflow_mock = make_mock_mlflow()
        with mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}):
            pub = MlflowPublisher.create(FAKE_URI, run_name="local-run")
        self.assertIsNotNone(pub)
        self.assertEqual(mlflow_mock.calls[0], ("set_tracking_uri", (FAKE_URI,), {}))

    def test_non_arn_uri_silences_plugin_header_registry_logger(self):
        # sagemaker-mlflow's header provider warns per-request on non-ARN
        # URIs; create() bumps that logger to ERROR for such destinations.
        import logging
        reg = logging.getLogger("mlflow.tracking.request_header.registry")
        old_level = reg.level
        try:
            reg.setLevel(logging.NOTSET)
            mlflow_mock = make_mock_mlflow()
            with mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}):
                MlflowPublisher.create(FAKE_URI)
            self.assertEqual(reg.level, logging.ERROR)

            # ARN destinations leave the logger untouched.
            reg.setLevel(logging.NOTSET)
            mlflow_mock = make_mock_mlflow()
            with mock.patch.dict(sys.modules, {"mlflow": mlflow_mock}):
                MlflowPublisher.create(FAKE_ARN)
            self.assertEqual(reg.level, logging.NOTSET)
        finally:
            reg.setLevel(old_level)


class MetricsLoggerIntegrationTest(unittest.TestCase):
    """rl.train.MetricsLogger mirrors its record to the publisher."""

    def test_log_step_publishes_record(self):
        import tempfile
        from training_sessions_cookbook.rl.train import MetricsLogger

        mlflow_mock = make_mock_mlflow()
        pub = MlflowPublisher(mlflow_mock, "run-123")
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            ml = MetricsLogger(f.name, publisher=pub)
            record = ml.log_step(
                step=3,
                episodes=[],
                transitions=[],
                fb_result=None,
                sample_seconds=1.0,
                train_seconds=2.0,
                step_total_seconds=3.0,
                effective_lr=4e-5,
                kl_coef=0.0,
            )
        calls = [c for c in mlflow_mock.calls if c[0] == "log_metrics"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], {"step": 3})
        published = calls[0][1][0]
        self.assertIn("time/step_total_seconds", published)
        self.assertNotIn("step", published)
        # Every published key exists in the JSONL record with the same value.
        for k, v in published.items():
            self.assertEqual(float(record[k]), v)


class RolloutTraceTest(unittest.TestCase):
    """log_rollout_traces emits one trace per episode with per-turn spans."""

    def _rollout(self, n_turns=2, episode=0):
        return {"episode": episode, "turns": [
            {"prompt": f"p{i}", "response": f"r{i}", "reward": 1.0,
             "advantage": 0.5, "done": i == n_turns - 1}
            for i in range(n_turns)
        ]}

    def test_traces_structure(self):
        mlflow_mock = make_mock_mlflow()
        pub = MlflowPublisher(mlflow_mock, "run-123")
        pub.log_rollout_traces(3, [self._rollout(2, 0), self._rollout(1, 1)])
        spans = [c for c in mlflow_mock.calls if c[0] == "start_span"]
        # ep0: root + 2 turns; ep1: root + 1 turn
        names = [s[1][0] for s in spans]
        self.assertEqual(names, ["rollout_step3_ep0", "turn_0", "turn_1",
                                 "rollout_step3_ep1", "turn_0"])
        root = spans[0][2]
        self.assertEqual(root["span_type"], "AGENT")
        self.assertEqual(root["attributes"]["step"], 3)
        self.assertEqual(root["attributes"]["reward"], 2.0)
        turn = spans[1][2]
        self.assertEqual(turn["span_type"], "LLM")
        self.assertEqual(turn["inputs"], {"prompt": "p0"})
        self.assertEqual(turn["outputs"], {"response": "r0"})

    def test_trace_failure_is_isolated(self):
        mlflow_mock = make_mock_mlflow()

        def boom(*a, **kw):
            raise RuntimeError("span exploded")

        mlflow_mock.start_span = boom
        pub = MlflowPublisher(mlflow_mock, "run-123")
        pub.log_rollout_traces(0, [self._rollout()])  # must not raise

    def test_end_flushes_traces_before_ending_run(self):
        mlflow_mock = make_mock_mlflow()
        pub = MlflowPublisher(mlflow_mock, "run-123")
        pub.end("FINISHED")
        names = [c[0] for c in mlflow_mock.calls]
        self.assertEqual(names, ["flush_trace_async_logging", "end_run"])

    def test_end_survives_missing_flush(self):
        mlflow_mock = make_mock_mlflow()
        del mlflow_mock.flush_trace_async_logging  # older mlflow builds
        pub = MlflowPublisher(mlflow_mock, "run-123")
        pub.end("FINISHED")
        self.assertEqual([c[0] for c in mlflow_mock.calls], ["end_run"])


class EpisodesToRolloutDictsTest(unittest.TestCase):
    """rl.train.episodes_to_rollout_dicts decodes and caps episodes."""

    def test_decode_and_cap(self):
        from training_sessions_cookbook.rl.train import episodes_to_rollout_dicts

        class Tok:
            def decode(self, toks):
                return "text:" + ",".join(map(str, toks))

        class T:
            def __init__(self):
                self.obs_tokens = [1, 2]
                self.act_tokens = [3]
                self.reward = 1.0
                self.advantage = 0.25
                self.done = True

        episodes = [[T()], [T(), T()], [T()]]
        out = episodes_to_rollout_dicts(episodes, Tok(), cap=2)
        self.assertEqual(len(out), 2)  # capped
        self.assertEqual(out[0]["turns"][0]["prompt"], "text:1,2")
        self.assertEqual(out[0]["turns"][0]["response"], "text:3")
        self.assertEqual(out[1]["episode"], 1)
        self.assertEqual(len(out[1]["turns"]), 2)
        # -1 (or any negative) = ALL episodes; 0 = none.
        self.assertEqual(len(episodes_to_rollout_dicts(episodes, Tok(), cap=-1)), 3)
        self.assertEqual(len(episodes_to_rollout_dicts(episodes, Tok(), cap=0)), 0)

    def test_decoder_failure_degrades(self):
        from training_sessions_cookbook.rl.train import episodes_to_rollout_dicts

        class BadTok:
            def decode(self, toks):
                raise ValueError("nope")

        class T:
            obs_tokens = [1]
            act_tokens = [2]
            reward = 0.0
            advantage = 0.0
            done = False

        out = episodes_to_rollout_dicts([[T()]], BadTok(), cap=5)
        self.assertEqual(out[0]["turns"][0]["prompt"], "[1]")


if __name__ == "__main__":
    unittest.main()
