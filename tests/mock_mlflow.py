"""A mock ``mlflow`` module for the test suite.

``make_mock_mlflow`` builds a mock ``mlflow`` module that tests inject into
``sys.modules`` (``mock.patch.dict(sys.modules, {"mlflow": mock_module})``) so
``MlflowPublisher.create``'s lazy import picks it up. No network and no real
mlflow dependency are needed.
"""

from __future__ import annotations

import contextlib
import types
from unittest import mock

FAKE_ARN = (
    "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/test-app"
)
FAKE_URI = "http://localhost:5000"


def make_mock_mlflow(experiment_tags=None):
    """A minimal mock of the mlflow fluent API that records every call.

    ``experiment_tags`` seeds the tags that the mock_module ``MlflowClient``'s
    ``get_experiment`` reports (default: none, mimics a fresh experiment).
    """
    mock_module = types.ModuleType("mlflow")
    mock_module.calls = []
    mock_module.experiment_tags = dict(experiment_tags or {})

    def _record(name):
        def fn(*a, **kw):
            mock_module.calls.append((name, a, kw))
            if name == "start_run":
                run = mock.Mock()
                run.info.run_id = "run-123"
                run.info.experiment_id = "exp-1"
                return run
            return None
        return fn

    for name in ("set_tracking_uri", "set_experiment", "start_run",
                 "log_params", "log_metrics", "set_tags", "log_artifact",
                 "end_run"):
        setattr(mock_module, name, _record(name))

    class _FakeClient:
        def get_experiment(self, exp_id):
            mock_module.calls.append(("get_experiment", (exp_id,), {}))
            exp = mock.Mock()
            exp.tags = dict(mock_module.experiment_tags)
            return exp

        def set_experiment_tag(self, exp_id, key, value):
            mock_module.calls.append(("set_experiment_tag", (exp_id, key, value), {}))
            mock_module.experiment_tags[key] = value

    mock_module.MlflowClient = _FakeClient

    # Spans: record every start_span with its kwargs plus the set_* payloads.

    @contextlib.contextmanager
    def start_span(name="span", span_type="UNKNOWN", attributes=None, **kw):
        span = mock.Mock()
        record = {"name": name, "span_type": span_type,
                  "attributes": dict(attributes or {}), "inputs": None, "outputs": None}
        span.set_inputs.side_effect = lambda v: record.update(inputs=v)
        span.set_outputs.side_effect = lambda v: record.update(outputs=v)
        mock_module.calls.append(("start_span", (name,), record))
        yield span

    mock_module.start_span = start_span
    mock_module.flush_trace_async_logging = _record("flush_trace_async_logging")
    return mock_module
