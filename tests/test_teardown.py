"""Session-lifecycle (teardown) tests for BOTH trainers.

The contract, symmetric across SFT and RL: success deletes the session (no
stopped records pile up); failure stops it but keeps the record for
debugging; a teardown error never masks the original exception; and a
failure during setup (after the session opens, before the loop owns it)
stops the session too.

No network / no SDK: session clients are mocked throughout.
"""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import training_sessions_cookbook.sft.train as sft_train


def _mock_args(tmpdir):
    """Minimal args namespace: no mlflow flags -> RunLog runs pure-local."""
    return mock.Mock(
        training_session_name="crash-unit-test",
        log_path=tmpdir,
        mlflow_tracking_uri=None,
        mlflow_experiment_name=None,
        mlflow_run_name=None,
    )


class SftStopOnFailureTest(unittest.TestCase):
    def _run_main(self, tmpdir, train_side_effect, stop_side_effect=None):
        """Run sft.train.main() with session open + training loop mocked."""
        svc = mock.Mock()
        training_client = mock.Mock()
        training_client.training_session_name = "crash-unit-test"
        if stop_side_effect is not None:
            training_client.stop.side_effect = stop_side_effect
        tokenizer = mock.Mock()

        with mock.patch.object(sft_train, "parse_args",
                               return_value=_mock_args(tmpdir)), \
             mock.patch.object(sft_train, "open_session",
                               return_value=(svc, training_client, tokenizer)), \
             mock.patch.object(sft_train, "_train",
                               side_effect=train_side_effect):
            sft_train.main()
        return svc, training_client

    def test_training_failure_stops_session_and_reraises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError) as ctx:
                self._run_main(d, train_side_effect=RuntimeError("oom mid-step"))
            # Original exception surfaces unchanged...
            self.assertEqual(str(ctx.exception), "oom mid-step")

    def test_stop_called_exactly_once_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            svc = mock.Mock()
            training_client = mock.Mock()
            training_client.training_session_name = "crash-unit-test"
            with mock.patch.object(sft_train, "parse_args",
                                   return_value=_mock_args(d)), \
                 mock.patch.object(sft_train, "open_session",
                                   return_value=(svc, training_client, mock.Mock())), \
                 mock.patch.object(sft_train, "_train",
                                   side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    sft_train.main()
            training_client.stop.assert_called_once()
            # Failure path stops, never deletes: the record must survive.
            svc.delete_session.assert_not_called()

    def test_teardown_failure_does_not_mask_original_exception(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError) as ctx:
                self._run_main(
                    d,
                    train_side_effect=RuntimeError("the real error"),
                    stop_side_effect=ConnectionError("stop API down"),
                )
            self.assertEqual(str(ctx.exception), "the real error")

    def test_success_path_does_not_stop(self):
        with tempfile.TemporaryDirectory() as d:
            svc, training_client = self._run_main(d, train_side_effect=None)
            training_client.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()


# --- RL (ServiceBackend) lifecycle ----------------------------------------


def test_service_backend_teardown_semantics():
    """delete_session() deletes (stop_if_active) and makes close() a no-op;
    close() alone stops without deleting (failure path keeps the record)."""
    from unittest import mock
    from training_sessions_cookbook.rl.backends.service import ServiceBackend

    def make(svc, tc):
        b = ServiceBackend.__new__(ServiceBackend)
        b._svc, b._tc = svc, tc
        b._session_deleted = False
        b._session_name = "run-x"
        return b

    # Success path: delete, then the finally-block close() must not stop.
    svc, tc = mock.Mock(), mock.Mock()
    b = make(svc, tc)
    b.delete_session()
    b.close()
    svc.delete_session.assert_called_once_with("run-x", stop_if_active=True)
    tc.stop.assert_not_called()

    # Failure path: close() only stops; nothing is deleted.
    svc, tc = mock.Mock(), mock.Mock()
    b = make(svc, tc)
    b.close()
    tc.stop.assert_called_once()
    svc.delete_session.assert_not_called()


def test_create_stops_session_when_setup_fails():
    """A failure after the session opens but before the backend exists must
    stop the session (nothing else holds a handle to it) and re-raise."""
    from unittest import mock

    from training_sessions_cookbook.rl.backends import service as svc_mod

    svc, tc = mock.Mock(), mock.Mock()
    tc.save_weights_for_sampler.return_value.result.side_effect = RuntimeError(
        "checkpoint load failed"
    )
    args = mock.Mock(base_model_arn="arn:x", role_arn="arn:r",
                     model_package_group_arn="arn:g", output_s3_path="s3://b/",
                     lora_rank=8, lora_alpha=16, run_name="t")
    with mock.patch.object(svc_mod, "open_session",
                           return_value=(svc, tc, mock.Mock())):
        try:
            svc_mod.ServiceBackend.create(args)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "checkpoint load failed" in str(e)
    tc.stop.assert_called_once()


# --- resume batch cursor ----------------------------------------------------


class _Future:
    def __init__(self, value):
        self._value = value

    async def result_async(self):
        return self._value

    def __await__(self):
        return self.result_async().__await__()


class _FakeResult:
    """Shape returned by forward_backward: metrics dict + model package arn."""
    metrics = {"mean_nll": 1.0}
    loss = 1.0
    loss_fn_outputs = [{"actor/loss": 1.0}]
    model_package_arn = "arn:fake"

    def to_dict(self):
        return {"mean_nll": 1.0}


class _RecordingClient:
    """Fake TrainingClient recording which batches it trained."""

    training_session_name = "resume-cursor-test"

    def __init__(self):
        self.trained = []  # batch markers, in training order

    async def forward_backward_async(self, batch, loss_fn=None, **kw):
        self.trained.append(batch[0]["marker"])
        return _Future(_FakeResult())

    async def optim_step_async(self, adam_params):
        return _Future(_FakeResult())

    async def forward_async(self, batch, loss_fn=None, **kw):
        return _Future(_FakeResult())

    async def save_state_async(self, description="", ttl_seconds=None):
        return _Future(_FakeResult())

    async def save_weights_for_sampler_async(self, description="", ttl_seconds=None):
        return _Future(_FakeResult())

    def stop(self):
        pass


def test_resume_continues_at_the_untrained_batches(tmp_path):
    """Resume must fast-forward the batch cursor: a run checkpointed after
    step N continues with batch N, not by retraining batches 0..k."""
    import asyncio
    import training_sessions_cookbook.sft.train as sft_train

    def _args(max_steps, resume):
        ns = sft_train.parse_args([
            "--dataset_path", "unused.jsonl", "--base_model_arn", "arn:m",
            "--model_package_group_arn", "arn:g", "--role_arn", "arn:r",
            "--output_s3_path", "s3://b/", "--batch_size", "1",
            "--max_steps", str(max_steps), "--save_every", "0",
            "--eval_every", "0", "--log_every", "0", "--num_epochs", "1",
            "--log_path", str(tmp_path),
        ])
        ns.resume = resume
        return ns

    batches = [[{
        "marker": i,
        "modelInput": {"chunks": [{"tokens": [1, 2, 3]}]},
    }] for i in range(6)]

    def run_once(max_steps, resume, client):
        with mock.patch.object(
            sft_train, "open_session",
            return_value=(mock.Mock(), client, mock.Mock()),
        ):
            asyncio.run(sft_train.run_async(
                _args(max_steps, resume), train_batches=batches))

    # First run: trains batches 0..2, final checkpoint records step 2.
    first = _RecordingClient()
    run_once(3, resume=False, client=first)
    assert first.trained == [0, 1, 2]

    # Resumed run: must continue with batches 3..5, never retrain 0..2.
    second = _RecordingClient()
    run_once(6, resume=True, client=second)
    assert second.trained == [3, 4, 5]
