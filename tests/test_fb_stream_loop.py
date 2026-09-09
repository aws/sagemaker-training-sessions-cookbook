"""Loop-level contracts for --fb_stream_chunks inside run_training.

Complements tests/test_fb_stream.py (which pins the streamer/collector in
isolation) with the contracts that only show up at the loop level:

* fallback: a backend WITHOUT the streaming protocol runs the monolithic
  train_step untouched even when the flag is set (custom backends written
  before this feature keep working);
* streaming: a protocol backend gets N submissions + one finish per step,
  and train_step is never called;
* the optimizer barrier: a step that submits zero fb chunks (every rollout
  failed) must never reach finish_train_step - otherwise gradients from a
  half-streamed step would sit accumulated in the engine and bleed into the
  next step's update.
"""
import asyncio

from tests.test_backend_loop import MockBackend, _args, _gsm8k_dataset, _run


class StreamingMockBackend(MockBackend):
    """MockBackend + the optional streaming protocol, call-recording."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fb_submissions = []      # datum-list sizes, submission order
        self.finish_calls = 0

    async def submit_forward_backward(self, datums, *, loss_name):
        self.calls.append(("submit_fb", len(datums)))
        self.fb_submissions.append(len(datums))
        return f"op-{len(self.fb_submissions)}"

    async def finish_train_step(self, fb_ops, **kw):
        self.calls.append(("finish_train_step", len(fb_ops)))
        self.finish_calls += 1

        class _R:
            metrics = {"loss": 0.5, "grad_norm": 0.01}
        return _R()


def test_flag_with_plain_backend_falls_back_to_train_step(tmp_path):
    """No streaming protocol -> monolithic path, no errors, normal training."""
    backend = MockBackend()
    args = _args(tmp_path, fb_stream_chunks=4)
    _run(args, backend)
    # 4 prompts / batch_size 2 with drop_last -> the dataset exhausts at 2 steps
    assert backend.train_steps == 2
    assert not any(name == "submit_fb" for name, _ in backend.calls)


def test_flag_with_streaming_backend_streams_and_never_calls_train_step(tmp_path):
    """Protocol backend: 2 submissions per step (2 groups, chunks=2), one
    finish per step, monolithic train_step never used."""
    backend = StreamingMockBackend()
    args = _args(tmp_path, fb_stream_chunks=2)   # batch_size=2 groups -> 1 group/chunk
    _run(args, backend)
    N_STEPS = 2   # 4 prompts / batch_size 2, drop_last
    assert backend.train_steps == 0
    assert backend.finish_calls == N_STEPS
    # 2 groups x (2 episodes x 1 transition) per step, 1 group per submission
    per_step = len(backend.fb_submissions) // N_STEPS
    assert per_step == 2, backend.fb_submissions
    assert all(size == 2 for size in backend.fb_submissions)
    # ordering within each step: every submit precedes its finish
    names = [n for n, _ in backend.calls if n in ("submit_fb", "finish_train_step")]
    per = len(names) // N_STEPS
    for s in range(N_STEPS):
        step_calls = names[s * per:(s + 1) * per]
        assert step_calls[-1] == "finish_train_step" and set(step_calls[:-1]) == {"submit_fb"}


def test_all_failed_rollouts_never_reach_optimizer(tmp_path):
    """Zero usable transitions -> zero submissions -> finish_train_step never
    called: no orphaned gradient accumulation can leak into a later step."""
    class AllFailBackend(StreamingMockBackend):
        async def sample_async(self, sampling_client, prompt, **kw):
            raise RuntimeError("sampler down")   # every episode fails

    backend = AllFailBackend()
    args = _args(tmp_path, fb_stream_chunks=2)
    # The loop fails LOUDLY when nothing trained (an all-steps-skipped run must
    # not end "successfully" with an untrained checkpoint) - expect the raise,
    # and the streaming invariants must hold up to that point.
    import pytest
    with pytest.raises(Exception):
        _run(args, backend)
    assert backend.fb_submissions == []
    assert backend.finish_calls == 0
    assert backend.train_steps == 0
