"""Tests for the off-by-K async (overlapped) RL training loop.

Two layers, both SDK-free (no external SDK / no sagemaker):

  * unit tests on ``_StalenessBatcher``: the pure staleness-filter +
    accumulate-to-``batch_size`` decision, testable without the async plumbing:
    groups staler than the cap are dropped; a full batch is released exactly
    once per ``batch_size`` fresh-enough groups.
  * end-to-end tests driving the real ``run_training_async`` against the
    in-memory ``MockBackend`` (reused from ``test_backend_loop``): the loop
    trains the expected number of steps, writes the staleness metrics keys, and
    the producer/consumer shutdown/drain path terminates cleanly with no hung
    asyncio tasks.

Because the loop is backend-agnostic, exercising it with the mock is a
meaningful proxy for the real service backend; they differ only in the
hook bodies, not in how the loop schedules producers/consumers.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from training_sessions_cookbook.rl import train as rl  # noqa: E402
from tests.test_backend_loop import MockBackend, _args  # noqa: E402
from training_sessions_cookbook.rl.runstate import run_artifacts  # noqa: E402


# --- Helpers --------------------------------------------------------------


def _wg(birth_step, n_episodes=1, reward=1.0):
    """Build a WrappedGroup with `n_episodes` single-turn episodes.

    The reward drives the group's return; with distinct rewards across a group
    GRPO advantages are non-degenerate, so the group yields trainable datums.
    """
    episodes = []
    for i in range(n_episodes):
        episodes.append([
            rl.Transition(
                obs_tokens=[1, 2, 3],
                act_tokens=[4, 5],
                act_logprobs=[-0.1, -0.2],
                reward=reward + i, # vary so group std > 0
                done=True,
                env_type="gsm8k",
            )
        ])
    return rl.WrappedGroup(episodes=episodes, birth_step=birth_step)


# --- Unit tests: _StalenessBatcher ---------------------------------------


def test_batcher_drops_groups_over_the_cap():
    # cap=1: a group born at step 0, offered at current_step=3, is 3 stale > 1.
    batcher = rl._StalenessBatcher(batch_size=4, max_steps_off_policy=1)
    batch, stale = batcher.offer(_wg(birth_step=0), current_step=3)
    assert batch is None and stale is None  # drop policy: no prompt returned
    assert batcher.num_dropped == 1
    # exactly-at-cap survives (3 - 2 == 1, not > 1); it accumulates, no batch yet.
    batch, stale = batcher.offer(_wg(birth_step=2), current_step=3)
    assert batch is None and stale is None
    assert batcher.num_dropped == 1
    assert len(batcher.pending) == 1


def test_batcher_releases_exactly_one_batch_per_batch_size():
    batcher = rl._StalenessBatcher(batch_size=3, max_steps_off_policy=100)
    # First two accumulate (all fresh: current_step == birth_step).
    batch, _ = batcher.offer(_wg(birth_step=0), current_step=0)
    assert batch is None
    batch, _ = batcher.offer(_wg(birth_step=0), current_step=0)
    assert batch is None
    # Third completes the batch -> released, accumulator resets.
    batch, _ = batcher.offer(_wg(birth_step=0), current_step=0)
    assert batch is not None and len(batch) == 3
    assert batcher.pending == []
    # A fourth starts a fresh accumulation (no double-release).
    batch, _ = batcher.offer(_wg(birth_step=0), current_step=0)
    assert batch is None
    assert len(batcher.pending) == 1


def test_batcher_stale_drops_do_not_count_toward_batch():
    # Interleave stale + fresh; only the fresh ones fill the batch.
    batcher = rl._StalenessBatcher(batch_size=2, max_steps_off_policy=0)
    batch, _ = batcher.offer(_wg(birth_step=0), current_step=1)
    assert batch is None  # stale (1>0)
    batch, _ = batcher.offer(_wg(birth_step=5), current_step=5)
    assert batch is None  # fresh, 1/2
    assert batcher.num_dropped == 1
    batch, _ = batcher.offer(_wg(birth_step=5), current_step=5)  # fresh, 2/2
    assert batch is not None and len(batch) == 2
    assert batcher.num_dropped == 1


def test_batcher_requeue_returns_prompt():
    # requeue policy: stale group returns its prompt for requeueing.
    batcher = rl._StalenessBatcher(batch_size=4, max_steps_off_policy=1, stale_policy="requeue")
    prompt_sentinel = {"question": "test_q", "answer": "42"}
    wg = rl.WrappedGroup(episodes=[[]], birth_step=0, prompt=prompt_sentinel)
    batch, stale_prompt = batcher.offer(wg, current_step=3)
    assert batch is None
    assert stale_prompt is prompt_sentinel
    assert batcher.num_requeued == 1
    assert batcher.num_dropped == 0


# --- End-to-end: run_training_async over MockBackend ----------------------


def _async_args(tmp_path, **overrides):
    base = dict(
        max_steps_off_policy=100, # generous cap: nothing drops in the happy path
        sampling_seed=None,
        record_trajectories=False,
        num_rollout_workers=2,
        batch_size=2,
        responses_per_prompt=2,
        max_steps=10,
        refresh_every=1,
        val_at_start=False,
    )
    base.update(overrides)
    return _args(tmp_path, **base)


def _gsm8k_dataset(n=8):
    # MockBackend emits "#### 42", so the correct answer scores 1.0.
    return [{"question": f"q{i}", "answer": "42"} for i in range(n)]


def _run_async_with_task_guard(args, backend, dataset, timeout=30.0):
    """Run run_training_async and assert no asyncio tasks are left pending.

    Wraps the loop in a coroutine so we can inspect ``asyncio.all_tasks()``
    from inside the event loop right after the loop returns, the shutdown
    cascade must have cancelled+awaited the dataloader and every rollout worker.
    A timeout guards against a shutdown deadlock surfacing as a hang.
    """
    async def _driver():
        env_factory = rl.get_env_factory(args.env_type, args)
        await asyncio.wait_for(
            rl.run_training_async(
                args,
                backend,
                dataset,
                tokenizer=backend.tokenizer,
                env_factory=env_factory,
                template_fn=rl.apply_qwen3_general,
            ),
            timeout=timeout,
        )
        # Only this driver task should remain; producers must be gone.
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        leftover = [t for t in others if not t.done()]
        assert not leftover, f"hung tasks after shutdown: {leftover}"

    asyncio.run(_driver())


def test_async_loop_exhausts_dataset_and_trains_expected_batches(tmp_path):
    # 8 prompts -> 8 groups; batch_size=2 -> 4 training batches; max_steps=10
    # is not the limit, so the dataset-exhaustion shutdown path fires.
    args = _async_args(tmp_path, max_steps=10, batch_size=2)
    backend = MockBackend()
    _run_async_with_task_guard(args, backend, _gsm8k_dataset(8))

    with open(run_artifacts(args).metrics_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    step_records = [r for r in records if "train/loss" in r]
    assert len(step_records) == 4, [r["step"] for r in step_records]
    assert backend.train_steps == 4


def test_async_loop_stops_at_max_steps(tmp_path):
    # Plenty of data (epochs high), small max_steps: the max_steps guard stops
    # the consumer before exhaustion.
    args = _async_args(tmp_path, max_steps=3, batch_size=2, epochs=50)
    backend = MockBackend()
    _run_async_with_task_guard(args, backend, _gsm8k_dataset(8))
    assert backend.train_steps == 3


def test_async_loop_writes_staleness_and_core_metrics(tmp_path):
    args = _async_args(tmp_path, max_steps=10, batch_size=2)
    backend = MockBackend()
    _run_async_with_task_guard(args, backend, _gsm8k_dataset(8))

    with open(run_artifacts(args).metrics_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert records
    r = records[0]
    for key in (
        "step",
        "rollout/reward_mean",
        "train/loss",
        "sampler/step_min",
        "sampler/step_max",
        "sampler/step_mean",
        "sampler/max_steps_off_policy",
        "rollout/num_stale_dropped",
    ):
        assert key in r, f"missing {key!r} in {sorted(r)}"
    # Staleness is always non-negative and bounded by the cap.
    assert r["sampler/step_min"] >= 0
    assert r["sampler/step_max"] <= args.max_steps_off_policy
    assert r["sampler/step_min"] <= r["sampler/step_mean"] <= r["sampler/step_max"]


def test_async_loop_reward_is_one_when_answer_matches(tmp_path):
    args = _async_args(tmp_path, max_steps=10, batch_size=2)
    backend = MockBackend(answer_text="#### 42")
    _run_async_with_task_guard(args, backend, _gsm8k_dataset(8))
    with open(run_artifacts(args).metrics_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert records
    assert all(rec["rollout/reward_mean"] == 1.0 for rec in records)


def test_async_loop_rejects_monte_carlo(tmp_path):
    args = _async_args(tmp_path, advantage_mode="monte_carlo")
    backend = MockBackend()
    env_factory = rl.get_env_factory(args.env_type, args)
    with pytest.raises(SystemExit):
        asyncio.run(
            rl.run_training_async(
                args,
                backend,
                _gsm8k_dataset(8),
                tokenizer=backend.tokenizer,
                env_factory=env_factory,
                template_fn=rl.apply_qwen3_general,
            )
        )
