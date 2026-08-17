"""Unit tests for metrics aggregation functions and MetricsLogger.

"""
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from training_sessions_cookbook.rl.train import (
    MetricsLogger,
    Transition,
    compute_advantage_stats,
    compute_episode_length_mean,
    compute_reward_stats,
    count_episodes,
)


def _make_transition(reward: float = 0.0, advantage: float = 0.0) -> Transition:
    """Create a minimal Transition for testing."""
    return Transition(
        obs_tokens=[1, 2, 3],
        act_tokens=[4, 5],
        act_logprobs=[-0.5, -0.3],
        reward=reward,
        done=True,
        env_type="gsm8k",
        advantage=advantage,
    )


class TestComputeRewardStats:
    def test_empty_episodes_returns_zeros(self):
        result = compute_reward_stats([])
        assert result == {"reward_mean": 0.0, "reward_std": 0.0, "reward_min": 0.0, "reward_max": 0.0}

    def test_all_empty_episodes_returns_zeros(self):
        result = compute_reward_stats([[], [], []])
        assert result == {"reward_mean": 0.0, "reward_std": 0.0, "reward_min": 0.0, "reward_max": 0.0}

    def test_single_episode_single_transition(self):
        ep = [_make_transition(reward=1.0)]
        result = compute_reward_stats([ep])
        assert result["reward_mean"] == 1.0
        assert result["reward_std"] == 0.0
        assert result["reward_min"] == 1.0
        assert result["reward_max"] == 1.0

    def test_multiple_episodes(self):
        ep1 = [_make_transition(reward=0.5), _make_transition(reward=0.5)]  # sum = 1.0
        ep2 = [_make_transition(reward=0.0)]  # sum = 0.0
        ep3 = [_make_transition(reward=1.0)]  # sum = 1.0
        result = compute_reward_stats([ep1, ep2, ep3])
        rewards = np.array([1.0, 0.0, 1.0])
        assert np.isclose(result["reward_mean"], rewards.mean())
        assert np.isclose(result["reward_std"], rewards.std())
        assert result["reward_min"] == 0.0
        assert result["reward_max"] == 1.0

    def test_empty_episodes_excluded(self):
        ep1 = [_make_transition(reward=1.0)]
        result = compute_reward_stats([ep1, []])
        assert result["reward_mean"] == 1.0


class TestComputeAdvantageStats:
    def test_empty_transitions_returns_zeros(self):
        result = compute_advantage_stats([])
        assert result == {"advantage_mean": 0.0, "advantage_std": 0.0, "advantage_min": 0.0, "advantage_max": 0.0}

    def test_single_transition(self):
        t = _make_transition(advantage=0.5)
        result = compute_advantage_stats([t])
        assert result["advantage_mean"] == 0.5
        assert result["advantage_std"] == 0.0
        assert result["advantage_min"] == 0.5
        assert result["advantage_max"] == 0.5

    def test_multiple_transitions(self):
        ts = [_make_transition(advantage=-1.0), _make_transition(advantage=0.0), _make_transition(advantage=2.0)]
        result = compute_advantage_stats(ts)
        arr = np.array([-1.0, 0.0, 2.0])
        assert np.isclose(result["advantage_mean"], arr.mean())
        assert np.isclose(result["advantage_std"], arr.std())
        assert result["advantage_min"] == -1.0
        assert result["advantage_max"] == 2.0


class TestComputeEpisodeLengthMean:
    def test_empty_returns_zero(self):
        assert compute_episode_length_mean([]) == 0.0

    def test_all_empty_episodes_returns_zero(self):
        assert compute_episode_length_mean([[], []]) == 0.0

    def test_single_episode(self):
        ep = [_make_transition(), _make_transition(), _make_transition()]
        assert compute_episode_length_mean([ep]) == 3.0

    def test_multiple_episodes_with_empty(self):
        ep1 = [_make_transition(), _make_transition()]  # len 2
        ep2 = [_make_transition()]  # len 1
        # Empty episode excluded
        assert compute_episode_length_mean([ep1, [], ep2]) == 1.5


class TestCountEpisodes:
    def test_empty_list(self):
        assert count_episodes([]) == (0, 0)

    def test_all_successful(self):
        ep1 = [_make_transition()]
        ep2 = [_make_transition()]
        assert count_episodes([ep1, ep2]) == (2, 0)

    def test_all_failed(self):
        assert count_episodes([[], [], []]) == (0, 3)

    def test_mixed(self):
        ep1 = [_make_transition()]
        assert count_episodes([ep1, [], [_make_transition()], []]) == (2, 2)


class TestMetricsLogger:
    def test_log_step_writes_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            metrics_path = f.name

        try:
            logger = MetricsLogger(metrics_path)
            episodes = [[_make_transition(reward=1.0)], [_make_transition(reward=0.0)]]
            transitions = [_make_transition(advantage=0.5), _make_transition(advantage=-0.5)]

            fb_result = MagicMock()
            fb_result.metrics = {"loss": 0.42, "grad_norm": 0.9}

            record = logger.log_step(
                step=1,
                episodes=episodes,
                transitions=transitions,
                fb_result=fb_result,
                sample_seconds=5.0,
                train_seconds=1.0,
                step_total_seconds=6.0,
                effective_lr=4e-5,
                kl_coef=0.0,
            )

            # Verify returned record has expected keys
            assert record["step"] == 1
            assert "rollout/reward_mean" in record
            assert "rollout/advantage_mean" in record
            assert "rollout/episode_length_mean" in record
            assert record["rollout/num_episodes"] == 2
            assert record["rollout/num_failed_rollouts"] == 0
            assert record["train/loss"] == 0.42
            assert record["train/grad_norm"] == 0.9
            assert record["time/sample_seconds"] == 5.0
            assert record["time/train_seconds"] == 1.0
            assert record["time/step_total_seconds"] == 6.0
            assert record["config/learning_rate"] == 4e-5

            # Verify JSONL was written
            with open(metrics_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            written = json.loads(lines[0])
            assert written == record
        finally:
            os.unlink(metrics_path)

    def test_log_step_appends_multiple_steps(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            metrics_path = f.name

        try:
            logger = MetricsLogger(metrics_path)
            episodes = [[_make_transition(reward=1.0)]]
            transitions = [_make_transition(advantage=0.0)]

            for step in range(3):
                logger.log_step(
                    step=step,
                    episodes=episodes,
                    transitions=transitions,
                    fb_result=None,
                    sample_seconds=1.0,
                    train_seconds=0.5,
                    step_total_seconds=1.5,
                    effective_lr=1e-4,
                    kl_coef=0.0,
                )

            with open(metrics_path) as f:
                lines = f.readlines()
            assert len(lines) == 3
            for i, line in enumerate(lines):
                record = json.loads(line)
                assert record["step"] == i
        finally:
            os.unlink(metrics_path)

    def test_metrics_namespaced_under_train(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            metrics_path = f.name

        try:
            logger = MetricsLogger(metrics_path)
            fb_result = MagicMock()
            fb_result.metrics = {"policy_loss": 0.1, "value_loss": 0.2, "entropy": 0.5}

            record = logger.log_step(
                step=0,
                episodes=[[_make_transition()]],
                transitions=[_make_transition()],
                fb_result=fb_result,
                sample_seconds=1.0,
                train_seconds=0.5,
                step_total_seconds=1.5,
                effective_lr=1e-4,
                kl_coef=0.0,
            )

            assert record["train/policy_loss"] == 0.1
            assert record["train/value_loss"] == 0.2
            assert record["train/entropy"] == 0.5
            # Original keys should NOT appear without prefix
            assert "policy_loss" not in record
            assert "value_loss" not in record
        finally:
            os.unlink(metrics_path)

    def test_no_fb_result_skips_train_metrics(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            metrics_path = f.name

        try:
            logger = MetricsLogger(metrics_path)
            record = logger.log_step(
                step=0,
                episodes=[[_make_transition()]],
                transitions=[_make_transition()],
                fb_result=None,
                sample_seconds=1.0,
                train_seconds=0.5,
                step_total_seconds=1.5,
                effective_lr=1e-4,
                kl_coef=0.0,
            )

            # No train/ prefixed keys should exist
            train_keys = [k for k in record if k.startswith("train/")]
            assert len(train_keys) == 0
        finally:
            os.unlink(metrics_path)
