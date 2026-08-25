"""Token/turn rollout stats: values and presence in the step record.

These metrics drive the ``tokens`` figure family. They were silently lost when
the metrics helpers moved into ``rl/metrics.py``, leaving that family
unpopulated for every run after the move (caught 2026-08-25 when a fresh prod
run produced an empty figure next to a populated reference).
"""

from training_sessions_cookbook.rl.metrics import compute_token_turn_stats
from training_sessions_cookbook.rl.rollouts import Transition


def _t(obs_n, act_n, done=False):
    return Transition(
        obs_tokens=list(range(obs_n)), act_tokens=list(range(act_n)),
        act_logprobs=[0.0] * act_n, reward=0.0, done=done, env_type="code_rl",
    )


def test_empty_input_yields_a_stable_zero_schema():
    stats = compute_token_turn_stats([], [])
    assert stats["total_turns"] == 0 and stats["turns_per_episode"] == 0.0
    # schema stability matters: the plotter reads these keys every step
    for key in ("ac_tokens_per_turn", "ob_tokens_per_turn", "total_ac_tokens",
                "total_ob_tokens", "frac_max_turns_reached",
                "response_tokens_min", "response_tokens_max"):
        assert key in stats


def test_values_match_hand_computation():
    # two episodes: one single-turn, one two-turn ending on done (= hit the cap)
    episodes = [[_t(10, 4)], [_t(10, 6), _t(20, 10, done=True)]]
    stats = compute_token_turn_stats(episodes, [t for ep in episodes for t in ep])
    assert stats["total_turns"] == 3
    assert stats["total_episodes"] == 2
    assert stats["total_ac_tokens"] == 20          # 4 + 6 + 10
    assert stats["total_ob_tokens"] == 40          # 10 + 10 + 20
    assert stats["ac_tokens_per_turn"] == 20 / 3
    assert stats["turns_per_episode"] == 1.5       # (1 + 2) / 2
    assert stats["response_tokens_min"] == 4.0
    assert stats["response_tokens_max"] == 10.0
    assert stats["frac_max_turns_reached"] == 0.5  # 1 of 2 episodes


def test_stats_land_in_the_logged_step_record(tmp_path):
    from training_sessions_cookbook.rl.metrics import MetricsLogger

    logger = MetricsLogger(str(tmp_path / "metrics.jsonl"))
    episodes = [[_t(10, 4)], [_t(10, 6), _t(20, 10, done=True)]]
    record = logger.log_step(
        step=0,
        episodes=episodes,
        transitions=[t for ep in episodes for t in ep],
        groups=[],
        fb_result=None,
        sample_seconds=1.0,
        train_seconds=1.0,
        step_total_seconds=2.0,
        effective_lr=1e-5,
        kl_coef=0.0,
    )
    assert record["rollout/total_ac_tokens"] == 20
    assert record["rollout/turns_per_episode"] == 1.5
    assert record["rollout/frac_max_turns_reached"] == 0.5
