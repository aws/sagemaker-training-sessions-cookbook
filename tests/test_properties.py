"""Property-based tests for the RL cookbook.

Uses Hypothesis to verify correctness properties of the training helpers.
"""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

# Add the repo root to path so ``rl`` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings
from hypothesis import strategies as st

from training_sessions_cookbook.rl.train import load_dataset


# =============================================================================
# JSONL Dataset Parsing Round-Trip
# =============================================================================
# For any list of dictionaries with `question` and `answer` string fields, writing them as JSONL (one json.dumps per line) and then parsing with load_dataset SHALL produce an identical list of dictionaries.
#
# =============================================================================

# Strategy: generate lists of dicts with question and answer string fields
gsm8k_record = st.fixed_dictionaries({
    "question": st.text(min_size=1),
    "answer": st.text(min_size=1),
})

gsm8k_dataset = st.lists(gsm8k_record, min_size=0, max_size=20)


@given(records=gsm8k_dataset)
@settings(max_examples=200)
def test_jsonl_parsing_round_trip(records):
    """JSONL Dataset Parsing Round-Trip.

    Writing dicts as JSONL then parsing with load_dataset produces identical data.

    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
        temp_path = f.name

    try:
        result = load_dataset(temp_path, None)
        assert result == records, (
            f"Round-trip mismatch:\n"
            f"  Original: {records}\n"
            f"  Loaded:   {result}"
        )
    finally:
        os.unlink(temp_path)


@given(records=st.lists(gsm8k_record, min_size=1, max_size=20), max_prompts=st.integers(min_value=0, max_value=20))
@settings(max_examples=200)
def test_jsonl_parsing_round_trip_with_max_prompts(records, max_prompts):
    """Extension: max_prompts truncation works correctly.

    Writing dicts as JSONL then parsing with load_dataset(path, max_prompts)
    produces the first max_prompts items from the original list.

    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
        temp_path = f.name

    try:
        result = load_dataset(temp_path, max_prompts)
        expected = records[:max_prompts]
        assert result == expected, (
            f"Truncation mismatch with max_prompts={max_prompts}:\n"
            f"  Expected: {expected}\n"
            f"  Got:      {result}"
        )
    finally:
        os.unlink(temp_path)


# --- Linear Warmup Learning Rate Schedule ---

import math
from hypothesis import assume

from training_sessions_cookbook.rl.train import compute_learning_rate


class TestProperty2LinearWarmupLearningRate:
    """Linear Warmup Learning Rate Schedule.


    For any base_lr > 0, warmup_steps > 0, and step in [0, warmup_steps), compute_learning_rate(step, base_lr, warmup_steps) SHALL equal (step / warmup_steps) * base_lr. For any step >= warmup_steps, it SHALL equal base_lr.
    """

    @given(
        step=st.integers(min_value=0, max_value=10000),
        base_lr=st.floats(min_value=1e-8, max_value=1.0, allow_nan=False, allow_infinity=False),
        warmup_steps=st.integers(min_value=1, max_value=5000),
    )
    @settings(max_examples=100)
    def test_warmup_phase_lr_equals_linear_interpolation(self, step, base_lr, warmup_steps):
        """During warmup (step < warmup_steps), LR = (step / warmup_steps) * base_lr."""
        assume(step < warmup_steps)

        result = compute_learning_rate(step, base_lr, warmup_steps)
        expected = (step / warmup_steps) * base_lr

        assert math.isclose(result, expected, rel_tol=1e-9), (
            f"step={step}, base_lr={base_lr}, warmup_steps={warmup_steps}: "
            f"got {result}, expected {expected}"
        )

    @given(
        step=st.integers(min_value=0, max_value=10000),
        base_lr=st.floats(min_value=1e-8, max_value=1.0, allow_nan=False, allow_infinity=False),
        warmup_steps=st.integers(min_value=1, max_value=5000),
    )
    @settings(max_examples=100)
    def test_post_warmup_lr_equals_base_lr(self, step, base_lr, warmup_steps):
        """After warmup (step >= warmup_steps), LR = base_lr."""
        assume(step >= warmup_steps)

        result = compute_learning_rate(step, base_lr, warmup_steps)

        assert math.isclose(result, base_lr, rel_tol=1e-9), (
            f"step={step}, base_lr={base_lr}, warmup_steps={warmup_steps}: "
            f"got {result}, expected {base_lr}"
        )

    @given(
        step=st.integers(min_value=0, max_value=10000),
        base_lr=st.floats(min_value=1e-8, max_value=1.0, allow_nan=False, allow_infinity=False),
        warmup_steps=st.integers(min_value=-100, max_value=0),
    )
    @settings(max_examples=100)
    def test_no_warmup_when_warmup_steps_le_zero(self, step, base_lr, warmup_steps):
        """When warmup_steps <= 0, LR always equals base_lr (no warmup)."""
        result = compute_learning_rate(step, base_lr, warmup_steps)

        assert math.isclose(result, base_lr, rel_tol=1e-9), (
            f"step={step}, base_lr={base_lr}, warmup_steps={warmup_steps}: "
            f"got {result}, expected {base_lr}"
        )


# =============================================================================
# Run Folder Naming Pattern
# =============================================================================
# For any non-empty run_name without path separators, the created run folder path SHALL match the pattern runs/{run_name}_{YYYYMMDD_HHMMSS} where the timestamp portion consists of exactly 8 digits, an underscore, and 6 digits.
#
# =============================================================================

import re
import argparse

from training_sessions_cookbook.rl.train import create_run_folder, dump_config

# Non-empty strings without path separators (no '/' or '\') and filesystem-safe.
# We use printable ASCII minus path separators, null, and OS-problematic chars to ensure the generated names can actually be created on any filesystem.
_SAFE_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_.+= @#%&()[]{}!~"
)
run_name_strategy = st.text(
    alphabet=_SAFE_CHARS,
    min_size=1,
    max_size=50,
).filter(lambda s: len(s.strip()) > 0)


class TestProperty8RunFolderNaming:
    """Run Folder Naming Pattern.


    For any non-empty run_name without path separators, the created path matches runs/{run_name}_{YYYYMMDD_HHMMSS} where the timestamp portion consists of exactly 8 digits, an underscore, and 6 digits.
    """

    @given(run_name=run_name_strategy)
    @settings(max_examples=100)
    def test_run_folder_matches_naming_pattern(self, run_name):
        """For any valid run_name, created folder matches runs/{run_name}_{YYYYMMDD_HHMMSS}."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                result = create_run_folder(run_name)

                # The result should be an absolute path
                assert os.path.isabs(result)

                # The directory should exist
                assert os.path.isdir(result)

                # Extract the basename of the run folder
                basename = os.path.basename(result)

                # The parent directory should be "runs"
                parent_dir = os.path.basename(os.path.dirname(result))
                assert parent_dir == "runs", f"Expected parent 'runs', got '{parent_dir}'"

                # The basename should match: {run_name}_{YYYYMMDD}_{HHMMSS}
                # The timestamp is YYYYMMDD_HHMMSS = 8 digits + underscore + 6 digits
                expected_pattern = re.escape(run_name) + r"_\d{8}_\d{6}$"
                assert re.match(expected_pattern, basename), (
                    f"Folder basename '{basename}' does not match pattern "
                    f"'{run_name}_YYYYMMDD_HHMMSS'"
                )
            finally:
                os.chdir(original_cwd)


# =============================================================================
# Config JSON CLI Argument Round-Trip
# =============================================================================
# For any argparse namespace containing the full set of CLI arguments, dump_config followed by reading config.json SHALL produce a dictionary containing every argument name as a key with its corresponding value (modulo JSON serialization of types), plus a git_hash string field.
#
# =============================================================================

# JSON-serializable values for config round-trip
json_value_strategy = st.one_of(
    st.integers(min_value=-(2**31), max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(min_size=0, max_size=30),
    st.booleans(),
    st.none(),
)

# Strategy for CLI argument names (valid Python identifiers, typical argparse style)
arg_name_strategy = st.from_regex(r"[a-z][a-z0-9_]{0,19}", fullmatch=True)


class TestProperty9ConfigRoundTrip:
    """Config JSON CLI Argument Round-Trip.


    For any argparse namespace containing CLI arguments, dump_config followed by reading config.json contains every argument name as a key with its corresponding value (modulo JSON serialization), plus a git_hash string field.
    """

    @given(
        data=st.dictionaries(
            keys=arg_name_strategy,
            values=json_value_strategy,
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=100)
    def test_config_round_trip_preserves_all_keys(self, data):
        """dump_config then reading config.json has every CLI arg key + git_hash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build an argparse Namespace from the generated data
            args = argparse.Namespace(**data)

            dump_config(args, tmp_dir)

            # Read back the config
            config_path = os.path.join(tmp_dir, "config.json")
            assert os.path.exists(config_path), "config.json was not created"

            with open(config_path) as f:
                config = json.load(f)

            # Every argument key must be present
            for key in data:
                assert key in config, f"Key '{key}' missing from config.json"

            # Values must match after JSON serialization round-trip
            for key, value in data.items():
                expected = json.loads(json.dumps(value, default=str))
                assert config[key] == expected, (
                    f"Key '{key}': expected {expected!r}, got {config[key]!r}"
                )

            # git_hash must be present and be a string
            assert "git_hash" in config, "git_hash field missing from config.json"
            assert isinstance(config["git_hash"], str), "git_hash must be a string"
            assert len(config["git_hash"]) > 0, "git_hash must be non-empty"


# =============================================================================
# Metrics Aggregation Correctness
# =============================================================================
# For any non-empty list of episodes (each a list of Transitions with numeric rewards and advantages), the computed reward_mean, reward_std, reward_min, reward_max, advantage_mean, advantage_std, advantage_min, advantage_max, and episode_length_mean SHALL match the corresponding numpy statistical operations on the same data.
#
# =============================================================================

import numpy as np

from training_sessions_cookbook.rl.train import (
    Transition,
    compute_advantage_stats,
    compute_episode_length_mean,
    compute_reward_stats,
    count_episodes,
    MetricsLogger,
)

# Strategy: generate a single Transition with random reward and advantage
transition_strategy = st.builds(
    Transition,
    obs_tokens=st.just([1, 2, 3]),
    act_tokens=st.just([4, 5]),
    act_logprobs=st.just([-0.5, -0.3]),
    reward=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    done=st.booleans(),
    env_type=st.just("gsm8k"),
    advantage=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)

# Strategy: generate a non-empty episode (list of Transitions)
non_empty_episode_strategy = st.lists(transition_strategy, min_size=1, max_size=10)

# Strategy: generate episodes (list of lists of Transitions, at least one non-empty)
episodes_with_at_least_one_non_empty = st.lists(
    non_empty_episode_strategy, min_size=1, max_size=10
)


class TestProperty3MetricsAggregation:
    """Metrics Aggregation Correctness.


    For any non-empty list of episodes, the computed stats SHALL match the corresponding numpy operations on the same data.
    """

    @given(episodes=episodes_with_at_least_one_non_empty)
    @settings(max_examples=100)
    def test_reward_stats_match_numpy(self, episodes):
        """compute_reward_stats matches numpy on per-episode returns."""
        result = compute_reward_stats(episodes)

        # Reference: compute per-episode returns using numpy
        rewards = [sum(t.reward for t in ep) for ep in episodes if ep]
        arr = np.array(rewards)

        assert math.isclose(result["reward_mean"], float(arr.mean()), rel_tol=1e-6, abs_tol=1e-10), (
            f"reward_mean mismatch: got {result['reward_mean']}, expected {float(arr.mean())}"
        )
        assert math.isclose(result["reward_std"], float(arr.std()), rel_tol=1e-6, abs_tol=1e-10), (
            f"reward_std mismatch: got {result['reward_std']}, expected {float(arr.std())}"
        )
        assert math.isclose(result["reward_min"], float(arr.min()), rel_tol=1e-6, abs_tol=1e-10), (
            f"reward_min mismatch: got {result['reward_min']}, expected {float(arr.min())}"
        )
        assert math.isclose(result["reward_max"], float(arr.max()), rel_tol=1e-6, abs_tol=1e-10), (
            f"reward_max mismatch: got {result['reward_max']}, expected {float(arr.max())}"
        )

    @given(episodes=episodes_with_at_least_one_non_empty)
    @settings(max_examples=100)
    def test_advantage_stats_match_numpy(self, episodes):
        """compute_advantage_stats matches numpy on all transitions."""
        # Flatten all transitions from all non-empty episodes
        transitions = [t for ep in episodes for t in ep if ep]

        result = compute_advantage_stats(transitions)

        # Reference: compute using numpy
        advantages = [t.advantage for t in transitions]
        arr = np.array(advantages)

        assert math.isclose(result["advantage_mean"], float(arr.mean()), rel_tol=1e-6, abs_tol=1e-10), (
            f"advantage_mean mismatch: got {result['advantage_mean']}, expected {float(arr.mean())}"
        )
        assert math.isclose(result["advantage_std"], float(arr.std()), rel_tol=1e-6, abs_tol=1e-10), (
            f"advantage_std mismatch: got {result['advantage_std']}, expected {float(arr.std())}"
        )
        assert math.isclose(result["advantage_min"], float(arr.min()), rel_tol=1e-6, abs_tol=1e-10), (
            f"advantage_min mismatch: got {result['advantage_min']}, expected {float(arr.min())}"
        )
        assert math.isclose(result["advantage_max"], float(arr.max()), rel_tol=1e-6, abs_tol=1e-10), (
            f"advantage_max mismatch: got {result['advantage_max']}, expected {float(arr.max())}"
        )

    @given(episodes=episodes_with_at_least_one_non_empty)
    @settings(max_examples=100)
    def test_episode_length_mean_matches_numpy(self, episodes):
        """compute_episode_length_mean matches numpy mean of episode lengths."""
        result = compute_episode_length_mean(episodes)

        # Reference: compute mean episode length using numpy
        non_empty = [ep for ep in episodes if ep]
        lengths = [len(ep) for ep in non_empty]
        expected = float(np.mean(lengths))

        assert math.isclose(result, expected, rel_tol=1e-6, abs_tol=1e-10), (
            f"episode_length_mean mismatch: got {result}, expected {expected}"
        )


# =============================================================================
# Training-metric namespacing
# =============================================================================
# For any dictionary of string keys and numeric values provided as fb_result.metrics, the resulting metrics JSONL record contains every key prefixed with train/ and no input key is missing from the output.
# =============================================================================

# Strategy for training-metric keys: simple alphanumeric keys with underscores
metric_key = st.from_regex(r"[a-z][a-z0-9_]{0,19}", fullmatch=True)

# Strategy for training-metric values: numeric floats
metric_value = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)

# Strategy for fb_result.metrics dictionaries
metrics_dict = st.dictionaries(
    keys=metric_key,
    values=metric_value,
    min_size=1,
    max_size=10,
)


class TestTrainingMetricNamespacing:
    """For any dictionary of string keys and numeric values provided as fb_result.metrics, the resulting metrics JSONL record contains every key prefixed with train/ and no input key is missing from the output.
    """

    @given(metrics_dict=metrics_dict)
    @settings(max_examples=100)
    def test_all_metrics_appear_with_train_prefix(self, metrics_dict):
        """Every key in fb_result.metrics appears as train/{key} in the output record."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            metrics_path = f.name

        try:
            logger = MetricsLogger(metrics_path)

            # Create a mock fb_result with the generated metrics
            fb_result = MagicMock()
            fb_result.metrics = metrics_dict

            # Create a minimal episode so the logger doesn't hit the empty case
            episode = [Transition(
                obs_tokens=[1], act_tokens=[2], act_logprobs=[-0.5],
                reward=1.0, done=True, env_type="gsm8k", advantage=0.0,
            )]

            record = logger.log_step(
                step=0,
                episodes=[episode],
                transitions=episode,
                fb_result=fb_result,
                sample_seconds=1.0,
                train_seconds=0.5,
                step_total_seconds=1.5,
                effective_lr=1e-4,
                kl_coef=0.0,
            )

            # Verify every key from metrics_dict appears prefixed with train/
            for key in metrics_dict:
                prefixed_key = f"train/{key}"
                assert prefixed_key in record, (
                    f"Key '{key}' from fb_result.metrics not found as "
                    f"'{prefixed_key}' in output record. Record keys: {list(record.keys())}"
                )
                assert math.isclose(record[prefixed_key], metrics_dict[key], rel_tol=1e-9, abs_tol=1e-12), (
                    f"Value mismatch for '{prefixed_key}': "
                    f"got {record[prefixed_key]}, expected {metrics_dict[key]}"
                )
        finally:
            os.unlink(metrics_path)


# =============================================================================
# Episode Counting
# =============================================================================
# For any list of episodes where some are empty lists (failed rollouts) and some are non-empty, count_episodes SHALL return num_episodes equal to the count of non-empty episodes and num_failed_rollouts equal to the count of empty episodes.
#
# =============================================================================

# Strategy: generate a mix of empty and non-empty episodes
mixed_episodes_strategy = st.lists(
    st.one_of(
        # Non-empty episode (successful rollout)
        st.lists(transition_strategy, min_size=1, max_size=5),
        # Empty episode (failed rollout)
        st.just([]),
    ),
    min_size=1,
    max_size=20,
)


class TestProperty5EpisodeCounting:
    """Episode Counting.


    For any list of episodes where some are empty lists (failed rollouts) and some are non-empty, count_episodes SHALL return num_episodes equal to the count of non-empty episodes and num_failed_rollouts equal to the count of empty episodes.
    """

    @given(episodes=mixed_episodes_strategy)
    @settings(max_examples=100)
    def test_count_episodes_correct(self, episodes):
        """count_episodes returns correct counts of non-empty and empty episodes."""
        num_episodes, num_failed = count_episodes(episodes)

        # Reference: directly count
        expected_non_empty = sum(1 for ep in episodes if ep)
        expected_empty = sum(1 for ep in episodes if not ep)

        assert num_episodes == expected_non_empty, (
            f"num_episodes mismatch: got {num_episodes}, expected {expected_non_empty}"
        )
        assert num_failed == expected_empty, (
            f"num_failed_rollouts mismatch: got {num_failed}, expected {expected_empty}"
        )

    @given(episodes=mixed_episodes_strategy)
    @settings(max_examples=100)
    def test_count_episodes_sum_equals_total(self, episodes):
        """num_episodes + num_failed_rollouts always equals total episode count."""
        num_episodes, num_failed = count_episodes(episodes)

        assert num_episodes + num_failed == len(episodes), (
            f"Sum of counts ({num_episodes} + {num_failed} = {num_episodes + num_failed}) "
            f"does not equal total episodes ({len(episodes)})"
        )


# =============================================================================
# Validation Accuracy Computation
# =============================================================================
# For any list of reward floats, compute_validation_accuracy SHALL return the fraction of values exactly equal to 1.0. When the list is empty, it SHALL return 0.0.
#
# =============================================================================

from training_sessions_cookbook.rl.train import ValidationExample, compute_validation_accuracy

# Strategy: reward floats that produce 1.0 frequently so the accuracy fraction exercises a meaningful mix of matches and non-matches. We include exact 1.0, exact 0.0, intermediate values, and arbitrary floats.
reward_value_strategy = st.one_of(
    st.sampled_from([0.0, 1.0, 0.5]),
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)

reward_list_strategy = st.lists(reward_value_strategy, min_size=0, max_size=50)


class TestProperty6ValidationAccuracy:
    """Validation Accuracy Computation.


    For any list of reward floats, compute_validation_accuracy SHALL return the fraction of values exactly equal to 1.0. When the list is empty, it SHALL return 0.0.
    """

    @given(rewards=reward_list_strategy)
    @settings(max_examples=200)
    def test_accuracy_equals_fraction_of_exact_ones(self, rewards):
        """compute_validation_accuracy equals (count of r == 1.0) / len(rewards)."""
        result = compute_validation_accuracy(rewards)

        if not rewards:
            expected = 0.0
        else:
            expected = sum(1 for r in rewards if r == 1.0) / len(rewards)

        assert math.isclose(result, expected, rel_tol=1e-12, abs_tol=1e-12), (
            f"accuracy mismatch: got {result}, expected {expected} "
            f"for rewards={rewards}"
        )

    @given(rewards=reward_list_strategy)
    @settings(max_examples=200)
    def test_accuracy_in_unit_interval(self, rewards):
        """The computed accuracy is always a fraction in [0.0, 1.0]."""
        result = compute_validation_accuracy(rewards)
        assert 0.0 <= result <= 1.0, f"accuracy {result} outside [0, 1] for {rewards}"

    def test_empty_list_returns_zero(self):
        """An empty rewards list returns exactly 0.0."""
        assert compute_validation_accuracy([]) == 0.0

    @given(n=st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_all_ones_returns_one(self, n):
        """A list of all 1.0 rewards returns accuracy 1.0."""
        assert compute_validation_accuracy([1.0] * n) == 1.0

    @given(n=st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_no_ones_returns_zero(self, n):
        """A list with no 1.0 rewards returns accuracy 0.0."""
        assert compute_validation_accuracy([0.0] * n) == 0.0


# =============================================================================
# Validation Examples JSONL Schema Completeness
# =============================================================================
# For any validation example written to validation_examples.jsonl, the JSON object SHALL contain all required fields (step, prompt, response, expected_answer, reward) and parsing the file back SHALL yield identical field values.
#
# =============================================================================

_REQUIRED_VALIDATION_FIELDS = ("step", "prompt", "response", "expected_answer", "reward")

# Strategy: build ValidationExample objects with random field values.
validation_example_strategy = st.builds(
    ValidationExample,
    step=st.integers(min_value=0, max_value=100000),
    prompt=st.text(min_size=0, max_size=200),
    response=st.text(min_size=0, max_size=500),
    expected_answer=st.text(min_size=0, max_size=50),
    reward=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    improved=st.booleans(),
)

validation_examples_list_strategy = st.lists(
    validation_example_strategy, min_size=1, max_size=20
)


class TestProperty7ValidationExamplesSchema:
    """Validation Examples JSONL Schema Completeness.


    For any validation example written to validation_examples.jsonl, the JSON object SHALL contain all required fields (step, prompt, response, expected_answer, reward) and parsing the file back SHALL yield identical field values.
    """

    @given(examples=validation_examples_list_strategy)
    @settings(max_examples=100)
    def test_examples_round_trip_with_required_fields(self, examples):
        """Writing examples as JSONL then reading back preserves all required fields."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for example in examples:
                f.write(json.dumps(example.to_record()) + "\n")
            temp_path = f.name

        try:
            parsed = []
            with open(temp_path) as fh:
                for line in fh:
                    if line.strip():
                        parsed.append(json.loads(line))

            assert len(parsed) == len(examples), (
                f"Record count mismatch: wrote {len(examples)}, read {len(parsed)}"
            )

            for example, record in zip(examples, parsed):
                # Every required field must be present.
                for field in _REQUIRED_VALIDATION_FIELDS:
                    assert field in record, (
                        f"Required field '{field}' missing from record: {record}"
                    )

                # Field values must round-trip identically.
                assert record["step"] == example.step
                assert record["prompt"] == example.prompt
                assert record["response"] == example.response
                assert record["expected_answer"] == example.expected_answer
                assert math.isclose(
                    record["reward"], example.reward, rel_tol=1e-12, abs_tol=1e-12
                ), f"reward mismatch: got {record['reward']}, expected {example.reward}"

                # The `improved` field is NOT part of the serialized schema.
                assert "improved" not in record, (
                    "'improved' must not be serialized into the JSONL schema"
                )
        finally:
            os.unlink(temp_path)

    @given(example=validation_example_strategy)
    @settings(max_examples=100)
    def test_to_record_has_exactly_required_fields(self, example):
        """to_record() produces exactly the required fields and nothing else."""
        record = example.to_record()
        assert set(record.keys()) == set(_REQUIRED_VALIDATION_FIELDS), (
            f"to_record keys {set(record.keys())} != required "
            f"{set(_REQUIRED_VALIDATION_FIELDS)}"
        )
