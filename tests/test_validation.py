"""Unit tests for validation utilities.

Covers the ValidationExample dataclass, accuracy computation, and improvement
detection.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_sessions_cookbook.rl.train import (
    ValidationExample,
    ValidationRunner,
    compute_validation_accuracy,
)


# --- compute_validation_accuracy (Requirement 7.5) -----------------------

def test_accuracy_empty_returns_zero():
    assert compute_validation_accuracy([]) == 0.0


def test_accuracy_all_perfect():
    assert compute_validation_accuracy([1.0, 1.0, 1.0]) == 1.0


def test_accuracy_fraction():
    # Only values exactly equal to 1.0 count.
    assert compute_validation_accuracy([1.0, 0.0, 1.0, 0.5]) == 0.5


# --- ValidationExample dataclass (Requirement 7.7) -----------------------

def test_validation_example_defaults_not_improved():
    ex = ValidationExample(step=1, prompt="p", response="r", expected_answer="a", reward=1.0)
    assert ex.improved is False


def test_validation_example_to_record_schema():
    ex = ValidationExample(
        step=50, prompt="2+2?", response="4", expected_answer="4",
        reward=1.0, improved=True,
    )
    record = ex.to_record()
    # Exactly the required JSONL fields, and `improved` is excluded.
    assert record == {
        "step": 50,
        "prompt": "2+2?",
        "response": "4",
        "expected_answer": "4",
        "reward": 1.0,
    }
    assert "improved" not in record


# --- improvement detection (Requirement 7.8) -----------------------------

def _runner() -> ValidationRunner:
    return ValidationRunner(val_dataset=[], val_examples=5, run_folder=".", env_type="gsm8k")


def test_check_improvements_tags_zero_to_one():
    runner = _runner()
    # Step 0: prompt scores 0.0
    runner._check_improvements(0, [
        ValidationExample(step=0, prompt="q", response="", expected_answer="a", reward=0.0),
    ])
    # Step 10: same prompt now scores 1.0 -> improved
    results = runner._check_improvements(10, [
        ValidationExample(step=10, prompt="q", response="ans", expected_answer="a", reward=1.0),
    ])
    assert results[0].improved is True


def test_check_improvements_no_tag_when_already_correct():
    runner = _runner()
    runner._check_improvements(0, [
        ValidationExample(step=0, prompt="q", response="ans", expected_answer="a", reward=1.0),
    ])
    results = runner._check_improvements(10, [
        ValidationExample(step=10, prompt="q", response="ans", expected_answer="a", reward=1.0),
    ])
    assert results[0].improved is False


def test_check_improvements_first_validation_not_improved():
    runner = _runner()
    results = runner._check_improvements(0, [
        ValidationExample(step=0, prompt="q", response="ans", expected_answer="a", reward=1.0),
    ])
    assert results[0].improved is False


# --- run_validation end-to-end on GSM8K (regression) ---------------------
#
# The pure-helper tests above never drove run_validation/_evaluate_one, so a
# bug where the GSM8K branch referenced prompt_text/expected_answer without
# assigning them (UnboundLocalError, crashing every GSM8K validation incl.
# --val_at_start) went uncaught. This exercises the real path over a tiny mock.

import asyncio
import json
from types import SimpleNamespace

from training_sessions_cookbook.rl import train as rl


class _Seq:
    def __init__(self, tokens):
        self.tokens = tokens
        self.logprobs = [-0.1] * len(tokens)


class _Tok:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class _ValMockBackend:
    """Minimal backend for run_validation: always emits '#### 42'."""

    def __init__(self):
        self.tokenizer = _Tok()

    def wrap_prompt(self, tokens):
        return list(tokens)

    def make_sampling_params(self, *, max_tokens, temperature, seed=None):
        return SimpleNamespace(max_tokens=max_tokens, temperature=temperature)

    async def sample_async(self, sampling_client, prompt, *, num_samples=1, sampling_params=None):
        toks = self.tokenizer.encode("#### 42")
        return SimpleNamespace(samples=[_Seq(toks)])


def _val_args():
    return SimpleNamespace(
        max_tokens=32, temperature=1.0, num_rollout_workers=4,
        answer_format="hash", env_type="gsm8k",
    )


def test_run_validation_gsm8k_scores_and_does_not_crash(tmp_path):
    # GSM8K items are {"question","answer"} (see tools/download_gsm8k.py).
    val_dataset = [
        {"question": "6*7?", "answer": "42"}, # mock answers 42 -> reward 1.0
        {"question": "2+2?", "answer": "4"}, # mock answers 42 -> reward 0.0
    ]
    runner = ValidationRunner(val_dataset, val_examples=2,
                              run_folder=str(tmp_path), env_type="gsm8k")
    metrics = asyncio.run(runner.run_validation(
        step=0,
        sampling_client=object(),
        tokenizer=_Tok(),
        template_fn=rl.apply_qwen3_general,
        args=_val_args(),
        backend=_ValMockBackend(),
    ))
    # Half correct -> accuracy 0.5, and it did not raise UnboundLocalError.
    assert metrics["val/accuracy"] == 0.5
    assert metrics["val/reward_mean"] == 0.5
    # The examples file carries the extracted prompt/expected_answer.
    rec_path = os.path.join(str(tmp_path), "validation_examples.jsonl")
    assert os.path.exists(rec_path)
    with open(rec_path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    prompts = {r["prompt"] for r in recs}
    assert "6*7?" in prompts and "2+2?" in prompts
    assert {r["expected_answer"] for r in recs} == {"42", "4"}
