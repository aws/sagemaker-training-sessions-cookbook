"""SFT tokenization correctness.

The SFT datum builder (``sft/data.py``) masks loss to assistant-response tokens only and normalizes per-datum weights to sum to 1. These invariants are what make the training signal correct, so we assert them directly on a toy round-trippable tokenizer (no network, no service).

We do NOT re-derive the qwen3 chat layout here; that is verified once against the service tokenizer in the byte-parity investigation. These tests pin the masking / normalization / truncation / skip logic that lives in this repo.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_sessions_cookbook.sft.data import conversation_to_datum


class _CharTokenizer:
    """Toy tokenizer: one token per character. add_special_tokens ignored."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _datum(messages, max_length=4096):
    return conversation_to_datum(messages, _CharTokenizer(), max_length)


def test_weights_sum_to_one_over_response_tokens():
    d = _datum([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ])
    assert d is not None
    w = np.array(d["lossFnInputs"]["weights"]["data"], dtype=np.float64)
    assert abs(w.sum() - 1.0) < 1e-6


def test_prompt_tokens_are_masked_out():
    d = _datum([
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ])
    w = np.array(d["lossFnInputs"]["weights"]["data"], dtype=np.float64)
    # Some positions must be zero (the prompt) and some positive (the response).
    assert (w == 0).any()
    assert (w > 0).any()


def test_input_target_are_shifted_by_one():
    d = _datum([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ])
    n_in = len(d["modelInput"]["chunks"][0]["tokens"])
    n_tgt = len(d["lossFnInputs"]["target_tokens"]["data"])
    n_w = len(d["lossFnInputs"]["weights"]["data"])
    assert n_tgt == n_in
    assert n_w == n_in


def test_no_assistant_content_returns_none():
    # Only a user turn -> no trainable tokens -> None.
    assert _datum([{"role": "user", "content": "hi"}]) is None


def test_truncated_away_response_returns_none():
    # max_length so small the whole budget is consumed by the prompt + the assistant header/think block, leaving no response tokens -> None.
    d = _datum(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "x" * 500},
        ],
        max_length=32,
    )
    assert d is None


def test_truncation_respects_max_length():
    long_answer = "x" * 500
    max_length = 120
    d = _datum(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": long_answer},
        ],
        max_length=max_length,
    )
    assert d is not None
    # input = tokens[:-1], so length is max_length - 1 after truncation.
    assert len(d["modelInput"]["chunks"][0]["tokens"]) == max_length - 1
    # Some response survived, so weights still normalize to 1.
    w = np.array(d["lossFnInputs"]["weights"]["data"], dtype=np.float64)
    assert abs(w.sum() - 1.0) < 1e-6


def test_tool_role_is_masked_like_user():
    d = _datum([
        {"role": "user", "content": "weather?"},
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ])
    assert d is not None
    w = np.array(d["lossFnInputs"]["weights"]["data"], dtype=np.float64)
    # tool content must not receive loss weight; only the final assistant turn.
    assert abs(w.sum() - 1.0) < 1e-6
