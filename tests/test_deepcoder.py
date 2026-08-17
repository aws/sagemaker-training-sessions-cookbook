"""Tests for ``envs/deepcoder.py`` task construction + test normalization.

Covers the pure, network-free helpers of the DeepCoder loader on synthetic rows:
  * ``_ensure_dict``: metadata coercion (dict / JSON string / junk)
  * ``_normalize_tests``: taco conversion, JSON-string parsing, single-dict wrapping, functional ``func_name`` handling, drops
  * ``_build_question``: LCB prompt building + ``question``/``prompt``/``problem`` fallback + starter-code fencing
  * ``DeepcoderTask``: frozen dataclass field defaults

We deliberately do NOT touch ``load_deepcoder_tasks`` / ``_load_deepcoder_split`` (they need the ``datasets`` lib + network). ``deepcoder.py`` is import-light (``datasets`` is imported lazily), so we only add the repo root to ``sys.path`` so ``envs`` resolves.
"""
import json
import os
import sys

# Add the repo root to path so the ``envs`` package resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings
from hypothesis import strategies as st

from training_sessions_cookbook.rl.envs.deepcoder import (
    DeepcoderTask,
    _build_question,
    _ensure_dict,
    _normalize_tests,
)
from training_sessions_cookbook.rl.envs.lcb_utils import (
    LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE,
    LCB_SYSTEM_MESSAGE_GENERIC,
)

_REQUIRED_KEYS = {"input", "output", "testtype", "metadata"}


# ===========================================================================
# _ensure_dict
# ===========================================================================


def test_ensure_dict_passthrough_dict():
    d = {"a": 1, "b": "x"}
    assert _ensure_dict(d) == {"a": 1, "b": "x"}


def test_ensure_dict_parses_json_string():
    assert _ensure_dict('{"a": 1}') == {"a": 1}


def test_ensure_dict_invalid_string_returns_empty():
    assert _ensure_dict("not json {") == {}


def test_ensure_dict_non_dict_non_str_returns_empty():
    assert _ensure_dict(None) == {}
    assert _ensure_dict(5) == {}
    assert _ensure_dict([1, 2, 3]) == {}


def test_ensure_dict_json_string_encoding_non_dict_returns_empty():
    # Valid JSON, but decodes to a list (not a dict) -> {}.
    assert _ensure_dict("[1, 2, 3]") == {}


@settings(max_examples=100)
@given(st.dictionaries(st.text(max_size=8), st.integers(), max_size=5))
def test_ensure_dict_json_roundtrip(d):
    # Any dict, serialized to a JSON string, round-trips back to that dict.
    assert _ensure_dict(json.dumps(d)) == d


# ===========================================================================
# _normalize_tests
# ===========================================================================


def test_normalize_taco_inputs_outputs_dict():
    raw = {"inputs": ["1\n", "2\n"], "outputs": ["a\n", "b\n"]}
    result = _normalize_tests(raw, metadata={})

    assert len(result) == 2
    for case, inp, out in zip(result, raw["inputs"], raw["outputs"]):
        assert set(case.keys()) == _REQUIRED_KEYS
        assert case["input"] == inp
        assert case["output"] == out
        assert case["testtype"] == "stdin_stdout"


def test_normalize_json_string_list_of_tests():
    raw = json.dumps(
        [
            {"input": "1", "output": "2", "testtype": "stdin_stdout"},
            {"input": "3", "output": "4", "testtype": "stdin_stdout"},
        ]
    )
    result = _normalize_tests(raw, metadata={})

    assert len(result) == 2
    assert result[0]["input"] == "1"
    assert result[0]["output"] == "2"
    assert result[1]["input"] == "3"
    assert all(set(c.keys()) == _REQUIRED_KEYS for c in result)


def test_normalize_single_dict_wrapped_in_list():
    raw = {"input": "x", "output": "y", "testtype": "stdin_stdout"}
    result = _normalize_tests(raw, metadata={})

    assert len(result) == 1
    assert result[0]["input"] == "x"
    assert result[0]["output"] == "y"


def test_normalize_functional_func_name_from_test_metadata():
    raw = {
        "input": "[1, 2]",
        "output": "3",
        "testtype": "functional",
        "metadata": {"func_name": "add"},
    }
    result = _normalize_tests(raw, metadata={})

    assert len(result) == 1
    assert result[0]["testtype"] == "functional"
    assert result[0]["metadata"]["func_name"] == "add"
    assert isinstance(result[0]["metadata"]["func_name"], str)


def test_normalize_functional_func_name_from_outer_metadata():
    raw = {"input": "[1, 2]", "output": "3", "testtype": "functional"}
    result = _normalize_tests(raw, metadata={"func_name": "solve"})

    assert len(result) == 1
    assert result[0]["testtype"] == "functional"
    assert result[0]["metadata"]["func_name"] == "solve"
    assert isinstance(result[0]["metadata"]["func_name"], str)


def test_normalize_invalid_json_string_dropped():
    assert _normalize_tests("not valid json {", metadata={}) == []


def test_normalize_skips_non_dict_elements():
    raw = [
        {"input": "a", "output": "b", "testtype": "stdin_stdout"},
        "not a dict",
        5,
        None,
    ]
    result = _normalize_tests(raw, metadata={})

    assert len(result) == 1
    assert result[0]["input"] == "a"


def test_normalize_defaults_missing_input_output_and_testtype():
    # Missing input/output coerced to "", missing testtype defaults to stdin_stdout.
    result = _normalize_tests([{}], metadata={})
    assert len(result) == 1
    assert result[0]["input"] == ""
    assert result[0]["output"] == ""
    assert result[0]["testtype"] == "stdin_stdout"
    assert result[0]["metadata"] == {"func_name": None}


@settings(max_examples=100)
@given(
    st.lists(
        st.tuples(
            st.text(max_size=10).map(lambda s: s.replace("`", "")),
            st.text(max_size=10).map(lambda s: s.replace("`", "")),
        ),
        max_size=8,
    )
)
def test_normalize_taco_preserves_pairs(pairs):
    inputs = [p[0] for p in pairs]
    outputs = [p[1] for p in pairs]
    result = _normalize_tests({"inputs": inputs, "outputs": outputs}, metadata={})

    assert len(result) == len(inputs)
    for case, inp, out in zip(result, inputs, outputs):
        assert set(case.keys()) == _REQUIRED_KEYS
        assert case["input"] == inp
        assert case["output"] == out
        assert case["testtype"] == "stdin_stdout"


# ===========================================================================
# _build_question
# ===========================================================================


def test_build_question_with_question_key():
    prompt = _build_question({"question": "Sum two numbers."})
    assert prompt is not None
    assert LCB_SYSTEM_MESSAGE_GENERIC in prompt
    assert "Sum two numbers." in prompt


def test_build_question_falls_back_to_prompt_key():
    prompt = _build_question({"prompt": "Only prompt text."})
    assert prompt is not None
    assert "Only prompt text." in prompt


def test_build_question_falls_back_to_problem_key():
    prompt = _build_question({"problem": "Only problem text."})
    assert prompt is not None
    assert "Only problem text." in prompt


def test_build_question_with_starter_code_fenced():
    starter = "class Solution:\n    def solve(self):\n        pass"
    prompt = _build_question({"question": "Implement solve.", "starter_code": starter})
    assert prompt is not None
    assert LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE in prompt
    assert f"```python\n{starter}\n```" in prompt


def test_build_question_empty_or_missing_returns_none():
    assert _build_question({}) is None
    assert _build_question({"question": ""}) is None
    assert _build_question({"question": "   "}) is None
    assert _build_question({"question": 123}) is None


# ===========================================================================
# DeepcoderTask
# ===========================================================================


def test_deepcoder_task_fields_and_default_starter_code():
    tests = [{"input": "1", "output": "2", "testtype": "stdin_stdout", "metadata": {}}]
    task = DeepcoderTask(problem="prob", tests=tests)

    assert task.problem == "prob"
    assert task.tests == tests
    assert task.starter_code is None


def test_deepcoder_task_with_starter_code():
    task = DeepcoderTask(problem="p", tests=[], starter_code="def f(): pass")
    assert task.starter_code == "def f(): pass"


def test_deepcoder_task_is_frozen():
    task = DeepcoderTask(problem="p", tests=[])
    try:
        task.problem = "mutated"
    except Exception as e:  # FrozenInstanceError subclasses dataclasses error
        assert "frozen" in str(type(e)).lower() or "frozen" in str(e).lower()
    else:
        raise AssertionError("DeepcoderTask should be immutable (frozen dataclass)")
