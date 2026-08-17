"""Tests for ``envs/lcb_utils.py``.

Covers:
  * LCB normalization round-trip (`taco_to_lcb_format`)
  * Code extraction (`extract_code_from_model`)
  * `fetch_live_code_bench_system_prompt`, `postprocess_lcb_sample`

``lcb_utils`` is stdlib-only and import-light, so we only add the repo root to ``sys.path`` so ``envs`` is importable.
"""
import json
import os
import sys

# Add the repo root to path so the import-light ``envs`` package (and ``envs.lcb_utils``) resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings
from hypothesis import strategies as st

from training_sessions_cookbook.rl.envs.lcb_utils import (
    LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE,
    LCB_FORMATTING_WITHOUT_STARTER_CODE,
    LCB_SYSTEM_MESSAGE_GENERIC,
    extract_code_from_model,
    fetch_live_code_bench_system_prompt,
    postprocess_lcb_sample,
    taco_to_lcb_format,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Short, plain text. Backticks are stripped so generated code bodies and surrounding text can never accidentally form a fenced block.
_no_backtick_text = st.text(max_size=30).map(lambda s: s.replace("`", ""))

# Identifier-ish function names (non-empty word characters).
_fn_names = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789",
    min_size=1,
    max_size=12,
)


@st.composite
def _equal_length_io(draw):
    """Generate equal-length ``inputs``/``outputs`` lists of short strings."""
    pairs = draw(st.lists(st.tuples(_no_backtick_text, _no_backtick_text), max_size=8))
    inputs = [p[0] for p in pairs]
    outputs = [p[1] for p in pairs]
    return inputs, outputs


# ===========================================================================
# LCB Normalization Round-Trip
# ===========================================================================
# For any {inputs, outputs} of equal length, taco_to_lcb_format produces exactly one case per pair, preserving the input/output values and assigning the correct testtype (stdin_stdout when no fn_name, functional otherwise).
#
# ===========================================================================


@settings(max_examples=100)
@given(_equal_length_io())
def test_taco_to_lcb_round_trip_stdin_stdout(io_pair):
    inputs, outputs = io_pair
    result = taco_to_lcb_format({"inputs": inputs, "outputs": outputs})

    assert len(result) == len(inputs)
    for i, case in enumerate(result):
        assert case["input"] == inputs[i]
        assert case["output"] == outputs[i]
        assert case["testtype"] == "stdin_stdout"


@settings(max_examples=100)
@given(_equal_length_io(), _fn_names)
def test_taco_to_lcb_round_trip_functional(io_pair, fn_name):
    inputs, outputs = io_pair
    result = taco_to_lcb_format(
        {"inputs": inputs, "outputs": outputs, "fn_name": fn_name}
    )

    assert len(result) == len(inputs)
    for i, case in enumerate(result):
        assert case["input"] == inputs[i]
        assert case["output"] == outputs[i]
        assert case["testtype"] == "functional"
        assert case["metadata"]["func_name"] == fn_name


def test_taco_to_lcb_list_output_takes_first_element():
    """When an output element is a list, the case output is its first element."""
    result = taco_to_lcb_format({"inputs": ["a", "b"], "outputs": [["x", "y"], ["z"]]})
    assert result[0]["output"] == "x"
    assert result[1]["output"] == "z"


def test_taco_to_lcb_empty():
    assert taco_to_lcb_format({"inputs": [], "outputs": []}) == []


# ===========================================================================
# Code Extraction
# ===========================================================================
# For any text containing at least one fenced code block, extract_code_from_model returns the contents of the LAST block exactly (stripped); None when absent.
#
# ===========================================================================


def _fenced(body: str) -> str:
    return f"```python\n{body}\n```"


@settings(max_examples=100)
@given(
    body=_no_backtick_text,
    prefix=_no_backtick_text,
    suffix=_no_backtick_text,
    earlier=st.lists(_no_backtick_text, max_size=3),
)
def test_extract_code_returns_last_block(body, prefix, suffix, earlier):
    # Build text: prefix, some EARLIER fenced blocks, then the target block last.
    parts = [prefix]
    parts.extend(_fenced(b) for b in earlier)
    parts.append(_fenced(body))
    parts.append(suffix)
    text = "\n".join(parts)

    assert extract_code_from_model(text) == body.strip()


def test_extract_code_no_fence_returns_none():
    assert extract_code_from_model("no code here at all") is None
    assert extract_code_from_model("") is None


def test_extract_code_multiple_blocks_returns_last():
    text = (
        "first:\n```python\nprint('one')\n```\n"
        "second:\n```python\nprint('two')\n```\n"
    )
    assert extract_code_from_model(text) == "print('two')"


# ===========================================================================
# Example tests: fetch_live_code_bench_system_prompt
# ===========================================================================


def test_system_prompt_without_starter():
    prompt = fetch_live_code_bench_system_prompt("Solve the problem.")
    assert LCB_SYSTEM_MESSAGE_GENERIC in prompt
    assert "Solve the problem." in prompt
    assert LCB_FORMATTING_WITHOUT_STARTER_CODE in prompt
    assert "```python\n# YOUR CODE HERE\n```" in prompt


def test_system_prompt_with_starter():
    starter = "class Solution:\n    def solve(self):\n        pass"
    prompt = fetch_live_code_bench_system_prompt("Solve it.", starter_code=starter)
    assert LCB_SYSTEM_MESSAGE_GENERIC in prompt
    assert LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE in prompt
    assert f"```python\n{starter}\n```" in prompt


# ===========================================================================
# Example tests: postprocess_lcb_sample
# ===========================================================================


def test_postprocess_stdin_stdout_shape():
    sample = [
        {"input": "1\n2\n", "output": "3\n", "testtype": "stdin_stdout", "metadata": {}},
        {"input": "4\n5\n", "output": "9\n", "testtype": "stdin_stdout", "metadata": {}},
    ]
    result = postprocess_lcb_sample(sample)
    assert set(result.keys()) == {"input_output"}

    parsed = json.loads(result["input_output"])
    assert parsed["inputs"] == ["1\n2\n", "4\n5\n"]
    assert parsed["outputs"] == ["3\n", "9\n"]
    assert "fn_name" not in parsed


def test_postprocess_functional_includes_fn_name():
    sample = [
        {
            "input": "[1, 2]",
            "output": "3",
            "testtype": "functional",
            "metadata": {"func_name": "add"},
        }
    ]
    result = postprocess_lcb_sample(sample)
    parsed = json.loads(result["input_output"])
    assert parsed["inputs"] == ["[1, 2]"]
    assert parsed["outputs"] == ["3"]
    assert parsed["fn_name"] == "add"


def test_postprocess_from_taco_functional_round_trip():
    """taco_to_lcb_format output should flow into postprocess_lcb_sample."""
    cases = taco_to_lcb_format(
        {"inputs": ["[1, 2]"], "outputs": ["3"], "fn_name": "add"}
    )
    result = postprocess_lcb_sample(cases)
    parsed = json.loads(result["input_output"])
    assert parsed["inputs"] == ["[1, 2]"]
    assert parsed["outputs"] == ["3"]
    assert parsed["fn_name"] == "add"
