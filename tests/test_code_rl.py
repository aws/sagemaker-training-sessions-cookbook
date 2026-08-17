"""Tests for ``envs/code_rl.py`` reward + ``CodeRLEnv`` stepping.

Covers:
  * Reward formula: ``deepcoder_reward`` returns exactly ``format_coef * (has_code - 1.0) + correct`` for any ``format_coef`` in [0, 1], any ``has_code`` in {True, False} (controlled by whether the input text contains a fenced code block), and any ``passed`` in {True, False}.
  * Single-turn ``CodeRLEnv`` stepping with a *mocked* async sandbox: reset shape, terminal grade, code-block vs no-code-block reward, and the synchronous ``step`` raising ``NotImplementedError``.
  * A small multi-turn smoke test (2 turns) over the agent-tool loop with a mocked sandbox, plus the failed-parse termination path.

Grading is async (``sandbox_check_correctness`` is a coroutine), so the async env methods are driven with ``asyncio.run(...)`` inside plain test functions; the suite does not depend on ``pytest-asyncio``. No live sandbox / ``aiohttp`` is required: a fake async ``sandbox_check`` is injected via the ``CodeRLEnv`` constructor.
"""
import asyncio
import math
import os
import sys

# Add the repo root to path so the ``envs`` package resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings
from hypothesis import strategies as st

from training_sessions_cookbook.rl.envs.code_rl import CodeRLEnv, deepcoder_reward
from training_sessions_cookbook.rl.envs.deepcoder import DeepcoderTask
from training_sessions_cookbook.rl.envs.tool_render import render_tool_call


# ---------------------------------------------------------------------------
# Fake async sandbox checks (no live sandbox / aiohttp)
# ---------------------------------------------------------------------------


async def fake_pass(tests, code, timeout):
    """A sandbox check that always reports a pass."""
    return (True, {})


async def fake_fail(tests, code, timeout):
    """A sandbox check that always reports a failure."""
    return (False, {"error": "x"})


class _CountingSandbox:
    """An injectable sandbox check that records how many times it was called."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def __call__(self, tests, code, timeout):
        self.calls += 1
        return self.result


def _task():
    """A minimal single-test DeepcoderTask whose prompt text is ``"prob"``."""
    return DeepcoderTask(
        problem="prob",
        tests=[{"input": "1", "output": "2", "testtype": "stdin_stdout", "metadata": {}}],
    )


# ===========================================================================
# Reward Formula  (Hypothesis, >=100 examples)
# ===========================================================================

# Text without any fenced block: no backticks => extract_code_from_model is None.
_no_fence = st.text(st.characters(blacklist_characters="`"), max_size=100)
# Inner code body that cannot accidentally close/open a fence.
_inner_code = st.text(st.characters(blacklist_characters="`"), max_size=100)


@settings(max_examples=100)
@given(
    format_coef=st.floats(min_value=0, max_value=1, allow_nan=False),
    has_code=st.booleans(),
    passed=st.booleans(),
    inner=_inner_code,
    prose=_no_fence,
)
def test_reward_formula_property(format_coef, has_code, passed, inner, prose):
    """reward == ``format_coef * (has_code - 1.0) + float(passed)`` and metrics == ``{"format": float(has_code), "correct": float(passed)}``.
    """
    if has_code:
        text = f"```python\n{inner}\n```"
        expected_format = 1.0
    else:
        text = prose
        expected_format = 0.0

    reward, metrics = deepcoder_reward(text, passed, format_coef)

    expected = format_coef * (expected_format - 1.0) + float(passed)
    assert math.isclose(reward, expected, rel_tol=1e-9, abs_tol=1e-12)
    assert metrics == {"format": expected_format, "correct": float(passed)}


def test_reward_with_code_equals_correct():
    """With a fenced block the format term is zero => reward == float(passed)."""
    r_pass, m_pass = deepcoder_reward("```python\nx\n```", True)
    assert r_pass == 1.0
    assert m_pass == {"format": 1.0, "correct": 1.0}

    r_fail, m_fail = deepcoder_reward("```python\nx\n```", False)
    assert r_fail == 0.0
    assert m_fail == {"format": 1.0, "correct": 0.0}


def test_reward_no_code_applies_format_penalty():
    """With no fenced block the default format_coef (0.1) penalty applies."""
    r_fail, m_fail = deepcoder_reward("no code", False)
    assert math.isclose(r_fail, -0.1, abs_tol=1e-12)
    assert m_fail == {"format": 0.0, "correct": 0.0}

    r_pass, m_pass = deepcoder_reward("no code", True)
    assert math.isclose(r_pass, 0.9, abs_tol=1e-12)
    assert m_pass == {"format": 0.0, "correct": 1.0}


# ===========================================================================
# Single-turn CodeRLEnv with a mocked sandbox
# ===========================================================================


def test_single_turn_reset_returns_raw_prompt():
    env = CodeRLEnv(dataset=[_task()], max_turns=1, sandbox_check=fake_pass)
    obs, info = env.reset()
    assert obs == "prob"
    assert info == {"env_type": "code_rl"}


def test_single_turn_step_pass():
    env = CodeRLEnv(dataset=[_task()], max_turns=1, sandbox_check=fake_pass)
    env.reset()

    result = asyncio.run(env.step_async("```python\nprint(2)\n```"))

    assert isinstance(result, tuple) and len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert terminated is True
    assert truncated is True
    assert reward == 1.0
    assert info["passed"] is True


def test_single_turn_step_fail_with_code_block():
    env = CodeRLEnv(dataset=[_task()], max_turns=1, sandbox_check=fake_fail)
    env.reset()

    _, reward, terminated, truncated, info = asyncio.run(
        env.step_async("```python\nprint(2)\n```")
    )
    # has_code True, correct 0 => format_coef*(1-1)+0 == 0.0.
    assert reward == 0.0
    assert info["passed"] is False
    assert terminated is True and truncated is True


def test_single_turn_step_no_code_block_skips_sandbox():
    sandbox = _CountingSandbox((False, {"error": "x"}))
    env = CodeRLEnv(dataset=[_task()], max_turns=1, sandbox_check=sandbox)
    env.reset()

    _, reward, _, _, info = asyncio.run(env.step_async("just prose, no code block"))

    # No code extracted => sandbox is never invoked, passed False.
    assert sandbox.calls == 0
    assert info["passed"] is False
    # reward == format_coef*(0-1)+0 == -format_coef (default -0.1).
    assert math.isclose(reward, -0.1, abs_tol=1e-12)


def test_sync_step_raises_not_implemented():
    env = CodeRLEnv(dataset=[_task()], max_turns=1, sandbox_check=fake_pass)
    env.reset()
    try:
        env.step("x")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("synchronous step() should raise NotImplementedError")


# ===========================================================================
# Multi-turn smoke (mocked sandbox), 2 turns
# ===========================================================================


def test_multi_turn_reset_includes_tool_schema():
    env = CodeRLEnv(
        dataset=[_task()],
        max_turns=2,
        sandbox_check=fake_pass,
        model_name="openai/gpt-oss-20b",
    )
    obs, info = env.reset()
    assert "prob" in obs
    assert "check_solution" in obs
    assert info == {"env_type": "code_rl"}


def test_multi_turn_two_turns_then_done():
    env = CodeRLEnv(
        dataset=[_task()],
        max_turns=2,
        sandbox_check=fake_pass,
        model_name="openai/gpt-oss-20b",
    )
    env.reset()

    # Turn 1: a tool call -> episode continues, intermediate reward 0.0.
    action1 = render_tool_call("check_solution", {"code": "print(2)"})
    r1 = asyncio.run(env.step_async(action1))
    obs1, reward1, terminated1, _, _ = r1
    assert terminated1 is False
    assert reward1 == 0.0
    assert "passed" in obs1

    # Turn 2: a final fenced answer at max_turns -> terminal grade, reward 1.0.
    action2 = "```python\nprint(2)\n```"
    r2 = asyncio.run(env.step_async(action2))
    _, reward2, terminated2, truncated2, info2 = r2
    assert terminated2 is True
    assert truncated2 is True
    assert reward2 == 1.0


def test_multi_turn_failed_parse_terminates():
    env = CodeRLEnv(
        dataset=[_task()],
        max_turns=2,
        sandbox_check=fake_pass,
        model_name="openai/gpt-oss-20b",
    )
    env.reset()

    _, reward, terminated, truncated, info = asyncio.run(
        env.step_async("just prose no tool no code")
    )
    assert terminated is True
    assert truncated is True
    assert math.isclose(reward, env.failed_parse_reward, abs_tol=1e-12)
    assert reward == -0.1
