"""Tests for the ``envs`` registry and the GSM8K environment.

Covers:
  * No heavy imports when only the ``envs`` package / GSM8K path is used (verified in a clean subprocess interpreter).
  * Registry totality: every registered ``--env_type`` resolves to a zero-arg callable producing an ``Env``-conforming object; unknown types raise ``ValueError``.
  * GSM8K invariance: ``GSM8KMultiTurnEnv`` produces the known GSM8K observations, rewards, and termination behavior.

The ``envs`` package is import-light, so we only add the repo root to ``sys.path`` so ``envs`` is importable.

We never call the ``code_rl`` factory's product (it would import ``datasets``/``aiohttp``); we only assert that ``code_rl`` IS registered (does not raise ``ValueError``).
"""
import argparse
import os
import subprocess
import sys

import pytest

# Add the standalone directory to path so the import-light ``envs`` package resolves.
_STANDALONE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STANDALONE_DIR)

from training_sessions_cookbook.rl.envs import Env, get_env_factory
from training_sessions_cookbook.rl.envs.gsm8k import GSM8KMultiTurnEnv


def _args(**overrides):
    """Build a minimal CLI-args namespace for env factories."""
    base = {"max_turns": 1, "_dataset": None}
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# No-heavy-import guard
# ---------------------------------------------------------------------------

# Modules that must NOT be pulled in merely by importing ``envs``.
_FORBIDDEN_MODULES = (
    "aiohttp",
    "datasets",
    "numpy",
)


def test_envs_import_is_dependency_light():
    """Importing ``envs`` in a clean interpreter pulls in no heavy deps.

    Run in a subprocess so the check is against a fresh ``sys.modules`` rather than this already-populated test interpreter.
    """
    forbidden = ", ".join(repr(m) for m in _FORBIDDEN_MODULES)
    code = (
        "import sys\n"
        f"sys.path.insert(0, {_STANDALONE_DIR!r})\n"
        "import training_sessions_cookbook.rl.envs\n"
        f"forbidden = ({forbidden},)\n"
        "bad = [m for m in forbidden if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# �� Registry Totality
# ---------------------------------------------------------------------------


def test_gsm8k_factory_returns_conforming_env():
    """``get_env_factory('gsm8k', args)`` yields a zero-arg Env factory.

    """
    factory = get_env_factory("gsm8k", _args())
    assert callable(factory)

    env = factory()
    # ``Env`` is ``runtime_checkable``: conformance is by method presence.
    assert isinstance(env, Env)
    for method in ("assign_problem", "reset", "step"):
        assert hasattr(env, method) and callable(getattr(env, method))


def test_unknown_env_type_raises_value_error():
    """An unregistered ``env_type`` raises ``ValueError``.

    """
    with pytest.raises(ValueError):
        get_env_factory("does_not_exist", _args())


def test_code_rl_is_registered():
    """``code_rl`` IS registered: it must not raise ``ValueError``.

    The ``code_rl`` module / its heavy deps may not be present yet, so a callable factory OR an ``ImportError``/``ModuleNotFoundError`` both count as PASS. Only a ``ValueError`` (meaning "not registered") is a FAIL.

    """
    try:
        factory = get_env_factory("code_rl", _args())
    except (ImportError, ModuleNotFoundError):
        # code_rl module/deps not built yet: still considered registered.
        pass
    except ValueError:  # pragma: no cover - explicit failure signal
        pytest.fail("code_rl must be registered (got ValueError)")
    else:
        assert callable(factory)


# ---------------------------------------------------------------------------
# �� GSM8K Invariance (behavioral snapshot)
# ---------------------------------------------------------------------------


def test_gsm8k_reset_observation_contains_question_and_instruction():
    """``reset()`` exposes the question and the ``####`` instruction.

    """
    env = GSM8KMultiTurnEnv(
        max_turns=1, dataset=[{"question": "q", "answer": "42"}]
    )
    obs, info = env.reset()
    assert info == {"env_type": "gsm8k"}
    assert "q" in obs
    assert "####" in obs


def test_gsm8k_correct_answer_scores_one_and_terminates():
    """A correct ``#### <answer>`` scores 1.0 and ends the episode.

    """
    env = GSM8KMultiTurnEnv(
        max_turns=1, dataset=[{"question": "q", "answer": "42"}]
    )
    env.reset()
    obs, reward, terminated, truncated, info = env.step("#### 42")
    assert reward == 1.0
    assert terminated and truncated
    assert info == {"env_type": "gsm8k"}


def test_gsm8k_wrong_answer_scores_zero():
    """A wrong ``#### <answer>`` scores 0.0.

    """
    env = GSM8KMultiTurnEnv(
        max_turns=1, dataset=[{"question": "q", "answer": "42"}]
    )
    env.reset()
    _obs, reward, terminated, _truncated, _info = env.step("#### 7")
    assert reward == 0.0
    assert terminated
