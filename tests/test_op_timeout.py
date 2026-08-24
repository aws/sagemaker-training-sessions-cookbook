"""Tests for the service backend's per-operation wait ceilings.

The default is "wait for the service": the service bounds and retries training
operations itself, so a client-side ceiling below that envelope would end a
session that is merely queued behind other tenants. Ceilings are opt-in via
``OP_TIMEOUT_SECONDS`` (shared) or ``TRAIN_OP_TIMEOUT_SECONDS`` /
``SAMPLE_OP_TIMEOUT_SECONDS`` (per class), or injected as constructor params.

Resolution happens at BACKEND CONSTRUCTION, not at module import, so these
tests set env vars and construct - no ``importlib.reload`` gymnastics, and no
frozen-at-import surprishes for callers either.
"""

from __future__ import annotations

import pytest

from training_sessions_cookbook.rl.backends import service as service_mod
from training_sessions_cookbook.rl.backends.service import (
    ServiceBackend,
    _resolve_class_timeout,
    _resolve_op_timeout,
)


def _backend(**kwargs) -> ServiceBackend:
    """A ServiceBackend with inert collaborators (timeout logic needs none)."""
    return ServiceBackend(
        args=None, svc=None, tc=None, sampling_client=None, tokenizer=None,
        types={}, **kwargs,
    )


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, None),        # unset: wait for the service
        ("", None),          # empty: same as unset
        ("   ", None),       # whitespace only
        ("none", None),      # explicit opt-out
        ("NONE", None),      # case-insensitive
        ("0", None),         # zero means no ceiling, not "expire immediately"
        ("-5", None),        # a negative ceiling is meaningless
        ("1800", 1800.0),    # explicit ceiling
        ("600.5", 600.5),    # fractional seconds
    ],
)
def test_op_timeout_resolution(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: float | None
) -> None:
    if env_value is None:
        monkeypatch.delenv("OP_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("OP_TIMEOUT_SECONDS", env_value)
    assert _resolve_op_timeout() == expected


def test_invalid_value_is_not_silently_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed ceiling fails loudly rather than reverting to a default.

    Silently treating ``OP_TIMEOUT_SECONDS=abc`` as unbounded would hide a typo
    exactly when the operator most wanted a ceiling.
    """
    monkeypatch.setenv("OP_TIMEOUT_SECONDS", "abc")
    with pytest.raises(ValueError):
        _resolve_op_timeout()


def test_class_specific_env_var_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """TRAIN/SAMPLE_OP_TIMEOUT_SECONDS override the shared OP_TIMEOUT_SECONDS."""
    monkeypatch.setenv("OP_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("TRAIN_OP_TIMEOUT_SECONDS", "1200")
    monkeypatch.delenv("SAMPLE_OP_TIMEOUT_SECONDS", raising=False)
    assert _resolve_class_timeout("TRAIN_OP_TIMEOUT_SECONDS") == 1200.0
    assert _resolve_class_timeout("SAMPLE_OP_TIMEOUT_SECONDS") == 600.0  # shared fallback


def test_backend_resolves_env_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env changes take effect on the NEXT backend, no module reload needed.

    Resolution used to happen at import, freezing the values for the process
    lifetime; setting the env after import silently did nothing.
    """
    monkeypatch.delenv("OP_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("TRAIN_OP_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SAMPLE_OP_TIMEOUT_SECONDS", raising=False)
    assert _backend()._train_op_timeout is None

    monkeypatch.setenv("TRAIN_OP_TIMEOUT_SECONDS", "900")
    backend = _backend()
    assert backend._train_op_timeout == 900.0
    assert backend._sample_op_timeout is None  # class vars are independent


def test_constructor_params_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Programmatic callers can set ceilings without touching the process env."""
    monkeypatch.setenv("OP_TIMEOUT_SECONDS", "600")
    backend = _backend(train_op_timeout=300.0, sample_op_timeout=45.0)
    assert backend._train_op_timeout == 300.0
    assert backend._sample_op_timeout == 45.0


def test_explicit_none_means_unbounded_despite_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``None`` is a meaningful value (unbounded), distinct from "not passed".

    With a plain ``None`` default, a caller could never request unbounded
    waits while the environment set a ceiling; the ``_UNSET`` sentinel keeps
    the two cases apart.
    """
    monkeypatch.setenv("OP_TIMEOUT_SECONDS", "600")
    backend = _backend(train_op_timeout=None, sample_op_timeout=None)
    assert backend._train_op_timeout is None
    assert backend._sample_op_timeout is None
    # and not passing them still picks up the env
    assert _backend()._train_op_timeout == 600.0


def test_no_module_level_resolved_constants() -> None:
    """The module must not cache env-resolved timeouts at import time."""
    for name in ("OP_TIMEOUT_SECONDS", "TRAIN_OP_TIMEOUT_SECONDS", "SAMPLE_OP_TIMEOUT_SECONDS"):
        assert not hasattr(service_mod, name), (
            f"{name} resolved at import freezes the env for the process lifetime"
        )
