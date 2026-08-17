"""Tests for ``envs/sandbox.py`` with a mocked HTTP layer (no live sandbox).

Covers:
  * Sandbox payload fidelity: files base64 round-trip back to the originals and ``run_timeout == int(timeout)``; success only on ``status == "Success"``.
  * Behavior: success / failure / SandboxError / non-200 / exception handling.
  * ``sandbox_check_correctness`` request shape (file keys, harness contents, total-timeout formula ``(timeout+1)*num_tests+5``).

We never require ``aiohttp`` (or a live sandbox). The client uses ``aiohttp`` lazily inside ``_get_session``; we MOCK at that boundary by replacing ``_get_session`` with an async function returning a fake session whose ``.post(url, json=payload)`` returns an async context manager yielding a fake response.

The async client methods are driven with ``asyncio.run(...)`` inside plain test functions so the suite does not depend on ``pytest-asyncio``.
"""
import asyncio
import base64
import json
import os
import sys

# Add the repo root to path so the import-light ``envs`` package (and ``envs.sandbox`` / ``envs.lcb_utils``) resolves. ``envs.sandbox`` imports only the standard library at module load time, so this requires neither ``aiohttp`` nor a running sandbox.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings
from hypothesis import strategies as st

from training_sessions_cookbook.rl.envs import sandbox as sandbox_mod
from training_sessions_cookbook.rl.envs.lcb_utils import TEST_UTIL
from training_sessions_cookbook.rl.envs.sandbox import SandboxFusionClient


# ---------------------------------------------------------------------------
# Fake async HTTP layer
# ---------------------------------------------------------------------------


class _FakeResp:
    """Async-context-manager response stand-in for ``session.post(...)``."""

    def __init__(self, status, payload=None, text=""):
        self.status = status
        self._payload = payload
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class _FakeSession:
    """Captures the ``url`` and ``json=payload`` passed to ``.post``."""

    def __init__(self, resp, capture):
        self._resp = resp
        self._capture = capture

    def post(self, url, json=None):
        self._capture["url"] = url
        self._capture["payload"] = json
        return self._resp


class _RaisingSession:
    """A session whose ``.post`` raises, to exercise the exception path."""

    def __init__(self, exc):
        self._exc = exc

    def post(self, url, json=None):
        raise self._exc


def _client_with(resp, capture=None):
    """Build a client whose ``_get_session`` returns a fake session.

    ``_get_session`` is replaced with an async function (matching the awaited call site ``await self._get_session()``), so ``aiohttp`` is never imported.
    """
    if capture is None:
        capture = {}
    client = SandboxFusionClient(url="http://fake-sandbox/run_code")
    session = _FakeSession(resp, capture)

    async def _fake_get_session():
        return session

    client._get_session = _fake_get_session  # type: ignore[method-assign]
    return client, capture


# ---------------------------------------------------------------------------
# Sandbox payload fidelity (property-based, >=100 examples)
# ---------------------------------------------------------------------------

_filenames = st.text(min_size=1, max_size=20).filter(lambda s: s.strip() != "")
_contents = st.text(max_size=80)


@settings(max_examples=100)
@given(
    code=st.text(max_size=80),
    files=st.dictionaries(keys=_filenames, values=_contents, max_size=6),
    timeout=st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False),
)
def test_payload_fidelity_property(code, files, timeout):
    """For arbitrary code/files/timeout, the request payload's base64-encoded files decode (utf-8) back to the originals and ``run_timeout == int(timeout)``.
    """
    resp = _FakeResp(status=200, payload={"status": "Success"})
    client, capture = _client_with(resp)

    success, _ = asyncio.run(client.run(code=code, files=files, timeout=timeout))

    payload = capture["payload"]
    assert payload["code"] == code
    assert payload["language"] == "python"
    assert payload["run_timeout"] == int(timeout)
    assert isinstance(payload["run_timeout"], int)

    # base64 round-trip for every file.
    assert set(payload["files"].keys()) == set(files.keys())
    for name, content in files.items():
        decoded = base64.b64decode(payload["files"][name]).decode("utf-8")
        assert decoded == content

    # status == "Success" => success True.
    assert success is True


# ---------------------------------------------------------------------------
# Payload fidelity: concrete example (item 1)
# ---------------------------------------------------------------------------


def test_payload_fidelity_example():
    resp = _FakeResp(status=200, payload={"status": "Success"})
    client, capture = _client_with(resp)

    success, data = asyncio.run(
        client.run(code="print(1)", files={"a.py": "X\nY", "b.txt": "héllo"}, timeout=7)
    )

    payload = capture["payload"]
    assert payload["code"] == "print(1)"
    assert payload["language"] == "python"
    assert payload["run_timeout"] == 7
    assert isinstance(payload["run_timeout"], int)

    assert base64.b64decode(payload["files"]["a.py"]).decode("utf-8") == "X\nY"
    assert base64.b64decode(payload["files"]["b.txt"]).decode("utf-8") == "héllo"

    assert capture["url"] == "http://fake-sandbox/run_code"
    assert success is True


# ---------------------------------------------------------------------------
# Success / failure detection (item 2)
# ---------------------------------------------------------------------------


def test_run_success_status():
    data_in = {"status": "Success", "run_result": {"stdout": "ok"}}
    client, _ = _client_with(_FakeResp(status=200, payload=data_in))
    success, data = asyncio.run(client.run(code="x", files={}, timeout=1))
    assert success is True
    assert data == data_in


def test_run_failed_status():
    data_in = {"status": "Failed"}
    client, _ = _client_with(_FakeResp(status=200, payload=data_in))
    success, data = asyncio.run(client.run(code="x", files={}, timeout=1))
    assert success is False
    assert data == data_in


def test_run_sandbox_error_status():
    data_in = {"status": "SandboxError", "message": "x"}
    client, _ = _client_with(_FakeResp(status=200, payload=data_in))
    success, data = asyncio.run(client.run(code="x", files={}, timeout=1))
    assert success is False
    assert data["error"] == "x"


def test_run_non_200():
    client, _ = _client_with(_FakeResp(status=500, text="boom"))
    success, data = asyncio.run(client.run(code="x", files={}, timeout=1))
    assert success is False
    assert data == {"error": "HTTP 500: boom"}


def test_run_exception_in_post():
    client = SandboxFusionClient(url="http://fake-sandbox/run_code")
    session = _RaisingSession(RuntimeError("network down"))

    async def _fake_get_session():
        return session

    client._get_session = _fake_get_session  # type: ignore[method-assign]

    success, data = asyncio.run(client.run(code="x", files={}, timeout=1))
    assert success is False
    assert "error" in data
    assert "network down" in data["error"]


# ---------------------------------------------------------------------------
# sandbox_check_correctness request shape (item 3)
# ---------------------------------------------------------------------------


class _FakeClient:
    """Records ``run`` calls and returns a preset ``(success, data)``."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    async def run(self, code, files, timeout, language="python"):
        self.calls.append(
            {"code": code, "files": files, "timeout": timeout, "language": language}
        )
        return self.result

    async def close(self):
        pass


def test_sandbox_check_correctness_request_shape(monkeypatch):
    fake = _FakeClient((True, {"status": "Success"}))
    # Patch the module-level singleton so no network/session is created.
    monkeypatch.setattr(sandbox_mod, "_client", fake)

    sample = [{"input": "1", "output": "2", "testtype": "stdin_stdout", "metadata": {}}]
    result = asyncio.run(
        sandbox_mod.sandbox_check_correctness(sample=sample, generation="code")
    )

    # Return value propagates from the mocked run.
    assert result == (True, {"status": "Success"})

    assert len(fake.calls) == 1
    call = fake.calls[0]
    files = call["files"]
    assert set(files.keys()) == {"test_cases.txt", "code.py", "testing_util.py"}
    assert files["code.py"] == "code"
    assert files["testing_util.py"] == TEST_UTIL

    # total timeout = (timeout + 1) * num_tests + 5 ; default timeout=6, num_tests=1 => 12
    assert call["timeout"] == 12

    # test_cases.txt is the postprocessed LCB sample (one input/output pair).
    tc = json.loads(files["test_cases.txt"])
    parsed = json.loads(tc["input_output"])
    assert parsed["inputs"] == ["1"]
    assert parsed["outputs"] == ["2"]


def test_sandbox_check_correctness_propagates_failure(monkeypatch):
    fake = _FakeClient((False, {"error": "HTTP 500: boom"}))
    monkeypatch.setattr(sandbox_mod, "_client", fake)

    sample = [{"input": "1", "output": "2", "testtype": "stdin_stdout", "metadata": {}}]
    result = asyncio.run(
        sandbox_mod.sandbox_check_correctness(sample=sample, generation="code")
    )
    assert result == (False, {"error": "HTTP 500: boom"})
