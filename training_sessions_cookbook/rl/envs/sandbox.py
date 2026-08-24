"""SandboxFusion HTTP client for sandboxed code execution.

SandboxFusion is a Docker-based code execution sandbox. Run it locally with::

    docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609

Configuration via environment variables:
    SANDBOX_URL: Endpoint URL (default: http://localhost:8080/run_code)
    SANDBOX_MAX_CONCURRENCY: Max concurrent requests (default: 4)
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import json
import os
from typing import Any

import aiohttp

from .lcb_utils import TEST_CODE, TEST_UTIL, postprocess_lcb_sample


class SandboxFusionClient:
    """Async HTTP client for SandboxFusion code execution.

    Usage::

        client = SandboxFusionClient()
        success, response = await client.run(
            code="print('hello')",
            files={"data.txt": "some content"},
            timeout=30,
        )
        await client.close()

    The endpoint URL and max concurrency default to the ``SANDBOX_URL`` and ``SANDBOX_MAX_CONCURRENCY`` environment variables. A single connection-pooled session is shared across calls and is created lazily on first use.
    """

    def __init__(
        self,
        url: str | None = None,
        max_concurrency: int | None = None,
    ):
        self._url = url or os.getenv("SANDBOX_URL", "http://localhost:8080/run_code")
        self._max_concurrency = max_concurrency or int(os.getenv("SANDBOX_MAX_CONCURRENCY", "4"))
        self._session: Any | None = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> Any:
        """Get or create the shared HTTP session with connection pooling.

        The ``TCPConnector`` limits concurrent connections to ``max_concurrency``; when all connections are busy, additional requests wait in a queue until a connection becomes available.
        """
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=self._max_concurrency,
                    limit_per_host=self._max_concurrency,
                )
                timeout = aiohttp.ClientTimeout(total=6000)
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                )
            return self._session

    async def run(
        self,
        code: str,
        files: dict[str, str],
        timeout: float,
        language: str = "python",
    ) -> tuple[bool, dict[str, Any]]:
        """Execute code with supporting files in the sandbox.

        Args:
            code: Main code to execute (entry point).
            files: Additional files to include ``{filename: content}``.
            timeout: Execution timeout in seconds.
            language: Programming language (default: ``python``).

        Returns:
            Tuple of ``(success, response)``:
            - ``success`` is True only when the response ``status == "Success"``.
            - ``response`` contains the full API response or error details.
        """
        encoded_files = {
            k: base64.b64encode(v.encode("utf-8")).decode("utf-8") for k, v in files.items()
        }

        payload = {
            "code": code,
            "language": language,
            "run_timeout": int(timeout),
            "files": encoded_files,
        }

        try:
            session = await self._get_session()
            # Per-request deadline: the sandbox enforces `run_timeout` on the
            # execution itself, so a healthy response arrives within roughly
            # that budget plus transfer/queueing. Without this, a wedged
            # sandbox holds the rollout worker for the session-level backstop
            # (100 minutes) instead of failing this one episode quickly.
            request_timeout = aiohttp.ClientTimeout(total=timeout + 60)
            async with session.post(
                self._url, json=payload, timeout=request_timeout
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return False, {"error": f"HTTP {resp.status}: {error_text}"}

                data: dict[str, Any] = await resp.json()

                if data.get("status") == "SandboxError":
                    return False, {"error": data.get("message", "SandboxError"), **data}

                success = data.get("status") == "Success"
                return success, data

        except Exception as e:
            return False, {"error": str(e)}

    async def close(self) -> None:
        """Close the HTTP session and release resources."""
        async with self._session_lock:
            if self._session is not None and not self._session.closed:
                await self._session.close()
                self._session = None


# ---------------------------------------------------------------------------
# Grading entry point. The LCB harness strings and test normalization come from ``envs.lcb_utils`` (stdlib + numpy only).
# ---------------------------------------------------------------------------

# Module-level lazily-created shared client. A single pooled client is reused across all grading calls so connections are pooled per ``SANDBOX_MAX_CONCURRENCY``.
# One client per event loop, keyed by a weak reference to the loop.
#
# An ``aiohttp.ClientSession`` is bound to the event loop it was created on. A
# process-global singleton therefore breaks the second ``asyncio.run()`` in the
# same process (validation after training, back-to-back programmatic runs,
# test suites): the cached session's loop is closed, and every request fails
# with "Event loop is closed" / "attached to a different loop". Keying the
# cache by the RUNNING loop gives each loop its own session, and the weak keys
# let a dead loop's entry be collected.
import weakref

_clients: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_atexit_registered = False


def _close_clients_at_exit() -> None:
    """Close all cached clients' HTTP sessions at interpreter exit.

    Without this, aiohttp logs "Unclosed client session" ERROR lines when a
    still-open client is garbage-collected during teardown, alarming noise at
    the end of an otherwise clean run.
    """
    for client in list(_clients.values()):
        try:
            asyncio.run(client.close())
        except Exception:
            pass  # teardown is best-effort; never fail exit


def _get_client() -> SandboxFusionClient:
    """Get or create the SandboxFusion client for the current event loop."""
    global _atexit_registered
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None:
        client = SandboxFusionClient()
        _clients[loop] = client
        if not _atexit_registered:
            atexit.register(_close_clients_at_exit)
            _atexit_registered = True
    return client


async def sandbox_check_correctness(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int = 6,
) -> tuple[bool, dict[str, Any]]:
    """Check correctness of generated code using SandboxFusion execution.

    Runs the LiveCodeBench test harness (``TEST_CODE`` / ``TEST_UTIL`` from ``envs.lcb_utils``) inside the sandbox over ``code.py`` (the extracted solution) and ``test_cases.txt`` (the LCB-format test cases).

    Args:
        sample: List of test cases in LiveCodeBench format (at least one).
        generation: Generated code to test.
        timeout: Per-test timeout in seconds.

    Returns:
        Tuple of ``(all_passed, details)`` as returned by ``client.run``; on any error during setup/execution, returns ``(False, {"error": str(e)})``.
    """
    assert len(sample) >= 1, "Sample must contain at least one test case"

    test_cases = postprocess_lcb_sample(sample)

    try:
        test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
        total_timeout = (timeout + 1) * test_cnt + 5

        client = _get_client()
        return await client.run(
            code=TEST_CODE % {"timeout": timeout},
            files={
                "test_cases.txt": json.dumps(test_cases),
                "code.py": generation,
                "testing_util.py": TEST_UTIL,
            },
            timeout=total_timeout,
        )
    except Exception as e:
        return False, {"error": str(e)}
