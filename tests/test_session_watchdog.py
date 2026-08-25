"""Session-liveness watchdog: a server-side session failure must fail the
client loudly instead of hanging forever.

Observed on prod (2026-08-24): the service marked a session Failed (engine
checkpoint-load fault) while the client was awaiting sample results; with no
operation-path signal and no status polling, the run went silent for 80
minutes. The watchdog closes that hole.
"""

import asyncio

import pytest

from training_sessions_cookbook.rl.backends.service import ServiceBackend


class _Transport:
    def __init__(self, status="Running", reason=""):
        self.status, self.reason = status, reason

    def describe_training_session(self, TrainingSessionName):
        return {
            "TrainingSessionDetail": {
                "TrainingSessionStatus": self.status,
                "FailureReason": self.reason,
            }
        }


class _Svc:
    def __init__(self, transport):
        self._sagemaker_client = transport


from training_sessions_cookbook.service.session_watchdog import SessionWatchdog


def _backend(transport) -> ServiceBackend:
    b = ServiceBackend(
        args=None, svc=_Svc(transport), tc=None, sampling_client=None,
        tokenizer=None, types={},
    )
    b._session_name = "watchdog-test"
    return b


def test_shared_watchdog_is_usable_standalone():
    """The SFT loop constructs SessionWatchdog directly; same contract."""
    wd = SessionWatchdog(_Svc(_Transport(status="Running")), "sft-test")

    async def resolves():
        return "ok"

    async def scenario():
        assert await wd.guard(resolves(), what="op") == "ok"
        wd.stop()

    asyncio.run(scenario())


def test_guarded_wait_fails_fast_when_session_dies():
    transport = _Transport(status="Failed", reason="Failed to load checkpoint.")
    backend = _backend(transport)

    async def never_resolves():
        await asyncio.sleep(3600)

    async def scenario():
        # fast watchdog cadence for the test
        backend._ensure_watchdog()
        backend._watchdog._interval = 0.01
        with pytest.raises(RuntimeError) as exc:
            await asyncio.wait_for(
                backend._await_guarded(never_resolves(), what="sample request"),
                timeout=5,
            )
        assert "Failed to load checkpoint" in str(exc.value)
        assert "sample request abandoned" in str(exc.value)

    asyncio.run(scenario())


def test_guarded_wait_passes_results_through_when_alive():
    backend = _backend(_Transport(status="Running"))

    async def resolves():
        return 42

    async def scenario():
        assert await backend._await_guarded(resolves(), what="op") == 42
        backend.close()  # cancels the watchdog; no pending-task warnings

    asyncio.run(scenario())


def test_describe_errors_do_not_kill_the_session():
    """A failed poll (throttle, blip) is not a dead session."""
    class _FlakyTransport:
        def describe_training_session(self, TrainingSessionName):
            raise ConnectionError("blip")

    backend = _backend(_FlakyTransport())

    async def scenario():
        backend._ensure_watchdog()
        backend._watchdog._interval = 0.01
        backend._watchdog._ensure_started()

        async def probe():
            await asyncio.sleep(3600)

        # several failed polls happen while we wait; the guard must still be live
        waiter = asyncio.ensure_future(
            backend._await_guarded(probe(), what="op")
        )
        await asyncio.sleep(0.1)
        assert not waiter.done(), "flaky describes must not kill the wait"
        assert not backend._watchdog._dead.is_set()
        waiter.cancel()
        backend.close()

    asyncio.run(scenario())



def test_result_with_liveness_returns_result_when_alive():
    from training_sessions_cookbook.service.session_watchdog import result_with_liveness

    class _Future:
        def result(self, timeout=None):
            return "done"

    assert result_with_liveness(_Future(), _Svc(_Transport()), "s") == "done"


def test_result_with_liveness_raises_on_dead_session():
    """The initial-save hang: future never resolves, session dies -> loud fail.

    Without this, create() blocked forever on a result that could never
    arrive WHILE the session kept billing (no exception, so the stop-on-
    failure cleanup never fired).
    """
    from training_sessions_cookbook.service.session_watchdog import result_with_liveness

    class _NeverFuture:
        def result(self, timeout=None):
            raise TimeoutError()

    transport = _Transport(status="Failed", reason="engine fault")
    with pytest.raises(RuntimeError) as exc:
        result_with_liveness(
            _NeverFuture(), _Svc(transport), "doomed", poll_interval=0.01
        )
    assert "engine fault" in str(exc.value)


def test_result_with_liveness_honours_total_timeout():
    from training_sessions_cookbook.service.session_watchdog import result_with_liveness

    class _NeverFuture:
        def result(self, timeout=None):
            raise TimeoutError()

    with pytest.raises(TimeoutError):
        result_with_liveness(
            _NeverFuture(), _Svc(_Transport()), "slow",
            timeout=0.05, poll_interval=0.01,
        )



def test_guard_raises_the_distinct_dead_session_type():
    from training_sessions_cookbook.service.session_watchdog import SessionDeadError

    transport = _Transport(status="Failed", reason="engine gone")
    backend = _backend(transport)

    async def never_resolves():
        await asyncio.sleep(3600)

    async def scenario():
        backend._ensure_watchdog()
        backend._watchdog._interval = 0.01
        with pytest.raises(SessionDeadError):
            await asyncio.wait_for(
                backend._await_guarded(never_resolves(), what="op"), timeout=5
            )

    asyncio.run(scenario())


def test_rollout_best_effort_does_not_swallow_dead_session():
    """A dead session must terminate the rollout, not requeue forever.

    The generic best-effort handler (warn + empty episode) is correct for a
    flaky sample; applied to SessionDeadError it produced a two-hour retry
    storm ending in an OOM kill (2026-08-25). The handler must re-raise it.
    """
    from training_sessions_cookbook.rl.rollouts import rollout_one
    from training_sessions_cookbook.service.session_watchdog import SessionDeadError

    class _DeadBackend:
        def wrap_prompt(self, toks):
            return toks

        async def sample_async(self, *a, **kw):
            raise SessionDeadError("sample request abandoned: session Failed")

    class _Env:
        def reset(self):
            return "q", {}

        def assign_problem(self, question, answer):
            pass

    class _Tok:
        def encode(self, t, **kw):
            return [1, 2]

        def decode(self, toks):
            return "x"

    with pytest.raises(SessionDeadError):
        asyncio.run(
            rollout_one(
                prompt={"question": "q", "answer": "1"},
                env=_Env(),
                template_fn=lambda o: o,
                sampling_client=None,
                sampling_params=None,
                max_turns=1,
                tokenizer=_Tok(),
                backend=_DeadBackend(),
            )
        )
