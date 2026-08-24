"""Structural pins for teardown resilience.

On Ctrl-C, ``asyncio.run`` cancels the main task and every ``await`` inside a
``finally`` block is a cancellation point. If session teardown awaits anything,
the stop request can be killed mid-flight and a billed training session is
left running with nothing left to stop it. These tests pin the property that
the teardown path is synchronous (uncancellable by task cancellation).
"""

import ast
import inspect
import textwrap

from training_sessions_cookbook.rl import train as train_mod


def _finally_bodies(func_source: str):
    tree = ast.parse(textwrap.dedent(func_source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.finalbody:
            yield node.finalbody


def test_run_async_teardown_finally_never_awaits_backend_calls():
    """The finally that stops the session must not await.

    A synchronous ``backend.close()`` runs to completion under task
    cancellation; ``await asyncio.to_thread(backend.close)`` does not. This
    inspects the real source of ``run_async`` so the property cannot silently
    regress in a refactor.
    """
    src = inspect.getsource(train_mod.run_async)
    saw_close = False
    for body in _finally_bodies(src):
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Await):
                    call_src = ast.unparse(node)
                    assert "backend" not in call_src, (
                        "teardown finally awaits a backend call - cancellable "
                        f"on Ctrl-C, can orphan a billed session: {call_src}"
                    )
                if isinstance(node, ast.Call) and ast.unparse(node).startswith(
                    "backend.close"
                ):
                    saw_close = True
    assert saw_close, "run_async's finally no longer calls backend.close()"


def test_backend_close_is_synchronous():
    """close() (and delete_session) must stay sync so the finally can shield them."""
    from training_sessions_cookbook.rl.backends.service import ServiceBackend

    assert not inspect.iscoroutinefunction(ServiceBackend.close)
    assert not inspect.iscoroutinefunction(ServiceBackend.delete_session)
