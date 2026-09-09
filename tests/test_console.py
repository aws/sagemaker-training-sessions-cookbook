"""Tests for service.console: the shared-console output layer.

Pins the properties the trainers rely on:

* logs and printed output share one ordered stdout stream
* arbitrary model output (bracket-heavy text) never crashes rendering
* no ANSI escape codes when stdout is not a terminal (redirected logs)
* setup is idempotent and respects a caller-configured console handler
* every helper works without ``rich`` installed
"""

from __future__ import annotations

import logging
import sys
import unittest
from unittest import mock

from training_sessions_cookbook.service import console as console_mod
from training_sessions_cookbook.service.console import (
    console_print,
    console_rule,
    print_run_banner,
    setup_console_logging,
)


class _RootLoggerSandbox(unittest.TestCase):
    """Save/restore root logger handlers so tests never leak configuration."""

    def setUp(self):
        self._root = logging.getLogger()
        self._saved_handlers = self._root.handlers[:]
        self._saved_level = self._root.level
        self._root.handlers = []

    def tearDown(self):
        self._root.handlers = self._saved_handlers
        self._root.setLevel(self._saved_level)


class SingleStreamOrderingTest(_RootLoggerSandbox):
    def test_logs_and_prints_share_stdout_in_order(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            setup_console_logging()
            logging.getLogger("t").info("first: a log line")
            console_print("second: a printed row")
            logging.getLogger("t").info("third: a teardown line")
        out = buf.getvalue()
        self.assertIn("first: a log line", out)
        self.assertIn("second: a printed row", out)
        self.assertIn("third: a teardown line", out)
        self.assertLess(out.index("first:"), out.index("second:"))
        self.assertLess(out.index("second:"), out.index("third:"))

    def test_no_ansi_when_stdout_is_not_a_terminal(self):
        import contextlib
        import io

        buf = io.StringIO()  # not a tty
        with contextlib.redirect_stdout(buf):
            console_print("plain row")
            console_rule("section")
            print_run_banner("run", {"model": "m", "steps": 2})
        self.assertNotIn("\x1b[", buf.getvalue())

    def test_bracket_heavy_model_output_renders_literally(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            console_print("Response: [unclosed [red] [/pseudo] [0; 1)")
        self.assertIn("[unclosed [red] [/pseudo] [0; 1)", buf.getvalue())


class SetupIdempotencyTest(_RootLoggerSandbox):
    def test_second_setup_adds_no_handler(self):
        setup_console_logging()
        n = len(logging.getLogger().handlers)
        setup_console_logging()
        self.assertEqual(len(logging.getLogger().handlers), n)

    def test_respects_caller_configured_console_handler(self):
        mine = logging.StreamHandler(sys.stderr)
        logging.getLogger().addHandler(mine)
        setup_console_logging()
        self.assertEqual(logging.getLogger().handlers, [mine])


class NoRichFallbackTest(_RootLoggerSandbox):
    """Everything must work with rich absent (plain-text degradation)."""

    def test_all_helpers_work_without_rich(self):
        import contextlib
        import io

        buf = io.StringIO()
        with mock.patch.object(console_mod, "_RICH_AVAILABLE", False), \
             mock.patch.object(console_mod, "_console", None), \
             contextlib.redirect_stdout(buf):
            setup_console_logging()
            logging.getLogger("t").info("fallback log line")
            console_print("fallback row")
            console_rule("fallback section")
            print_run_banner("run", {"model": "m"})
        out = buf.getvalue()
        for expected in ("fallback log line", "fallback row",
                         "fallback section", "model: m"):
            self.assertIn(expected, out)
        self.assertNotIn("\x1b[", out)


if __name__ == "__main__":
    unittest.main()


def test_echo_file_mirrors_console_layer_without_ansi(tmp_path, monkeypatch):
    """With an echo file set, banner/rule/rows/tables land in the file as
    plain text (zero ANSI), and unsetting stops the mirror and releases it.

    Runs under Jupyter's color-forcing environment: kernels export
    FORCE_COLOR/CLICOLOR_FORCE, which rich honors OVER the file's non-tty
    status - a notebook-launched training once wrote raw ANSI into
    training.log because of exactly this.
    """
    from training_sessions_cookbook.service import console as c

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("CLICOLOR_FORCE", "1")
    echo = tmp_path / "training.log"
    c.add_console_echo_file(str(echo))
    try:
        c.print_run_banner("Echo run", {"model": "m", "steps": 2})
        c.console_rule("Validation after 0 steps")
        c.console_print("07:00:00 INFO step row " + "x" * 200)
        c.print_kv_table("Step 1 metrics", {"train/loss": "0.500000"})
    finally:
        c.remove_console_echo_file(str(echo))
    text = echo.read_text()
    for needle in ("Echo run", "Validation after 0 steps", "step row",
                   "Step 1 metrics", "train/loss", "0.500000"):
        assert needle in text, needle
    assert "\x1b[" not in text
    # long console_print lines stay single-line in the file
    row_lines = [l for l in text.splitlines() if "step row" in l]
    assert len(row_lines) == 1 and len(row_lines[0]) > 200
    # after removal, printing must not touch the file
    size = echo.stat().st_size
    c.console_print("not echoed")
    assert echo.stat().st_size == size


def test_echo_registry_supports_concurrent_runs(tmp_path):
    """Two live echo targets (concurrent runs in one process): both receive
    output, and removing one neither closes nor silences the other."""
    from training_sessions_cookbook.service import console as c

    a, b = tmp_path / "a.log", tmp_path / "b.log"
    c.add_console_echo_file(str(a))
    c.add_console_echo_file(str(b))
    try:
        c.console_print("both runs see this")
        c.remove_console_echo_file(str(a))
        c.console_print("only b sees this")
    finally:
        c.remove_console_echo_file(str(a))
        c.remove_console_echo_file(str(b))
    assert "both runs see this" in a.read_text()
    assert "only b sees this" not in a.read_text()
    text_b = b.read_text()
    assert "both runs see this" in text_b and "only b sees this" in text_b
