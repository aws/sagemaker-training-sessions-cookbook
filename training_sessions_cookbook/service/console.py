"""Console output for the training entry points.

One shared console carries everything the trainers show: log records, the
run banner, per-step metric rows, and validation examples. Routing all of it
through a single stream (stdout) keeps output ordered everywhere, including
when redirected to a file, piped, or run under ``nohup``. Previously log
records went to stderr while tables went to stdout, so redirected logs
interleaved wrongly (teardown lines appearing before the training table).

Log records are always plain single lines (colorized level names only when
stdout is a real terminal), so files stay grep-able and nothing ever wraps.
``rich`` renders only the deliberately visual elements: the run banner,
section rules, and table styling. Without ``rich``, or when stdout is not a
terminal, those degrade to plain text automatically, so files and CI logs
never contain ANSI escape codes.
``{run_folder}/training.log`` gets log records via its own FileHandler and,
once :func:`add_console_echo_file` points at it, a plain-text (never ANSI)
mirror of everything printed through this module. It stays plain: it gets its own
file handler from ``runlog.setup_file_logging``.

Design rules for extending this module:

* Content and rendering stay separate. Metric rows and validation examples
  are plain strings built elsewhere; this module only decides how they reach
  the terminal. New output kinds should follow the same split so tests can
  pin content without a terminal.
* Everything printed here may contain arbitrary model output, so rich markup
  is never interpreted (``markup=False``); a stray ``[`` in a completion must
  render literally, not crash the run.
* Every helper must work without ``rich`` installed.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Mapping, Optional

try:
    from rich.console import Console as _RichConsole

    _RICH_AVAILABLE = True
except ImportError:  # plain-text fallback everywhere
    _RICH_AVAILABLE = False

logger = logging.getLogger(__name__)

_console: Optional["_RichConsole"] = None

# Echo targets: file path -> (stream, rich Console or None). A registry (not
# a single slot) so concurrent runs in one process (a gather'd sweep, several
# notebook runs) each keep a complete mirror: adding a second target neither
# closes nor redirects the first, and removing one leaves the others intact.
_echo_targets: dict = {}


def add_console_echo_file(path: str) -> None:
    """Mirror everything printed through this module into ``path``.

    The run's ``training.log`` already receives log records via a
    ``logging.FileHandler``; this adds the console layer (banner, rules,
    metric rows and tables) so the file is a complete, plain-text record of
    what the terminal showed. ANSI is never written. Idempotent per path;
    remove with :func:`remove_console_echo_file`.
    """
    key = os.path.abspath(path)
    if key in _echo_targets:
        return
    # Explicit utf-8: model output (decoded completions) flows through here,
    # and the platform default encoding may not be able to represent it.
    stream = open(key, "a", buffering=1, encoding="utf-8", errors="replace")
    console = None
    if _RICH_AVAILABLE:
        # A file is not a terminal: rich emits no color/ANSI. Fixed width
        # keeps table geometry stable regardless of the launching terminal.
        console = _RichConsole(file=stream, width=100)
    _echo_targets[key] = (stream, console)


def remove_console_echo_file(path: str) -> None:
    """Stop mirroring into ``path`` and close it. Idempotent."""
    entry = _echo_targets.pop(os.path.abspath(path), None)
    if entry is not None:
        try:
            entry[0].close()
        except OSError:
            pass


def _echo_print(*values: Any, end: str = "\n") -> None:
    for stream, console in _echo_targets.values():
        if console is not None:
            console.print(*values, markup=False, highlight=False, end=end,
                          soft_wrap=True)
        else:
            print(*values, end=end, file=stream, flush=True)


def _get_console():
    """The shared ``rich`` console (lazily created), or ``None`` without rich.

    Created with ``file=None`` so it resolves ``sys.stdout`` at write time;
    stream redirection (tests, tools that capture output) keeps working.
    """
    global _console
    if not _RICH_AVAILABLE:
        return None
    if _console is None:
        _console = _RichConsole(file=None)
    return _console


_LEVEL_COLORS = {
    "DEBUG": "\x1b[2m",       # dim
    "INFO": "\x1b[32m",       # green
    "WARNING": "\x1b[33m",    # yellow
    "ERROR": "\x1b[31m",      # red
    "CRITICAL": "\x1b[1;31m", # bold red
}
_RESET = "\x1b[0m"


class _ConsoleFormatter(logging.Formatter):
    """Single-line log records; the level name is colorized only on a TTY.

    Log records must stay one line each and never wrap, so files remain
    grep-able and long messages survive intact. This is why log records do
    not go through a rich handler, which wraps and pads to a fixed width
    when output is redirected.
    """

    def __init__(self, colorize: bool):
        super().__init__("%(asctime)s %(levelname)s %(message)s",
                         datefmt="%H:%M:%S")
        self._colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        if self._colorize and record.levelname in _LEVEL_COLORS:
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = (
                f"{_LEVEL_COLORS[record.levelname]}{record.levelname}{_RESET}"
            )
        return super().format(record)


def setup_console_logging(level: int = logging.INFO) -> None:
    """Attach a plain single-line console log handler on stdout.

    Sharing stdout with :func:`console_print` keeps log records ordered with
    the banner, tables, and validation output. Idempotent, and a no-op when
    the caller already configured a console handler, so embedding the
    trainers in an application that owns logging keeps that setup.
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if type(handler).__name__ == "RichHandler":
            return  # caller chose rich log rendering; respect it
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            return
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_ConsoleFormatter(colorize=sys.stdout.isatty()))
    root.addHandler(handler)


def console_print(*values: Any, end: str = "\n") -> None:
    """Print through the shared console so output stays ordered with logs.

    Rich markup is never interpreted: the values may contain arbitrary model
    output. Without ``rich`` this is ``print`` with an explicit flush, which
    keeps ordering correct when stdout is block-buffered (redirected files).
    """
    console = _get_console()
    if console is not None:
        # soft_wrap: emit one logical line per value, however long; wrapping
        # belongs to the terminal, not to the log file.
        console.print(*values, markup=False, highlight=False, end=end,
                      soft_wrap=True)
    else:
        print(*values, end=end, flush=True)
    _echo_print(*values, end=end)


def console_rule(title: str) -> None:
    """Print a horizontal rule with a title, framing a section of output."""
    console = _get_console()
    if console is not None:
        from rich.text import Text

        console.rule(Text(title))
    else:
        print(f"---- {title} ----", flush=True)
    for stream, echo_console in _echo_targets.values():
        if echo_console is not None:
            from rich.text import Text

            echo_console.rule(Text(title))
        else:
            print(f"---- {title} ----", file=stream, flush=True)


def format_metric_values(record: Mapping[str, Any],
                         omit: tuple = ("step",)) -> dict:
    """Format a metrics record for a key/value table, sorted by name.

    Integers (counters) render bare; floats render with six decimals, except
    values too small for that precision (like a decayed learning rate of
    5e-13), which switch to scientific notation instead of collapsing to a
    misleading ``0.000000``. A float that IS zero still reads ``0.000000``,
    not as a count. Everything else goes through ``str``. Keys in ``omit``
    are dropped (``step`` by default: it is the table's title, not a
    metric).
    """
    out = {}
    for key in sorted(record):
        if key in omit:
            continue
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            out[key] = str(value)
        elif isinstance(value, int):
            out[key] = str(value)
        elif value != 0.0 and abs(value) < 1e-4:
            out[key] = f"{value:.2e}"
        else:
            out[key] = f"{value:.6f}"
    return out


def print_kv_table(title: str, rows: Mapping[str, str]) -> None:
    """Print a two-column key/value table with a title.

    Used for the per-step full-metrics view. Values arrive pre-formatted
    (content and formatting decisions belong to the caller; this function
    only renders). Falls back to aligned plain-text lines without ``rich``.
    """
    console = _get_console()
    if console is not None:
        from rich import box
        from rich.table import Table
        from rich.text import Text

        table = Table(title=Text(title), box=box.ROUNDED, title_justify="left")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right")
        for key, value in rows.items():
            table.add_row(Text(str(key)), Text(str(value)))
        console.print(table)
        for _, echo_console in _echo_targets.values():
            if echo_console is not None:
                echo_console.print(table)
    else:
        print(f"---- {title} ----", flush=True)
        width = max((len(str(k)) for k in rows), default=0)
        for key, value in rows.items():
            print(f"  {str(key):<{width}}  {value}", flush=True)
        for stream, echo_console in _echo_targets.values():
            if echo_console is None:
                print(f"---- {title} ----", file=stream)
                for key, value in rows.items():
                    print(f"  {str(key):<{width}}  {value}", file=stream, flush=True)


def print_run_banner(title: str, settings: Mapping[str, Any]) -> None:
    """Print a panel summarizing the run's key settings at start-up.

    ``settings`` maps label to value and should stay small: what a reader
    needs to identify the run (model, dataset, steps, learning rate, run
    folder), not the full config, which is in ``config.json``.
    """
    console = _get_console()
    if console is not None:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", justify="right", no_wrap=True)
        grid.add_column()
        for key, value in settings.items():
            grid.add_row(Text(str(key)), Text(str(value)))
        console.print(Panel(grid, title=Text(title), expand=False))
        for _, echo_console in _echo_targets.values():
            if echo_console is not None:
                echo_console.print(Panel(grid, title=Text(title), expand=False))
    else:
        print(f"=== {title} ===", flush=True)
        for key, value in settings.items():
            print(f"  {key}: {value}", flush=True)
        for stream, echo_console in _echo_targets.values():
            if echo_console is None:
                print(f"=== {title} ===", file=stream)
                for key, value in settings.items():
                    print(f"  {key}: {value}", file=stream, flush=True)
