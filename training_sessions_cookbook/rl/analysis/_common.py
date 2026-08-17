"""Shared plumbing for the RL comparison plots.

Every plot reads the same uniform layout: one ``metrics.jsonl`` per run under
``results/<experiment>/<run>/`` (the same shape the SFT cookbooks use), so the
three plot scripts differ only in *which* runs they compare and *which* panels
they draw, never in how data is loaded. All plots run fully offline off the
committed ``results/`` tree; there is no remote/EC2 fetch anywhere in this
package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# <repo>/rl/analysis/_common.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_rows(experiment: str, run: str) -> list[dict[str, Any]]:
    """Load ``results/<experiment>/<run>/metrics.jsonl`` as a list of dicts.

    Raises ``FileNotFoundError`` with the resolved path if the metrics file is
    absent, so a missing run is a loud failure rather than a silently empty
    series.
    """
    path = RESULTS_DIR / experiment / run / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing metrics file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def series(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """Return step-sorted ``(steps, values)`` for rows carrying ``step`` + ``key``.

    Rows missing either field (or with a ``None`` value) are skipped, so a run
    that never logged a metric simply contributes an empty series rather than
    erroring; the caller decides whether to draw it.
    """
    pairs = [
        (r["step"], r[key])
        for r in rows
        if "step" in r and r.get(key) is not None
    ]
    pairs.sort(key=lambda t: t[0])
    if not pairs:
        return [], []
    s, v = zip(*pairs)
    return list(s), list(v)


def plots_dir(experiment: str) -> Path:
    """The experiment's own folder; plots are written alongside their data."""
    d = RESULTS_DIR / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d
