"""Shared dataset file readers for the RL and SFT paths.

Both paths accept local ``.jsonl`` datasets. The parsing (one JSON object per non-blank line) is identical, so it lives here once; each path then interprets the parsed records in its own shape (RL: flat prompt dicts; SFT: conversations extracted from ``messages`` / ``conversations``).
"""

from __future__ import annotations

import json


def read_jsonl(path: str) -> list[dict]:
    """Read ``path`` as JSONL, returning one parsed object per non-blank line."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
