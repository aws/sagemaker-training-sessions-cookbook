"""DeepCoder dataset loader."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Any, Literal

from datasets import concatenate_datasets, load_dataset

from .lcb_utils import fetch_live_code_bench_system_prompt, taco_to_lcb_format

logger = logging.getLogger(__name__)

# Bump when the DeepcoderTask schema or the processing logic changes so stale
# caches are not silently reused.
_CACHE_VERSION = 1


def _cache_enabled() -> bool:
    """Task-list disk cache is on by default; disable with DEEPCODER_NO_CACHE=1."""
    return os.getenv("DEEPCODER_NO_CACHE", "").strip().lower() not in ("1", "true", "yes")


def _cache_dir() -> str:
    """Directory for the built-task cache (override via DEEPCODER_CACHE_DIR)."""
    return os.getenv(
        "DEEPCODER_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "deepcoder_tasks"),
    )


def _cache_path(split: str, seed: int, max_prompts: int | None, task_offset: int = 0) -> str:
    """Cache file path keyed by everything that changes the built task list.

    The key folds in the split, seed, cap, the ``DEEPCODER_SPLITS`` override
    (which selects which HF configs are loaded), and a schema version. Two runs
    that would build the same list share a cache entry; anything else misses.
    """
    payload = {
        "split": split,
        "seed": seed,
        "max_prompts": max_prompts,
        "task_offset": task_offset,
        "splits_env": os.getenv("DEEPCODER_SPLITS", "").strip(),
        "version": _CACHE_VERSION,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(_cache_dir(), f"deepcoder-{split}-{digest}.pkl")


@dataclass(frozen=True)
class DeepcoderTask:
    """A single code task with problem statement and test cases."""

    problem: str
    tests: list[dict[str, Any]]
    starter_code: str | None = None


def _ensure_dict(metadata: Any) -> dict[str, Any]:
    """Coerce ``metadata`` into a dict, JSON-decoding strings; {} on failure."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning("Failed to deserialize metadata: %s", metadata)
            return {}
    if isinstance(metadata, dict):
        return metadata
    return {}


def _normalize_tests(raw_tests: Any, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize test cases to a unified LCB format."""
    tests = raw_tests
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            logger.warning("Failed to deserialize tests. Dropping sample.")
            return []
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        tests = taco_to_lcb_format(tests)
    if isinstance(tests, dict):
        tests = [tests]

    normalized: list[dict[str, Any]] = []
    for test in tests or []:
        if not isinstance(test, dict):
            continue
        testtype = test.get("testtype") or "stdin_stdout"
        test_metadata = _ensure_dict(test.get("metadata", {}))
        if testtype == "functional":
            func_name = test_metadata.get("func_name") or metadata.get("func_name")
            if func_name is not None:
                test_metadata["func_name"] = str(func_name)
        normalized.append(
            {
                "input": str(test.get("input", "")),
                "output": str(test.get("output", "")),
                "testtype": testtype,
                "metadata": test_metadata or {"func_name": None},
            }
        )
    return normalized


def _build_question(example: dict[str, Any]) -> str | None:
    """Build the question text with the LCB system prompt."""
    question = example.get("question") or example.get("prompt") or example.get("problem")
    if not isinstance(question, str) or not question.strip():
        return None
    starter_code = example.get("starter_code")
    if isinstance(starter_code, str) and starter_code.strip():
        return fetch_live_code_bench_system_prompt(question, starter_code)
    return fetch_live_code_bench_system_prompt(question)


def _load_deepcoder_split(split: Literal["train", "test"]) -> Any:
    """Load + concatenate the DeepCoder dataset splits.

    The default set (primeintellect + taco + lcbv5 for train, codeforces + lcbv5
    for test) can exceed 32 GB of RAM to materialize because TACO test cases
    are large. On memory-constrained hosts, set the env var
    ``DEEPCODER_SPLITS`` to a comma-separated subset (e.g. ``lcbv5``) to
    override for smoke tests. Applies to both train and test splits.
    """
    logger.info("Loading DeepCoder dataset split: %s", split)
    if split == "train":
        names: tuple[str, ...] = ("primeintellect", "taco", "lcbv5")
    else:
        names = ("codeforces", "lcbv5")

    override = os.getenv("DEEPCODER_SPLITS", "").strip()
    if override:
        names = tuple(s.strip() for s in override.split(",") if s.strip())
        logger.info("DEEPCODER_SPLITS override active: %s", names)

    datasets = []
    for name in names:
        logger.info("  Loading %s...", name)
        ds = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split=split)
        datasets.append(ds)

    return concatenate_datasets(datasets)


def load_deepcoder_tasks(
    split: Literal["train", "test"] = "train",
    seed: int = 0,
    max_prompts: int | None = None,
    task_offset: int = 0,
) -> list[DeepcoderTask]:
    """Load tasks from the DeepCoder dataset.

    Args:
        split: Which split to load ("train" or "test").
        seed: Random seed for shuffling (train split only).
        max_prompts: Optional cap on the number of tasks returned.
        task_offset: Skip this many tasks of the seeded shuffle before taking
            ``max_prompts``. Windows ``[offset, offset+max_prompts)`` over the
            same seed are strictly disjoint; used by load tests so concurrent
            sessions on one engine never share prompts (KV-cache isolation).

    Returns:
        List of DeepcoderTask instances with normalized test cases.

    Performance notes:
        * The split load and the per-row processing loop are timed separately
          and logged, so the "how long did data prep take?" question is
          answerable from ``training.log`` alone.
        * We deliberately avoid ``datasets.shuffle()`` here. On a memory-mapped
          Arrow dataset ``shuffle`` returns an index-permuted view, so the
          subsequent ``for item in ds`` reads rows in random file order. Over
          the large TACO rows that random-access pattern is pathologically slow.
          Instead we read rows in ascending (sequential) file order and shuffle
          the *built task list* afterward, which yields the same randomized
          training order at a fraction of the I/O cost.
        * When ``max_prompts`` is set we pick a seeded random subset of row
          indices up front and read only those rows, instead of processing the
          entire corpus and truncating at the end. This is what makes a capped
          smoke run prep in seconds rather than minutes.
        * The final built list is cached to disk (keyed by split/seed/cap/
          split-override). A cache hit skips both the HF load and the
          processing loop, so repeat runs prep in well under a second. Disable
          with ``DEEPCODER_NO_CACHE=1``; relocate with ``DEEPCODER_CACHE_DIR``.
    """
    cache_path = _cache_path(split, seed, max_prompts, task_offset)
    if _cache_enabled() and os.path.exists(cache_path):
        t_cache = time.perf_counter()
        try:
            with open(cache_path, "rb") as fh:
                tasks = pickle.load(fh)
            logger.info(
                "Loaded %s tasks from cache in %.2fs: %s",
                len(tasks),
                time.perf_counter() - t_cache,
                cache_path,
            )
            return tasks
        except Exception as exc:  # corrupt/incompatible cache -> rebuild
            logger.warning(
                "Ignoring unreadable task cache %s (%r); rebuilding.",
                cache_path,
                exc,
            )

    t_load = time.perf_counter()
    ds = _load_deepcoder_split(split)
    load_secs = time.perf_counter() - t_load

    n = len(ds)
    # Decide which rows to read, and in what order, without random access over
    # the full corpus (see "Performance notes" above).
    if max_prompts is not None and (task_offset + max_prompts) < n:
        indices = list(range(n))
        random.Random(seed).shuffle(indices)
        # Read the chosen subset in ascending file order for I/O locality; the
        # training order is randomized below by shuffling the built list.
        # task_offset windows into the seeded shuffle: same seed, different
        # offsets -> strictly disjoint task subsets.
        indices = sorted(indices[task_offset : task_offset + max_prompts])
        source = ds.select(indices)
    else:
        source = ds

    logger.info(
        "Processing %s examples into tasks (split=%s, loaded in %.1fs)...",
        len(source),
        split,
        load_secs,
    )
    t_proc = time.perf_counter()
    tasks: list[DeepcoderTask] = []
    for item in source:
        row: dict[str, Any] = dict(item)

        # Extract and normalize metadata
        metadata = _ensure_dict(row.get("metadata", {}))

        # Normalize test cases
        raw_tests = row.get("tests") or row.get("ground_truth")
        tests = _normalize_tests(raw_tests, metadata)
        if not tests:
            continue

        # Build problem prompt
        problem = _build_question(row)
        if problem is None:
            continue

        # Extract starter code if present
        starter_code = row.get("starter_code")
        if isinstance(starter_code, str) and not starter_code.strip():
            starter_code = None

        tasks.append(
            DeepcoderTask(
                problem=problem,
                tests=tests,
                starter_code=starter_code if isinstance(starter_code, str) else None,
            )
        )

    proc_secs = time.perf_counter() - t_proc

    # Randomize training order deterministically by shuffling the built list
    # (not the Arrow dataset), so the read above stays sequential.
    if split == "train":
        random.Random(seed).shuffle(tasks)

    rows_per_sec = (len(source) / proc_secs) if proc_secs > 0 else 0.0
    logger.info(
        "Built %s tasks in %.1fs (%.1f rows/s); split load %.1fs, total %.1fs.",
        len(tasks),
        proc_secs,
        rows_per_sec,
        load_secs,
        load_secs + proc_secs,
    )

    if _cache_enabled():
        try:
            os.makedirs(_cache_dir(), exist_ok=True)
            tmp_path = f"{cache_path}.tmp-{os.getpid()}"
            with open(tmp_path, "wb") as fh:
                pickle.dump(tasks, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)  # atomic publish
            logger.info("Wrote task cache: %s", cache_path)
        except Exception as exc:  # caching is best-effort
            logger.warning("Failed to write task cache %s (%r).", cache_path, exc)

    return tasks
