"""Prompt-dataset loading and epoch iteration for the RL loop."""

from __future__ import annotations

import pyarrow.parquet as pq
from typing import Optional

from training_sessions_cookbook.service import read_jsonl

from .envs.gsm8k import BUILTIN_GSM8K_20
from .config import cfg


# === Training-datum builder ===
#
# Builds a training datum from a single rollout `Transition`. Slicing:
#
#   input_tokens  = obs_tokens + act_tokens[:-1]
#   target_tokens = obs_tokens[1:] + act_tokens
#
# Per-position fields (`weights`, `logprobs`, `advantages`) are zero over the `obs_n - 1` observation positions and active over the trailing `act_n` action positions. `weights = 0` over observation positions guarantees the loss only counts action tokens. `temperature` is a scalar `[1]` tensor.


# Datum construction lives on the backend: the service backend builds its PySDK datum shape in `backends/service.py`. The training loop calls `backend.build_datum(transition, temperature=...)` so the datum the SDK expects is produced from the rollout `Transition`.


# === Dataset loader ===
#
# Loads the prompt dataset from one of three sources (builtin / .jsonl / .parquet) and yields sequential round-robin batches across epochs.


def load_dataset(path: Optional[str], max_prompts: Optional[int]) -> list[dict]:
    """Load prompts from `path`, or fall back to the builtin GSM8K subset.

    Branches:
      - `path is None`   -> return a copy of `BUILTIN_GSM8K_20`.
      - `path` ends in `.jsonl` -> JSONL parsing, one object per line.
      - `path` ends in `.parquet` -> `pq.read_table(path).to_pylist()`.
      - Anything else raises `ValueError`.

    `max_prompts` truncates the loaded list when not `None`.
    """
    if path is None:
        data = list(BUILTIN_GSM8K_20)
    elif path.endswith(".jsonl"):
        data = read_jsonl(path)
    elif path.endswith(".parquet"):
        data = pq.read_table(path).to_pylist()
    else:
        raise ValueError(f"Unsupported dataset extension: {path}")
    if max_prompts is not None:
        data = data[:max_prompts]
    return data


def _build_val_dataset(args):
    """Build the validation set from CLI flags, or return None to skip val.

    Precedence (code_rl):
      1. ``--val_code_split`` in {test, train}: the REAL DeepCoder split,
         returned as a ``list[DeepcoderTask]`` and graded via SandboxFusion
         (see ``ValidationRunner._grade_deepcoder``) exactly like training.
         Capped by ``--val_max_prompts``.
      2. ``--val_dataset_path``: a JSONL/Parquet file (the legacy toy code set
         or a GSM8K set), graded by the local subprocess / GSM8K scorer.

    For non-code envs only (2) applies. Returns ``None`` when neither is set.
    """
    if (
        cfg(args, "env_type") == "code_rl"
        and cfg(args, "val_code_split") != "none"
    ):
        # Lazy import so the GSM8K path never pulls in datasets.
        from .envs.deepcoder import load_deepcoder_tasks

        return load_deepcoder_tasks(
            split=args.val_code_split,
            seed=cfg(args, "seed"),
            max_prompts=cfg(args, "val_max_prompts"),
        )
    if cfg(args, "val_dataset_path"):
        return load_dataset(
            args.val_dataset_path, cfg(args, "val_max_prompts")
        )
    return None


class _EpochIter:
    """Sequential round-robin cursor over `dataset` for `epochs` passes.

    Persistent state used by `next_batch`:
      - `_epoch` counts completed passes (0..epochs).
      - `_idx`   is the current position inside the active pass.
      - `drop_last` is honored by `next_batch` (not by the cursor itself).
    """

    __slots__ = ("dataset", "epochs", "drop_last", "_epoch", "_idx")

    def __init__(self, dataset: list[dict], epochs: int, drop_last: bool) -> None:
        self.dataset = dataset
        self.epochs = epochs
        self.drop_last = drop_last
        self._epoch = 0
        self._idx = 0

    def _next_one(self) -> Optional[dict]:
        """Advance one position. Returns `None` when all epochs are exhausted."""
        if not self.dataset or self._epoch >= self.epochs:
            return None
        item = self.dataset[self._idx]
        self._idx += 1
        if self._idx >= len(self.dataset):
            self._idx = 0
            self._epoch += 1
        return item


def make_epoch_iter(dataset: list[dict], epochs: int, drop_last: bool) -> _EpochIter:
    """Construct the persistent iteration state passed into `next_batch`.

    The returned object preserves cursor position across `next_batch` calls so batches are sequential and round-robin across epochs.
    """
    return _EpochIter(dataset, epochs, drop_last)


def next_batch(dataset: list[dict], n: int, epoch_iter: _EpochIter) -> list[dict]:
    """Pull the next `n` prompts from the persistent `epoch_iter`.

    Honors `--drop_last` and `--epochs` via the state in `epoch_iter`:
      - When the iterator is fully exhausted, returns `[]`.
      - When the iterator yields fewer than `n` trailing items and `epoch_iter.drop_last` is True, returns `[]` to keep batch size constant.
      - When `drop_last` is False, returns the partial batch as-is.

    `dataset` is accepted for signature symmetry with the design snippet but is unused; the cursor state lives entirely in `epoch_iter`.
    """
    del dataset  # state lives in epoch_iter
    batch: list[dict] = []
    for _ in range(n):
        item = epoch_iter._next_one()
        if item is None:
            break
        batch.append(item)
    if len(batch) < n and epoch_iter.drop_last:
        return []
    return batch

