"""Shared SageMaker Training Sessions plumbing for the RL and SFT paths.

Both training paths (``rl/`` and ``sft/``) run on the same service. The pieces they genuinely share live here so neither re-implements them:

* :mod:`service.session`: opening a session and resolving its tokenizer.
* :mod:`service.datum`: the PySDK datum dict shape.
* :mod:`service.schedule`: learning-rate warmup and decay.
* :mod:`service.runlog`: run folder, config snapshot, logging, metrics log.
* :mod:`service.mlflow_log`: optional MLflow (SageMaker MLflow App) publishing.
* :mod:`service.data_io`: shared dataset file readers.
"""

from __future__ import annotations

from . import checkpoint_utils
from .data_io import read_jsonl
from .datum import make_datum, tensordata
from .mlflow_log import MlflowPublisher, add_mlflow_args
from .runlog import (
    RunLog,
    append_metrics,
    create_run_folder,
    dump_config,
    git_hash,
    setup_file_logging,
)
from .schedule import learning_rate
from .session import open_session

__all__ = [
    "checkpoint_utils",
    "read_jsonl",
    "open_session",
    "make_datum",
    "tensordata",
    "learning_rate",
    "create_run_folder",
    "dump_config",
    "setup_file_logging",
    "append_metrics",
    "git_hash",
    "RunLog",
    "MlflowPublisher",
    "add_mlflow_args",
]
