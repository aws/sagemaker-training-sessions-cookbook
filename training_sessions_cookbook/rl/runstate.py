"""Typed accessors for run-scoped state that rides on the args namespace.

Two pieces of mutable state are attached to ``args`` after parsing, because
the training loops, the env registry, and the metrics sinks all already
receive ``args`` and re-threading a second parameter through every signature
is the job of the planned module split, not of this contract:

* **Run artifacts** (``run_folder`` / ``metrics_path`` / ``mlflow``): created
  once in ``run()`` after the run folder is set up, read by the training loops
  and metrics sinks.
* **Shared dataset**: the prompt dataset handed from ``run()`` to the env
  registry; the code_rl factory REPLACES it with DeepCoder tasks so the
  central sampler iterates code tasks (see ``build_code_rl_env_factory``).

These helpers are the only sanctioned way to touch that state. They exist so
the contract is written down in exactly one place, reads are typo-proof
(an attribute name lives here, not at 18 call sites), and absent state has a
defined meaning (``None`` fields / ``None`` dataset) instead of whatever a
hand-typed ``getattr`` default said.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RunArtifacts:
    """Where a run writes its outputs, and the optional MLflow publisher.

    ``run_folder`` and ``metrics_path`` are ``None`` only when the training
    loop is driven without ``run()`` (unit tests, programmatic callers that
    skip folder setup); consumers fall back to deriving paths from
    ``args.log_dir`` in that case.
    """

    run_folder: Optional[str] = None
    metrics_path: Optional[str] = None
    mlflow: Optional[Any] = None


def attach_run_artifacts(
    args: Any,
    *,
    run_folder: Optional[str],
    metrics_path: Optional[str],
    mlflow: Optional[Any],
) -> None:
    """Attach the run's output locations and publisher to ``args`` (once, in ``run()``)."""
    args._run_artifacts = RunArtifacts(
        run_folder=run_folder, metrics_path=metrics_path, mlflow=mlflow
    )


def run_artifacts(args: Any) -> RunArtifacts:
    """The run's artifacts, or an all-``None`` instance when never attached."""
    return getattr(args, "_run_artifacts", None) or RunArtifacts()


def set_shared_dataset(args: Any, dataset: Any) -> None:
    """Publish the dataset the central sampler iterates (see module docstring)."""
    args._dataset = dataset


def shared_dataset(args: Any) -> Any:
    """The dataset published via :func:`set_shared_dataset`, or ``None``."""
    return getattr(args, "_dataset", None)
