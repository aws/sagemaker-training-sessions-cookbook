"""Training backend for the RL cookbook.

The training loop is backend-agnostic: it drives rollouts, advantage computation, metrics, and validation without knowing *how* the model is trained or sampled. Everything backend-specific (client construction, datum shape, sampler refresh, forward/backward + optimizer step, and deploy/invoke) lives behind the :class:`TrainingBackend` protocol (``base.py``).

The concrete backend:

* ``service``: SageMaker Training Sessions via the ``sagemaker-train`` PySDK.

The backend module is imported lazily inside :func:`make_backend` so building the
CLI (or running SDK-free tests) never requires the sagemaker-train SDK to be installed.

``make_backend(name, args)`` is the single entry point ``main()`` uses.
"""

from __future__ import annotations

from .base import TrainingBackend


def make_backend(name: str, args) -> "TrainingBackend":
    """Construct the backend selected by ``--backend``.

    Args:
        name: ``"service"``.
        args: The parsed argparse namespace (the backend reads the knobs it needs off it).

    Returns:
        A ready-to-use :class:`TrainingBackend` with clients already created.

    Raises:
        ValueError: if ``name`` is not a known backend.
    """
    if name == "service":
        from .service import ServiceBackend
        return ServiceBackend.create(args)
    raise ValueError(f"unknown --backend {name!r}; expected 'service'")


__all__ = ["TrainingBackend", "make_backend"]
