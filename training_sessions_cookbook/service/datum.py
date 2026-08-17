"""Datum shaping for the SageMaker Training Sessions PySDK.

Both training paths hand the service the same datum dict: a single encoded-text chunk of input tokens plus a ``lossFnInputs`` map of per-position tensors. RL supplies ``logprobs`` and ``advantages`` alongside ``target_tokens`` and ``weights``; SFT supplies only the latter two. :func:`make_datum` builds either shape so the slicing and tensor encoding live in one place.
"""

from __future__ import annotations

from typing import Sequence


def tensordata(data: Sequence, dtype: str) -> dict:
    """Wrap a flat sequence as a PySDK TensorData dict (1-D shape)."""
    values = list(data)
    return {"data": values, "dtype": dtype, "shape": [len(values)]}


def make_datum(
    input_tokens: Sequence[int],
    target_tokens: Sequence[int],
    weights: Sequence[float],
    *,
    logprobs: Sequence[float] | None = None,
    advantages: Sequence[float] | None = None,
) -> dict:
    """Assemble one datum dict from pre-sliced parallel sequences.

    ``input_tokens``, ``target_tokens``, and ``weights`` must be the same length; ``logprobs`` / ``advantages`` are optional and, when given, must match too. The caller owns the slicing (typically ``input = tokens[:-1]``, ``target = tokens[1:]``) and the masking convention for ``weights``.
    """
    loss_inputs = {
        "target_tokens": tensordata([int(t) for t in target_tokens], "int64"),
        "weights": tensordata(weights, "float32"),
    }
    if logprobs is not None:
        loss_inputs["logprobs"] = tensordata(logprobs, "float32")
    if advantages is not None:
        loss_inputs["advantages"] = tensordata(advantages, "float32")

    return {
        "modelInput": {"chunks": [{"tokens": list(input_tokens), "type": "encoded_text"}]},
        "lossFnInputs": loss_inputs,
    }
