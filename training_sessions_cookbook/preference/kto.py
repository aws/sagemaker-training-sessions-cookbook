"""KTO (Kahneman-Tversky Optimization) math, independent of the service API.

KTO (Ethayarajh et al., 2024, https://arxiv.org/abs/2402.01306) aligns a model
from a **binary** signal - each ``(prompt, completion)`` is either *desirable*
or *undesirable* - with no need for the *paired* chosen-vs-rejected data DPO
requires. It is derived from prospect theory: treat the implicit reward as a
human "utility" measured against a reference point, with loss aversion.

This module holds the parts that are pure arithmetic on log-probabilities, so
they can be unit-tested without a training session. The training loop
(:mod:`training_sessions_cookbook.preference.train`) obtains the per-sequence
policy and reference log-probabilities from the service and calls into here.

Why we can train KTO with only ``forward`` + ``forward_backward`` and no custom
server-side loss kernel
------------------------------------------------------------------------------
KTO's loss is non-linear (a sigmoid of the policy-vs-reference log-ratio, with a
batch-level KL reference point). But the model enters it **only through the
per-sequence completion log-probability**

    logp_i = log pi_theta(y_i | x_i) = sum_t log pi_theta(y_{i,t} | ...) .

So the gradient is linear in the per-token log-probs by the chain rule:

    dL/dtheta = sum_i (dL/dlogp_i) * (dlogp_i/dtheta)
              = sum_i s_i * sum_{t in y_i} d(log pi_theta(y_{i,t}))/dtheta ,

where ``s_i = dL_i/dlogp_i`` is a **single scalar per sequence** the client
computes from a no-grad forward pass. Feeding ``advantage = -s_i`` (broadcast
over the completion tokens) through the service's REINFORCE/importance-sampling
loss - whose gradient at ratio 1 is ``-sum advantage_t * dlogp_t/dtheta`` - makes
the service compute exactly ``dL/dtheta``. This is the manual
vector-Jacobian-product ("surrogate loss") trick; see the recipe README.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class KtoTerm:
    """The KTO loss and its per-sequence gradient for one example.

    ``loss`` is the example's contribution to the KTO objective (for logging).
    ``grad_wrt_logp`` is ``s_i = dL_i / d log pi_theta(y_i)`` - the scalar that,
    broadcast over the completion tokens as ``advantage = -s_i``, drives the
    surrogate ``forward_backward``. ``reward`` is the implicit reward
    ``r_i = logp_policy - logp_ref`` (KTO's "utility" input), surfaced for
    metrics. ``value`` is ``v(x, y)`` in the paper's notation, in ``[0, 1]``.
    """

    loss: float
    grad_wrt_logp: float
    reward: float
    value: float


def kto_term(
    logp_policy: float,
    logp_ref: float,
    *,
    desirable: bool,
    z_ref: float,
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
) -> KtoTerm:
    """KTO loss and per-sequence gradient for a single ``(x, y)`` example.

    Implements, with ``r = logp_policy - logp_ref`` and sigmoid ``s(.)``:

        desirable:    v = s(beta * (r - z_ref)),   L = lambda_D * (1 - v)
        undesirable:  v = s(beta * (z_ref - r)),   L = lambda_U * (1 - v)

    and returns ``s_i = dL/dlogp_policy`` (note ``dr/dlogp_policy = 1``):

        desirable:    s_i = -lambda_D * beta * v * (1 - v)   (< 0: push logp UP)
        undesirable:  s_i = +lambda_U * beta * v * (1 - v)   (> 0: push logp DOWN)

    ``z_ref`` (the KL reference point) is treated as a detached constant, exactly
    as in the paper - no gradient flows through it.
    """
    reward = logp_policy - logp_ref
    if desirable:
        v = sigmoid(beta * (reward - z_ref))
        loss = desirable_weight * (1.0 - v)
        grad = -desirable_weight * beta * v * (1.0 - v)
    else:
        v = sigmoid(beta * (z_ref - reward))
        loss = undesirable_weight * (1.0 - v)
        grad = undesirable_weight * beta * v * (1.0 - v)
    return KtoTerm(loss=loss, grad_wrt_logp=grad, reward=reward, value=v)


def advantage_from_grad(grad_wrt_logp: float) -> float:
    """Per-token advantage for the surrogate loss: ``-s_i``.

    The service's REINFORCE / importance-sampling loss increases a token's
    log-prob for a *positive* advantage (its gradient at ratio 1 is
    ``-advantage * dlogp/dtheta``). We want the update to move ``theta`` down the
    KTO gradient ``s_i * dlogp/dtheta``, i.e. push log-prob **up** when
    ``s_i < 0`` (desirable) and **down** when ``s_i > 0`` (undesirable).
    Setting ``advantage = -s_i`` achieves both.
    """
    return -grad_wrt_logp


def kl_reference_point(
    policy_kl_logps: Sequence[float],
    ref_kl_logps: Sequence[float],
) -> float:
    """KTO's KL reference point ``z_ref = max(0, mean(logp_policy - logp_ref))``.

    Estimated over **mismatched** ``(x_i, y_j)`` pairs (see
    :mod:`training_sessions_cookbook.preference.data`), following the paper's
    ``z_0`` term, then clamped at 0. The value is a detached constant: the two
    log-prob sequences come from no-grad forward passes.

    Returns ``0.0`` for an empty batch (equivalent to disabling the KL term).
    """
    n = len(policy_kl_logps)
    if n == 0:
        return 0.0
    if len(ref_kl_logps) != n:
        raise ValueError(
            f"policy_kl_logps ({n}) and ref_kl_logps ({len(ref_kl_logps)}) "
            "must be the same length"
        )
    mean_kl = sum(p - r for p, r in zip(policy_kl_logps, ref_kl_logps)) / n
    return max(0.0, mean_kl)


def sum_completion_logprob(
    per_token_logprobs: Sequence[float],
    completion_mask: Sequence[float],
) -> float:
    """Sum per-token log-probs over the completion positions (mask > 0).

    ``per_token_logprobs`` are the target-aligned log-probs the service returns
    for one datum; ``completion_mask`` is the same-length 0/1 mask marking the
    completion tokens (0 over the prompt). Their masked sum is
    ``log pi(y | x)``.
    """
    if len(per_token_logprobs) != len(completion_mask):
        raise ValueError(
            f"per_token_logprobs ({len(per_token_logprobs)}) and completion_mask "
            f"({len(completion_mask)}) must be the same length"
        )
    return sum(lp for lp, m in zip(per_token_logprobs, completion_mask) if m > 0.0)
