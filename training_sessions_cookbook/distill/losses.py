"""On-policy distillation math, independent of the service API.

On-policy distillation trains the student on its own samples, graded token by
token by a teacher. The student samples ``y ~ q(. | c_S)``; the teacher reads
the same tokens in one prefill and returns ``log p(y_t | c_T, y_<t)``. The three
recipes differ only in the teacher and its context ``c_T``:

* **OPD**: a stronger model, same prompt as the student.
* **OPSD**: the student's frozen base, prompt + a privileged reference solution.
* **SDPO**: the student itself (frozen or lagged), prompt + feedback on this attempt.

Why one sampled token per position is enough
--------------------------------------------
Reverse KL is an expectation over the student's own samples::

    KL(q || p) = E_{y ~ q} [ log q(y) - log p(y) ]

so ``log q(y_t) - log p(y_t)`` at the sampled token is an unbiased (if noisy)
per-position estimate, and the noise averages out over thousands of positions.
Using its negative as a per-token advantage::

    A_t = log p(y_t | c_T, y_<t) - log q(y_t | c_S, y_<t)

turns distillation into ordinary policy-gradient RL: the service's
importance-sampling loss has gradient ``-sum_t A_t * dlog q(y_t)/dtheta`` (with
the IS ratio correcting for sampler/trainer mismatch), which pushes up tokens
the teacher likes more than the student does and down those it likes less.
Each token's advantage is its own (discount 0), the convention of Thinking
Machines' OPD, MiMo's MOPD and SDPO's token form.

Tempered teacher
----------------
``--teacher_temperature T`` grades against ``p_T(v) ∝ p(v)^(1/T)``. Its
log-prob is ``(1/T) * log p(y_t) - log Z_t``, and ``compute_logprobs`` returns
only ``log p(y_t)``, not the normalizer. ``log Z_t`` is the same for every
candidate token at position ``t``, so it is a per-position baseline:
``E_{y~q}[log Z_t * dlog q(y_t)/dtheta] = 0``. Dropping it leaves the expected
gradient unchanged, so ``A_t = (1/T) log p(y_t) - log q(y_t)`` trains toward
the tempered teacher exactly in expectation. ``T < 1`` sharpens the teacher;
with the student's own model as the teacher, that is the only source of
signal (at ``T = 1`` the advantage is zero). ``T -> 0`` is degenerate: every
non-argmax token gets ``-inf``.

Both log-probs come from the inference engine (the student's from the sampler
that generated the tokens, the teacher's from ``compute_logprobs``), so the
advantage compares like with like. The gradient itself is computed by the
training engine on the student's tokens.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


def reverse_kl_advantages(
    student_logprobs: Sequence[float],
    teacher_logprobs: Sequence[Optional[float]],
    *,
    clip: Optional[float] = None,
    teacher_temperature: float = 1.0,
) -> list[float]:
    """Per-token advantages ``A_t = log p_teacher / T - log q_student``.

    ``teacher_temperature`` (``T``) tempers the teacher; the normalizer it
    would add is a per-position baseline and is dropped (see module docs).

    ``teacher_logprobs`` may contain ``None`` (or non-finite values) where the
    teacher returned no score; those positions get advantage ``0.0`` (no
    gradient), rather than a spurious huge signal. ``clip`` (when set) bounds
    each advantage to ``[-clip, clip]``: a token the teacher finds nearly
    impossible otherwise produces an arbitrarily large negative advantage.
    """
    if len(student_logprobs) != len(teacher_logprobs):
        raise ValueError(
            f"student_logprobs ({len(student_logprobs)}) and teacher_logprobs "
            f"({len(teacher_logprobs)}) must be the same length"
        )
    if teacher_temperature <= 0:
        raise ValueError(f"teacher_temperature must be > 0 (got {teacher_temperature})")
    inv_t = 1.0 / teacher_temperature
    out: list[float] = []
    for s, t in zip(student_logprobs, teacher_logprobs):
        if t is None or not math.isfinite(t) or not math.isfinite(s):
            out.append(0.0)
            continue
        a = inv_t * float(t) - float(s)
        if clip is not None:
            a = max(-clip, min(clip, a))
        out.append(a)
    return out


def mix_task_advantage(
    distill_advantages: Sequence[float],
    task_advantage: float,
    alpha: float,
) -> list[float]:
    """``A_t + alpha * A_task``: add a (scalar, per-episode) task-reward advantage.

    ``A_task`` is the GRPO group-centered reward advantage of the episode. With
    ``alpha = 0`` this is pure distillation; ``alpha > 0`` is the MiMo / GKD+RL
    mix, which lets the student move past the teacher on the task reward.
    """
    if alpha == 0.0:
        return [float(a) for a in distill_advantages]
    bonus = alpha * float(task_advantage)
    return [float(a) + bonus for a in distill_advantages]


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[idx])


class DistillStats:
    """Accumulates per-token distillation diagnostics across one training step.

    ``reverse_kl`` is the sampled-token estimate ``log q - log p`` averaged over
    scored tokens; it should fall as the student matches the teacher. With a
    tempered teacher (``T != 1``) ``log p`` is the unnormalized ``log p / T``,
    so the value is offset by the dropped normalizer: watch its trend, not its
    level. Tokens
    the teacher did not score, and sequences with no teacher context (SDPO with
    nothing to say about an attempt), are counted separately.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._kl: list[float] = []
        self._teacher_sum = 0.0
        self._student_sum = 0.0
        self._clipped = 0
        self._unscored_tokens = 0
        self._sequences = 0
        self._unscored_sequences = 0

    def add_sequence(
        self,
        student_logprobs: Sequence[float],
        teacher_logprobs: Sequence[Optional[float]],
        *,
        clip: Optional[float] = None,
        teacher_temperature: float = 1.0,
    ) -> None:
        self._sequences += 1
        inv_t = 1.0 / teacher_temperature
        for s, t in zip(student_logprobs, teacher_logprobs):
            if t is None or not math.isfinite(t) or not math.isfinite(s):
                self._unscored_tokens += 1
                continue
            t = inv_t * float(t)
            self._kl.append(float(s) - t)
            self._teacher_sum += t
            self._student_sum += float(s)
            if clip is not None and abs(t - float(s)) > clip:
                self._clipped += 1

    def add_unscored_sequence(self, num_tokens: int) -> None:
        self._sequences += 1
        self._unscored_sequences += 1
        self._unscored_tokens += int(num_tokens)

    def summary(self) -> dict:
        n = len(self._kl)
        kl_sorted = sorted(self._kl)
        mean = lambda total: total / n if n else float("nan")
        return {
            "distill/reverse_kl": mean(sum(self._kl)),
            "distill/reverse_kl_p50": _percentile(kl_sorted, 0.50),
            "distill/reverse_kl_p95": _percentile(kl_sorted, 0.95),
            "distill/teacher_logprob": mean(self._teacher_sum),
            "distill/student_logprob": mean(self._student_sum),
            "distill/clip_frac": self._clipped / n if n else 0.0,
            "distill/num_scored_tokens": n,
            "distill/num_unscored_tokens": self._unscored_tokens,
            "distill/frac_sequences_scored": (
                (self._sequences - self._unscored_sequences) / self._sequences
                if self._sequences else float("nan")
            ),
        }
