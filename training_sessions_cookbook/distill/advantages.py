"""The teacher-scoring stage that replaces GRPO advantages in the RL loop.

:class:`DistillAdvantageFn` plugs into the RL loop's ``advantage_fn`` seam
(``rl/loops.py``). For each completed group of rollouts it:

1. builds the teacher's view of every transition (:mod:`.context`);
2. scores all of them concurrently under the teacher (:mod:`.teacher`);
3. writes the per-token advantage ``A_t = log p_teacher - log q_student``
   (optionally clipped, optionally plus ``alpha * A_task``) onto each
   transition (:mod:`.losses`).

Rollouts, datum building, ``forward_backward(importance_sampling)``,
checkpoints, validation and metrics are the RL loop's, unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Optional

from training_sessions_cookbook.rl.advantages import compute_grouped_advantages
from training_sessions_cookbook.service.session_watchdog import SessionDeadError

from .context import TeacherContext
from .losses import DistillStats, mix_task_advantage, reverse_kl_advantages
from .teacher import Teacher

logger = logging.getLogger(__name__)

#: Mean |sampler logprob - compute_logprobs| above which the step-0 check warns.
LOGPROB_CHECK_TOLERANCE = 0.05


class DistillAdvantageFn:
    """``advantage_fn`` for on-policy distillation (OPD / OPSD / SDPO).

    Args:
        teacher: Scores token sequences (see :func:`.teacher.make_teacher`).
        context: Builds the teacher's prompt for each episode.
        adv_clip: Bound on ``|A_t|`` (``None`` = unclipped).
        teacher_temperature: Temper the teacher by scaling its log-probs by
            ``1/T`` (see :mod:`.losses`); ``1`` = the teacher as served.
        alpha_task: Weight of the GRPO task-reward advantage mixed into every
            token's advantage (``0`` = pure distillation).
        logprob_check_samples: On the first call, re-score this many student
            transitions with the student's own sampler via ``compute_logprobs``
            and compare against the sampled log-probs (``0`` disables). The
            advantage subtracts a sampler log-prob from a ``compute_logprobs``
            one, so the two must be on the same scale; see the recipe README.
    """

    def __init__(self, *, teacher: Teacher, context: TeacherContext,
                 adv_clip: Optional[float] = None, alpha_task: float = 0.0,
                 teacher_temperature: float = 1.0,
                 logprob_check_samples: int = 8) -> None:
        self.teacher = teacher
        self.context = context
        self.adv_clip = adv_clip
        self.alpha_task = float(alpha_task)
        self.teacher_temperature = float(teacher_temperature)
        self.logprob_check_samples = int(logprob_check_samples)
        self._stats = DistillStats()
        self._check_metrics: dict = {}
        self._checked = False

    async def __call__(self, groups, *, step: int, sampling_client: Any = None):
        await self.teacher.update(step, sampling_client)
        # GRPO's grouped advantages give the flat transition order the loop
        # expects and, when alpha_task > 0, the per-episode task advantage.
        flat = compute_grouped_advantages(groups)
        task_advantage = {id(t): float(t.advantage) for t in flat}

        if not self._checked:
            self._checked = True
            await self._logprob_check(flat, sampling_client)

        jobs = []
        for group in groups:
            for episode in group:
                queries = self.context.queries(episode, group)
                if queries is None:
                    for t in episode:
                        self._stats.add_unscored_sequence(len(t.act_tokens))
                        t.advantage = mix_task_advantage(
                            [0.0] * len(t.act_tokens), task_advantage[id(t)], self.alpha_task)
                    continue
                for t, q in zip(episode, queries):
                    jobs.append((t, q))

        scores = await asyncio.gather(*(self._score(q) for _, q in jobs))
        for (t, _), teacher_lp in zip(jobs, scores):
            if teacher_lp is None:
                self._stats.add_unscored_sequence(len(t.act_tokens))
                t.advantage = mix_task_advantage(
                    [0.0] * len(t.act_tokens), task_advantage[id(t)], self.alpha_task)
                continue
            distill = reverse_kl_advantages(t.act_logprobs, teacher_lp, clip=self.adv_clip,
                                            teacher_temperature=self.teacher_temperature)
            t.advantage = mix_task_advantage(distill, task_advantage[id(t)], self.alpha_task)
            self._stats.add_sequence(t.act_logprobs, teacher_lp, clip=self.adv_clip,
                                     teacher_temperature=self.teacher_temperature)
        return flat

    async def _score(self, query):
        """Teacher log-probs for ``query``, or ``None`` if the request failed.

        Best-effort like rollouts: one failed request leaves its sequence
        unscored instead of failing the step. A dead session still raises.
        """
        try:
            return await self.teacher.score(query.tokens, query.n_tail)
        except SessionDeadError:
            raise
        except Exception as exc:
            logger.warning("teacher scoring failed (%d tokens): %r", len(query.tokens), exc)
            return None

    async def _logprob_check(self, transitions, sampling_client) -> None:
        """Compare sampler log-probs with ``compute_logprobs`` on the same weights."""
        n = min(self.logprob_check_samples, len(transitions))
        if n <= 0 or sampling_client is None:
            return
        probe = Teacher(sampling_client, name="logprob-check")
        sample = transitions[:n]
        try:
            rescored = await asyncio.gather(*(
                probe.score(list(t.obs_tokens) + list(t.act_tokens), len(t.act_tokens))
                for t in sample
            ))
        except SessionDeadError:
            raise
        except Exception as exc:  # a diagnostic; never fail the run over it
            logger.warning("logprob check skipped: %r", exc)
            return
        diffs = [abs(float(s) - float(r))
                 for t, lp in zip(sample, rescored)
                 for s, r in zip(t.act_logprobs, lp)
                 if r is not None and math.isfinite(r) and math.isfinite(s)]
        if not diffs:
            return
        mean_diff = sum(diffs) / len(diffs)
        self._check_metrics = {"distill/logprob_check_abs_diff": mean_diff}
        if mean_diff > LOGPROB_CHECK_TOLERANCE:
            logger.warning(
                "logprob check: sampler log-probs and compute_logprobs differ by "
                "%.4f nats/token on the same weights (tolerance %.2f). The "
                "distillation advantage subtracts one from the other, so this "
                "offset biases every token. Sample at --temperature 1.0 with "
                "--top_p/--top_k unset so sampled log-probs are unprocessed.",
                mean_diff, LOGPROB_CHECK_TOLERANCE)
        else:
            logger.info("logprob check: sampler vs compute_logprobs agree "
                        "(mean |diff| %.4f nats/token over %d tokens)", mean_diff, len(diffs))

    def pop_step_metrics(self) -> dict:
        metrics = {**self._stats.summary(), **self._check_metrics}
        self._stats.reset()
        self._check_metrics = {}
        return metrics

    def close(self) -> None:
        self.teacher.close()
