"""On-policy distillation (OPD, OPSD, SDPO) over SageMaker Training Sessions.

Peer of :mod:`training_sessions_cookbook.rl`, and built on its loop: the
student samples rollouts and trains with ``forward_backward(importance_sampling)``
exactly as in RL, but each sampled token's advantage comes from a teacher that
re-reads the rollout in one ``compute_logprobs`` prefill, instead of one GRPO
scalar per episode. Needs no service change.

* :mod:`.losses` - the pure per-token advantage math and diagnostics.
* :mod:`.context` - what the teacher reads (same prompt / privileged solution /
  feedback), spliced in front of the student's exact token IDs.
* :mod:`.teacher` - teacher clients (the frozen base in the student's session;
  a checkpoint, a lagged self, or a different base model in a second session).
* :mod:`.advantages` - the teacher-scoring stage plugged into the RL loop.
* :mod:`.train` - parser and entry points used by the ``opd_math``,
  ``opsd_math`` and ``sdpo_code`` recipes.
"""
