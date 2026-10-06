# Self-Distillation Policy Optimization (SDPO) on code

[SDPO](https://arxiv.org/abs/2601.20802) is RL with rich feedback. When a
rollout fails, the environment usually says *why*: a failing test, a runtime
error, a wrong output. SDPO turns that into a per-token signal: the teacher is
the student itself, re-reading its attempt behind a prompt that includes the
feedback (and, when a sibling rollout passed, that sibling's solution), and
grading each token. Tokens the informed self would not have written are pushed
down. Unlike [OPSD](../opsd_math/), no reference solutions are needed: the
signal comes from the environment, and from the student's own successes.

This recipe runs SDPO on DeepCoder with sandboxed grading, reusing the
[`code_rl`](../code_rl/) environment. Everything except the teacher and its
context is shared with [`opd_math`](../opd_math/README.md#how-it-works).

## Run

Start SandboxFusion and install the `code-rl` extras as for
[`code_rl`](../code_rl/README.md), then:

```bash
bash examples/sdpo_code.sh

# Or explicitly
python -m training_sessions_cookbook.recipes.sdpo_code.train \
    --config training_sessions_cookbook/recipes/sdpo_code/configs/sdpo_code_oss20b.json \
    --role_arn "$ROLE_ARN" \
    --model_package_group_arn "$MODEL_PACKAGE_GROUP_ARN" \
    --output_s3_path "$OUTPUT_S3_PATH"
```

One session (the preset's frozen teacher shares it).

## The teacher

**Weights** (`--teacher base`, preset): the student's frozen base, an
ARN-less client in the student's own session. SDPO reports that grading with
the current weights diverges (36.1), while a frozen teacher (48.8) is close
to its best EMA (49.3) and trust-region (50.6) teachers.

`--teacher lagged` approximates a slowly moving teacher (the service has no
EMA adapter): a second, sampling-only session on the student's base model
starts at the base, and every `--teacher_refresh_every` steps binds its
sampling path to the student's latest sampler checkpoint
(`load_weights_for_sampling`). It needs its own session because a session
has a single bound sampling path, which the student's per-step refresh
re-binds. The second session costs extra; the lagged teacher's benefit over
frozen is the thing to measure.

**Context** ([`distill/context.py`](../../distill/context.py),
`SDPO_TEMPLATE`): for each rollout in a group of `--responses_per_prompt`,

```
<problem>

Correct solution:            <- a *different* rollout in the group that passed
<its code block>

The following is feedback from your unsuccessful earlier attempt:
<sandbox output: first failing test, expected vs actual, stderr>   <- this rollout failed

Correctly solve the original question.
```

followed by the student's exact sampled tokens. Either section is omitted
when it does not apply. The student's own attempt is never shown to the
teacher (SDPO found that hurts). A rollout with nothing to add, for example
the only passing rollout in its group, is left unscored (zero advantage), so
`distill/frac_sequences_scored` is below 1 by design.

- `--no-sdpo_sibling_solutions` / `--no-sdpo_env_feedback` turn either source
  off.
- `--feedback_max_chars` truncates sandbox output (default 2000).
- `--alpha_task A` adds the GRPO task-reward advantage, which also trains the
  rollouts SDPO leaves unscored.

## What to watch

The `distill/*` metrics from [`opd_math`](../opd_math/README.md#what-to-watch),
plus:

| Metric | Healthy trend |
|---|---|
| `val/accuracy` (DeepCoder test split, `--val_code_split test`) | rising |
| `rollout/reward_mean` | rising |
| `distill/frac_sequences_scored` | well above 0; low means few failures carry feedback and few groups have a success |
| `rollout/ac_tokens_per_turn` | SDPO reports 3-11x shorter answers than GRPO |

## Expected results

Not yet validated on the service. SDPO's gains are reported to emerge with
scale (it trails GRPO at 1.5B); the preset uses gpt-oss-20b, the model the
code_rl recipe is tuned for.

| Student | Steps | `val/accuracy` | `rollout/reward_mean` |
|---|---|---|---|
| gpt-oss-20b | 50 | TBD | TBD |

## Differences from the paper

- **Sampled-token** form (SDPO's "token form", which drops into RLVR by
  replacing the advantage). The paper's logit form uses top-100 log-probs plus
  a tail term, which needs top-K log-probs the API does not return yet.
- Frozen teacher by default; `--teacher lagged` is a periodic snapshot, not an
  EMA or trust region.
- Single-turn (`--max_turns 1`): feedback comes from the final grade. In
  multi-turn code_rl the student already sees `check_solution` results between
  turns.
