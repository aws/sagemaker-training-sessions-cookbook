# On-Policy Distillation (OPD) on GSM8K

On-policy distillation trains a small student on its **own** samples, with a
stronger teacher grading every token: the student writes a solution, the
teacher reads it in one forward pass (a prefill, no decoding) and reports how
likely it would have been to write each token. The student is pushed toward
tokens the teacher prefers and away from the rest. It trains like RL (on the
model's own attempts) but with a dense per-token signal instead of one reward
per episode, which is why it reaches RL-level quality at a fraction of the
compute ([Qwen3 report, Table 21](https://arxiv.org/abs/2505.09388);
[Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)).

This recipe runs OPD on GSM8K using only the stock API: `sample`,
`compute_logprobs`, and `forward_backward(importance_sampling)` with
per-token advantages. No service change is needed.

**Current preset: same-model teacher.** Training Sessions does not yet serve a
small Qwen student next to a larger same-family teacher, so the preset trains
Qwen3.6-27B against **its own frozen base, tempered to T = 0.5**
(`--teacher base --teacher_temperature 0.5`; one session). An untempered
same-model teacher on the same prompt gives exactly zero signal; tempering
sharpens it, so the student is pushed toward the base model's more confident
choices (a self-sharpening / entropy-reducing objective rather than knowledge
transfer). For real transfer, point the teacher at a stronger model of the
same family: `--teacher checkpoint --teacher_model_package_arn <adapter>`
(e.g. the final checkpoint of a [`gsm8k_rl`](../gsm8k_rl/) run, the
"same-size checkpoint teacher" frontier labs use) or, once a second model is
available, `--teacher session --teacher_base_model_arn <ARN>`. Its siblings swap the teacher:
[`opsd_math`](../opsd_math/) (the student's own base, shown the reference
solution) and [`sdpo_code`](../sdpo_code/) (the student's base, shown sandbox
feedback).

## Run

Set up the environment and account config first (see the
[repo README](../../../README.md)), fetch GSM8K once
(`python tools/download_gsm8k.py`), then:

```bash
bash examples/opd_gsm8k.sh            # preset + .env account values

# Or explicitly
python -m training_sessions_cookbook.recipes.opd_math.train \
    --config training_sessions_cookbook/recipes/opd_math/configs/opd_gsm8k_qwen3_6_27b.json \
    --role_arn "$ROLE_ARN" \
    --model_package_group_arn "$MODEL_PACKAGE_GROUP_ARN" \
    --output_s3_path "$OUTPUT_S3_PATH"
```

With `--teacher session`, the run opens **two** Training Sessions: the
student's, and a teacher session on `--teacher_base_model_arn` that is only
ever scored with `compute_logprobs` (never trained). The teacher session is
stopped and deleted when the run ends, on success or failure. The same
applies to `--teacher checkpoint`, whose adapter is bound in its own session
(a session has one bound sampling path, and the student's per-step sampler
refresh re-binds it). The preset's `base` teacher shares the student's
session.

## How it works

Each step reuses the RL loop ([`rl/`](../../rl/)) end to end, with the GRPO
advantage stage replaced by a teacher-scoring stage
([`distill/`](../../distill/)):

1. **Rollouts**: the student samples `--responses_per_prompt` solutions per
   prompt; the sampler returns each token's log-prob `log q(y_t)`.
2. **Teacher scoring**: for each rollout, the teacher scores
   `prompt + student tokens` with `compute_logprobs_async`. The student's token
   IDs are reused as-is, never re-tokenized, so teacher and student must share
   a tokenizer; this is checked when the teacher session opens.
3. **Per-token advantage**: `A_t = log p_teacher(y_t) / T - log q_student(y_t)`
   (`T = --teacher_temperature`, default 1). Positive where the teacher likes a
   token more than the student did.
4. **Update**: `forward_backward(importance_sampling)` with those per-token
   advantages (sampler log-probs as the old log-probs), then `optim_step`.

Why one sampled token is enough: reverse KL is an expectation over the
student's own samples, `KL(q||p) = E_{y~q}[log q(y) - log p(y)]`, so the
log-prob gap at the sampled token is an unbiased per-position estimate, and
using its negative as the advantage makes the importance-sampling loss follow
the reverse-KL gradient ([`distill/losses.py`](../../distill/losses.py) has
the derivation and the design doc's 3-token check is a unit test). The noise
averages out over thousands of positions per step.

Both log-probs in the advantage come from the inference engine (sampler and
`compute_logprobs`), so they are on the same scale. On the first step the run
re-scores a few rollouts under the student's own sampler and reports the gap
as `distill/logprob_check_abs_diff`; it should be close to 0. If it is not,
the two calls disagree on how they compute log-probs and every advantage is
offset. Sample at `--temperature 1.0` with `--top_p` / `--top_k` unset, so the
sampled log-probs are those of the student's actual distribution.

## Choosing the teacher

| `--teacher` | Weights | Notes |
|---|---|---|
| `base` (preset) | the student's frozen base | needs `--teacher_temperature < 1` for a signal; also recovery / KL anchor |
| `checkpoint` | a saved adapter of the student's base (`--teacher_model_package_arn`) | second session on the same base; a stronger same-size teacher, e.g. a `gsm8k_rl` checkpoint; recovery OPD from a pre-fine-tune checkpoint |
| `session` (CLI default) | a different base model (`--teacher_base_model_arn`) | second session; must be a model Training Sessions supports, same tokenizer family |

### Teacher temperature

`--teacher_temperature T` grades against the teacher's distribution tempered
to `T`, `p_T ∝ p^(1/T)`. `compute_logprobs` returns only the sampled token's
log-prob, so the recipe uses `log p(y_t) / T` and drops the normalizer
`log Z_t`. That is exact in expectation: `log Z_t` is the same for every
token at a position, so it acts as a baseline and contributes zero expected
gradient ([`distill/losses.py`](../../distill/losses.py)). Two consequences:

- `T` must be > 0. At `T -> 0` the teacher is greedy and every non-argmax
  token gets `-inf`; the run rejects `0`. `0.5` (preset) to `0.8` sharpens
  without that blow-up; `--adv_clip 5` bounds the large negative advantages
  rare tokens get.
- With `T != 1`, `distill/reverse_kl` and `distill/teacher_logprob` are offset
  by the dropped normalizer. Read their trend, not their level.

Same-family pairs keep the tokenizer identical (Qwen3.5 and Qwen3.6 share a
248,044-token vocabulary). Larger is not necessarily better: published OPD
results favor teachers ~1-6x the student.

## What to watch

| Metric | Meaning | Healthy trend |
|---|---|---|
| `distill/reverse_kl` | mean `log q - log p` over sampled tokens | falling toward 0 |
| `distill/reverse_kl_p95` | tail of the per-token gap | falling |
| `distill/teacher_logprob` / `distill/student_logprob` | mean per-token log-probs | student rising toward the teacher |
| `distill/logprob_check_abs_diff` | step-0 sampler vs `compute_logprobs` gap | ~0 |
| `distill/frac_sequences_scored` | rollouts that got a teacher score | 1.0 |
| `val/accuracy` | GSM8K test accuracy (sampled at the training temperature) | rising |
| `rollout/reward_mean` | task reward (logged only; not trained on unless `--alpha_task > 0`) | rising |

## Knobs

- `--alpha_task A`: add `A x` the GRPO task-reward advantage to every token
  (MiMo-style OPD + RL); `0` is pure distillation.
- `--adv_clip C`: clip each per-token advantage to `[-C, C]` if early steps are
  unstable (a token the teacher finds nearly impossible gives a large negative
  advantage).
- `--responses_per_prompt`: OPD needs only 1-4 samples per prompt.
- Every RL knob (off-by-k with `--max_steps_off_policy`, streamed
  `forward_backward`, validation, checkpoints, MLflow) works unchanged.

## Expected results

Not yet validated on the service. The design targets the published pattern:
`distill/reverse_kl` falling and GSM8K accuracy rising within tens of steps,
reaching RL's accuracy at a fraction of RL's sampled tokens. Fill in this
table from a reference run before relying on the preset's hyperparameters
(learning rate in particular is untuned).

| Student / teacher | Steps | `distill/reverse_kl` | `val/accuracy` |
|---|---|---|---|
| Qwen3.6-27B / own base, T = 0.5 | 50 | TBD | TBD |

## Limitations

- A same-model teacher (the preset) only sharpens the student; it cannot
  teach it anything the base does not already prefer. Watch `val/accuracy`
  and response diversity, and compare with a `checkpoint` teacher.
- A `session` teacher must be a model Training Sessions supports for
  training, since a session is opened on it (it is never trained). Teacher-only (inference-only)
  sessions are a service ask.
- Sampled-token reverse KL only. Top-K variants (forward KL, JSD) need top-K
  log-probs from `compute_logprobs`, which the API does not return yet.
- The teacher session bills for its whole lifetime, including the minutes the
  student spends sampling.
