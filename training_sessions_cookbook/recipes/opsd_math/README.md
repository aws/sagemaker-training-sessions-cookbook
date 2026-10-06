# On-Policy Self-Distillation (OPSD) on GSM8K

[OPSD](https://arxiv.org/abs/2601.18734) is on-policy distillation with no
second model: the teacher is the student's own frozen base model, given
**privileged** information the student never sees, here the dataset's worked
solution. The student solves the problem unaided; the teacher re-reads the
student's tokens behind a prompt that includes the reference solution, and
grades each token. Knowing the answer makes the base model a sharper grader of
its own reasoning. The privileged context is used only for scoring; it never
enters the student's prompt, so the student cannot learn to depend on it.

Everything except the teacher and its context is shared with
[`opd_math`](../opd_math/README.md#how-it-works): read that README for the
per-token advantage, the log-prob check and the metrics.

## Run

Fetch GSM8K with worked solutions once (`python tools/download_gsm8k.py`;
the `answer` field is the full solution ending `#### N`), then:

```bash
bash examples/opsd_gsm8k.sh

# Or explicitly
python -m training_sessions_cookbook.recipes.opsd_math.train \
    --config training_sessions_cookbook/recipes/opsd_math/configs/opsd_gsm8k_qwen3_6_27b.json \
    --role_arn "$ROLE_ARN" \
    --model_package_group_arn "$MODEL_PACKAGE_GROUP_ARN" \
    --output_s3_path "$OUTPUT_S3_PATH"
```

One session. The teacher is an ARN-less sampling client on the student's
session, which serves the untouched base weights for the whole run (the LoRA
adapter starts at zero, so the base *is* the student at initialization).

## The teacher's prompt

The student sees the normal GSM8K prompt. The teacher sees
([`distill/context.py`](../../distill/context.py), `OPSD_TEMPLATE`):

```
<problem>

Here is a reference solution to this problem:

<dataset answer field>

After understanding the reference solution, solve the problem yourself using
your own approach.
```

rendered with the same chat template, followed by the student's exact sampled
tokens.

- `--privileged_field` picks the row field (default `answer`).
- `--privileged_mode final_answer` keeps only the text after `####`, a much
  weaker hint. 2026 follow-ups report short hints can beat full solutions;
  this is one of the open questions to measure.
- `--teacher_context_template` overrides the prompt (placeholders
  `{observation}`, `{privileged}`).
- Rows without the field are left unscored (`distill/frac_sequences_scored`
  drops below 1). The built-in 20-problem subset only has bare numeric
  answers; the run warns if you use it.

## What to watch

The `distill/*` metrics from [`opd_math`](../opd_math/README.md#what-to-watch),
plus GSM8K `val/accuracy`. Two OPSD-specific expectations:

- `distill/reverse_kl` is non-zero from step 0, even though student and
  teacher are the same weights: they see different prompts. Its magnitude
  should fall as the student absorbs what the solution tells the teacher.
- Responses often get shorter (`rollout/ac_tokens_per_turn`).

## Expected results

Not yet validated on the service. OPSD's gain over GRPO is reported to shrink
with model size (large at 1.7B, small by 8B), so the preset's 27B student
mainly validates the plumbing; add a smaller student to measure the effect.

| Student | Steps | `distill/reverse_kl` | `val/accuracy` |
|---|---|---|---|
| Qwen3.6-27B | 50 | TBD | TBD |

## Differences from the paper

- **Sampled-token** reverse KL. The paper's best result uses full-vocabulary
  forward KL with per-entry clipping, which needs top-K or full-distribution
  log-probs the API does not return yet. `--adv_clip` (5.0 in the preset)
  clips the per-token advantage instead.
- The frozen-at-init teacher is the base model. When the student starts from a
  non-zero adapter, use `--teacher checkpoint --teacher_model_package_arn` to
  freeze the starting adapter instead.
