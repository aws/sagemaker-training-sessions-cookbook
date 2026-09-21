# KTO Preference Alignment

[KTO](https://arxiv.org/abs/2402.01306) (Kahneman-Tversky Optimization) aligns a
model from a **binary** signal - each `(prompt, completion)` is labelled
*desirable* or *undesirable* - with no need for the *paired* chosen-vs-rejected
data DPO requires. That makes data far cheaper to collect: thumbs up/down,
pass/fail graders, or accepted/rejected logs are enough.

This recipe trains KTO on a managed Training Session using **only** the stock
`forward` / `forward_backward` API - no custom server-side loss kernel - via the
surrogate-gradient trick described below.

## Run

Set up the environment and account config first (see the
[repo README](../../../README.md)), then:

```bash
python -m training_sessions_cookbook.recipes.preference_kto.train \
    --base_model_arn "$BASE_MODEL_ARN" \
    --model_package_group_arn "$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn "$ROLE_ARN" \
    --output_s3_path "$OUTPUT_S3_PATH" \
    --dataset_path trl-lib/kto-mix-14k \
    --beta 0.1 --batch_size 32 --learning_rate 5e-6

# Or drive everything from the preset
python -m training_sessions_cookbook.recipes.preference_kto.train \
    --config training_sessions_cookbook/recipes/preference_kto/configs/kto_mix.json
```

The dataset needs `prompt` / `completion` / `label` columns (the
`trl-lib/kto-mix-*` shape); a local `.jsonl` with the same keys also works
(`label` truthy = desirable). Prompt and completion may each be a chat-message
list or a bare string.

## What to watch

| Metric | Meaning | Healthy trend |
|---|---|---|
| `kto/loss` | mean KTO objective over the batch | decreasing |
| `kto/reward_desirable` | mean `log pi_theta - log pi_ref` on desirable | rising |
| `kto/reward_undesirable` | same on undesirable | falling |
| `kto/reward_margin` | desirable minus undesirable reward | widening (> 0) |
| `kto/accuracy` | fraction on the correct side of `z_ref` | rising toward 1 |
| `kto/z_ref` | KL reference point | small, stable |

A widening `kto/reward_margin` with a stable `kto/z_ref` is the signal that the
model is learning to prefer desirable completions without drifting far from the
reference.

## How the loop works

The loop lives in [`preference/`](../../preference/). KTO's loss is non-linear
(a sigmoid of the policy-vs-reference log-ratio), but the model enters it **only
through the per-sequence completion log-probability**
`logp_i = log pi_theta(y_i | x_i)`. So its gradient is linear in the per-token
log-probs by the chain rule, and each sequence contributes a single scalar
`s_i = dL_i / dlogp_i`. Each step:

1. **forward** (no grad) on the matched `(x_i, y_i)` batch → per-token policy
   log-probs → `log pi_theta(y_i)`.
2. **forward** (no grad) on a *mismatched* KL batch `(x_i, y_p(i))` → the KL
   reference point `z_ref = max(0, mean(log pi_theta - log pi_ref))`
   (`--kl_estimator none` skips this pass and sets `z_ref = 0`).
3. Compute the implicit reward `r_i` and `s_i` per example, and set the
   completion-token **advantage to `-s_i`**.
4. **forward_backward** on the matched batch with the importance-sampling loss,
   with `old_log_probs` set to the step-1 log-probs so the importance ratio is
   1. The service's gradient `-Σ advantage_t · dlogp_t/dθ` then equals the exact
   KTO gradient `Σ s_i · dlog pi_theta(y_i)/dθ`. Then **optim_step**.

This is the manual vector-Jacobian-product ("surrogate loss") pattern: the
client computes `dL/dlogp`, the service computes `dlogp/dθ`, and the weighted
`forward_backward` multiplies them. It works for *any* differentiable loss whose
dependence on the model is through per-sequence log-probs.

### Reference model

The reference is the frozen base model (a LoRA adapter starts at zero delta), so
the reference log-probs are computed **once before the first optimizer step**
and cached to `{run_folder}/kto_reference.json`, which `--resume` reloads (after
resuming, the adapter is no longer zero, so they cannot be recomputed on the
fly).

### Cost

Per step: two no-grad forwards (matched + KL) plus one `forward_backward`, i.e.
~3× a plain SFT step; `--kl_estimator none` drops it to ~2×. Reference log-probs
add a one-time pass over the dataset (two with the KL term) at startup.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--beta` | `0.1` | temperature on the implicit reward; higher = sharper |
| `--desirable_weight` / `--undesirable_weight` | `1.0` / `1.0` | class loss weights `λ_D` / `λ_U`; raise the minority class under imbalance |
| `--kl_estimator` | `shuffled` | `shuffled` (faithful `z_ref`) or `none` (`z_ref = 0`, one fewer forward) |
| `--learning_rate` | `5e-6` | preference tuning wants a smaller LR than SFT |

## Checkpoints

Every saved checkpoint is a SageMaker model package recorded in
`{run_folder}/checkpoints.jsonl`. `--save_every N` adds periodic full
checkpoints; the final checkpoint (weights + optimizer, resumable) is always
saved. `--resume --log_path <run_folder>` continues an interrupted run.
