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

Launcher: [`examples/kto_mix.sh`](../../../examples/kto_mix.sh) (sources `.env`
for account ARNs; optional MLflow publishing).

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
| `kto/logps_desirable` / `kto/logps_undesirable` | mean completion log-prob per class (TRL's `logps/chosen`, `logps/rejected`) | steady — a crash signals collapse |
| `kto/accuracy` | fraction on the correct side of `z_ref` | rising toward 1 |
| `kto/z_ref` | KL reference point | small, stable |

Every `--eval_every` steps a **held-out** pass adds `kto/eval_reward_margin`,
`kto/eval_accuracy`, `kto/eval_reward_desirable` / `_undesirable`, and
`kto/eval_logps_desirable` / `_undesirable` to that step's record (forward-only,
using the step's `z_ref` as the accuracy threshold). The held-out
`kto/eval_reward_margin` is the metric to trust for whether KTO is actually
helping — training margins can widen while the model degrades. The preset trains
for 300 steps: a 400-step reference run showed every metric plateaued by step
~300 (eval margin and accuracy flat from there on). On other datasets, stop
once `kto/eval_reward_margin` is flat across two consecutive eval passes.

A widening `kto/reward_margin` (train **and** eval) with a stable `kto/z_ref`
and non-collapsing `kto/logps_*` is the signal that the model is learning to
prefer desirable completions without drifting far from the reference. If
`kto/logps_desirable` falls sharply, the model is degrading generations — lower
the learning rate or `--beta`.

## Expected results

Assessed on `trl-lib/kto-mix-14k` with an instruct base model. Absolute values
depend on the model and data; what matters is the **direction** over training:

| Metric | Start | Healthy trend | Reference run (400 steps) |
|---|---|---|---|
| `kto/eval_reward_margin` | ~0 | rises and stays > 0 | -0.1 → ~98 |
| `kto/eval_accuracy` | ~0.5 | rises toward 0.7–0.9 | 0.47 → 0.69 |
| `kto/z_ref` | small | stays small and stable (no runaway KL) | 0 throughout |
| `kto/eval_logps_desirable` | — | does **not** collapse | -333 → -354 |

Reference run: `configs/kto_mix.json` on qwen3.6-27b; metrics and plots in
[`results/preference_kto/`](../../../results/preference_kto/)
(regenerate with `python -m training_sessions_cookbook.recipes.preference_kto.plots`).

> For a
> downstream check beyond these preference metrics, evaluate win-rate vs the
> reference model (model-judged) or a standard instruction benchmark on the
> saved adapter — those measure whether the alignment transferred to generation
> quality, which the reward metrics only proxy.

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

The reference is the frozen base model (a LoRA adapter starts at zero delta).
`--reference_source` picks how its log-probs are obtained:

- **`precompute`** (default): one training-forward pass at step 0 (while the
  adapter delta is still zero, so the training client scores the base), cached
  to `{run_folder}/kto_reference.json` and reloaded by `--resume` (after
  resuming, the adapter is no longer zero, so they cannot be recomputed on the
  fly). The KL pairing is fixed at build time so its reference side is
  precomputable too.
- **`compute_logprobs`**: score the frozen base **on demand** via an ARN-less
  sampling client (`create_sampling_client()` with no checkpoint bound serves
  the base model), so the KL pairing is re-randomized every step (TRL-style).
  No reference cache is needed — the base client is always available.

> **Engine-mismatch caveat.** The ARN-less client runs on the **inference**
> engine, while the policy matched log-probs come from the **training** forward,
> so the reward `r = logπ_θ(train) − logπ_ref(inference)` crosses two engines
> (the rollout-vs-training log-prob gap). For KTO's `z_ref` — a detached, clamped
> baseline — that is fine. The policy side of the reward is **never** taken from
> the inference path, because the engine gap and the sampler-checkpoint lag would
> bias the actual gradient.

### Cost

Per step: two no-grad forwards (matched + KL) plus one `forward_backward`, i.e.
~3× a plain SFT step; `--kl_estimator none` drops it to ~2×. Reference log-probs
add a one-time pass over the dataset (two with the KL term) at startup.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--beta` | `0.1` | temperature on the implicit reward; higher = sharper |
| `--desirable_weight` / `--undesirable_weight` | `1.0` / `1.0` | class loss weights `λ_D` / `λ_U`; raise the minority class under imbalance |
| `--kl_estimator` | `shuffled` | `shuffled` (faithful `z_ref`) or `none` (`z_ref = 0`, one fewer forward). In the reference run `z_ref` stayed 0 throughout — if you see the same, `none` is free |
| `--reference_source` | `precompute` | `precompute` (step-0 forward, cached, fixed KL pairing) or `compute_logprobs` (score base on demand, re-randomized KL pairing) |
| `--eval_every` | `50` | held-out eval pass cadence in steps (`0` disables); split sized by `--eval_size` / `--eval_ratio` |
| `--learning_rate` | `5e-6` | preference tuning wants a smaller LR than SFT; TRL recommends `5e-7`–`5e-6` at `beta=0.1` |

## Checkpoints

Every saved checkpoint is a SageMaker model package recorded in
`{run_folder}/checkpoints.jsonl`. `--save_every N` adds periodic full
checkpoints; the final checkpoint (weights + optimizer, resumable) is always
saved. `--resume --log_path <run_folder>` continues an interrupted run.
