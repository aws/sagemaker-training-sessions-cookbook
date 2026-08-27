# GSM8K RL

Policy-gradient RL on grade-school math: sample rollouts from the current
policy, reward the final `#### <answer>` line, compute grouped (GRPO-style) or
Monte-Carlo advantages, and train with `--loss ppo`, `importance_sampling`,
or `cispo`. A built-in 20-problem subset makes smoke runs dataset-free; use
`tools/download_gsm8k.py` for the full dataset.

## Run

Set up the environment and account config first (see the [repo README](../../../README.md)),
then fetch the dataset once (the preset below trains on the full GSM8K train
split, which this writes to `data/gsm8k_train.jsonl` plus the test split used
for validation):

```bash
python tools/download_gsm8k.py
```

Omit `--dataset_path` entirely to skip this and run on the built-in
20-problem subset instead (useful for a smoke test, not for the numbers below).

Then:

```bash
python -m training_sessions_cookbook.recipes.gsm8k_rl.train \
    --base_model_arn "$BASE_MODEL_ARN" \
    --model_package_group_arn "$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn "$ROLE_ARN" \
    --output_s3_path "$OUTPUT_S3_PATH" \
    --loss ppo --advantage_mode grouped \
    --batch_size 128 --responses_per_prompt 8 --learning_rate 4e-5

# Or drive everything from the preset
python -m training_sessions_cookbook.recipes.gsm8k_rl.train --config training_sessions_cookbook/recipes/gsm8k_rl/configs/gsm8k_oss20b.json
```

Launcher: [`examples/rl_gsm8k.sh`](../../../examples/rl_gsm8k.sh).

## Expected results

| Model | Steps | `rollout/reward_mean` |
|---|---|---|
| gpt-oss-20b | 40 | 0.71 → 0.90 |

## How the loop works

The training loop lives in [`rl/`](../../rl/) and is shared with the code-RL
recipe:

- **Rollouts**: each step samples `--responses_per_prompt` completions per
  prompt for `--batch_size` prompts, concurrently (`rl/rollouts.py`).
- **Advantages**: `grouped` centers rewards within each prompt's response
  group; `monte_carlo` uses a global baseline (`rl/advantages.py`).
- **Sampler refresh**: every `--refresh_every` steps the loop publishes the
  latest weights, controlling how on-policy the data is.
- **Overlapped training (off-by-k)**: `--max_steps_off_policy K` lets rollout
  workers keep generating while the trainer trains; stale groups are dropped
  or requeued per `--stale_policy` (`rl/staleness.py`).
- **Validation**: `--val_dataset_path` scores sampled completions (at the
  training temperature) on a held-out set on a cadence (`rl/validation.py`).

## Checkpoints

Every saved checkpoint is a SageMaker model package, recorded in
`{run_folder}/checkpoints.jsonl`. By default the run saves a full checkpoint
at the end (`save_state`: weights + optimizer, resumable) plus a rolling
crash-recovery checkpoint every 10 steps (state-only, saved in the background;
each one deletes its predecessor so storage stays bounded -
`--rolling_save_every` tunes the cadence, `0` disables). `--save_every N` adds
full periodic checkpoints mid-run, and `--save_weights_at_end` additionally
publishes a deployable adapter package.

## Bring your own environment

Implement the `Env` protocol in [`rl/envs/base.py`](../../rl/envs/base.py) and
register it with `envs.register_env("my_task", factory_builder,
add_cli_args=...)`; the loop, advantages, and metrics all work unchanged, and
your flags appear in your recipe's `--help`. See the
[recipes README](../README.md) for the full pattern; `rl/envs/gsm8k.py` is the
smallest complete example.
