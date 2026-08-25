# Supervised Fine-Tuning

Fine-tune a base model on conversational data with the `cross_entropy` loss.
Loss is masked to assistant-response tokens; any HuggingFace conversational
dataset (or a local JSONL file) works.

## Run

Set up the environment and account config first (see the [repo README](../../../README.md)),
then:

```bash
# Tulu 3 mixture
python -m training_sessions_cookbook.recipes.chat_sft.train \
    --base_model_arn "$BASE_MODEL_ARN" \
    --model_package_group_arn "$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn "$ROLE_ARN" \
    --output_s3_path "$OUTPUT_S3_PATH" \
    --dataset_path allenai/tulu-3-sft-mixture \
    --batch_size 128 --learning_rate 5e-5 --lora_rank 64 \
    --save_every 100 --eval_every 50

# UltraChat
python -m training_sessions_cookbook.recipes.chat_sft.train ... \
    --dataset_path HuggingFaceH4/ultrachat_200k \
    --train_split train_sft --eval_split test_sft

# Or drive everything from a preset config
python -m training_sessions_cookbook.recipes.chat_sft.train \
    --config training_sessions_cookbook/recipes/chat_sft/configs/sft_tulu3.json \
    --role_arn "$ROLE_ARN" --model_package_group_arn "$MODEL_PACKAGE_GROUP_ARN" \
    --output_s3_path "$OUTPUT_S3_PATH"
```

Any flag passed explicitly overrides the preset. A finished run's
`config.json` is the same shape, so a run can be reproduced with
`--config results/<run>/config.json`.

Ready-made launchers: [`examples/sft_tulu3.sh`](../../../examples/sft_tulu3.sh),
[`examples/sft_ultrachat.sh`](../../../examples/sft_ultrachat.sh).

## Expected results

| Dataset | Model | Steps | `train/mean_nll` |
|---|---|---|---|
| Tulu 3 | Qwen3.6-27B | 400 | 1.23 → 0.65 |
| UltraChat | Qwen3.6-27B | 300 | 2.93 → 0.79 |

Analyze a finished run with `python -m training_sessions_cookbook.recipes.chat_sft.analyze
--metrics_path <run_folder>/metrics.jsonl` (loss curves, token stats; needs the
`plots` extra).

## Checkpoints and resume

Every `--save_every` steps the loop saves a full checkpoint (optimizer state +
sampler weights) as SageMaker model packages and records the ARNs in
`{run_folder}/checkpoints.jsonl`. Checkpoints are kept until you delete
their model packages (rolling ones delete themselves as newer ones land).

- Resume an interrupted run by re-running with `--resume --log_path
  <run_folder>`. The session reopens from the newest recorded checkpoint and
  continues at the next step.
- `--rolling_save_every N` adds cheap resume-only checkpoints between periodic
  saves; each one deletes its predecessor so storage stays bounded.

## Bring your own dataset

`data.py` loads any HF conversational dataset (`messages`-style records) or a
local JSONL file with the same shape. Pass `--dataset_path` accordingly;
`--train_split` / `--eval_split` select explicit splits, or `--eval_ratio`
carves out a held-out slice.

For anything else, bypass the loader entirely: `run(args,
train_batches=...)` accepts pre-built datum batches or a callable
`(tokenizer) -> batches` (the service-resolved tokenizer arrives once the
session opens). See the [recipes README](../README.md) for the programmatic
pattern.
