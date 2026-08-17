#!/bin/bash
# Analyze SFT results: loss plots (+ optional before/after inference).
# Run from the repo root: bash examples/sft_analyses.sh
#
# Plot only (no session needed):
#     bash examples/sft_analyses.sh
# Plot + before/after inference (pass extra flags through):
#     bash examples/sft_analyses.sh --run_inference \
#         --model_package_arn arn:aws:sagemaker:us-west-2:ACCOUNT:model-package/NAME/9
# For datasets with named eval splits (e.g. ultrachat's test_sft), pass --eval_split:
#     bash examples/sft_analyses.sh --run_inference --eval_split test_sft \
#         --dataset_path HuggingFaceH4/ultrachat_200k \
#         --model_package_arn arn:aws:sagemaker:us-west-2:ACCOUNT:model-package/NAME/9
set -x
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/sft_analyses_$(date +%m%dT%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Point at the metrics.jsonl written by the SFT run.
export METRICS_PATH="${METRICS_PATH:-results/sft_tulu3_service/metrics.jsonl}"

# SageMaker configuration: only needed with --run_inference. Override via env.
export BASE_MODEL_ARN="${BASE_MODEL_ARN:-arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/huggingface-vlm-qwen3-6-27b/1.8.0}"
export MODEL_PACKAGE_GROUP_ARN="${MODEL_PACKAGE_GROUP_ARN:-arn:aws:sagemaker:us-west-2:ACCOUNT_ID:model-package-group/sft-tulu3}"
export ROLE_ARN="${ROLE_ARN:-arn:aws:iam::ACCOUNT_ID:role/SageMakerRole}"
export OUTPUT_S3_PATH="${OUTPUT_S3_PATH:-s3://YOUR_BUCKET/training-sessions-cookbook/sft-output/}"

python -m training_sessions_cookbook.recipes.chat_sft.analyze \
    --metrics_path="$METRICS_PATH" \
    --region=us-west-2 \
    --dataset_path=allenai/tulu-3-sft-mixture \
    --base_model_arn="$BASE_MODEL_ARN" \
    --model_package_group_arn="$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn="$ROLE_ARN" \
    --output_s3_path="$OUTPUT_S3_PATH" \
    "$@"
