#!/bin/bash
# SFT on a SageMaker Training Session: ultrachat_200k.
# Run from the repo root: bash examples/sft_ultrachat.sh
set -x
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/sft_ultrachat_$(date +%m%dT%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# SageMaker configuration: override via env for your account.
export TRAINING_SESSION_NAME="${TRAINING_SESSION_NAME:-sft-ultrachat-$(date +%s)}"
export BASE_MODEL_ARN="${BASE_MODEL_ARN:-arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/huggingface-vlm-qwen3-6-27b/1.8.0}"
export MODEL_PACKAGE_GROUP_ARN="${MODEL_PACKAGE_GROUP_ARN:-arn:aws:sagemaker:us-west-2:ACCOUNT_ID:model-package-group/sft-ultrachat}"
export ROLE_ARN="${ROLE_ARN:-arn:aws:iam::ACCOUNT_ID:role/SageMakerRole}"
export OUTPUT_S3_PATH="${OUTPUT_S3_PATH:-s3://YOUR_BUCKET/training-sessions-cookbook/sft-output/ultrachat}"

RESULTS_DIR="results/sft_ultrachat"
mkdir -p "$RESULTS_DIR"

python -m training_sessions_cookbook.recipes.chat_sft.train \
    --training_session_name="$TRAINING_SESSION_NAME" \
    --base_model_arn="$BASE_MODEL_ARN" \
    --model_package_group_arn="$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn="$ROLE_ARN" \
    --output_s3_path="$OUTPUT_S3_PATH" \
    --region=us-west-2 \
    --dataset_path=HuggingFaceH4/ultrachat_200k \
    --train_split=train_sft \
    --eval_split=test_sft \
    --lora_rank=64 \
    --lora_alpha=128 \
    --learning_rate=5e-5 \
    --batch_size=128 \
    --max_length=16384 \
    --num_epochs=1 \
    --max_steps=300 \
    --save_every=100 \
    --eval_every=50 \
    --lr_schedule=linear \
    --log_path="$RESULTS_DIR" \
    --max_conversations=50000 \
    --eval_size=1024 \
    "$@"
