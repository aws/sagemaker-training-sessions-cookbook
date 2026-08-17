#!/bin/bash
# SFT on a SageMaker Training Session: tulu-3-sft-mixture.
# Run from the repo root: bash examples/sft_tulu3.sh
set -x
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/sft_tulu3_$(date +%m%dT%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# SageMaker configuration: override via env for your account.
export TRAINING_SESSION_NAME="${TRAINING_SESSION_NAME:-sft-tulu3-$(date +%s)}"
export BASE_MODEL_ARN="${BASE_MODEL_ARN:-arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/huggingface-vlm-qwen3-6-27b/1.8.0}"
export MODEL_PACKAGE_GROUP_ARN="${MODEL_PACKAGE_GROUP_ARN:-arn:aws:sagemaker:us-west-2:ACCOUNT_ID:model-package-group/sft-tulu3}"
export ROLE_ARN="${ROLE_ARN:-arn:aws:iam::ACCOUNT_ID:role/SageMakerRole}"
export OUTPUT_S3_PATH="${OUTPUT_S3_PATH:-s3://YOUR_BUCKET/training-sessions-cookbook/sft-output/}"

RESULTS_DIR="results/sft_tulu3"
mkdir -p "$RESULTS_DIR"

# Optional MLflow publishing (requires: pip install -e '.[mlflow]').
# Set MLFLOW_TRACKING_URI to a SageMaker managed MLflow tracking App
# ARN (arn:aws:sagemaker:...:mlflow-app/<name>) or any MLflow tracking URI
# (e.g. http://localhost:5000).
MLFLOW_ARGS=()
if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
    MLFLOW_ARGS+=(--mlflow_tracking_uri="$MLFLOW_TRACKING_URI")
    MLFLOW_ARGS+=(--mlflow_experiment_name="${MLFLOW_EXPERIMENT_NAME:-cookbook-sft}")
    [ -n "${MLFLOW_RUN_NAME:-}" ] && MLFLOW_ARGS+=(--mlflow_run_name="$MLFLOW_RUN_NAME")
fi

python -m training_sessions_cookbook.recipes.chat_sft.train \
    --training_session_name="$TRAINING_SESSION_NAME" \
    --base_model_arn="$BASE_MODEL_ARN" \
    --model_package_group_arn="$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn="$ROLE_ARN" \
    --output_s3_path="$OUTPUT_S3_PATH" \
    --region=us-west-2 \
    --dataset_path=allenai/tulu-3-sft-mixture \
    --lora_rank=64 \
    --lora_alpha=128 \
    --learning_rate=5e-5 \
    --batch_size=128 \
    --max_length=16384 \
    --eval_size=1024 \
    --num_epochs=1 \
    --max_steps=400 \
    --save_every=100 \
    --eval_every=50 \
    --lr_schedule=linear \
    --log_path="$RESULTS_DIR" \
    --max_conversations=60000 \
    "${MLFLOW_ARGS[@]}" \
    "$@"
