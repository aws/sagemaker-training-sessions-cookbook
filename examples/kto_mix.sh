#!/bin/bash
# KTO preference alignment on a SageMaker Training Session: trl-lib/kto-mix-14k.
# Run from the repo root: bash examples/kto_mix.sh
set -x
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/kto_mix_$(date +%m%dT%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Auto-load account-specific config from .env if present (BASE_MODEL_ARN,
# ROLE_ARN, MODEL_PACKAGE_GROUP_ARN, OUTPUT_S3_PATH). Copy env.example.sh -> .env
# and fill in your values once.
if [ -f .env ]; then
    # shellcheck disable=SC1091
    source .env
fi

# SageMaker configuration: override via env for your account.
export TRAINING_SESSION_NAME="${TRAINING_SESSION_NAME:-kto-mix-$(date +%s)}"
export BASE_MODEL_ARN="${BASE_MODEL_ARN:-arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/huggingface-vlm-qwen3-6-27b/1.23.0}"
export MODEL_PACKAGE_GROUP_ARN="${MODEL_PACKAGE_GROUP_ARN:-arn:aws:sagemaker:us-west-2:ACCOUNT_ID:model-package-group/cookbook-preference}"
export ROLE_ARN="${ROLE_ARN:-arn:aws:iam::ACCOUNT_ID:role/SageMakerRole}"
export OUTPUT_S3_PATH="${OUTPUT_S3_PATH:-s3://YOUR_BUCKET/training-sessions-cookbook/kto-output/kto-mix}"

# Optional MLflow publishing (requires: pip install -e '.[mlflow]').
# Set MLFLOW_TRACKING_URI to a SageMaker managed MLflow tracking App ARN
# (arn:aws:sagemaker:...:mlflow-app/<name>) or any MLflow tracking URI.
MLFLOW_ARGS=()
if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
    MLFLOW_ARGS+=(--mlflow_tracking_uri="$MLFLOW_TRACKING_URI")
    MLFLOW_ARGS+=(--mlflow_experiment_name="${MLFLOW_EXPERIMENT_NAME:-cookbook-preference}")
    [ -n "${MLFLOW_RUN_NAME:-}" ] && MLFLOW_ARGS+=(--mlflow_run_name="$MLFLOW_RUN_NAME")
fi

RESULTS_DIR="results/kto_mix"
mkdir -p "$RESULTS_DIR"

python -m training_sessions_cookbook.recipes.preference_kto.train \
    --training_session_name="$TRAINING_SESSION_NAME" \
    --base_model_arn="$BASE_MODEL_ARN" \
    --model_package_group_arn="$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn="$ROLE_ARN" \
    --output_s3_path="$OUTPUT_S3_PATH" \
    --region=us-west-2 \
    --dataset_path=trl-lib/kto-mix-14k \
    --lora_rank=64 \
    --lora_alpha=128 \
    --beta=0.1 \
    --desirable_weight=1.0 \
    --undesirable_weight=1.0 \
    --kl_estimator=shuffled \
    --reference_source=precompute \
    --learning_rate=5e-6 \
    --batch_size=32 \
    --max_length=4096 \
    --num_epochs=1 \
    --max_steps=300 \
    --warmup_steps=20 \
    --save_every=100 \
    --eval_every=50 \
    --eval_ratio=0.05 \
    --eval_size=320 \
    --lr_schedule=linear \
    --log_path="$RESULTS_DIR" \
    "${MLFLOW_ARGS[@]}" \
    "$@"
