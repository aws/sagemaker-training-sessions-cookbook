#!/bin/bash
# RL on a SageMaker Training Session: GSM8K, single-turn, GRPO grouped advantages.
# Run from the repo root: bash examples/rl_gsm8k.sh
#
# The validation block needs data/gsm8k_test.jsonl (held-out prompts), which is
# NOT checked in. Generate it once with: python tools/download_gsm8k.py
set -x
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/rl_gsm8k_$(date +%m%dT%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Auto-load account-specific config from .env if present (BASE_MODEL_ARN,
# ROLE_ARN, MODEL_PACKAGE_GROUP_ARN, OUTPUT_S3_PATH). Copy .env.example -> .env
# and fill in your values once.
if [ -f .env ]; then
    # shellcheck disable=SC1091
    source .env
fi

# SageMaker configuration: override via env for your account.
export TRAINING_SESSION_NAME="${TRAINING_SESSION_NAME:-}"
export BASE_MODEL="${BASE_MODEL:-}"
export BASE_MODEL_ARN="${BASE_MODEL_ARN:-}"
export MODEL_PACKAGE_GROUP_ARN="${MODEL_PACKAGE_GROUP_ARN:-}"
export ROLE_ARN="${ROLE_ARN:-}"
export OUTPUT_S3_PATH="${OUTPUT_S3_PATH:-}"

# Optional MLflow publishing (requires: pip install -e '.[mlflow]').
# Set MLFLOW_TRACKING_URI to a SageMaker managed MLflow tracking App
# ARN (arn:aws:sagemaker:...:mlflow-app/<name>) or any MLflow tracking URI
# (e.g. http://localhost:5000).
MLFLOW_ARGS=()
if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
    MLFLOW_ARGS+=(--mlflow_tracking_uri="$MLFLOW_TRACKING_URI")
    MLFLOW_ARGS+=(--mlflow_experiment_name="${MLFLOW_EXPERIMENT_NAME:-cookbook-rl}")
    [ -n "${MLFLOW_RUN_NAME:-}" ] && MLFLOW_ARGS+=(--mlflow_run_name="$MLFLOW_RUN_NAME")
fi

python -m training_sessions_cookbook.recipes.gsm8k_rl.train \
    --backend=service \
    --region=us-west-2 \
    --training_session_name="$TRAINING_SESSION_NAME" \
    --base_model="$BASE_MODEL" \
    --base_model_arn="$BASE_MODEL_ARN" \
    --model_package_group_arn="$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn="$ROLE_ARN" \
    --output_s3_path="$OUTPUT_S3_PATH" \
    --env_type=gsm8k \
    --template=auto \
    --dataset_path=data/gsm8k_train.jsonl \
    --max_prompts=1000 \
    --lora_rank=32 \
    --lora_alpha=64 \
    --learning_rate=4e-5 \
    --loss=ppo \
    --advantage_mode=grouped \
    --batch_size=64 \
    --responses_per_prompt=4 \
    --num_rollout_workers=32 \
    --refresh_every=3 \
    --max_turns=1 \
    --max_tokens=1024 \
    --epochs=1 \
    --max_steps=40 \
    --log_every=1 \
    --val_dataset_path=data/gsm8k_test.jsonl \
    --val_max_prompts=50 \
    --val_every=5 \
    --val_examples=5 \
    --run_name=gsm8k-service \
    "${MLFLOW_ARGS[@]}" \
    "$@"
