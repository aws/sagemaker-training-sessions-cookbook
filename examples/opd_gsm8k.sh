#!/bin/bash
# On-policy distillation (OPD) on GSM8K: Qwen3.6-27B, teacher = its own frozen
# base sharpened to temperature 0.5 (--teacher_temperature).
# Run from the repo root: bash examples/opd_gsm8k.sh
#
# One session. For a stronger teacher, pass --teacher checkpoint
# --teacher_model_package_arn <adapter> (e.g. a gsm8k_rl run; opens a second
# session on the same base), or, once the service supports a second model,
# --teacher session --teacher_base_model_arn.
# Needs data/gsm8k_{train,test}.jsonl: python tools/download_gsm8k.py
#
# Model ARNs and hyperparameters come from the preset; account values from
# .env. Any flag passed to this script overrides the preset, e.g.
#   bash examples/opd_gsm8k.sh --max_steps 5 --batch_size 8
set -x
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/opd_gsm8k_$(date +%m%dT%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Auto-load account-specific config from .env if present (ROLE_ARN,
# MODEL_PACKAGE_GROUP_ARN, OUTPUT_S3_PATH). Copy env.example.sh -> .env and
# fill in your values once.
if [ -f .env ]; then
    # shellcheck disable=SC1091
    source .env
fi

# Optional MLflow publishing (requires: pip install -e '.[mlflow]').
MLFLOW_ARGS=()
if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
    MLFLOW_ARGS+=(--mlflow_tracking_uri="$MLFLOW_TRACKING_URI")
    MLFLOW_ARGS+=(--mlflow_experiment_name="${MLFLOW_EXPERIMENT_NAME:-cookbook-distill}")
    [ -n "${MLFLOW_RUN_NAME:-}" ] && MLFLOW_ARGS+=(--mlflow_run_name="$MLFLOW_RUN_NAME")
fi

python -m training_sessions_cookbook.recipes.opd_math.train \
    --config=training_sessions_cookbook/recipes/opd_math/configs/opd_gsm8k_qwen3_6_27b.json \
    --role_arn="$ROLE_ARN" \
    --model_package_group_arn="$MODEL_PACKAGE_GROUP_ARN" \
    --output_s3_path="$OUTPUT_S3_PATH" \
    "${MLFLOW_ARGS[@]}" \
    "$@"
