#!/bin/bash
# Self-distillation policy optimization (SDPO) on DeepCoder: gpt-oss-20b, teacher = frozen self + sandbox feedback.
# Run from the repo root: bash examples/sdpo_code.sh
#
# One session: the teacher is the student's frozen base, shown sandbox test
# feedback on the attempt and a sibling rollout that passed (--teacher lagged
# re-snapshots the student instead, in a second session). Needs the
# code-rl extras and a SandboxFusion endpoint (see the code_rl recipe README).
#
# Model ARNs and hyperparameters come from the preset; account values from
# .env. Any flag passed to this script overrides the preset, e.g.
#   bash examples/sdpo_code.sh --max_steps 5 --batch_size 8
set -x
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/sdpo_code_$(date +%m%dT%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Auto-load account-specific config from .env if present (ROLE_ARN,
# MODEL_PACKAGE_GROUP_ARN, OUTPUT_S3_PATH). Copy env.example.sh -> .env and
# fill in your values once.
if [ -f .env ]; then
    # shellcheck disable=SC1091
    source .env
fi

export SANDBOX_URL="${SANDBOX_URL:-http://localhost:8080/run_code}"
export SANDBOX_MAX_CONCURRENCY="${SANDBOX_MAX_CONCURRENCY:-4}"

# Optional MLflow publishing (requires: pip install -e '.[mlflow]').
MLFLOW_ARGS=()
if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
    MLFLOW_ARGS+=(--mlflow_tracking_uri="$MLFLOW_TRACKING_URI")
    MLFLOW_ARGS+=(--mlflow_experiment_name="${MLFLOW_EXPERIMENT_NAME:-cookbook-distill}")
    [ -n "${MLFLOW_RUN_NAME:-}" ] && MLFLOW_ARGS+=(--mlflow_run_name="$MLFLOW_RUN_NAME")
fi

python -m training_sessions_cookbook.recipes.sdpo_code.train \
    --config=training_sessions_cookbook/recipes/sdpo_code/configs/sdpo_code_oss20b.json \
    --role_arn="$ROLE_ARN" \
    --model_package_group_arn="$MODEL_PACKAGE_GROUP_ARN" \
    --output_s3_path="$OUTPUT_S3_PATH" \
    "${MLFLOW_ARGS[@]}" \
    "$@"
