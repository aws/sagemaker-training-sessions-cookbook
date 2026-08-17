#!/bin/bash
# Code RL SMOKE run: tiny config to measure per-step timing (sampling vs
# training) and validate the pipeline end-to-end WITHOUT a multi-hour data-prep
# or rollout phase.
#
# What this is for:
#   * Estimate average time/sample_seconds and time/train_seconds per step from
#     runs/<run>/metrics.jsonl (log_every=1 writes every step).
#   * Confirm the training session does NOT sit idle long enough to be reaped
#     (the full run's failure mode). With the capped dataset + small rollout
#     batch, step 0 reaches the first forward_backward quickly.
#
# Why it's fast:
#   * DEEPCODER_SPLITS=lcbv5 loads only the smallest split.
#   * --max_prompts=64 now caps the ROWS READ during data prep (the improved
#     loader reads a seeded subset in file order instead of the whole corpus).
#   * batch_size=8 x responses_per_prompt=2 = 16 rollouts/step (vs 1024 in full).
#   * --max_steps=2 so you get two timed steps and stop.
#
# Prerequisites (same as examples/rl_coderl.sh):
#   1. SandboxFusion reachable at $SANDBOX_URL (bash tools/sandbox_up.sh).
#   2. pip install -e '.[code-rl]'.
#   3. Service creds on the EC2 box; multi-turn (--max_turns>1) is gpt-oss only.
#
# Run from the repo root:  bash examples/rl_code_rl_smoke.sh
set -x
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/rl_code_rl_smoke_$(date +%m%dT%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Auto-load account-specific config from .env if present.
if [ -f .env ]; then
    # shellcheck disable=SC1091
    source .env
fi

# SandboxFusion endpoint.
export SANDBOX_URL="${SANDBOX_URL:-http://localhost:8080/run_code}"
export SANDBOX_MAX_CONCURRENCY="${SANDBOX_MAX_CONCURRENCY:-4}"

# Smallest DeepCoder split for a fast smoke. Override to "" for the full corpus.
export DEEPCODER_SPLITS="${DEEPCODER_SPLITS:-lcbv5}"

# SageMaker config: defaults target the same resources as the full run.
# Override any of these via env (or .env) for your account.
export TRAINING_SESSION_NAME="${TRAINING_SESSION_NAME:-}"
export REGION="${REGION:-us-west-2}"
export BASE_MODEL="${BASE_MODEL:-openai/gpt-oss-20b}"
export BASE_MODEL_ARN="${BASE_MODEL_ARN:-arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/openai-reasoning-gpt-oss-20b/3.29.0}"
export ROLE_ARN="${ROLE_ARN:-arn:aws:iam::ACCOUNT_ID:role/SageMakerRole}"
export MODEL_PACKAGE_GROUP_ARN="${MODEL_PACKAGE_GROUP_ARN:-arn:aws:sagemaker:us-west-2:ACCOUNT_ID:model-package-group/cookbook-rl}"
export OUTPUT_S3_PATH="${OUTPUT_S3_PATH:-s3://YOUR_BUCKET/training-sessions-cookbook/}"

# Tiny config. max_tokens kept moderate (8192) so a smoke step is quick; note
# sampling time scales with max_tokens, so extrapolate to the full run's 24576
# accordingly. Bump --max_turns to 2 to mirror the full multi-turn config.
python -m training_sessions_cookbook.recipes.code_rl.train \
    --backend=service \
    --region="$REGION" \
    --training_session_name="$TRAINING_SESSION_NAME" \
    --base_model="$BASE_MODEL" \
    --base_model_arn="$BASE_MODEL_ARN" \
    --model_package_group_arn="$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn="$ROLE_ARN" \
    --output_s3_path="$OUTPUT_S3_PATH" \
    --no-save_weights_at_end \
    --no-deploy_and_invoke \
    --env_type=code_rl \
    --template=gpt_oss \
    --lora_rank=32 \
    --lora_alpha=64 \
    --learning_rate=4e-5 \
    --loss=importance_sampling \
    --advantage_mode=grouped \
    --batch_size=8 \
    --responses_per_prompt=2 \
    --num_rollout_workers=8 \
    --refresh_every=1 \
    --max_turns=1 \
    --max_tokens=8192 \
    --code_timeout=6 \
    --format_coef=0.1 \
    --context_overflow_reward=-0.1 \
    --max_trajectory_tokens=32768 \
    --kl_coef=0.0 \
    --epochs=1 \
    --max_prompts=64 \
    --max_steps=2 \
    --save_every=2 \
    --log_every=1 \
    --run_name=code-rl-smoke \
    --seed=0 \
    --val_code_split=test \
    --val_max_prompts=16 \
    --val_every=2 \
    --val_examples=3 \
    --val_at_start \
    "$@"

# --val_code_split=test uses the REAL DeepCoder held-out split (codeforces+lcbv5),
# graded with the same SandboxFusion harness as training, so val/reward_mean is
# comparable to rollout/reward_mean. Drop it (or set =none) to skip validation.
echo "Smoke done."
echo "  Per-step timing:  runs/code-rl-smoke_*/metrics.jsonl (time/sample_seconds, time/train_seconds)"
echo "  Validation:       val/reward_mean, val/accuracy in the same metrics.jsonl (steps 0 and 2)"
echo "  Val examples:     runs/code-rl-smoke_*/validation_examples.jsonl"
