#!/bin/bash
# Basic codeRL launcher: RL on a SageMaker Training Session, Code RL
# (DeepCoder + LiveCodeBench grading via SandboxFusion), multi-turn with the
# gpt-oss tool-use loop. This is the straightforward "just run codeRL"
# example: on-policy, all hyperparameters as explicit CLI flags, importance-
# sampling loss. For the off-by-k (drop/requeue) staleness experiment with a
# control/treatment MODE toggle, see examples/rl_coderl_offbyk_experiment.sh.
# Run from the repo root: bash examples/rl_coderl.sh
#
# Prerequisites:
#   1. SandboxFusion running locally. Start it with:
#        bash tools/sandbox_up.sh
#      The default endpoint is http://localhost:8080/run_code and is
#      configurable via SANDBOX_URL below.
#   2. Extras installed:  pip install -e '.[code-rl]'
#      (adds aiohttp + datasets; the GSM8K path does not need them).
#   3. Multi-turn (--max_turns > 1) only supports the gpt_oss family today.
#      Keep --base_model on a gpt-oss-* model, or set --max_turns 1 to run
#      the single-turn path (which works on any base model).
set -x
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/rl_coderl_$(date +%m%dT%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Auto-load account-specific config from .env if present (BASE_MODEL_ARN,
# ROLE_ARN, MODEL_PACKAGE_GROUP_ARN, OUTPUT_S3_PATH, SANDBOX_URL, etc).
# Copy .env.example -> .env and fill in your values once.
if [ -f .env ]; then
    # shellcheck disable=SC1091
    source .env
fi

# SandboxFusion endpoint. Point at a remote sandbox by exporting SANDBOX_URL.
export SANDBOX_URL="${SANDBOX_URL:-http://localhost:8080/run_code}"
export SANDBOX_MAX_CONCURRENCY="${SANDBOX_MAX_CONCURRENCY:-4}"

# DeepCoder split subset: leave unset to load the full corpus:
#   train: primeintellect + taco + lcbv5
#   test:  codeforces + lcbv5
# For memory-constrained hosts (<32 GB RAM), export DEEPCODER_SPLITS=lcbv5
# before running to load just the smaller lcbv5 config for both splits.
export DEEPCODER_SPLITS="${DEEPCODER_SPLITS:-}"

# SageMaker configuration: override via env for your account.
export TRAINING_SESSION_NAME="${TRAINING_SESSION_NAME:-}"
export BASE_MODEL="${BASE_MODEL:-openai/gpt-oss-20b}"
export BASE_MODEL_ARN="${BASE_MODEL_ARN:-}"
export MODEL_PACKAGE_GROUP_ARN="${MODEL_PACKAGE_GROUP_ARN:-}"
export ROLE_ARN="${ROLE_ARN:-}"
export OUTPUT_S3_PATH="${OUTPUT_S3_PATH:-}"

# HPs mirror the reference DeepCoder recipe README:
#   Qwen3-4B-Instruct-2507, group_size=8, groups_per_batch=128,
#   lr=4e-5, lora_rank=32, max_tokens=24576, max_turns=2, format_coef=0.1,
#   code_timeout=6, context_overflow_reward=-0.1, max_trajectory_tokens=32*1024,
#   kl=0, save_every=20.
# Base model differs: the multi-turn tool renderer only supports gpt-oss today,
# so we default to gpt-oss-20b. Use --max_turns=1 to run single-turn on Qwen.
python -m training_sessions_cookbook.recipes.code_rl.train \
    --backend=service \
    --region=us-west-2 \
    --training_session_name="$TRAINING_SESSION_NAME" \
    --base_model="$BASE_MODEL" \
    --base_model_arn="$BASE_MODEL_ARN" \
    --model_package_group_arn="$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn="$ROLE_ARN" \
    --output_s3_path="$OUTPUT_S3_PATH" \
    --env_type=code_rl \
    --template=gpt_oss \
    --lora_rank=32 \
    --lora_alpha=64 \
    --learning_rate=4e-5 \
    --loss=importance_sampling \
    --advantage_mode=grouped \
    --batch_size=128 \
    --responses_per_prompt=8 \
    --num_rollout_workers=32 \
    --refresh_every=1 \
    --max_turns=2 \
    --max_tokens=24576 \
    --code_timeout=6 \
    --format_coef=0.1 \
    --context_overflow_reward=-0.1 \
    --max_trajectory_tokens=32768 \
    --kl_coef=0.0 \
    --epochs=1 \
    --max_steps=100 \
    --save_every=20 \
    --log_every=1 \
    --run_name=code-rl-parity \
    "$@"
