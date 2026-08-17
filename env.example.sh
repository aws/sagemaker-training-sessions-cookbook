# Copy this template to `.env` and fill in your account-specific values:
#   cp env.example.sh .env
# The launcher scripts (examples/rl_*.sh) auto-source `.env` if it exists,
# so a plain `bash examples/rl_gsm8k.sh` works after you fill this in once.
#
# .env is gitignored: keep account ids / ARNs out of source control.
# This env.example.sh template IS committed.

# --- SageMaker service configuration ------------------------------------------
# JumpStart hub-content ARN of the base model. Must match --base_model below.
export BASE_MODEL_ARN="arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/openai-reasoning-gpt-oss-20b/3.29.0"

# IAM execution role the training session assumes.
export ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/SageMakerRole"

# Model package group for checkpoints. Region-scoped; must match --region.
export MODEL_PACKAGE_GROUP_ARN="arn:aws:sagemaker:us-west-2:ACCOUNT_ID:model-package-group/cookbook-rl"

# S3 bucket + prefix for run outputs (metrics.jsonl, checkpoints, etc).
export OUTPUT_S3_PATH="s3://YOUR_BUCKET/training-sessions-cookbook/"

# --- Model / run-name defaults ------------------------------------------------
# The launchers use these when --base_model / --run_name are not passed.
# Override per-run with `BASE_MODEL=... bash examples/rl_gsm8k.sh`.
export BASE_MODEL="openai/gpt-oss-20b"

# --- Sandbox (Code RL only) ---------------------------------------------------
# Endpoint of the SandboxFusion container. Change if you moved the port or
# are pointing at a remote sandbox.
export SANDBOX_URL="http://localhost:8080/run_code"
export SANDBOX_MAX_CONCURRENCY="4"
