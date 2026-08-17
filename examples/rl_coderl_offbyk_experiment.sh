#!/bin/bash
# ============================================================================
# codeRL off-by-k experiment harness (SageMaker Training Sessions PySDK).
#
# This is the experiment counterpart to the basic examples/rl_coderl.sh
# launcher. It runs the SAME codeRL recipe (DeepCoder + LCB grading via
# SandboxFusion) but is built to compare an on-policy CONTROL against an
# off-by-k TREATMENT: it is config-driven (cispo loss + clip thresholds), and
# a MODE toggle selects the loop. Use rl_coderl.sh for a plain on-policy run;
# use this when you want the drop/requeue off-by-k staleness comparison.
#
# All hyperparameters come from training_sessions_cookbook/recipes/code_rl/configs/coderl_oss20b_service.json:
#   env_type code_rl, batch_size 128, responses_per_prompt 8, loss cispo,
#   clip_low/high_threshold 1.0/6.0, max_turns 2, max_tokens 12288,
#   reasoning medium, template gpt_oss, val_code_split test, advantage_mode
#   grouped, refresh_every 1, num_rollout_workers 64, gpt-oss-20b, LoRA r32/a64.
# The service fields (region us-west-2, base_model_arn, role_arn,
# model_package_group_arn, output_s3_path) also live in that config.
# This script overrides ONLY the loop-mode block + run_name. Explicit CLI flags
# win over the JSON (rl/train.py::_apply_config_defaults).
#
# ===== The experiment: on-policy control vs off-by-k treatment =====
# One axis differs between the two variants:
#
#   MODE=onpolicy (CONTROL): --max_steps_off_policy 0
#     Strict serial barrier loop (run_training): refresh -> collect ALL rollouts
#     -> train -> repeat. Sampling and the gradient step never overlap.
#
#   MODE=offbyk  (TREATMENT): --max_steps_off_policy K
#     Overlapped bounded-staleness loop (run_training_async): rollout workers
#     generate continuously while the trainer trains; a group sampled against
#     sampler version t may be consumed up to K steps later (t+K), its shift
#     corrected by the cispo IS ratio. Groups staler than K are dropped.
#
# Run BOTH, then overlay:
#   python -m training_sessions_cookbook.rl.analysis.plot_runs \
#     coderl-oss20b-service-onpolicy_<TS> coderl-oss20b-service-offbyk-k3_<TS>
#
# Select the variant with MODE (default offbyk) and the staleness cap with
# MAX_STEPS_OFF_POLICY (default 3):
#   MODE=onpolicy bash examples/rl_coderl_offbyk_experiment.sh          # control
#   MODE=offbyk MAX_STEPS_OFF_POLICY=3 bash examples/rl_coderl_offbyk_experiment.sh
#
# ============================================================================
# ===== PREFLIGHT (do these first) =====
#
# TARGET REGION: us-west-2 (set in training_sessions_cookbook/recipes/code_rl/configs/coderl_oss20b_service.json).
#
# 1. AWS creds. The service backend needs outbound HTTPS to the SageMaker
#    endpoints and IAM creds resolving to account ACCOUNT_ID (the account that
#    owns the role/MPG/bucket in the config): NOT pod reachability.
#
# 2. Confirm the base model exists in us-west-2 before running:
#        python -c "from sagemaker.train.training_session import ServiceClient; \
#          print(ServiceClient(region='us-west-2').list_base_models())"
#    (The cispo loss, the loss_fn_config clip [1,6], and the off-by-k async path
#    are treated as already-supported/tested on the service.)
#
# 3. Install the PySDK + code-RL deps:
#        pip install sagemaker-train-*.whl aiohttp datasets numpy
#
# 4. SandboxFusion container for code grading, reachable at $SANDBOX_URL:
#        docker run -d -p 8080:8080 volcengine/sandbox-fusion:server-20250609
#        curl -sS -o /dev/null -w "sandbox HTTP %{http_code}\n" \
#          http://localhost:8080/run_code
#    Grading runs client-side on the launcher (same as the other runs), so the
#    sandbox is needed regardless of backend.
#
# ===== RECOMMENDED: smoke first (service scheduling can be slow at full scale) =
#   MODE=onpolicy MAX_PROMPTS=16 bash examples/rl_coderl_offbyk_experiment.sh \
#     --batch_size 16 --max_steps 2 --no-val_at_start
#   Confirms: session schedules (not stuck Pending), cispo loads, clip [1,6]
#   applied (no "does not accept loss_fn_config" warning), sandbox grades. Then
#   launch the full runs below.
# ============================================================================

set -x

# ----- Activate the conda env if not already active (self-contained) --------
# So the script works from a bare tmux/SSM shell. No-op if already in the env.
if [ "${CONDA_DEFAULT_ENV:-}" != "cookbook" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null && conda activate cookbook
fi

# ----- SandboxFusion (client-side code grading; same as rl_coderl.sh) -----
export SANDBOX_URL="${SANDBOX_URL:-http://localhost:8080/run_code}"
export SANDBOX_MAX_CONCURRENCY="${SANDBOX_MAX_CONCURRENCY:-16}"

# ----- Op poll ceiling for the service SDK futures --------------------------
# Bounds each .result()/result_async() poll so a server-side stall fails fast
# instead of hanging forever. codeRL steps are long (sample-bound); give plenty
# of headroom (default in service.py is 600s). Raise if steps legitimately run
# longer than this.
export OP_TIMEOUT_SECONDS="${OP_TIMEOUT_SECONDS:-1800}"

# ----- Experiment mode ------------------------------------------------------
# MODE=onpolicy -> strict serial control; MODE=offbyk -> overlapped treatment.
MODE="${MODE:-offbyk}"
MAX_STEPS_OFF_POLICY="${MAX_STEPS_OFF_POLICY:-3}"   # staleness cap K (offbyk only)

# NOTE: run_name must be <=29 characters due to service naming constraints.
# Longer names can leave the session stuck in Pending -> TimeoutError.
case "$MODE" in
  onpolicy)
    # CONTROL: serial barrier; max_steps_off_policy 0 selects run_training.
    LOOP_ARGS=(--max_steps_off_policy 0)
    RUN_NAME="${RUN_NAME:-coderl-svc-onpolicy}"       # 19 chars, within limit
    ;;
  offbyk)
    # TREATMENT: overlapped bounded-staleness loop (run_training_async).
    LOOP_ARGS=(--max_steps_off_policy "$MAX_STEPS_OFF_POLICY")
    RUN_NAME="${RUN_NAME:-coderl-svc-offbyk-k${MAX_STEPS_OFF_POLICY}}"  # 20 chars, within limit
    ;;
  *)
    echo "MODE must be 'onpolicy' or 'offbyk' (got '$MODE')" >&2
    exit 2
    ;;
esac

# Optional training-set cap for a fast smoke (e.g. MAX_PROMPTS=384). Unset =
# full DeepCoder corpus.
EXTRA_ARGS=()
if [ -n "${MAX_PROMPTS:-}" ]; then
    EXTRA_ARGS+=(--max_prompts "$MAX_PROMPTS")
fi

python -m training_sessions_cookbook.recipes.code_rl.train \
  --config training_sessions_cookbook/recipes/code_rl/configs/coderl_oss20b_service.json \
  --backend service \
  --run_name "$RUN_NAME" \
  "${LOOP_ARGS[@]}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
  "$@"
