# Code RL (DeepCoder)

Policy-gradient RL on competitive-programming problems: generated solutions
are executed against unit tests in a local sandbox and rewarded on pass rate.
Needs the `code-rl` extra and a running sandbox (`bash tools/sandbox_up.sh`).

## Run

Set up the environment and account config first (see the [repo README](../../../README.md)), then:

```bash
python -m training_sessions_cookbook.recipes.code_rl.train \
    --config training_sessions_cookbook/recipes/code_rl/configs/coderl_oss20b_service.json \
    --base_model_arn "$BASE_MODEL_ARN" \
    --model_package_group_arn "$MODEL_PACKAGE_GROUP_ARN" \
    --role_arn "$ROLE_ARN" \
    --output_s3_path "$OUTPUT_S3_PATH"
```

Launchers: [`examples/rl_coderl.sh`](../../../examples/rl_coderl.sh) (plain
on-policy) and [`examples/rl_coderl_offbyk_experiment.sh`](../../../examples/rl_coderl_offbyk_experiment.sh)
(on-policy control vs off-by-k treatment comparison).

## Expected results

| Model | Steps | `rollout/reward_mean` |
|---|---|---|
| gpt-oss-20b | 50 | 0.46 → 0.60 |

## How it works

Shares the training loop with the [GSM8K-RL recipe](../gsm8k_rl/README.md);
see that README for the loop mechanics (rollouts, advantages, sampler
refresh, off-by-k, validation). This recipe adds:

- **Sandboxed grading**: `rl/envs/code_rl.py` submits generated code to a
  SandboxFusion container and scores unit-test pass rate.
- **Multi-turn tool rendering**: `rl/envs/tool_render.py` formats execution
  feedback for follow-up turns.
- **DeepCoder dataset**: `rl/envs/deepcoder.py` loads and filters the corpus.
