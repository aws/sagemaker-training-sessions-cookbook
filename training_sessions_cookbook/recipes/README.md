# Recipes

Each recipe wires a task onto the shared library loops and is launched the
same way: `python -m training_sessions_cookbook.recipes.<name>.train`.

| Recipe | Task | Library loop |
|---|---|---|
| [`chat_sft/`](chat_sft/) | Supervised fine-tuning on chat datasets | `sft/` |
| [`gsm8k_rl/`](gsm8k_rl/) | RL on grade-school math (GSM8K) | `rl/` |
| [`code_rl/`](code_rl/) | RL on competitive programming, sandbox-graded | `rl/` |
| [`preference_kto/`](preference_kto/) | KTO preference alignment from binary labels | `preference/` |
| [`opd_math/`](opd_math/) | On-policy distillation (GSM8K; same-model tempered teacher, or a checkpoint / second-model teacher) | `distill/` over `rl/` |
| [`opsd_math/`](opsd_math/) | On-policy self-distillation with reference solutions (GSM8K) | `distill/` over `rl/` |
| [`sdpo_code/`](sdpo_code/) | Self-distillation from sandbox feedback (code) | `distill/` over `rl/` |

Every run writes `{run_folder}/metrics.jsonl` (one line per step) and
`{run_folder}/checkpoints.jsonl` (the model-package ARN of every saved
checkpoint). SFT runs resume from an interruption with
`--resume --log_path <run_folder>`.

## Adding a recipe

Create a folder here with a `train.py` over one of the library loops, a README
with runnable commands and expected results, and any recipe-specific configs.
Each entry builds a parser restricted to its own surface (see
`gsm8k_rl/train.py`):

```python
from training_sessions_cookbook.rl.train import build_parser, parse_with_config, run


def main() -> None:
    parser = build_parser(envs=("my_task",))  # single env: also the default
    args = parse_with_config(parser)
    run(args)


if __name__ == "__main__":
    main()
```

**Bring your own environment.** Register a class implementing the small `Env`
protocol (`rl/envs/base.py`) and the whole RL loop (rollouts, advantages,
off-by-k, metrics, checkpointing) works unchanged. An env can also ship its
own CLI flags:

```python
from training_sessions_cookbook.rl import envs

envs.register_env(
    "my_task",
    lambda args: (lambda: MyEnv(args.dataset_path)),
    add_cli_args=lambda p: p.add_argument("--my_knob", type=int, default=7),
)
```

**Run programmatically** (a notebook, a sweep script) by constructing args and
calling the loop directly, with no CLI involved:

```python
args = parse_with_config(build_parser(envs=("gsm8k",)),
                         argv=["--base_model", "openai/gpt-oss-20b"])
args.base_model_arn = "..."; args.role_arn = "..."; args.max_steps = 10
run(args)            # scripts
await run_async(args)  # notebooks / async code (Jupyter already runs a loop)
```

`run` also takes optional overrides: `env_factory=` (bypass the registry) and
`validation_runner=` (any object with the `ValidationRunner.run_validation`
signature) to plug in your own evaluator. The SFT loop's `run` accepts
`train_batches=` / `eval_batches=` (a list, or a callable `(tokenizer) ->
batches`) to bypass dataset loading entirely.
