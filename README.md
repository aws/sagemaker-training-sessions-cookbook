# Training Sessions Cookbook

>Training recipes using SageMaker AI Training Sessions

A Training Session is a live, GPU-backed session you drive from your own
Python process: your loop decides each step (a forward/backward pass on a
batch, an optimizer step, sampling from the current policy, saving or loading
a checkpoint) while the platform does all the heavy lifting such as handling
the GPUs, loading the base model, registering LoRA adapters, and managing
checkpoints. It is built for authoring your own training
loop, rather than handing a config to a managed job. The loss function is an
argument to each training call: `cross_entropy` for supervised fine-tuning;
`ppo`, `importance_sampling`, or `cispo` for RL. One session and one API
drive either kind of training, and anything else you can express as a loop
over them.

The core of the API fits in a few lines:

```python
from sagemaker.train.training_session import ServiceClient, AdamParams

svc = ServiceClient(region=..., role_arn=..., model_package_group_arn=..., s3_output_path=...)
tc = svc.create_lora_training_client(base_model, rank=64, alpha=128)

# Train
for batch in batches:
    fb = tc.forward_backward(batch, loss_fn="cross_entropy")   # returns a future
    opt = tc.optim_step(AdamParams(learning_rate=5e-5))        # pipelined behind it
    fb.result(), opt.result()

# Save current weights and perform sampling
sc = tc.save_weights_and_get_sampling_client()
out = sc.sample(prompt, num_samples=8).result()                # Sample
logprobs = sc.compute_logprobs(prompt).result()                # Get token logprobs

# Or sample any saved checkpoint directly
sc = svc.create_sampling_client(model_package_arn=...)

# Checkpoint (resumable)
print(tc.save_state().result().model_package_arn)
```

The recipes here are complete training loops built on those calls, covering
batching, advantages, evaluation, checkpointing with resume, and metrics.
They are written against the `sagemaker-train` SDK's async APIs for training
and sampling.

## Recipes

| Recipe | Task | Reference result |
|---|---|---|
| [`chat_sft`](training_sessions_cookbook/recipes/chat_sft/) | Supervised fine-tuning on chat datasets (Tulu 3, UltraChat, local JSONL) | Tulu 3 NLL 1.23 → 0.65 in 400 steps |
| [`gsm8k_rl`](training_sessions_cookbook/recipes/gsm8k_rl/) | RL on grade-school math (GSM8K) | reward 0.71 → 0.90 in 40 steps |
| [`code_rl`](training_sessions_cookbook/recipes/code_rl/) | RL on competitive programming, graded by sandboxed execution | reward 0.46 → 0.60 in 50 steps |

Each recipe README has the exact commands, hyperparameters, and expected
metrics. Shell launchers for all of them live in [`examples/`](examples/).

## Quickstart

1. **Install**: Training Sessions support requires the `sagemaker-train`
   package from PyPI, then install this package:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install --upgrade sagemaker-train
   pip install -e .
   ```

   Or let the setup script do it (creates a conda env, installs the SDK, the
   package, and all extras): `bash tools/setup_env.sh && conda activate cookbook`.

   Optional extras: `plots` (analysis scripts), `code-rl` (sandboxed code
   execution), `mlflow` (metric publishing), `dev` (test suite).

2. **Configure your account.** Copy [`env.example.sh`](env.example.sh) to
   `.env` and fill in the base model's hub content ARN, an execution role, a
   model package group for checkpoints, and an S3 output path. All of them
   must be in the same region as `--region` (default `us-west-2`).

3. **Fetch a dataset** (only the GSM8K recipe needs this; the SFT and code-RL
   recipes download theirs at runtime):

   ```bash
   python tools/download_gsm8k.py   # writes data/gsm8k_{train,test}.jsonl
   ```

4. **Run a recipe.**

   ```bash
   bash examples/sft_tulu3.sh        # or:
   python -m training_sessions_cookbook.recipes.gsm8k_rl.train --help
   ```

## What a run produces

Every run writes a run folder containing:

- `config.json`: every argument plus the git hash, for reproducibility
- `metrics.jsonl`: one JSON line per training step
- `checkpoints.jsonl`: the model-package ARN of every saved checkpoint
- `training.log`: the full log

Checkpoints are SageMaker model packages, visible in the console like any
other. SFT runs resume from an interruption with
`--resume --log_path <run_folder>`; see the
[chat_sft README](training_sessions_cookbook/recipes/chat_sft/README.md) for
the checkpoint cadence options.

## Deploy to SageMaker Inference

Saved checkpoints can be deployed to
[SageMaker real-time endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html).
Since every checkpoint is a LoRA adapter over a shared base model, the natural
fit is [adapter inference components](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-adapt.html):
host the base model once as an inference component and attach each fine-tune
as an adapter component on the same endpoint.
[`notebooks/deploy_to_sagemaker_inference.ipynb`](notebooks/deploy_to_sagemaker_inference.ipynb)
demonstrates the full flow.

## MLflow publishing (optional)

Pass `--mlflow_tracking_uri` (a SageMaker managed MLflow App ARN or any plain
MLflow URI) and the run's config, per-step metrics, checkpoint ARNs, and
artifacts are published as they happen. RL runs can additionally publish
decoded rollouts as traces with `--mlflow_traces_per_step N`. Each trace is
named `rollout_step30_ep7`, carries its reward as a filterable assessment, and
is tagged with step/episode/reward/num_turns — so the Traces view can pull up
exactly the rollouts that scored nothing (`feedback.reward = '0.0'`) or one
step's worth (`tags.step = '30'`) and you can read the model's own reasoning
turn by turn. Traces are a sampling tool: keep `N` small (large values overflow
MLflow's export queue, which then discards traces) and use
`--record_trajectories` when every episode must be captured.

```bash
python -m training_sessions_cookbook.recipes.chat_sft.train ... \
  --mlflow_tracking_uri arn:aws:sagemaker:us-west-2:<acct>:mlflow-app/<name> \
  --mlflow_experiment_name my-experiments
```

Publishing is best-effort: if MLflow is unreachable or not installed, training
carries on and the local files are still written. On exit the MLflow run is
closed `FINISHED` or `FAILED` to match the outcome.

## Writing your own recipe

A new objective (DPO, reward modeling etc) is a new loop over
the same client calls, reusing the session setup, datum shape, checkpoint
manager, and run logging in
[`service/`](training_sessions_cookbook/service/). The
[recipes README](training_sessions_cookbook/recipes/README.md) describes the
layout and what a recipe folder needs; `recipes/gsm8k_rl/train.py` shows how
thin an entry point is. Run logging (and the MLflow publishing above) is three
lines:

```python
from training_sessions_cookbook.service.runlog import RunLog

with RunLog.open(run_name, config=vars(args), output_dir=args.log_path) as runlog:
    ...  # your training loop
    runlog.log_metrics(record, step=step)
```

## Repository layout

```
training_sessions_cookbook/
├── recipes/      # runnable entry points: chat_sft, gsm8k_rl, code_rl
├── sft/          # supervised fine-tuning library loop
├── rl/           # RL library loop: rollouts, advantages, envs, backends
└── service/      # shared plumbing: session, checkpoints, run logging
examples/         # shell launchers for every recipe
tools/            # env setup, dataset download, plotting
tests/            # unit tests (no AWS access needed)
```

## Development

```bash
pip install -e '.[dev]'
python -m pytest tests/    # no network or credentials needed
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, code style,
and pull-request process.
