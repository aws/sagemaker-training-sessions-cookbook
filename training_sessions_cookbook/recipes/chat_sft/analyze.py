"""Analyze SFT training results: loss plots + before/after inference.

1. Visualize training and evaluation NLL loss (side-by-side subplots).
2. Optionally run before/after inference on sample prompts for a qualitative base-vs-finetuned comparison.

Usage:
    # Plot only (no session needed)
    python -m training_sessions_cookbook.recipes.chat_sft.analyze --metrics_path results/sft_service/metrics.jsonl

    # Plot + inference
    python -m training_sessions_cookbook.recipes.chat_sft.analyze \
        --metrics_path results/sft_service/metrics.jsonl \
        --model_package_arn arn:aws:sagemaker:us-west-2:ACCOUNT:model-package/NAME/9 \
        --run_inference --num_prompts 7
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import numpy as np
from sagemaker.train.training_session import SamplingParams, ServiceClient

from training_sessions_cookbook.sft.data import load_conversations


def load_metrics(path: str) -> list[dict]:
    metrics = []
    with open(path) as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def plot_losses(metrics: list[dict], output_path: str):
    steps = [m["step"] for m in metrics]
    train_nll = [m["train/mean_nll"] for m in metrics]

    eval_steps = [m["step"] for m in metrics if "eval/mean_nll" in m and not np.isnan(m["eval/mean_nll"])]
    eval_nll = [m["eval/mean_nll"] for m in metrics if "eval/mean_nll" in m and not np.isnan(m["eval/mean_nll"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(steps, train_nll, color="tab:blue", linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Training NLL Loss")
    ax1.grid(True, alpha=0.3)

    if eval_steps:
        ax2.plot(eval_steps, eval_nll, color="tab:orange", marker="o", markersize=4, linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("NLL Loss")
    ax2.set_title("Evaluation NLL Loss")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("SFT Training Loss", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def get_sample_prompts(dataset_path: str, num_prompts: int = 5, eval_split: str | None = None) -> list[dict]:
    """Pull sample conversations and their ground-truth final responses from the eval split."""
    conversations = load_conversations(dataset_path, split="eval", eval_size=2048, hf_split_name=eval_split)
    random.seed(0)
    random.shuffle(conversations)

    samples = []
    for conv in conversations:
        last_assistant_idx = max(
            (i for i, m in enumerate(conv) if m["role"] == "assistant"),
            default=-1,
        )
        context = conv[:last_assistant_idx]
        if last_assistant_idx > 0 and any(m["role"] == "user" for m in context):
            samples.append({
                "context": context,
                "ground_truth": conv[last_assistant_idx]["content"],
            })
        if len(samples) >= num_prompts:
            break
    return samples


def get_lora_config_from_checkpoint(
    model_package_arn: str, region: str
) -> dict:
    """Read rank and alpha from a checkpoint's adapter_config.json in S3.

    The ``sagemaker`` control-plane client auto-resolves its endpoint; S3 is
    regional and needs no extra routing.
    """
    session = boto3.Session(region_name=region)
    sm = session.client("sagemaker")
    s3 = session.client("s3")

    resp = sm.describe_model_package(ModelPackageName=model_package_arn)
    s3_uri = resp["InferenceSpecification"]["Containers"][0]["ModelDataSource"]["S3DataSource"]["S3Uri"]

    bucket = s3_uri.split("/")[2]
    prefix = "/".join(s3_uri.split("/")[3:])
    if not prefix.endswith("/"):
        prefix += "/"

    paginator = s3.get_paginator("list_objects_v2")
    config_key = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("adapter_config.json"):
                config_key = obj["Key"]
                break
        if config_key:
            break

    if not config_key:
        print(f"  Warning: adapter_config.json not found at {s3_uri}, using defaults")
        return {"r": 64, "lora_alpha": 128}

    obj = s3.get_object(Bucket=bucket, Key=config_key)
    config = json.loads(obj["Body"].read().decode())
    print(f"  Read from checkpoint: rank={config['r']}, alpha={config['lora_alpha']}")
    return config


def run_inference_comparison(
    model_package_arn: str,
    output_path: str,
    base_model_arn: str,
    model_package_group_arn: str,
    role_arn: str,
    output_s3_path: str,
    region: str = "us-west-2",
    dataset_path: str = "allenai/tulu-3-sft-mixture",
    num_prompts: int = 5,
    eval_split: str | None = None,
):
    sampling_params = SamplingParams(max_tokens=4096, temperature=0.6)
    samples = get_sample_prompts(dataset_path=dataset_path, num_prompts=num_prompts, eval_split=eval_split)

    lora_config = get_lora_config_from_checkpoint(model_package_arn, region)
    lora_rank = lora_config["r"]
    lora_alpha = lora_config["lora_alpha"]

    session_name = f"sft-analyze-{int(time.time())}"
    print(f"Creating session: {session_name}")
    svc = ServiceClient(region=region)
    training_client = svc.create_lora_training_client(
        training_session_name=session_name,
        base_model_arn=base_model_arn,
        model_package_group_arn=model_package_group_arn,
        role_arn=role_arn,
        s3_output_path=output_s3_path,
        rank=lora_rank,
        alpha=lora_alpha,
    )
    print(f"Session active: {session_name}")

    # Service-resolved tokenizer: identical to the training path's tokenizer.
    tokenizer = training_client.get_tokenizer()

    def format_context(context: list[dict]) -> list[int]:
        parts = []
        for msg in context:
            role = "user" if msg["role"] == "tool" else msg["role"]
            parts.append(f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return tokenizer.encode("".join(parts), add_special_tokens=False)

    def render_context_md(context: list[dict]) -> str:
        return "\n\n".join(f"**{m['role']}:** {m['content']}" for m in context)

    results = []
    try:
        print("Running base model inference...")
        base_sampling_client = training_client.create_sampling_client(model_package_arn="")
        for i, sample in enumerate(samples):
            prompt_tokens = format_context(sample["context"])
            response = base_sampling_client.sample(
                prompt=prompt_tokens,
                num_samples=1,
                sampling_params=sampling_params,
            )
            text = tokenizer.decode(response.samples[0].tokens, skip_special_tokens=True)
            results.append({
                "context": sample["context"],
                "ground_truth": sample["ground_truth"],
                "base_response": text,
            })
            print(f"  Base model: {i+1}/{len(samples)} done")

        print(f"Loading checkpoint {model_package_arn}...")
        training_client.load_weights_for_sampling(model_package_arn)
        ft_sampling_client = training_client.create_sampling_client(model_package_arn=model_package_arn)
        print("Running fine-tuned model inference...")
        for i, sample in enumerate(samples):
            prompt_tokens = format_context(sample["context"])
            response = ft_sampling_client.sample(
                prompt=prompt_tokens,
                num_samples=1,
                sampling_params=sampling_params,
            )
            text = tokenizer.decode(response.samples[0].tokens, skip_special_tokens=True)
            results[i]["finetuned_response"] = text
            print(f"  Fine-tuned: {i+1}/{len(samples)} done")

        with open(output_path, "w") as f:
            f.write("# SFT Before/After Comparison\n\n")
            for i, r in enumerate(results):
                f.write(f"## Conversation {i+1}\n\n")
                f.write(f"**Context:**\n\n{render_context_md(r['context'])}\n\n")
                f.write(f"**Ground truth:**\n{r['ground_truth']}\n\n")
                f.write(f"**Base model:**\n{r['base_response']}\n\n")
                f.write(f"**Fine-tuned:**\n{r['finetuned_response']}\n\n")
                f.write("---\n\n")
        print(f"Saved comparison to {output_path}")
    finally:
        print(f"Cleaning up session: {session_name}")
        svc.delete_session(session_name, stop_if_active=True)


def main():
    parser = argparse.ArgumentParser(description="Analyze SFT training results (service)")
    parser.add_argument("--metrics_path", type=str, required=True)
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument("--dataset_path", type=str, default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--eval_split", type=str, default=None, help="Explicit HF split name for eval (e.g. test_sft)")
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)

    # Session / model args (required for --run_inference)
    parser.add_argument("--model_package_arn", type=str, default=None)
    parser.add_argument("--base_model_arn", type=str, default=None)
    parser.add_argument("--model_package_group_arn", type=str, default=None)
    parser.add_argument("--role_arn", type=str, default=None)
    parser.add_argument("--output_s3_path", type=str, default=None)
    parser.add_argument("--region", type=str, default="us-west-2")
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.metrics_path).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.metrics_path)
    plot_path = os.path.join(output_dir, "sft_loss_plot.png")
    plot_losses(metrics, plot_path)

    print(f"\nTraining summary ({len(metrics)} steps):")
    print(f"  Initial train NLL: {metrics[0]['train/mean_nll']:.4f}")
    print(f"  Final train NLL:   {metrics[-1]['train/mean_nll']:.4f}")
    eval_metrics = [m for m in metrics if "eval/mean_nll" in m and not np.isnan(m["eval/mean_nll"])]
    if eval_metrics:
        print(f"  Initial eval NLL:  {eval_metrics[0]['eval/mean_nll']:.4f}")
        print(f"  Final eval NLL:    {eval_metrics[-1]['eval/mean_nll']:.4f}")

    if args.run_inference:
        missing = [
            f"--{k}" for k in ("model_package_arn", "base_model_arn", "model_package_group_arn", "role_arn", "output_s3_path")
            if not getattr(args, k)
        ]
        if missing:
            print("\nError: --run_inference requires: " + ", ".join(missing))
            return
        comparison_path = os.path.join(output_dir, "sft_comparison.md")
        run_inference_comparison(
            model_package_arn=args.model_package_arn,
            output_path=comparison_path,
            base_model_arn=args.base_model_arn,
            model_package_group_arn=args.model_package_group_arn,
            role_arn=args.role_arn,
            output_s3_path=args.output_s3_path,
            region=args.region,
            dataset_path=args.dataset_path,
            num_prompts=args.num_prompts,
            eval_split=args.eval_split,
        )


if __name__ == "__main__":
    main()
