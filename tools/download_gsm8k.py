#!/usr/bin/env python3
"""Download the full GSM8K dataset and save as JSONL.

This script downloads the GSM8K dataset from HuggingFace and writes both splits: the training split (7,473 problems) to data/gsm8k_train.jsonl and the test split (1,319 problems) to data/gsm8k_test.jsonl. The test split can be passed to ``rl.train`` via ``--val_dataset_path`` to enable periodic validation and qualitative example logging.

Requirements:
    pip install datasets

Usage:
    python tools/download_gsm8k.py

To regenerate the full dataset:
    1. Ensure the `datasets` package is installed: pip install datasets
    2. Run this script from the repo root:
       python tools/download_gsm8k.py
    3. The script will download from HuggingFace and write data/gsm8k_train.jsonl

The output file contains one JSON object per line with fields:
    - question: The math word problem text
    - answer: The step-by-step solution ending with "#### <numeric_answer>"
"""

import json
import os

from datasets import load_dataset


def main():
    # Determine output path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(project_dir, "data")
    output_path = os.path.join(output_dir, "gsm8k_train.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading GSM8K from HuggingFace...")
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")

    train_path = output_path
    test_path = os.path.join(output_dir, "gsm8k_test.jsonl")

    for dataset, path in ((train_dataset, train_path), (test_dataset, test_path)):
        print(f"Writing {len(dataset)} problems to {path}...")
        with open(path, "w") as f:
            for record in dataset:
                line = json.dumps(
                    {"question": record["question"], "answer": record["answer"]},
                    ensure_ascii=False,
                )
                f.write(line + "\n")
        print(f"Done. Wrote {len(dataset)} records to {path}")


if __name__ == "__main__":
    main()
