"""Data loading and tokenization for SFT via the SageMaker Training Session API.

The tokenizer is the one the service resolves for the base model (``training_client.get_tokenizer()``), so client-side and service-side tokenization cannot diverge. The renderer emits the qwen3 chat layout (``<|im_start|>`` headers, an empty ``<think>`` block for the final assistant turn) as raw text and masks loss to assistant-response tokens only.

Each conversation becomes a plain datum dict the PySDK passes through:
  {"modelInput": {"chunks": [{"tokens": [...], "type": "encoded_text"}]},
   "lossFnInputs": {"target_tokens": {...}, "weights": {...}}}
"""

from __future__ import annotations

import logging
import random

import datasets
import numpy as np

from training_sessions_cookbook.service import make_datum, read_jsonl

logger = logging.getLogger(__name__)


# === Dataset loader ===
#
# Load raw conversations from a HuggingFace dataset or a local JSONL file, resolving the train/eval split (dedicated split when present, otherwise a seeded carve-out of the train set).


def load_conversations(
    dataset_path: str,
    split: str = "train",
    eval_size: int | None = None,
    eval_ratio: float = 0.1,
    max_rows: int | None = None,
    hf_split_name: str | None = None,
) -> list[list[dict]]:
    """Load conversations from a HuggingFace dataset or a local JSONL file.

    When the dataset has no dedicated eval split, the train set is shuffled (seed=0) and split. Use eval_size for a fixed count, or eval_ratio (default 0.1) for a proportional split.

    hf_split_name: explicit HF split name to use (e.g. "train_sft", "test_sft"). When set, bypasses the automatic split resolution logic.
    """
    def _eval_count(total: int) -> int:
        if eval_size is not None:
            return min(eval_size, total)
        return max(1, int(total * eval_ratio))

    if dataset_path.endswith(".jsonl"):
        conversations = []
        for entry in read_jsonl(dataset_path):
            messages = entry.get("messages") or entry.get("conversations")
            if messages:
                conversations.append(messages)

        if split in ("train", "eval"):
            random.seed(0)
            random.shuffle(conversations)
            n_eval = _eval_count(len(conversations))
            if split == "eval":
                conversations = conversations[:n_eval]
            else:
                conversations = conversations[n_eval:]

        logger.info(f"Loaded {len(conversations)} conversations from {dataset_path} (split={split})")
        return conversations

    ds = datasets.load_dataset(dataset_path)
    if isinstance(ds, datasets.DatasetDict):
        if hf_split_name and hf_split_name in ds:
            hf_split = ds[hf_split_name]
            if split == "train":
                hf_split = hf_split.shuffle(seed=0)
        elif split in ds:
            hf_split = ds[split]
            if split == "train":
                hf_split = hf_split.shuffle(seed=0)
        else:
            train_key = hf_split_name if (hf_split_name and hf_split_name in ds) else "train"
            full = ds[train_key].shuffle(seed=0)
            n_eval = _eval_count(len(full))
            if split == "eval":
                hf_split = full.select(range(n_eval))
            else:
                hf_split = full.select(range(n_eval, len(full)))
    else:
        hf_split = ds

    # Cap a dedicated eval split to eval_size (the carve-from-train path already applies this via _eval_count).
    if split == "eval" and eval_size is not None and len(hf_split) > eval_size:
        hf_split = hf_split.select(range(eval_size))

    if max_rows is not None and len(hf_split) > max_rows:
        hf_split = hf_split.select(range(max_rows))

    conversations = []
    for row in hf_split:
        messages = row.get("messages") or row.get("conversations")
        if messages:
            conversations.append(messages)
    logger.info(f"Loaded {len(conversations)} conversations from HF:{dataset_path}/{split}")
    return conversations


# === Datum builder ===
#
# Render each conversation into the qwen3 chat layout, mask loss to the assistant-response tokens, and shape the result into a service datum dict.


def _render_supervised_tokens(
    messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[float]]:
    """Render a conversation into (tokens, weights) using the qwen3 layout."""
    last_user_idx = max(
        (i for i, m in enumerate(messages) if m["role"] == "user"),
        default=-1,
    )

    tokens: list[int] = []
    weights: list[float] = []

    def _emit(text: str, weight: float) -> None:
        chunk = tokenizer.encode(text, add_special_tokens=False)
        tokens.extend(chunk)
        weights.extend([weight] * len(chunk))

    for idx, message in enumerate(messages):
        role = "user" if message["role"] == "tool" else message["role"]
        is_assistant = role == "assistant"

        maybe_newline = "\n" if idx > 0 else ""
        header_str = f"{maybe_newline}<|im_start|>{role}\n"

        if is_assistant and idx > last_user_idx:
            content = message.get("content", "")
            has_think = isinstance(content, str) and "<think>" in content
            if not has_think:
                header_str += "<think>\n\n</think>\n\n"

        _emit(header_str, 0.0)

        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Only string message content is supported")
        _emit(content + "<|im_end|>", 1.0 if is_assistant else 0.0)

    return tokens, weights


def conversation_to_datum(
    messages: list[dict],
    tokenizer,
    max_length: int,
) -> dict | None:
    """Convert a single conversation to a datum dict for the service API.

    Returns None if the conversation is too short or has no trainable (assistant-response) tokens after masking.
    """
    tokens, weights = _render_supervised_tokens(messages, tokenizer)

    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
        weights = weights[:max_length]

    if len(tokens) < 2:
        return None

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights_arr = np.array(weights[1:], dtype=np.float32)

    weight_sum = weights_arr.sum()
    if weight_sum <= 0:
        return None
    weights_arr = weights_arr / weight_sum

    return make_datum(input_tokens, target_tokens, weights_arr.tolist())


# === Batching ===
#
# Tokenize every conversation into a datum and pack them into fixed-size contiguous batches for the training loop.


def build_sft_batches(
    dataset_path: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    split: str = "train",
    eval_size: int | None = None,
    eval_ratio: float = 0.1,
    max_conversations: int | None = None,
    hf_split_name: str | None = None,
) -> list[list[dict]]:
    """Load dataset, tokenize all conversations, and split into batches.

    ``tokenizer`` is the service-resolved tokenizer (``training_client.get_tokenizer()``).
    """
    conversations = load_conversations(dataset_path, split=split, eval_size=eval_size, eval_ratio=eval_ratio, max_rows=max_conversations, hf_split_name=hf_split_name)

    datums = []
    skipped = 0
    for conv in conversations:
        try:
            datum = conversation_to_datum(conv, tokenizer, max_length)
        except Exception:
            datum = None
        if datum is not None:
            datums.append(datum)
        else:
            skipped += 1

    if skipped > 0:
        logger.info(f"Skipped {skipped} conversations (too short or no assistant content)")
    logger.info(f"Tokenized {len(datums)} conversations into datums")

    batches = []
    for i in range(0, len(datums), batch_size):
        batch = datums[i : i + batch_size]
        if batch:
            batches.append(batch)

    return batches
