"""Data loading and datum shaping for KTO preference training.

KTO consumes an **unpaired** signal: each row is a ``(prompt, completion,
label)`` where ``label`` marks the completion desirable (True) or undesirable
(False). This module loads such rows (HuggingFace datasets in the
``trl-lib/kto-mix-14k`` shape, or a local JSONL), renders them into the qwen3
chat layout with the same piecewise tokenization the SFT path uses, and packs
them into batches of :class:`KtoExample`.

Each example carries its prompt and completion token ids separately so the
training loop can compose three datums from them (see the functions at the
bottom):

* **matched** ``(x_i, y_i)`` - the log-prob whose gradient we train on;
* **KL-partner** ``(x_i, y_{p(i)})`` - a *mismatched* completion (a fixed seeded
  derangement) used only to estimate KTO's KL reference point ``z_ref``.

The prompt log-probabilities are masked out everywhere (weights/mask 0 over the
prompt, 1 over the completion), so only completion tokens count.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from training_sessions_cookbook.service import make_datum, read_jsonl

logger = logging.getLogger(__name__)


# === Raw rows ===
#
# Load ``(prompt_messages, completion_messages, label)`` rows from a local
# JSONL file or a HuggingFace dataset. Both the chat-message shape
# (``prompt``/``completion`` as message lists) and a flat ``prompt``/``completion``
# string shape are accepted; ``label`` is coerced to bool.


def _as_messages(value, role: str) -> list[dict]:
    """Coerce a prompt/completion field to a list of chat messages.

    Accepts a ready message list (returned as-is) or a bare string (wrapped as a
    single message with ``role``).
    """
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    if isinstance(value, list):
        return value
    raise ValueError(f"Unsupported {role} field type: {type(value)!r}")


def load_preference_rows(
    dataset_path: str,
    split: str = "train",
    eval_size: int | None = None,
    eval_ratio: float = 0.1,
    max_rows: int | None = None,
    hf_split_name: str | None = None,
) -> list[tuple[list[dict], list[dict], bool]]:
    """Load ``(prompt_messages, completion_messages, label)`` triples.

    JSONL rows use keys ``prompt`` / ``completion`` / ``label`` (label truthy =
    desirable). HuggingFace datasets use the same column names (the
    ``trl-lib/kto-mix-*`` shape). When no dedicated eval split exists, the train
    set is shuffled (seed 0) and carved by ``eval_size`` or ``eval_ratio``.
    """
    def _eval_count(total: int) -> int:
        if eval_size is not None:
            return min(eval_size, total)
        return max(1, int(total * eval_ratio))

    def _triples(rows) -> list[tuple[list[dict], list[dict], bool]]:
        out = []
        for row in rows:
            if "prompt" not in row or "completion" not in row or "label" not in row:
                continue
            prompt = _as_messages(row["prompt"], "user")
            completion = _as_messages(row["completion"], "assistant")
            out.append((prompt, completion, bool(row["label"])))
        return out

    if dataset_path.endswith(".jsonl"):
        triples = _triples(read_jsonl(dataset_path))
        if split in ("train", "eval"):
            random.seed(0)
            random.shuffle(triples)
            n_eval = _eval_count(len(triples))
            triples = triples[:n_eval] if split == "eval" else triples[n_eval:]
    else:
        import datasets

        ds = datasets.load_dataset(dataset_path)
        if isinstance(ds, datasets.DatasetDict):
            pool_name = (
                hf_split_name if hf_split_name and hf_split_name in ds else "train"
            )
            held_out = next(
                (n for n in ("eval", "validation", "test") if n in ds and n != pool_name),
                None,
            )
            if held_out is not None:
                # Dedicated held-out split: eval uses it (capped by eval_size),
                # train keeps the full pool.
                if split == "eval":
                    hf_split = ds[held_out].shuffle(seed=0)
                    if eval_size is not None and eval_size < len(hf_split):
                        hf_split = hf_split.select(range(eval_size))
                else:
                    hf_split = ds[pool_name].shuffle(seed=0)
            else:
                # No held-out split: carve eval = head, train = tail from one
                # shuffle, so the two sides stay disjoint.
                full = ds[pool_name].shuffle(seed=0)
                n_eval = _eval_count(len(full))
                hf_split = (
                    full.select(range(n_eval))
                    if split == "eval"
                    else full.select(range(n_eval, len(full)))
                )
        else:
            hf_split = ds
        triples = _triples(hf_split)

    if max_rows is not None and len(triples) > max_rows:
        triples = triples[:max_rows]
    logger.info(
        "Loaded %d preference rows from %s (split=%s)",
        len(triples), dataset_path, split,
    )
    return triples


# === Rendering ===
#
# Render a ``(prompt, completion)`` pair into two token sequences using the
# qwen3 layout, mirroring ``sft/data.py``: piecewise ``encode(..., add_special
# _tokens=False)`` per segment so client- and service-side tokenization agree.
# The prompt text includes the trailing assistant header (with an empty
# ``<think>`` block), so the completion tokens are exactly the assistant
# response plus ``<|im_end|>``.


def render_prompt_completion(
    prompt_messages: list[dict],
    completion_messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[int]]:
    """Return ``(prompt_tokens, completion_tokens)`` in the qwen3 chat layout.

    ``prompt_tokens`` covers the context messages and the opening assistant
    header; ``completion_tokens`` covers the assistant reply content and its
    closing ``<|im_end|>``. Both are encoded piecewise (no special tokens) and
    concatenated, matching the SFT renderer.
    """
    def _encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prompt_tokens: list[int] = []
    for idx, message in enumerate(prompt_messages):
        role = "user" if message["role"] == "tool" else message["role"]
        maybe_newline = "\n" if idx > 0 else ""
        prompt_tokens += _encode(f"{maybe_newline}<|im_start|>{role}\n")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Only string message content is supported")
        prompt_tokens += _encode(content + "<|im_end|>")

    # Opening assistant header for the reply (belongs to the prompt: masked out).
    maybe_newline = "\n" if prompt_messages else ""
    prompt_tokens += _encode(f"{maybe_newline}<|im_start|>assistant\n<think>\n\n</think>\n\n")

    # The completion is the assistant reply content + turn end (loss-bearing).
    reply = " ".join(
        m.get("content", "") for m in completion_messages if isinstance(m.get("content"), str)
    )
    completion_tokens = _encode(reply + "<|im_end|>")
    return prompt_tokens, completion_tokens


# === Examples and batching ===


@dataclass
class KtoExample:
    """One KTO training example.

    ``prompt_tokens`` / ``completion_tokens`` are the matched pair; ``label`` is
    True for desirable. ``kl_completion_tokens`` is a *mismatched* completion
    (fixed at build time) paired with this example's prompt to estimate the KL
    reference point ``z_ref``.
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    label: bool
    kl_completion_tokens: list[int]


def _slice(prompt_tokens: list[int], completion_tokens: list[int]):
    """Shared slicing: ``(input_tokens, target_tokens, completion_mask)``.

    ``full = prompt + completion``; ``input = full[:-1]``; ``target = full[1:]``.
    The mask is 0 over the ``len(prompt) - 1`` prompt-target positions and 1 over
    the ``len(completion)`` completion targets, so a masked sum of the returned
    per-token log-probs is ``log pi(completion | prompt)``.
    """
    p, c = len(prompt_tokens), len(completion_tokens)
    if p < 1 or c < 1:
        raise ValueError(f"need prompt>=1 and completion>=1 tokens (got {p}, {c})")
    full = prompt_tokens + completion_tokens
    input_tokens = full[:-1]
    target_tokens = full[1:]
    completion_mask = [0.0] * (p - 1) + [1.0] * c
    assert len(completion_mask) == len(target_tokens) == p + c - 1
    return input_tokens, target_tokens, completion_mask


def matched_slice(ex: KtoExample):
    """``(input, target, completion_mask)`` for the matched pair ``(x_i, y_i)``."""
    return _slice(ex.prompt_tokens, ex.completion_tokens)


def kl_slice(ex: KtoExample):
    """``(input, target, completion_mask)`` for the KL pair ``(x_i, y_{p(i)})``."""
    return _slice(ex.prompt_tokens, ex.kl_completion_tokens)


def pair_slice(prompt_tokens, completion_tokens):
    """``(input, target, completion_mask)`` for an arbitrary ``(prompt, completion)``.

    Used for the ``compute_logprobs`` reference source, where the KL-partner
    completion is re-drawn per step rather than fixed on the example.
    """
    return _slice(prompt_tokens, completion_tokens)


def batch_kl_partners(batch: list["KtoExample"], rng) -> list[list[int]]:
    """A within-batch derangement of completions for the KL term.

    Returns, per example, *another* example's completion tokens (partner != self
    when the batch has >1 example), so the KL reference point is estimated from
    genuinely mismatched pairs. ``rng`` is a ``random.Random`` seeded per step,
    so the pairing is re-randomized each step (TRL-style) when the reference is
    scored on demand.
    """
    n = len(batch)
    idx = list(range(n))
    if n > 1:
        rng.shuffle(idx)
        for i in range(n):
            if idx[i] == i:
                j = (i + 1) % n
                idx[i], idx[j] = idx[j], idx[i]
    return [batch[idx[i]].completion_tokens for i in range(n)]


def forward_datum(input_tokens, target_tokens, completion_mask) -> dict:
    """A forward-only datum: log-probs are read back for every target token.

    ``weights`` carries the completion mask so a service that also returns
    ``actor/loss`` reports it over completion tokens only; the loop reads the
    per-token ``logprobs`` output, not the loss.
    """
    return make_datum(input_tokens, target_tokens, completion_mask)


def train_datum(input_tokens, target_tokens, completion_mask, *, advantage: float,
                old_logprobs: list[float]) -> dict:
    """A datum for the surrogate ``forward_backward`` (importance-sampling loss).

    ``advantages`` broadcasts the per-sequence ``-s_i`` over completion tokens
    (0 over the prompt); ``logprobs`` are the policy's own per-token log-probs
    from the matched forward pass, so the importance ratio is 1 at evaluation
    and the surrogate gradient equals the exact KTO gradient.
    """
    advantages = [a * advantage for a in completion_mask]  # completion_mask is 0/1
    return make_datum(
        input_tokens, target_tokens, completion_mask,
        logprobs=old_logprobs, advantages=advantages,
    )


def build_kto_batches(
    dataset_path: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    split: str = "train",
    eval_size: int | None = None,
    eval_ratio: float = 0.1,
    max_rows: int | None = None,
    hf_split_name: str | None = None,
    kl_seed: int = 0,
) -> list[list[KtoExample]]:
    """Load, render, and batch a preference dataset into :class:`KtoExample`s.

    Examples whose prompt+completion exceed ``max_length`` (or that render to
    <1 completion token) are dropped. The KL-partner completion is a fixed
    seeded derangement over the surviving examples, so the reference log-probs
    for both the matched and KL pairs can be precomputed once.
    """
    triples = load_preference_rows(
        dataset_path, split=split, eval_size=eval_size, eval_ratio=eval_ratio,
        max_rows=max_rows, hf_split_name=hf_split_name,
    )

    rendered: list[tuple[list[int], list[int], bool]] = []
    skipped = 0
    for prompt_messages, completion_messages, label in triples:
        try:
            pt, ct = render_prompt_completion(prompt_messages, completion_messages, tokenizer)
        except Exception:
            skipped += 1
            continue
        if len(pt) < 1 or len(ct) < 1 or len(pt) + len(ct) > max_length:
            skipped += 1
            continue
        rendered.append((pt, ct, label))

    if skipped:
        logger.info("Skipped %d rows (too long or empty completion)", skipped)
    n = len(rendered)
    if n == 0:
        return []

    # Fixed derangement for the KL-partner completions: partner(i) != i so the
    # KL term always uses a genuinely mismatched (prompt, completion) pair.
    rng = random.Random(kl_seed)
    partner = list(range(n))
    if n > 1:
        rng.shuffle(partner)
        for i in range(n):
            if partner[i] == i:  # break any fixed points against the next slot
                j = (i + 1) % n
                partner[i], partner[j] = partner[j], partner[i]

    examples = [
        KtoExample(
            prompt_tokens=pt,
            completion_tokens=ct,
            label=label,
            kl_completion_tokens=rendered[partner[i]][1],
        )
        for i, (pt, ct, label) in enumerate(rendered)
    ]

    n_desirable = sum(1 for e in examples if e.label)
    logger.info(
        "Built %d KTO examples (%d desirable, %d undesirable) into %d batches",
        n, n_desirable, n - n_desirable, (n + batch_size - 1) // batch_size,
    )

    return [examples[i : i + batch_size] for i in range(0, n, batch_size)]
