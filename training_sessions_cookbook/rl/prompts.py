"""Prompt templates and chat-template handling per model family.

``apply_auto`` picks the right template from the base-model id; the
``apply_*`` variants implement each family's conventions. ``encode_prompt``
turns a template result (string or pre-tokenized ids) into token ids.
"""

from __future__ import annotations



# === Prompt templates ===
#
# Each template is a function `apply_X(observation: str) -> str` that wraps the observation in the model family's chat format. `apply_auto` selects a template by substring match against `args.base_model` and falls back to `qwen3_general`. `TEMPLATE_FACTORY` is the dispatch dict keyed by the seven `--template` values exposed on the CLI.


def apply_qwen3_general(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n" "<|im_start|>assistant\n"


def apply_qwen3_game(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_gpt_oss(question: str, reasoning: str = "high") -> str:
    """Hand-rolled gpt-oss chat prompt with configurable reasoning level.

    `reasoning` is mapped directly into the `Reasoning: <level>` system-block field. `auto` resolves to `high`. `off` resolves to `low` so the model still produces a (minimal) analysis channel; the gpt-oss chat format does not have a zero-reasoning mode. Compare to `apply_hf(..., reasoning=...)` which uses the model-shipped jinja and adds the ChatGPT identity + date framing on top.
    """
    if reasoning == "auto":
        reasoning = "high"
    elif reasoning == "off":
        reasoning = "low"
    return (
        "<|start|>system<|message|>You are a helpful assistant.\n"
        f"Reasoning: {reasoning}\n"
        "# Valid channels: analysis, commentary, final. "
        "Channel must be included for every message.<|end|>"
        f"<|start|>user<|message|>{question}<|end|>"
        "<|start|>assistant"
    )


def apply_no(observation: str) -> str:
    return observation


def _normalize_chat_template_output(out) -> list[int]:
    """Coerce `apply_chat_template(..., tokenize=True)` output to `list[int]`.

    HF returns one of three shapes across (slow vs fast) tokenizers and HF versions: a plain `list[int]`, a `BatchEncoding` exposing `input_ids`, or a list of `tokenizers.Encoding` objects. Cover all three.
    """
    if hasattr(out, "input_ids"):
        return list(out.input_ids)
    if isinstance(out, list) and len(out) > 0 and hasattr(out[0], "ids"):
        return list(out[0].ids)
    return list(out)


def _apply_chat_template_progressive(tokenizer, messages, extra_kwargs: dict) -> list[int]:
    """Call `tokenizer.apply_chat_template` and degrade gracefully on TypeError.

    Different model templates accept different kwargs:
      - Gemma's jinja accepts `enable_thinking` but not `reasoning_effort`.
      - gpt-oss's jinja accepts `reasoning_effort` but not `enable_thinking`.
      - Older / unrelated templates accept neither.
    We try the full `extra_kwargs` first, then progressively drop pieces until something works. First success wins; behavior is identical for templates that accept everything we pass.
    """
    base = dict(tokenize=True, add_generation_prompt=True, return_tensors=None)
    attempts = [
        extra_kwargs,
        {k: v for k, v in extra_kwargs.items() if k == "enable_thinking"},
        {k: v for k, v in extra_kwargs.items() if k == "reasoning_effort"},
        {},
    ]
    last_exc: Exception = RuntimeError("apply_chat_template never attempted")
    for kw in attempts:
        try:
            return _normalize_chat_template_output(
                tokenizer.apply_chat_template(messages, **base, **kw)
            )
        except TypeError as exc:
            # The template's jinja rejected one of our kwargs. Try a smaller subset on the next iteration.
            last_exc = exc
            continue
    # Bare `{}` should always succeed; if it didn't, surface the failure.
    raise last_exc


def _reasoning_to_kwargs(reasoning: str) -> dict:
    """Translate the high-level `--reasoning` knob to chat-template kwargs.

    `auto` sets `enable_thinking=True` (so gemma runs its `<|think|>` channel) but leaves `reasoning_effort` unset so gpt-oss's HF jinja uses its own default level (`medium`). Explicit `low|medium|high` overrides both. `off` disables thinking entirely AND pins the gpt-oss level to `low`.
    """
    if reasoning == "off":
        return {"enable_thinking": False, "reasoning_effort": "low"}
    if reasoning in ("low", "medium", "high"):
        return {"enable_thinking": True, "reasoning_effort": reasoning}
    # auto: thinking on; no reasoning_effort so the template defaults apply.
    return {"enable_thinking": True}


def apply_gemma(observation: str, tokenizer, reasoning: str = "auto") -> list[int]:
    """Tokenize a Gemma user-turn prompt using the model-shipped chat template.

    Returns a list of token IDs (NOT a string) because Gemma's tokenizer ships role markers like `<|turn>` / `<turn|>` present in the vocab but not registered as no-split special tokens; the standard `tokenizer.encode("<|turn>...")` path splits them into character tokens and corrupts every prompt. Routing through `apply_chat_template(..., tokenize=True)` uses Gemma's own chat template and emits the correct special-token IDs.

    `reasoning` maps to Gemma's `enable_thinking` kwarg via `_reasoning_to_kwargs`: `off` → `enable_thinking=False` (the empty `<|channel>thought\\n<channel|>` skip-marker is emitted, telling the model to go straight to content); `low|medium|high` → `enable_thinking=True` (the system block opens with `<|think|>\\n` and the skip-marker is suppressed so the model can use its thinking channel). `auto` keeps whatever the chat template's own default is (typically thinking off, but we override that elsewhere via the --reasoning flag for math runs).
    """
    return _apply_chat_template_progressive(
        tokenizer,
        [{"role": "user", "content": observation}],
        _reasoning_to_kwargs(reasoning),
    )


def apply_hf(observation: str, tokenizer, reasoning: str = "auto") -> list[int]:
    """Generic HF chat-template wrapper for any model family.

    Drives `tokenizer.apply_chat_template` with the model's own jinja, threading the `--reasoning` knob through whatever kwargs the template happens to accept (`enable_thinking` for gemma-style, `reasoning_effort` for gpt-oss-style, neither for plain qwen/llama). Unknown kwargs are silently dropped by `_apply_chat_template_progressive` so this works on any tokenizer.

    Use this template by passing `--template hf` to opt into the model's canonical prompt format (including any identity / date framing the model card encodes in its jinja). Compare to the hand-rolled `apply_qwen3_general` / `apply_gpt_oss` paths which use a stripped-down format the cookbook authors wrote by hand.
    """
    return _apply_chat_template_progressive(
        tokenizer,
        [{"role": "user", "content": observation}],
        _reasoning_to_kwargs(reasoning),
    )


def apply_auto(observation: str, base_model: str, tokenizer=None, reasoning: str = "auto"):
    """Pick a template by substring match against `base_model`.

    Match precedence (first hit wins):
      - "gpt-oss"  -> apply_gpt_oss              (returns str)
      - "gemma"    -> apply_gemma                (returns list[int] tokens)
      - "qwen"     -> apply_qwen3_general        (returns str)
      - otherwise  -> apply_qwen3_general        (returns str)

    Return type is heterogeneous: `apply_gemma` returns token IDs directly because Gemma's tokenizer can't faithfully re-encode the chat markers. Callers must handle both via `encode_prompt(result, tokenizer)`.

    `reasoning` is forwarded to whichever path supports it. For non-thinking templates (qwen3_general, qwen3_game, no) it's silently ignored.

    `tokenizer` is required when the resolved template is Gemma; passing `None` will raise from inside `apply_gemma`. For non-Gemma models it's unused and may be omitted.

    Matching is case-insensitive and operates on the trailing path segment of `base_model` so values like "Qwen/Qwen3-1.7B-Base", "google/gemma-4-31B-it", or "arn:.../model/gpt-oss-20b" all resolve correctly.
    """
    name = (base_model or "").rstrip("/").split("/")[-1].lower()
    if "gpt-oss" in name:
        return apply_gpt_oss(observation, reasoning=reasoning)
    if "gemma" in name:
        return apply_gemma(observation, tokenizer, reasoning=reasoning)
    if "qwen" in name:
        return apply_qwen3_general(observation)
    return apply_qwen3_general(observation)


def encode_prompt(result, tokenizer) -> list[int]:
    """Normalize template output to a `list[int]` of token IDs.

    String templates (qwen3_general, qwen3_game, gpt_oss, no) return text and need a final `tokenizer.encode(...)`. Token-list templates (apply_gemma) already produced IDs via `apply_chat_template` and are passed through. This indirection keeps the call sites (validation, rollout_one, rollout_one_grouped) ignorant of which template was selected.
    """
    if isinstance(result, list):
        return list(result)
    return list(tokenizer.encode(result))


# Dispatch dict keyed by the seven `--template` values accepted on the CLI. `auto`, `gemma`, and `hf` are special-cased in `main()` because they need the tokenizer baked into the closure; the entries below are placeholders that raise if invoked directly without the closure wiring (they're never reached at runtime because `main()`'s `args.template` branch chooses the right closure first).
def _factory_needs_tokenizer(name: str):
    def _raise(_observation):
        raise RuntimeError(
            f"--template {name} must be resolved through main()'s closure "
            f"binding (it needs the tokenizer); reached the raw factory "
            f"entry, which is a wiring bug."
        )
    return _raise


TEMPLATE_FACTORY: "dict[str, object]" = {
    "qwen3_general": apply_qwen3_general,
    "qwen3_game": apply_qwen3_game,
    "gpt_oss": apply_gpt_oss,
    "gemma": _factory_needs_tokenizer("gemma"),
    "hf": _factory_needs_tokenizer("hf"),
    "no": apply_no,
    "auto": apply_auto,
}

