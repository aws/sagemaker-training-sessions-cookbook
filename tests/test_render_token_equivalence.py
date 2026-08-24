"""Token-level equivalence between our rendering and each model's own template.

The cookbook builds a multi-turn prompt incrementally, concatenating token ids:

    prompt_tokens = template(first_observation)
                  + sampled_assistant_tokens        # verbatim, never re-encoded
                  + encode(render_tool_result(...))
                  + ...

The sampled tokens are appended verbatim on purpose: the training datums and
their logprobs refer to exactly those ids, so re-rendering the conversation from
a message list each turn (and thereby re-encoding the assistant text) would
silently corrupt the learning signal.

That design is only safe if our hand-rolled frames tokenize to the same ids the
model was trained on. These tests prove that by comparing against the
authoritative source available locally - the model's own chat template - for an
equivalent conversation. A drift in either direction (our renderer, or a
tokenizer upgrade) fails here rather than showing up as an unexplained drop in
reward.

Tokenizer downloads are required, so each test skips when the model cannot be
loaded (offline CI); the assertions are exact when it can.
"""

from __future__ import annotations

import pytest

from training_sessions_cookbook.rl.envs.tool_render import (
    render_tool_call,
    render_tool_result,
    render_tool_schema,
)

TOOL = {
    "name": "check_solution",
    "description": "Run candidate code against tests.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source"}},
        "required": ["code"],
    },
}

# The same tool, in the OpenAI-style shape chat templates expect.
TEMPLATE_TOOLS = [{"type": "function", "function": TOOL}]

CODE = "print(1)"
TOOL_OUTPUT = '{"passed": true}'


def _tokenizer(model: str):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(model)
    except Exception as exc:  # network/gated/offline
        pytest.skip(f"tokenizer for {model} unavailable: {exc}")


def _ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def test_gpt_oss_tool_frames_tokenize_like_the_model_template():
    """Our harmony tool-call and tool-result frames match the model's own.

    Rather than compare whole prompts (the cookbook's system/developer preamble
    is intentionally its own), this pins the two frames that the renderer owns
    and that must be byte-exact for the model to recognise a tool exchange.
    """
    tok = _tokenizer("openai/gpt-oss-20b")
    reference = tok.apply_chat_template(
        [
            {"role": "user", "content": "Solve it."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "check_solution", "arguments": {"code": CODE}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "check_solution", "content": TOOL_OUTPUT},
        ],
        tools=TEMPLATE_TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )

    # The tool-result frame must appear in the reference verbatim.
    ours_result = render_tool_result("check_solution", TOOL_OUTPUT, "gpt_oss")
    frame_head = "<|start|>functions.check_solution to=assistant<|channel|>commentary<|message|>"
    assert frame_head in reference, "reference template changed shape; update this test"
    assert ours_result.startswith(frame_head) and ours_result.endswith("<|end|>")

    # And the declaration block must use the same TypeScript-like syntax.
    ours_schema = render_tool_schema([TOOL], "gpt_oss")
    for fragment in ("namespace functions {", "type check_solution = (_: {", "code: string,", "}) => any;"):
        assert fragment in ours_schema, f"our schema is missing {fragment!r}"
        assert fragment in reference, f"reference lost {fragment!r}; update this test"

    # Every frame we emit must tokenize without splitting harmony specials.
    for text in (ours_result, render_tool_call("check_solution", {"code": CODE}, "gpt_oss")):
        assert tok.decode(_ids(tok, text)) == text, "frame does not round-trip through the tokenizer"


def test_qwen3_tool_frames_round_trip_through_the_tokenizer():
    """qwen3 frames must survive encode/decode unchanged.

    The XML convention uses ordinary text plus ``<|im_start|>``/``<|im_end|>``
    specials; if a marker were split into pieces the model would see a different
    grammar than the one it was trained on.
    """
    tok = _tokenizer("Qwen/Qwen3.6-27B")
    for text in (
        render_tool_call("check_solution", {"code": CODE}, "qwen3"),
        render_tool_result("check_solution", TOOL_OUTPUT, "qwen3"),
        render_tool_schema([TOOL], "qwen3"),
    ):
        assert tok.decode(_ids(tok, text)) == text, f"frame does not round-trip: {text[:60]!r}"


def test_multi_turn_prompt_has_exactly_one_system_block():
    """A second turn must not re-open the conversation preamble.

    Regression test for the double-framing defect: the tool result returned by
    ``CodeRLEnv`` is already a complete chat frame, so passing it through the
    prompt template again injected a fresh system block and nested the tool
    frame inside a user message, malforming every turn after the first. The
    prompt is assembled here exactly as ``rollout_one_code`` assembles it.
    """
    from training_sessions_cookbook.rl.prompts import apply_gpt_oss

    first_turn = apply_gpt_oss("Solve it." + render_tool_schema([TOOL], "gpt_oss"), "medium")
    tool_frame = render_tool_result("check_solution", TOOL_OUTPUT, "gpt_oss")
    sampled_assistant = render_tool_call("check_solution", {"code": CODE}, "gpt_oss")

    prompt = first_turn + sampled_assistant + tool_frame

    assert prompt.count("<|start|>system<|message|>") == 1, (
        "the conversation preamble appears more than once: a later turn was "
        "re-templated instead of appended as a frame"
    )
    # The tool frame must sit at top level, not inside a user message.
    assert "<|start|>user<|message|><|start|>functions." not in prompt, (
        "tool result is nested inside a user message"
    )


# --- gemma ----------------------------------------------------------------

GEMMA_TOOL = {
    "name": "check_solution",
    "description": "Run tests.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source"},
            "retries": {"type": "integer", "description": "How many times"},
            "verbose": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "mode": {"type": "string", "enum": ["fast", "slow"]},
        },
        "required": ["code"],
    },
}
GEMMA_PING = {
    "name": "ping",
    "description": "No args.",
    "parameters": {"type": "object", "properties": {}},
}
GEMMA_ARGS = {"code": CODE, "retries": 2, "verbose": True, "tags": ["a", "b"]}


def _gemma_reference(tok) -> str:
    return tok.apply_chat_template(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "check_solution", "arguments": GEMMA_ARGS},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
        tools=[{"type": "function", "function": t} for t in (GEMMA_TOOL, GEMMA_PING)],
        tokenize=False,
        add_generation_prompt=True,
    )


def test_gemma_frames_match_the_model_template_exactly():
    """Our gemma frames must be substrings of what gemma's own template emits.

    Gemma's tool syntax has several details that are easy to get subtly wrong -
    keys are emitted in sorted order, JSON schema types are upper-cased, string
    scalars are wrapped in the ``<|"|>`` sentinel while numbers and booleans are
    bare, and a parameterless tool omits ``properties`` entirely. Rendering by
    hand is only acceptable while it provably agrees with the template, so this
    compares against it directly rather than against a frozen expected string.
    """
    tok = _tokenizer("google/gemma-4-31b-it")
    reference = _gemma_reference(tok)

    declaration = (
        render_tool_schema([GEMMA_TOOL, GEMMA_PING], "gemma")
        .split("system\n", 1)[1]
        .rsplit("<turn|>", 1)[0]
    )
    assert declaration in reference, "tool declarations diverge from gemma's template"
    assert render_tool_call("check_solution", GEMMA_ARGS, "gemma") in reference
    assert render_tool_result("check_solution", "ok", "gemma") in reference


def test_gemma_frames_round_trip_through_the_tokenizer():
    """Gemma's sentinels are single vocabulary tokens, so strings re-encode exactly.

    This is what allows gemma to be rendered as text fragments at all: the
    incremental prompt assembly in ``rollout_one_code`` encodes each frame, and
    the sampled assistant tokens are appended verbatim.
    """
    tok = _tokenizer("google/gemma-4-31b-it")
    for text in (
        render_tool_call("check_solution", GEMMA_ARGS, "gemma"),
        render_tool_result("check_solution", "ok", "gemma"),
        render_tool_schema([GEMMA_TOOL], "gemma"),
    ):
        assert tok.decode(_ids(tok, text)) == text, f"frame does not round-trip: {text[:60]!r}"


def test_gemma_parse_recovers_argument_types():
    """Parsing must restore Python types, not leave every value a string."""
    from training_sessions_cookbook.rl.envs.tool_render import parse_tool_calls

    rendered = render_tool_call("check_solution", GEMMA_ARGS, "gemma")
    calls = parse_tool_calls(rendered, family="gemma")
    assert len(calls) == 1
    assert calls[0].name == "check_solution"
    assert calls[0].arguments == GEMMA_ARGS


# --- turn-1 composition and turn-2 opener (the full-prompt seams) -----------


def test_turn1_prompt_places_declarations_outside_the_user_message():
    """Tools passed as data must land in their own frame, per family.

    Regression for the turn-1 double-framing defect: `reset()` used to embed a
    fully framed schema turn in the observation, which every template then
    wrapped inside its user message (a nested system/developer turn plus a
    doubled end-token), so all multi-turn episodes started malformed.
    """
    from training_sessions_cookbook.rl.prompts import apply_gpt_oss, apply_qwen3_general

    tools = [GEMMA_TOOL]

    oss = apply_gpt_oss("P", tools=tools)
    dev = oss.index("<|start|>developer<|message|>")
    user = oss.index("<|start|>user<|message|>")
    assert dev < user, "gpt-oss declarations must precede the user frame"
    assert "<|end|><|end|>" not in oss

    qwen = apply_qwen3_general("P", tools=tools)
    sys_turn = qwen.index("<|im_start|>system")
    user_turn = qwen.index("<|im_start|>user")
    assert sys_turn < user_turn, "qwen3 declarations must precede the user turn"
    assert "<|im_end|><|im_end|>" not in qwen


def test_gemma_turn1_uses_the_model_template_natively():
    """apply_gemma(tools=...) must let gemma's own jinja place the declarations."""
    from training_sessions_cookbook.rl.prompts import apply_gemma

    tok = _tokenizer("google/gemma-4-31b-it")
    ours = tok.decode(apply_gemma("P", tok, tools=[GEMMA_TOOL]))
    sys_end = ours.index("<turn|>")
    assert "<|tool>declaration:check_solution" in ours[:sys_end], (
        "declarations must be inside gemma's system turn"
    )
    assert ours.count("<|turn>user") == 1
    assert "<turn|><turn|>" not in ours


def test_generation_opener_matches_each_model_template():
    """After a tool result, sampling must start where the model template does.

    gpt-oss and qwen3 open a fresh assistant frame (their templates emit one
    with ``add_generation_prompt=True`` after a ``tool`` message); gemma
    continues the same model turn, so its opener is empty.
    """
    from training_sessions_cookbook.rl.envs.tool_render import generation_opener

    assert generation_opener("gpt_oss") == "<|start|>assistant"
    assert generation_opener("qwen3") == "<|im_start|>assistant\n"
    assert generation_opener("gemma") == ""


def test_gpt_oss_system_message_carries_the_commentary_routing_line():
    """With tools defined, the system message must include harmony's routing line.

    The spec (and gpt-oss's own template) emit exactly
    ``Calls to these tools must go to the commentary channel: 'functions'.``
    in the SYSTEM message whenever functions are declared; it is what routes
    tool calls to the commentary channel. Without tools the line must be absent.
    """
    from training_sessions_cookbook.rl.prompts import apply_gpt_oss

    line = "Calls to these tools must go to the commentary channel: 'functions'."
    with_tools = apply_gpt_oss("P", tools=[GEMMA_TOOL])
    system_msg = with_tools[: with_tools.index("<|end|>")]
    assert line in system_msg, "routing line missing from the system message"

    without_tools = apply_gpt_oss("P")
    assert line not in without_tools
