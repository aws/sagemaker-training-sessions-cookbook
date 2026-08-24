"""Tests for ``envs/tool_render.py`` (gpt-oss tool renderer).

Covers:
  * Tool-call render/parse round-trip (gpt-oss): rendering ``check_solution(code=X)`` then parsing recovers ``name == "check_solution"`` and ``arguments["code"] == X`` for arbitrary code strings X, including tricky cases (newlines, quotes, braces, backslashes) and embedded harmony control tokens (``<|call|>`` etc.).
  * Example tests: final-answer (no tool call) → ``[]``; fenced-fallback parse; schema/result rendering; ``family_for_model`` dispatch.

``tool_render`` is stdlib-only and import-light, so we only add the repo root to ``sys.path`` so ``envs`` is importable.
"""
import os
import sys

# Add the repo root to path so the import-light ``envs`` package (and ``envs.tool_render``) resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings
from hypothesis import strategies as st

from training_sessions_cookbook.rl.envs.base import ToolCall
from training_sessions_cookbook.rl.envs.tool_render import (
    check_solution_tool_spec,
    family_for_model,
    parse_tool_calls,
    render_tool_call,
    render_tool_result,
    render_tool_schema,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Harmony control tokens we want to make sure survive round-tripping when they appear *inside* an argument value (they must be JSON-escaped, never confused with the real frame markers).
_HARMONY_TOKENS = ["<|call|>", "<|message|>", "<|end|>", "<|channel|>", "<|start|>"]

# Arbitrary code strings, broadened to include tricky characters: newlines, quotes, braces, backslashes, and the full unicode character space.
_arbitrary_code = st.text(st.characters(), max_size=200)

# Code strings that explicitly embed one or more harmony control tokens, glued together with arbitrary text, so the round-trip is exercised on adversarial inputs as well as random ones.
_code_with_tokens = st.lists(
    st.one_of(st.sampled_from(_HARMONY_TOKENS), st.text(st.characters(), max_size=20)),
    min_size=1,
    max_size=6,
).map("".join)

# Union: plain arbitrary text or token-laced text.
_code_strategy = st.one_of(_arbitrary_code, _code_with_tokens)


# ---------------------------------------------------------------------------
# render -> parse round-trip
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(code=_code_strategy)
def test_round_trip_single_arg(code):
    """parse(render(check_solution, {code: X})) == [ToolCall(..., {code: X})].

    """
    rendered = render_tool_call("check_solution", {"code": code})
    parsed = parse_tool_calls(rendered)
    assert parsed == [ToolCall("check_solution", {"code": code})]
    # Be explicit about the bits the property names.
    assert len(parsed) == 1
    assert parsed[0].name == "check_solution"
    assert parsed[0].arguments["code"] == code


@settings(max_examples=100)
@given(code=_code_strategy)
def test_round_trip_multi_arg(code):
    """Multi-argument calls recover every argument exactly.

    """
    args = {"code": code, "lang": "python"}
    rendered = render_tool_call("check_solution", args)
    parsed = parse_tool_calls(rendered)
    assert parsed == [ToolCall("check_solution", args)]
    assert parsed[0].arguments["code"] == code
    assert parsed[0].arguments["lang"] == "python"


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_final_answer_has_no_tool_call():
    """A final-channel / plain-text answer parses to no tool calls."""
    final_msg = (
        "<|start|>assistant<|channel|>final<|message|>"
        "Here is the answer: 42.<|end|>"
    )
    assert parse_tool_calls(final_msg) == []
    assert parse_tool_calls("Just some plain prose, no tool call here.") == []
    assert parse_tool_calls("") == []


def test_fenced_fallback_parse():
    """The fenced ```tool_call block fallback parses name + arguments."""
    text = (
        '```tool_call\n'
        '{"name":"check_solution","arguments":{"code":"print(1)"}}\n'
        '```'
    )
    parsed = parse_tool_calls(text)
    assert parsed == [ToolCall("check_solution", {"code": "print(1)"})]


def test_render_tool_schema_contains_name_and_param():
    """The rendered schema mentions the tool name and the `code` param."""
    schema = render_tool_schema([check_solution_tool_spec()])
    assert "check_solution" in schema
    assert "code" in schema


def test_render_tool_result_contains_name_and_content():
    """The rendered tool result echoes the tool name and the content."""
    content = '{"passed": true}'
    rendered = render_tool_result("check_solution", content)
    assert "check_solution" in rendered
    assert content in rendered


def test_family_for_model_dispatch():
    """Known names resolve to their family; unknown names resolve to None.

    ``None`` rather than a default family is deliberate: handing an
    unrecognised model gpt-oss tool syntax produces a silently mis-trained run
    instead of an error (see envs/tool_render.family_for_model).
    """
    assert family_for_model("openai/gpt-oss-20b") == "gpt_oss"
    assert family_for_model("Qwen/Qwen3.6-27B") == "qwen3"
    assert family_for_model("google/gemma-4-31B-it") == "gemma"
    assert family_for_model(None) is None
    assert family_for_model("mistralai/Mistral-7B-Instruct-v0.3") is None


def test_harmony_token_in_argument_round_trips():
    """Argument values containing harmony control tokens still round-trip."""
    for token in ("<|call|>", "<|message|>"):
        code = f"x = 1  # {token} embedded\nprint(x)"
        rendered = render_tool_call("check_solution", {"code": code})
        parsed = parse_tool_calls(rendered)
        assert parsed == [ToolCall("check_solution", {"code": code})]
        assert parsed[0].arguments["code"] == code

    # And a value made entirely of stitched-together control tokens.
    combo = "".join(_HARMONY_TOKENS)
    rendered = render_tool_call("check_solution", {"code": combo})
    assert parse_tool_calls(rendered) == [ToolCall("check_solution", {"code": combo})]


# ---------------------------------------------------------------------------
# qwen3 (Qwen3.5+/3.6 XML convention)
# ---------------------------------------------------------------------------

from training_sessions_cookbook.rl.envs.tool_render import SUPPORTED_FAMILIES

# The qwen3 format carries parameter values as raw text (no escaping exists in
# the format), so the round-trip guarantee excludes values containing the
# format's own closing tags; such values are unrepresentable for the model
# too. Everything else (newlines, quotes, braces, unicode, im_start markers)
# must survive verbatim, modulo the framing newlines the renderer adds.
_QWEN_CLOSERS = ["</parameter>", "</function>", "</tool_call>"]
_qwen_code = st.text(st.characters(), max_size=200).filter(
    lambda s: not any(c in s for c in _QWEN_CLOSERS)
    and s == s.strip("\n")  # framing newlines are stripped on parse
)


@settings(max_examples=200, deadline=None)
@given(code=_qwen_code)
def test_qwen3_round_trip_single_arg(code):
    rendered = render_tool_call("check_solution", {"code": code}, family="qwen3")
    calls = parse_tool_calls(rendered, family="qwen3")
    assert len(calls) == 1
    assert calls[0].name == "check_solution"
    assert calls[0].arguments == {"code": code}


def test_qwen3_realistic_model_output_parses():
    """Reasoning text before the call (allowed by the template), then XML."""
    out = (
        "Let me verify my solution first.\n\n"
        "<tool_call>\n<function=check_solution>\n"
        "<parameter=code>\nimport sys\nprint(sys.stdin.read()[::-1])\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = parse_tool_calls(out, family="qwen3")
    assert calls == [
        ToolCall(name="check_solution",
                 arguments={"code": "import sys\nprint(sys.stdin.read()[::-1])"})
    ]


def test_qwen3_multi_parameter_call():
    rendered = render_tool_call("f", {"a": "1\n2", "b": "x"}, family="qwen3")
    (call,) = parse_tool_calls(rendered, family="qwen3")
    assert call.arguments == {"a": "1\n2", "b": "x"}


def test_qwen3_schema_matches_model_template_conventions():
    """Pin the invariants the Qwen3.6 chat template mandates."""
    schema = render_tool_schema([check_solution_tool_spec()], family="qwen3")
    assert schema.startswith("<|im_start|>system")
    assert schema.rstrip().endswith("<|im_end|>")
    assert "<tools>" in schema and "</tools>" in schema
    assert '"name": "check_solution"' in schema
    # invocation format is the XML function/parameter form, NOT a JSON payload
    assert "<function=example_function_name>" in schema
    assert '"arguments"' not in schema.split("</tools>")[1]


def test_qwen3_result_is_tool_response_in_user_turn():
    """The template detects multi-step turns by this exact wrapping."""
    result = render_tool_result("check_solution", '{"passed": false}', family="qwen3")
    assert result.startswith("<|im_start|>user\n<tool_response>")
    assert result.rstrip().endswith("</tool_response><|im_end|>")
    assert '{"passed": false}' in result


def test_qwen3_final_answer_has_no_tool_call():
    assert parse_tool_calls("The answer is 42.", family="qwen3") == []


def test_qwen3_fenced_fallback_shared():
    out = '```tool_call\n{"name": "check_solution", "arguments": {"code": "x"}}\n```'
    (call,) = parse_tool_calls(out, family="qwen3")
    assert call.name == "check_solution" and call.arguments == {"code": "x"}


def test_unsupported_family_raises_with_supported_list():
    for fn in (
        lambda: render_tool_schema([], family="llama3"),
        lambda: render_tool_call("f", {}, family="llama3"),
        lambda: render_tool_result("f", "x", family="llama3"),
        lambda: parse_tool_calls("y", family="llama3"),
    ):
        try:
            fn()
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "supported:" in str(e)
    assert set(SUPPORTED_FAMILIES) == {"gpt_oss", "qwen3", "gemma"}
