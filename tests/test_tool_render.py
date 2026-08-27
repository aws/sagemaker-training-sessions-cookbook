"""Tests for ``envs/tool_render.py`` (gpt-oss tool renderer).

Covers:
  * Tool-call render/parse round-trip (gpt-oss): rendering ``check_solution(code=X)`` then parsing recovers ``name == "check_solution"`` and ``arguments["code"] == X`` for arbitrary code strings X, including tricky cases (newlines, quotes, braces, backslashes) and embedded harmony control tokens (``<|call|>`` etc.).
  * Example tests: final-answer (no tool call) → ``[]``; fenced-fallback parse; schema/result rendering; ``family_for_model`` dispatch.

``tool_render`` is stdlib-only and import-light, so we only add the repo root to ``sys.path`` so ``envs`` is importable.

Everything here runs offline. Verification that needs real tokenizers (comparing
our frames token-for-token against each model's own chat template) lives in
``test_render_token_equivalence.py``, which skips when a model is unavailable.
"""
import os
import sys

# Add the repo root to path so the import-light ``envs`` package (and ``envs.tool_render``) resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

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

# ---------------------------------------------------------------------------
# Format fidelity: what each model family was actually trained on
# ---------------------------------------------------------------------------
#
# Fidelity tests for tool-call rendering and parsing.
#
# These tests assert what each model family was actually trained on, verified
# against primary sources rather than inference:
#
# * gpt-oss: the harmony response format specification
#   (https://github.com/openai/harmony/blob/main/docs/format.md), cross-checked
#   against ``openai/gpt-oss-20b``'s own chat template.
# * qwen3: ``Qwen/Qwen3.6-27B``'s chat template.
# * gemma: ``google/gemma-4-31B-it``'s chat template.
#
# They exist because rendering defects in this layer are silent: a malformed
# prompt or a mis-parsed tool call does not raise, it produces a low reward, so a
# multi-hour run looks like a weak model instead of a bug.
#
# Tests that require a tokenizer download are marked ``needs_tokenizer`` and skip
# when the model is unavailable; the rest are pure string assertions.


FIDELITY_TOOLS = [
    {
        "name": "check_solution",
        "description": "Run candidate code against tests.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source"}},
            "required": ["code"],
        },
    }
]


# --- format fidelity: things the spec pins exactly ------------------------


def test_gpt_oss_tool_call_matches_harmony_spec():
    """Tool-call header ordering and the ``<|constrain|>json`` marker are spec.

    Spec example (docs/format.md, "Receiving tool calls"):
        <|start|>assistant<|channel|>commentary to=functions.get_weather \
<|constrain|>json<|message|>{"location":"San Francisco"}<|call|>
    """
    rendered = render_tool_call("check_solution", {"code": "print(1)"}, "gpt_oss")
    assert rendered.startswith("<|start|>assistant<|channel|>commentary to=functions.check_solution ")
    assert "<|constrain|>json<|message|>" in rendered
    assert rendered.endswith("<|call|>")


def test_gpt_oss_tool_result_matches_harmony_spec():
    """Tool output frame is ``<|start|>{toolname} to=assistant...``.

    Spec: ``<|start|>functions.get_weather to=assistant<|channel|>commentary\
<|message|>{output}<|end|>``
    """
    rendered = render_tool_result("check_solution", '{"passed": true}', "gpt_oss")
    assert rendered == (
        "<|start|>functions.check_solution to=assistant<|channel|>commentary"
        '<|message|>{"passed": true}<|end|>'
    )


def test_gpt_oss_schema_uses_typescript_syntax_not_raw_json():
    """Function schemas must be TypeScript-like, not an inlined JSON schema.

    The spec is explicit about the shape and about why it matters ("It's
    important to stick to this format closely to improve accuracy of function
    calling"):

        type get_current_weather = (_: {
        // The city and state
        location: string,
        }) => any;

    Emitting the raw JSON schema inside the signature gives the model a
    grammar it was never trained on, and the failure is silent: worse tool-call
    accuracy, no error.
    """
    rendered = render_tool_schema(FIDELITY_TOOLS, "gpt_oss")
    assert "namespace functions {" in rendered
    assert "type check_solution = (_: {" in rendered
    assert "code: string," in rendered, "argument should be a TypeScript field"
    assert '"properties"' not in rendered, "raw JSON schema must not be inlined"
    assert '"type": "object"' not in rendered, "raw JSON schema must not be inlined"
    assert rendered.rstrip().endswith("<|end|>")


# --- F1: unknown families must not silently become gpt_oss ---------------


@pytest.mark.parametrize(
    "model",
    [
        "google/gemma-4-31B-it",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "some-internal/unreleased-model",
    ],
)
def test_family_for_model_does_not_silently_default(model):
    """An unrecognised model must not be handed gpt-oss tool grammar.

    Defaulting means a multi-turn run on an unsupported model trains against
    the wrong tool syntax for hours and reports it as poor reward, which is the
    hardest class of bug to notice. Either the family is known (gemma is, as of
    the gemma support work) or resolution fails loudly.
    """
    family = family_for_model(model)
    if family is None:
        return  # explicit "unknown" is the acceptable answer
    assert family != "gpt_oss" or "gpt" in model.lower() or "oss" in model.lower(), (
        f"{model!r} resolved to gpt_oss tool grammar; unknown models must resolve "
        "to their own family or to None, never to a silent default"
    )


# --- F4: a parser must reject what it cannot parse, never truncate --------


def test_qwen3_parser_rejects_closing_tag_inside_parameter_value():
    """Code containing a literal ``</parameter>`` must not be silently truncated.

    A code-writing model emitting the string ``"</parameter>"`` inside its
    solution is plausible. Truncating at the first match yields a syntactically
    broken program that then fails its tests, so the model is punished for a
    parser bug. Rejecting (or round-trip-verifying) is correct; guessing is not.
    """
    code = 'print("</parameter>")\nprint("rest")'
    xml = (
        "<tool_call><function=check_solution>"
        f"<parameter=code>{code}</parameter>"
        "</function></tool_call>"
    )
    calls = parse_tool_calls(xml, family="qwen3")
    if calls:
        assert calls[0].arguments.get("code") == code, (
            "parser truncated the value at an embedded closing tag instead of "
            "recovering the full argument"
        )


def test_qwen3_parser_handles_multiple_tool_calls():
    """Two calls in one message must both be returned, not silently reduced."""
    xml = (
        "<tool_call><function=check_solution><parameter=code>a</parameter>"
        "</function></tool_call>"
        "<tool_call><function=check_solution><parameter=code>b</parameter>"
        "</function></tool_call>"
    )
    calls = parse_tool_calls(xml, family="qwen3")
    assert len(calls) == 2, f"expected both calls, got {len(calls)}"
    assert [c.arguments.get("code") for c in calls] == ["a", "b"]


def test_qwen3_parser_reports_empty_parameter_as_empty_not_missing():
    """An explicitly empty argument is different from an absent one."""
    xml = (
        "<tool_call><function=check_solution><parameter=code></parameter>"
        "</function></tool_call>"
    )
    calls = parse_tool_calls(xml, family="qwen3")
    assert calls, "a well-formed call with an empty value must still parse"
    assert calls[0].arguments.get("code") == ""


# --- F3: a failed parse must be distinguishable from "no tool call" -------


@pytest.mark.parametrize(
    ("text", "description"),
    [
        ("<tool_call><function=check_solution><parameter=code>x</parameter></function>",
         "missing </tool_call>"),
        ("<tool_call><function=check_solution><parameter=code>x</function></tool_call>",
         "missing </parameter>"),
        ("<tool_call>garbage</tool_call>", "no function element"),
    ],
)
def test_malformed_tool_call_is_reported_not_swallowed(text, description):
    """A broken tool call must be surfaced, not returned as "no calls".

    Downstream, an empty parse result means "the model gave a final answer",
    so a fumbled call is graded as a deliberate answer. The reward signal then
    encodes a parser outcome rather than model behaviour. Some channel must
    distinguish the two: a second return value, an ``UnparsedToolCall``
    sentinel, or a raised error the env converts into corrective feedback.
    """
    result = parse_tool_calls(text, family="qwen3")
    assert result == [] or result, "sanity"
    unparsed = getattr(result, "unparsed", None)
    assert unparsed, (
        f"malformed call ({description}) returned {result!r} with no way for the "
        "caller to tell a parse failure from an intentional final answer"
    )
