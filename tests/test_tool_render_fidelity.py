"""Fidelity tests for tool-call rendering and parsing.

These tests assert what each model family was actually trained on, verified
against primary sources rather than inference:

* gpt-oss: the harmony response format specification
  (https://github.com/openai/harmony/blob/main/docs/format.md), cross-checked
  against ``openai/gpt-oss-20b``'s own chat template.
* qwen3: ``Qwen/Qwen3.6-27B``'s chat template.
* gemma: ``google/gemma-4-31B-it``'s chat template.

They exist because rendering defects in this layer are silent: a malformed
prompt or a mis-parsed tool call does not raise, it produces a low reward, so a
multi-hour run looks like a weak model instead of a bug.

Tests that require a tokenizer download are marked ``needs_tokenizer`` and skip
when the model is unavailable; the rest are pure string assertions.
"""

from __future__ import annotations

import pytest

from training_sessions_cookbook.rl.envs.tool_render import (
    family_for_model,
    parse_tool_calls,
    render_tool_call,
    render_tool_result,
    render_tool_schema,
)

TOOLS = [
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
    rendered = render_tool_schema(TOOLS, "gpt_oss")
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
