"""Tool-call render/parse layer for the multi-turn Code RL loop.

Covers the slice of renderer behaviour the agent-tool loop needs:

* declare a tool schema into the system/developer prompt,
* render an assistant tool call,
* render a tool result fed back on the next turn,
* parse tool calls back out of model output.

Three families are implemented, each matching the format its model was actually
trained on. The shapes below were read off the models' own chat templates (and,
for gpt-oss, the published harmony specification) rather than inferred, because
a mismatch here does not raise: the model simply stops recognising tool calls,
which surfaces as a mediocre reward rather than an error.

* **gpt-oss (harmony)** declares functions in a ``developer`` message using
  harmony's TypeScript-like syntax (``type check_solution = (_: {code:
  string,}) => any;`` inside ``namespace functions``), emits calls on the
  ``commentary`` channel addressed to ``functions.<name>`` with a JSON
  arguments object, and returns results as a ``functions.<name>``-role message.
* **qwen3** declares tools as JSON signatures inside ``<tools>...</tools>`` in a
  system turn; the model invokes them with an XML block::

      <tool_call>
      <function=name>
      <parameter=key>
      raw multi-line value
      </parameter>
      </function>
      </tool_call>

  and results return inside ``<tool_response>...</tool_response>`` in a user
  turn, using the same ``<|im_start|>role ... <|im_end|>`` markers as the repo's
  other Qwen prompts.
* **gemma** uses its own sentinel tokens - declarations as
  ``<|tool>declaration:NAME{...}<tool|>`` in a system turn, calls as
  ``<|tool_call>call:NAME{code:<|"|>src<|"|>}<tool_call|>``, results as
  ``<|tool_response>response:NAME{value:<|"|>...<|"|>}<tool_response|>``. Object
  keys are emitted in sorted order and schema types upper-cased, matching the
  template; string scalars are wrapped in the ``<|"|>`` sentinel while numbers
  and booleans are bare. Every sentinel is a single vocabulary token, so a
  rendered frame re-encodes to exactly the ids the model saw in training - which
  is what lets gemma be rendered as text fragments at all.

Round-trip guarantees, measured rather than assumed (see
tests/test_render_token_equivalence.py and tests/test_tool_render_fidelity.py):

* gpt-oss arguments go through :func:`json.dumps` and are recovered exactly for
  arbitrary strings, including quotes, newlines and control tokens.
* qwen3 parameter values are raw text (the format has no escaping), and a value
  is recovered exactly unless it contains the literal sequence that both closes
  one parameter and opens the next (``</parameter><parameter=``). A lone
  ``</parameter>``, ``</function>`` or ``</tool_call>`` inside a value survives.
* gemma values are recovered exactly unless the value contains gemma's own
  ``<|"|>`` quote sentinel, which the format cannot represent.

Dispatch is keyed by model family; callers pass ``family=...`` or resolve it once
via :func:`family_for_model`, which returns ``None`` for an unrecognised model
rather than guessing a default - handing a model the wrong tool grammar trains
silently against syntax it never saw. Unknown families raise ``ValueError``
naming :data:`SUPPORTED_FAMILIES`.

:func:`parse_tool_calls` returns a :class:`ParsedToolCalls` (a ``list``
subclass, so existing callers are unaffected) whose ``unparsed`` attribute
carries text that looked like a tool call but could not be parsed, so a
malformed call can be told apart from a deliberate final answer.

Standard library only, plus the shared :class:`~envs.base.ToolCall` type.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .base import ToolCall

# ---------------------------------------------------------------------------
# Family dispatch
# ---------------------------------------------------------------------------

# Maps a substring found in a model name (lower-cased) to a renderer family.
# Ordered by match precedence (first hit wins). The qwen3 family implements the
# Qwen3.5+/3.6 XML tool convention (older Qwen3 checkpoints used a JSON payload
# and would need their own family).
FAMILY: "list[tuple[str, str]]" = [
    ("gpt-oss", "gpt_oss"),
    ("gpt_oss", "gpt_oss"),
    ("qwen", "qwen3"),
    ("gemma", "gemma"),
]


# Families with a full render/parse implementation below.
SUPPORTED_FAMILIES = ("gpt_oss", "qwen3", "gemma")


def _require_supported(family: str) -> None:
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"unsupported tool-renderer family: {family!r} "
            f"(supported: {', '.join(SUPPORTED_FAMILIES)})"
        )


def family_for_model(model_name: Optional[str]) -> Optional[str]:
    """Resolve a renderer family from a model name, or ``None`` if unknown.

    Matching is case-insensitive and operates on the trailing path segment of
    ``model_name``, so ``"openai/gpt-oss-20b"`` and
    ``"arn:.../model/gpt-oss-20b"`` both resolve to ``"gpt_oss"``.

    Returns ``None`` for an unrecognised or empty name rather than falling back
    to a default family. A silent default is the worst outcome available here:
    the run does not fail, it trains for hours against tool syntax the model was
    never taught, and the damage surfaces only as a low reward that looks like
    model weakness. Callers that need a concrete family for multi-turn tool use
    must treat ``None`` as a configuration error (see
    :class:`~training_sessions_cookbook.rl.envs.code_rl.CodeRLEnv`); callers
    doing single-turn work can ignore it, because no tool grammar is involved.
    """

    name = (model_name or "").rstrip("/").split("/")[-1].lower()
    for needle, family in FAMILY:
        if needle in name:
            return family
    return None


# ---------------------------------------------------------------------------
# Tool schema representation
# ---------------------------------------------------------------------------


def make_tool_spec(name: str, description: str, parameters: dict) -> dict:
    """Build a single tool spec dict ``{name, description, parameters}``.

    ``parameters`` is a JSON-schema-style object describing the tool's arguments (``{"type": "object", "properties": {...}, "required": [...]}``).
    """

    return {"name": name, "description": description, "parameters": parameters}


def check_solution_tool_spec() -> dict:
    """Return the schema for the ``check_solution(code: str)`` tool.

    This is the single tool the Code RL multi-turn loop exposes: the model submits a candidate solution as a string ``code`` argument; the env runs it in the sandbox and returns the grading result as a tool message.
    """

    return make_tool_spec(
        name="check_solution",
        description=(
            "Run the candidate solution against the problem's test cases in a "
            "sandbox and return whether it passes. Submit the complete program "
            "as the `code` argument."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The complete solution source code to execute.",
                }
            },
            "required": ["code"],
        },
    )


# ---------------------------------------------------------------------------
# gpt-oss (harmony) constants
# ---------------------------------------------------------------------------

# Harmony control tokens (kept as named constants so render/parse agree).
_START = "<|start|>"
_CHANNEL = "<|channel|>"
_CONSTRAIN = "<|constrain|>"
_MESSAGE = "<|message|>"
_CALL = "<|call|>"
_END = "<|end|>"

# ---------------------------------------------------------------------------
# qwen3 constants
# ---------------------------------------------------------------------------

# Qwen chat frames (same markers the repo's Qwen prompt templates emit) and
# the XML-style tags Qwen models use for tool declaration, invocation, and
# results.
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_TOOLS_OPEN = "<tools>"
_TOOLS_CLOSE = "</tools>"
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_TOOL_RESP_OPEN = "<tool_response>"
_TOOL_RESP_CLOSE = "</tool_response>"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# gemma
# ---------------------------------------------------------------------------
#
# Gemma frames tool use with its own sentinel tokens, each a single id in the
# vocabulary, so a rendered string re-encodes to exactly the ids the model was
# trained on (asserted in tests/test_render_token_equivalence.py):
#
#   declaration  <|tool>declaration:NAME{description:<|"|>D<|"|>,parameters:{...}}<tool|>
#   call         <|tool_call>call:NAME{code:<|"|>print(1)<|"|>,retries:2}<tool_call|>
#   response     <|tool_response>response:NAME{value:<|"|>ok<|"|>}<tool_response|>
#
# The shapes below were read off google/gemma-4-31B-it's own chat template
# rather than inferred: object keys are emitted in sorted order, JSON schema
# types are upper-cased, string scalars are wrapped in the <|"|> sentinel while
# numbers and booleans are bare, and a tool with no parameters omits the
# ``properties`` key entirely.

_GEMMA_QUOTE = '<|"|>'
_GEMMA_TOOL_OPEN, _GEMMA_TOOL_CLOSE = "<|tool>", "<tool|>"
_GEMMA_CALL_OPEN, _GEMMA_CALL_CLOSE = "<|tool_call>", "<tool_call|>"
_GEMMA_RESP_OPEN, _GEMMA_RESP_CLOSE = "<|tool_response>", "<tool_response|>"
_GEMMA_TURN_OPEN, _GEMMA_TURN_CLOSE = "<|turn>", "<turn|>"


def _gemma_str(value: str) -> str:
    """Wrap a string scalar in gemma's quote sentinel."""
    return f"{_GEMMA_QUOTE}{value}{_GEMMA_QUOTE}"


def _gemma_value(value: "Any") -> str:
    """Encode one argument value the way gemma's template does.

    Booleans are checked before numbers because ``bool`` is a subclass of ``int``
    and gemma renders them as bare ``true``/``false``, not ``1``/``0``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return _gemma_str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_gemma_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_gemma_value(v)}" for k, v in sorted(value.items())) + "}"
    if value is None:
        return _gemma_str("")
    return _gemma_str(str(value))


def _gemma_schema_node(node: dict) -> str:
    """Render one JSON-schema node in gemma's declaration syntax.

    Keys are emitted in sorted order and ``type`` is upper-cased, matching the
    template. ``enum`` and ``items`` are rendered recursively.
    """
    parts: "list[str]" = []
    for key in sorted(node):
        value = node[key]
        if key == "type" and isinstance(value, str):
            parts.append(f"type:{_gemma_str(value.upper())}")
        elif key == "properties" and isinstance(value, dict):
            inner = ",".join(f"{k}:{_gemma_schema_node(v or {})}" for k, v in sorted(value.items()))
            parts.append("properties:{" + inner + "}")
        elif key == "items" and isinstance(value, dict):
            parts.append("items:" + _gemma_schema_node(value))
        elif key == "enum" and isinstance(value, (list, tuple)):
            parts.append("enum:[" + ",".join(_gemma_value(v) for v in value) + "]")
        elif key == "required" and isinstance(value, (list, tuple)):
            if not value:
                continue
            parts.append("required:[" + ",".join(_gemma_str(str(v)) for v in value) + "]")
        else:
            parts.append(f"{key}:{_gemma_value(value)}")
    return "{" + ",".join(parts) + "}"


def _render_tool_schema_gemma(tools: "list[dict]") -> str:
    """Render gemma's tool declarations as a complete system turn."""
    blocks: "list[str]" = []
    for tool in tools:
        params = dict(tool.get("parameters") or {})
        # A parameterless tool declares only its type; an empty ``properties``
        # map is omitted, as the template does.
        if not params.get("properties"):
            params.pop("properties", None)
        fields = [f"description:{_gemma_str(tool.get('description', ''))}"]
        fields.append(f"parameters:{_gemma_schema_node(params)}")
        blocks.append(
            f"{_GEMMA_TOOL_OPEN}declaration:{tool['name']}{{{','.join(fields)}}}{_GEMMA_TOOL_CLOSE}"
        )
    return f"{_GEMMA_TURN_OPEN}system\n" + "".join(blocks) + f"{_GEMMA_TURN_CLOSE}\n"


def _render_tool_call_gemma(name: str, arguments: dict) -> str:
    """Render an assistant tool call in gemma's syntax (arguments sorted)."""
    body = ",".join(f"{k}:{_gemma_value(v)}" for k, v in sorted((arguments or {}).items()))
    return f"{_GEMMA_CALL_OPEN}call:{name}{{{body}}}{_GEMMA_CALL_CLOSE}"


def _render_tool_result_gemma(name: str, content: str) -> str:
    """Render a tool result in gemma's syntax."""
    return (
        f"{_GEMMA_RESP_OPEN}response:{name}{{value:{_gemma_str(content)}}}"
        f"{_GEMMA_RESP_CLOSE}"
    )


_GEMMA_CALL_RE = re.compile(
    r"<\|tool_call>call:(?P<name>[^{\s]+)\{(?P<body>.*?)\}<tool_call\|>",
    re.DOTALL,
)
# key:<|"|>value<|"|>  or  key:bare   (bare covers numbers, booleans, arrays)
_GEMMA_ARG_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*):"
    r'(?:<\|"\|>(?P<quoted>.*?)<\|"\|>|(?P<bare>\[[^\]]*\]|[^,}]+))',
    re.DOTALL,
)


def _gemma_parse_scalar(raw: str) -> "Any":
    """Turn a bare gemma scalar back into a Python value."""
    text = raw.strip()
    if text == "true":
        return True
    if text == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_tool_calls_gemma(assistant_text: str) -> "tuple[list[ToolCall], list[str]]":
    """Parse gemma tool calls, reporting fragments that look like broken calls."""
    calls: "list[ToolCall]" = []
    unparsed: "list[str]" = []
    for m in _GEMMA_CALL_RE.finditer(assistant_text):
        name = m.group("name").strip()
        body = m.group("body")
        arguments: dict = {}
        for am in _GEMMA_ARG_RE.finditer(body):
            key = am.group("key")
            if am.group("quoted") is not None:
                arguments[key] = am.group("quoted")
            else:
                bare = am.group("bare")
                if bare.strip().startswith("["):
                    items = [i for i in bare.strip()[1:-1].split(",") if i.strip()]
                    arguments[key] = [
                        i.strip().strip('<|"|>') if _GEMMA_QUOTE in i else _gemma_parse_scalar(i)
                        for i in items
                    ]
                else:
                    arguments[key] = _gemma_parse_scalar(bare)
        if not name:
            unparsed.append(m.group(0))
            continue
        if not arguments and body.strip():
            unparsed.append(m.group(0))
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls, unparsed


def _ts_type(schema: dict) -> str:
    """Map one JSON-schema property to the TypeScript-ish type harmony expects.

    The harmony specification declares tools with a TypeScript-like syntax and
    is explicit that adherence matters for tool-call accuracy, so the JSON
    schema is translated rather than inlined:

        {"type": "string"}                  -> string
        {"type": "integer"}                 -> number
        {"enum": ["celsius", "fahrenheit"]} -> "celsius" | "fahrenheit"
        {"type": "array", "items": {...}}   -> string[]

    Anything unrecognised degrades to ``any``, which is a valid declaration the
    model has seen, rather than leaking raw JSON into the signature.
    """
    if not isinstance(schema, dict):
        return "any"
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return " | ".join(json.dumps(v) for v in schema["enum"])
    json_type = schema.get("type")
    if json_type == "array":
        return f"{_ts_type(schema.get('items', {}))}[]"
    return {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "object": "object",
    }.get(json_type, "any")


def _render_function_declaration(tool: dict) -> "list[str]":
    """Render one function as harmony's TypeScript-like declaration.

    Follows the formatting practices the specification lists: no-argument
    functions are declared ``() => any``; functions with arguments name the
    parameter ``_`` and inline the field definitions; a field's description
    becomes a comment on the line above it; optional fields (those absent from
    ``required``) are marked with ``?``; the return type is always ``any``.
    """
    name = tool["name"]
    description = tool.get("description", "")
    params = tool.get("parameters", {}) or {}
    properties = params.get("properties") or {}
    required = set(params.get("required") or [])

    lines: "list[str]" = []
    if description:
        lines.append(f"// {description}")
    if not properties:
        lines.append(f"type {name} = () => any;")
        return lines

    lines.append(f"type {name} = (_: {{")
    for field, field_schema in properties.items():
        field_desc = (field_schema or {}).get("description")
        if field_desc:
            lines.append(f"// {field_desc}")
        optional = "" if field in required else "?"
        lines.append(f"{field}{optional}: {_ts_type(field_schema)},")
    lines.append("}) => any;")
    return lines


def render_tool_schema(tools: list[dict], family: str = "gpt_oss") -> str:
    """Render the tool-declaration block injected into the prompt.

    For gpt-oss this is a ``developer``-role message declaring each function in harmony's TypeScript-like syntax (see :func:`_render_function_declaration`), plus an instruction describing how to invoke a tool (write to the ``commentary`` channel with recipient ``functions.<name>`` and a JSON arguments object).

    For qwen3 it is a ``system``-role turn declaring each tool as a JSON
    signature inside ``<tools>...</tools>``, followed by the exact
    ``<tool_call>`` invocation format the model should reply with.

    For gemma it is a ``system`` turn of ``<|tool>declaration:NAME{...}<tool|>``
    blocks, with keys sorted and schema types upper-cased as gemma's own
    template emits them; a tool with no parameters omits ``properties``.

    Returns a string ready to be concatenated into the system/developer prompt.
    """

    _require_supported(family)
    if family == "qwen3":
        return _render_tool_schema_qwen3(tools)
    if family == "gemma":
        return _render_tool_schema_gemma(tools)

    lines: list[str] = []
    lines.append(f"{_START}developer{_MESSAGE}# Tools")
    lines.append("")
    lines.append("## functions")
    lines.append("")
    lines.append("namespace functions {")
    for tool in tools:
        lines.append("")
        lines.extend(_render_function_declaration(tool))
    lines.append("")
    lines.append("} // namespace functions")
    lines.append("")
    lines.append(
        "To call a function, respond on the commentary channel addressed to the "
        "function. Format the message exactly as:"
    )
    lines.append(
        f'{_START}assistant{_CHANNEL}commentary to=functions.<name> '
        f'{_CONSTRAIN}json{_MESSAGE}{{...arguments as JSON...}}{_CALL}'
    )
    lines.append(
        "Provide a final answer on the `final` channel when no tool call is needed."
    )
    block = "\n".join(lines) + _END
    return block


def _render_tool_schema_qwen3(tools: list[dict]) -> str:
    """Qwen3 tool declaration, mirroring the model's own chat template.

    A ``system`` turn: JSON tool signatures inside ``<tools>``, then the XML
    ``<function=...>`` / ``<parameter=...>`` invocation format the Qwen3.6
    template instructs the model to reply with.
    """
    lines: list[str] = []
    lines.append(f"{_IM_START}system")
    lines.append("# Tools")
    lines.append("")
    lines.append("You have access to the following functions:")
    lines.append("")
    lines.append(_TOOLS_OPEN)
    for tool in tools:
        signature = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }
        lines.append(json.dumps(signature))
    lines.append(_TOOLS_CLOSE)
    lines.append("")
    lines.append("If you choose to call a function ONLY reply in the following "
                 "format with NO suffix:")
    lines.append("")
    lines.append(_TOOL_CALL_OPEN)
    lines.append("<function=example_function_name>")
    lines.append("<parameter=example_parameter_1>")
    lines.append("value_1")
    lines.append("</parameter>")
    lines.append("</function>")
    lines.append(_TOOL_CALL_CLOSE)
    lines.append("")
    lines.append("Reminder:")
    lines.append("- Function calls MUST follow the specified format: an inner "
                 "<function=...></function> block must be nested within "
                 "<tool_call></tool_call> XML tags")
    lines.append("- Required parameters MUST be specified")
    lines.append("- Reply with your final answer directly (no tool call) when done.")
    return "\n".join(lines) + _IM_END


def render_tool_call(name: str, arguments: dict, family: str = "gpt_oss") -> str:
    """Render an assistant tool call (the inverse of :func:`parse_tool_calls`).

    For gpt-oss this emits a ``commentary``-channel assistant message addressed
    to ``functions.<name>`` whose body is ``json.dumps(arguments)``; the JSON
    escaping is what makes that family's round-trip hold for arbitrary strings,
    including quotes, newlines and harmony control tokens.

    For qwen3 it emits an assistant turn carrying a ``<tool_call>`` block whose
    ``<parameter=key>`` values are **raw text**, because that is what the format
    specifies - there is no escaping mechanism, so the round-trip has the narrow
    limit documented in the module docstring.

    For gemma it emits ``<|tool_call>call:NAME{key:value,...}<tool_call|>`` with
    arguments sorted, strings wrapped in the ``<|"|>`` sentinel, and numbers and
    booleans bare.
    """

    _require_supported(family)
    if family == "gemma":
        return _render_tool_call_gemma(name, arguments)
    if family == "qwen3":
        parts = [f"{_IM_START}assistant", _TOOL_CALL_OPEN, f"<function={name}>"]
        for key, value in arguments.items():
            parts.append(f"<parameter={key}>")
            # Values are raw text in this format (multi-line allowed, no
            # escaping); non-string values are serialized as JSON.
            parts.append(value if isinstance(value, str) else json.dumps(value))
            parts.append("</parameter>")
        parts.append("</function>")
        parts.append(_TOOL_CALL_CLOSE)
        return "\n".join(parts) + _IM_END

    body = json.dumps(arguments)
    return (
        f"{_START}assistant{_CHANNEL}commentary to=functions.{name} "
        f"{_CONSTRAIN}json{_MESSAGE}{body}{_CALL}"
    )


def render_tool_result(name: str, content: str, family: str = "gpt_oss") -> str:
    """Render a tool result message appended to the conversation next turn.

    For gpt-oss this is a ``functions.<name>``-role message on the
    ``commentary`` channel carrying ``content`` (typically a short JSON or text
    description of the sandbox grading outcome), addressed back to the
    assistant. For qwen3, per Qwen's chat convention, the result returns as a
    ``user`` turn wrapping ``content`` in ``<tool_response>`` tags (opening the
    next assistant frame is the caller's concern, as with the repo's other Qwen
    prompts). For gemma it is
    ``<|tool_response>response:NAME{value:<|"|>content<|"|>}<tool_response|>``.

    The returned string is a complete chat frame. Callers append it to the
    running prompt as-is; passing it through a prompt template again would nest
    it inside another message and malform the turn.
    """

    _require_supported(family)
    if family == "gemma":
        return _render_tool_result_gemma(name, content)
    if family == "qwen3":
        return (
            f"{_IM_START}user\n{_TOOL_RESP_OPEN}\n{content}\n"
            f"{_TOOL_RESP_CLOSE}{_IM_END}"
        )

    return (
        f"{_START}functions.{name} to=assistant{_CHANNEL}commentary"
        f"{_MESSAGE}{content}{_END}"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _scan_json_object(text: str, start: int) -> "Optional[tuple[Any, int]]":
    """Scan a single balanced JSON object starting at ``text[start]``.

    Returns ``(parsed_obj, end_index)`` where ``end_index`` is the index just past the closing brace, or ``None`` if no valid object can be parsed. The scan respects string literals and escapes so braces / control tokens inside a string value do not confuse the matcher. This is what lets us recover the arguments JSON exactly even when an argument value contains harmony control tokens like ``<|call|>``.
    """

    n = len(text)
    # Skip leading whitespace to find the opening brace.
    i = start
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] != "{":
        return None

    depth = 0
    in_string = False
    escaped = False
    for j in range(i, n):
        ch = text[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[i : j + 1]
                try:
                    return json.loads(candidate), j + 1
                except (ValueError, json.JSONDecodeError):
                    return None
    return None


# Matches the harmony tool-call header up to (and including) the `<|message|>` marker, capturing the recipient function name. Tolerant of optional `<|constrain|>json` and surrounding whitespace.
_HARMONY_CALL_RE = re.compile(
    r"to=functions\.(?P<name>[A-Za-z0-9_\.]+)\s*"
    r"(?:" + re.escape(_CONSTRAIN) + r"\s*\w+\s*)?"
    + re.escape(_MESSAGE),
)

# Fallback: a fenced ```tool_call\n{...}\n``` block carrying {"name", "arguments"}.
_FENCED_CALL_RE = re.compile(
    r"```(?:tool_call|json)?\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)

# qwen3 (Qwen3.5+/3.6 XML convention): a <tool_call> block wrapping a single
# <function=NAME> element whose parameters are raw-text <parameter=KEY> blocks.
_QWEN_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\n]+)>(?P<body>.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
# A parameter value ends at the ``</parameter>`` that is followed by the next
# ``<parameter=`` or by the end of the call body. Anchoring on what FOLLOWS the
# close tag (rather than stopping at the first one) keeps values that themselves
# contain the literal text ``</parameter>`` intact - plausible output for a
# code-writing model, and previously truncated into a syntactically broken
# program that then failed its tests, punishing the model for a parser bug.
_QWEN_PARAM_RE = re.compile(
    r"<parameter=(?P<key>[^>\n]+)>\n?(?P<value>.*?)\n?</parameter>"
    r"(?=\s*(?:<parameter=|$))",
    re.DOTALL,
)


def _parse_tool_calls_qwen3(assistant_text: str) -> "tuple[list[ToolCall], list[str]]":
    """Parse Qwen3.5+/3.6 XML tool calls.

    Parameter values are raw text framed by newlines in the rendering, so one
    leading and one trailing newline are stripped; interior whitespace and
    newlines are preserved verbatim.

    Returns ``(calls, unparsed)``. A call whose name is readable but whose
    parameter block is malformed (an unclosed ``<parameter=...>``, say) yields
    both a call with the arguments that did parse *and* an entry in ``unparsed``:
    the call is still worth dispatching, because the tool layer will answer with
    a specific "missing argument" message the model can act on, while the
    ``unparsed`` entry keeps the parse defect visible instead of leaving it to be
    inferred from a reward dip.
    """
    calls: list[ToolCall] = []
    unparsed: list[str] = []
    for m in _QWEN_CALL_RE.finditer(assistant_text):
        name = m.group("name").strip()
        body = m.group("body")
        arguments: dict = {}
        for pm in _QWEN_PARAM_RE.finditer(body):
            arguments[pm.group("key").strip()] = pm.group("value")
        if not name:
            unparsed.append(m.group(0))
            continue
        if not arguments and "<parameter=" in body:
            # Parameter syntax was attempted but none of it parsed.
            unparsed.append(m.group(0))
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls, unparsed


class ParsedToolCalls(list):
    """The tool calls found in one model turn, plus what could not be parsed.

    Subclasses ``list`` so existing callers - which treat the return value as a
    plain list of :class:`~envs.base.ToolCall` and read an empty list as "the
    model gave a final answer" - keep working unchanged.

    The :attr:`unparsed` list closes a reward-attribution hole. A malformed tool
    call used to be indistinguishable from no tool call at all, so a model that
    tried to call a tool and got the syntax slightly wrong was graded as if it
    had deliberately answered. Callers that care can now tell the two apart and
    feed corrective feedback back to the model (or count the occurrences)
    instead of silently mis-grading the turn.

    Attributes:
        unparsed: Raw text of each fragment that looked like a tool call but
            could not be parsed. Empty when the turn contained no tool-call
            syntax at all.
    """

    def __init__(self, calls=(), unparsed=()):
        super().__init__(calls)
        self.unparsed: "list[str]" = list(unparsed)


# Fragments that signal an attempted tool call, used to tell "no tool call" from
# "tool call the model got wrong". Checked per family.
_ATTEMPT_MARKERS = {
    "qwen3": ("<tool_call>", "<function=", "<parameter="),
    "gemma": ("<|tool_call>", "call:"),
    "gpt_oss": ("to=functions.", "<|call|>", "```tool_call"),
}


def _looks_like_attempted_call(text: str, family: str) -> bool:
    markers = _ATTEMPT_MARKERS.get(family, ())
    return any(marker in text for marker in markers)


def parse_tool_calls(assistant_text: str, family: str = "gpt_oss") -> ParsedToolCalls:
    """Extract tool calls from model output.

    For gpt-oss, finds a ``commentary``-channel message addressed to ``functions.<name>`` whose body is a JSON object and returns ``[ToolCall(name=<name>, arguments=<parsed json>)]``. Falls back to a simpler fenced ``` ```tool_call\\n{"name":..., "arguments":{...}}\\n``` ``` block so the round-trip stays robust across renderings.

    For qwen3, finds ``<tool_call><function=NAME>...</function></tool_call>`` blocks and collects their raw-text ``<parameter=KEY>`` values.

    Returns a :class:`ParsedToolCalls` - a list of :class:`ToolCall`, so callers
    that treat the result as a plain list are unaffected. When the turn contained
    tool-call syntax that could not be parsed, the result is empty but
    ``.unparsed`` holds the offending text, which lets a caller distinguish "the
    model answered" from "the model tried to call a tool and got it wrong"
    instead of grading the second as the first.
    """

    _require_supported(family)

    if not assistant_text:
        return ParsedToolCalls()

    if family == "gemma":
        calls, unparsed = _parse_tool_calls_gemma(assistant_text)
        if calls:
            return ParsedToolCalls(calls, unparsed)
        if _looks_like_attempted_call(assistant_text, "gemma"):
            return ParsedToolCalls((), [assistant_text])
        return ParsedToolCalls()

    if family == "qwen3":
        calls, unparsed = _parse_tool_calls_qwen3(assistant_text)
        if calls:
            return ParsedToolCalls(calls, unparsed)
        # Fall through to the fenced fallback shared across families.
        fenced = _parse_fenced_calls(assistant_text)
        if fenced:
            return ParsedToolCalls(fenced)
        if _looks_like_attempted_call(assistant_text, "qwen3"):
            return ParsedToolCalls((), [assistant_text])
        return ParsedToolCalls()

    calls: list[ToolCall] = []

    # 1) Harmony-style commentary tool calls. There may be more than one; scan each header occurrence and parse the JSON object that follows.
    for m in _HARMONY_CALL_RE.finditer(assistant_text):
        name = m.group("name")
        scanned = _scan_json_object(assistant_text, m.end())
        if scanned is None:
            continue
        args, _ = scanned
        if isinstance(args, dict):
            calls.append(ToolCall(name=name, arguments=args))

    if calls:
        return ParsedToolCalls(calls)

    # 2) Fallback: fenced tool_call block with explicit name + arguments.
    fenced = _parse_fenced_calls(assistant_text)
    if fenced:
        return ParsedToolCalls(fenced)
    if _looks_like_attempted_call(assistant_text, "gpt_oss"):
        return ParsedToolCalls((), [assistant_text])
    return ParsedToolCalls()


def _parse_fenced_calls(assistant_text: str) -> list[ToolCall]:
    """Shared fallback: fenced ```tool_call``` blocks with name + arguments."""
    calls: list[ToolCall] = []
    for m in _FENCED_CALL_RE.finditer(assistant_text):
        body = m.group("body").strip()
        try:
            obj = json.loads(body)
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments", {})
        if isinstance(name, str) and isinstance(args, dict):
            calls.append(ToolCall(name=name, arguments=args))
    return calls

# What to append after a tool-result frame so the model starts a fresh reply.
# Read off each model's own chat template with ``add_generation_prompt=True``
# after a ``tool`` message: gpt-oss and qwen3 open a new assistant frame; gemma
# continues the SAME model turn after ``<tool_response|>`` (its template emits
# no new ``<|turn>model``), so its opener is empty.
_GENERATION_OPENERS = {
    "gpt_oss": "<|start|>assistant",
    "qwen3": "<|im_start|>assistant\n",
    "gemma": "",
}


def generation_opener(family: str = "gpt_oss") -> str:
    """The frame opener to append after a tool result, before sampling again.

    Without this, from the second turn onward the model is asked to continue
    right after ``<|end|>`` / ``<|im_end|>`` with no open assistant frame -
    a token sequence it never saw in training.
    """
    _require_supported(family)
    return _GENERATION_OPENERS[family]
