"""Tool-call render/parse layer for the multi-turn Code RL loop.

Covers the slice of renderer behaviour the agent-tool loop needs:

* declare a tool schema into the system/developer prompt,
* render an assistant tool call,
* render a tool result fed back on the next turn,
* parse tool calls back out of model output.

The initial target family is **gpt-oss (harmony)**, which emits tool calls on the ``commentary`` channel addressed to a recipient like ``functions.check_solution`` with a JSON arguments object. Rendering and parsing round-trip: for any code string X, ``parse_tool_calls(render_tool_call("check_solution", {"code": X}))`` recovers ``[ToolCall("check_solution", {"code": X})]``. Argument encoding goes through :func:`json.dumps`, so quotes, newlines, and harmony control tokens inside X are escaped and recovered exactly.

Dispatch is keyed by model family so more families (e.g. ``qwen3``) can be added without changing call sites: callers pass ``family=...`` or resolve it once via :func:`family_for_model`.

Standard library only, plus the shared :class:`~envs.base.Message` / :class:`~envs.base.ToolCall` types.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .base import Message, ToolCall  # Message kept for parity / typing

# ---------------------------------------------------------------------------
# Family dispatch
# ---------------------------------------------------------------------------

# Default family used when nothing else matches. gpt-oss was the family used by the prior Code RL run, so it is the sensible default.
DEFAULT_FAMILY = "gpt_oss"

# Maps a substring found in a model name (lower-cased) to a renderer family. Ordered by match precedence (first hit wins). qwen3 is reserved for a later task; it is listed so call sites resolve consistently once it is implemented.
FAMILY: "list[tuple[str, str]]" = [
    ("gpt-oss", "gpt_oss"),
    ("gpt_oss", "gpt_oss"),
    ("qwen", "qwen3"),
]


def family_for_model(model_name: Optional[str]) -> str:
    """Resolve a renderer family from a model name (substring match).

    Matching is case-insensitive and operates on the trailing path segment of ``model_name`` so values like ``"openai/gpt-oss-20b"`` or ``"arn:.../model/gpt-oss-20b"`` resolve to ``"gpt_oss"``. Defaults to :data:`DEFAULT_FAMILY` (``"gpt_oss"``) when nothing matches or the name is empty.
    """

    name = (model_name or "").rstrip("/").split("/")[-1].lower()
    for needle, family in FAMILY:
        if needle in name:
            return family
    return DEFAULT_FAMILY


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
# Rendering
# ---------------------------------------------------------------------------


def render_tool_schema(tools: list[dict], family: str = "gpt_oss") -> str:
    """Render the tool-declaration block injected into the prompt.

    For gpt-oss this is a ``developer``-role message that lists each function name, description and a JSON schema of its parameters, plus an instruction describing how to invoke a tool (write to the ``commentary`` channel with recipient ``functions.<name>`` and a JSON arguments object).

    Returns a string ready to be concatenated into the system/developer prompt.
    """

    if family != "gpt_oss":
        raise ValueError(f"unsupported tool-renderer family: {family!r}")

    lines: list[str] = []
    lines.append(f"{_START}developer{_MESSAGE}# Tools")
    lines.append("")
    lines.append("## functions")
    lines.append("")
    lines.append("namespace functions {")
    for tool in tools:
        name = tool["name"]
        description = tool.get("description", "")
        params = tool.get("parameters", {})
        lines.append("")
        if description:
            lines.append(f"// {description}")
        lines.append(f"type {name} = (_: {json.dumps(params)}) => any;")
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


def render_tool_call(name: str, arguments: dict, family: str = "gpt_oss") -> str:
    """Render an assistant tool call (the inverse of :func:`parse_tool_calls`).

    For gpt-oss this emits a ``commentary``-channel assistant message addressed to ``functions.<name>`` whose body is ``json.dumps(arguments)``. Using ``json.dumps`` guarantees correct escaping of quotes / newlines / control tokens inside the argument values, which is what makes the render -> parse round-trip hold for arbitrary strings.
    """

    if family != "gpt_oss":
        raise ValueError(f"unsupported tool-renderer family: {family!r}")

    body = json.dumps(arguments)
    return (
        f"{_START}assistant{_CHANNEL}commentary to=functions.{name} "
        f"{_CONSTRAIN}json{_MESSAGE}{body}{_CALL}"
    )


def render_tool_result(name: str, content: str, family: str = "gpt_oss") -> str:
    """Render a tool result message appended to the conversation next turn.

    For gpt-oss this is a ``functions.<name>``-role message on the ``commentary`` channel carrying ``content`` (typically a short JSON or text description of the sandbox grading outcome), addressed back to the assistant.
    """

    if family != "gpt_oss":
        raise ValueError(f"unsupported tool-renderer family: {family!r}")

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


def parse_tool_calls(assistant_text: str, family: str = "gpt_oss") -> list[ToolCall]:
    """Extract tool calls from model output.

    For gpt-oss, finds a ``commentary``-channel message addressed to ``functions.<name>`` whose body is a JSON object and returns ``[ToolCall(name=<name>, arguments=<parsed json>)]``. Falls back to a simpler fenced ``` ```tool_call\\n{"name":..., "arguments":{...}}\\n``` ``` block so the round-trip stays robust across renderings.

    Returns ``[]`` when the model produced a final answer (no tool call) or when parsing fails, callers treat an empty list as "no tool call this turn".
    """

    if family != "gpt_oss":
        raise ValueError(f"unsupported tool-renderer family: {family!r}")

    if not assistant_text:
        return []

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
        return calls

    # 2) Fallback: fenced tool_call block with explicit name + arguments.
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
