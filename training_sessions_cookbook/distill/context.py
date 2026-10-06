"""Teacher contexts: what the teacher reads before grading the student's tokens.

The teacher never writes anything. It re-reads the student's exact sampled
token IDs, behind a prompt of the recipe's choosing, and returns a log-prob for
each one. The prompt is the design lever:

* :class:`SameContext` (OPD): the student's own prompt, verbatim.
* :class:`PrivilegedContext` (OPSD): the prompt plus a reference solution or
  hint from the dataset row, which the student never sees.
* :class:`FeedbackContext` (SDPO): the prompt plus feedback on this attempt
  (sandbox errors) and/or a sibling attempt that succeeded.

Alignment rule: the teacher's sequence is ``teacher_prompt + continuation``,
where ``continuation`` is the student's tokens after its first prompt (for
multi-turn episodes this includes earlier turns and tool results, which the
teacher sees exactly as the student did). The student's tokens are reused as
IDs and never re-tokenized, so position ``i`` of the teacher's tail is the same
token as position ``i`` of ``act_tokens``. This assumes the teacher shares the
student's tokenizer and chat template (same model family).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from training_sessions_cookbook.rl.prompts import encode_prompt


@dataclass(frozen=True)
class TeacherQuery:
    """One sequence for the teacher to score.

    ``tokens`` is the full sequence (teacher prompt + student continuation);
    the teacher's log-probs for the last ``n_tail`` tokens line up 1:1 with the
    transition's ``act_tokens``.
    """

    tokens: list[int]
    n_tail: int


def splice(teacher_prompt: list[int], student_prompt_len: int, transition) -> TeacherQuery:
    """Build a :class:`TeacherQuery` by swapping the student's first prompt for ``teacher_prompt``."""
    continuation = list(transition.obs_tokens[student_prompt_len:]) + list(transition.act_tokens)
    return TeacherQuery(list(teacher_prompt) + continuation, len(transition.act_tokens))


def row_field(row: Any, name: str) -> Any:
    """Read ``name`` off a dataset row (a dict, or an object such as ``DeepcoderTask``)."""
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def decode_text(tokenizer, tokens) -> str:
    """Decode without special tokens when the tokenizer supports it."""
    try:
        return tokenizer.decode(list(tokens), skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(list(tokens))


def _episode_info(episode) -> dict:
    return (episode[0].info or {}) if episode else {}


class TeacherContext:
    """Base class. Subclasses return the teacher's first observation, or ``None``.

    :meth:`queries` renders that observation with the run's chat template and
    splices every transition of the episode behind it. Returning ``None`` from
    :meth:`teacher_observation` means the teacher has nothing to add for this
    episode; it is left unscored (no distillation signal).
    """

    #: ``True`` when the teacher reads the student's prompt unchanged, so no
    #: re-rendering is needed.
    same_prompt = False

    def __init__(self, *, tokenizer, template_fn) -> None:
        self.tokenizer = tokenizer
        self.template_fn = template_fn

    def teacher_observation(self, episode, group) -> Optional[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def render(self, observation: str, tools=None) -> list[int]:
        rendered = self.template_fn(observation, tools=tools) if tools else self.template_fn(observation)
        return encode_prompt(rendered, self.tokenizer)

    def queries(self, episode, group) -> Optional[list[TeacherQuery]]:
        """Teacher queries for every transition of ``episode`` (``None`` = unscored)."""
        if not episode:
            return []
        if self.same_prompt:
            return [TeacherQuery(list(t.obs_tokens) + list(t.act_tokens), len(t.act_tokens))
                    for t in episode]
        observation = self.teacher_observation(episode, group)
        if observation is None:
            return None
        teacher_prompt = self.render(observation, _episode_info(episode).get("tools"))
        student_prompt_len = len(episode[0].obs_tokens)
        return [splice(teacher_prompt, student_prompt_len, t) for t in episode]


class SameContext(TeacherContext):
    """OPD: the teacher reads exactly the student's prompt."""

    same_prompt = True

    def teacher_observation(self, episode, group) -> Optional[str]:
        return _episode_info(episode).get("observation")


#: OPSD teacher prompt, following Zhao et al. (2026): the problem, the
#: reference solution, then an instruction to solve it independently.
OPSD_TEMPLATE = (
    "{observation}\n\n"
    "Here is a reference solution to this problem:\n\n"
    "{privileged}\n\n"
    "After understanding the reference solution, solve the problem yourself "
    "using your own approach."
)


def final_answer_only(text: str) -> str:
    """Reduce a GSM8K-style solution (``... #### 72``) to its final answer."""
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    return text.strip()


class PrivilegedContext(TeacherContext):
    """OPSD: prompt + privileged text from the dataset row (``row[field]``).

    ``mode="solution"`` passes the field through (GSM8K's ``answer`` is the full
    worked solution ending ``#### N``); ``mode="final_answer"`` keeps only the
    part after ``####``, a much weaker hint. Rows without the field are left
    unscored.
    """

    def __init__(self, *, tokenizer, template_fn, field: str = "answer",
                 mode: str = "solution", template: str = OPSD_TEMPLATE) -> None:
        super().__init__(tokenizer=tokenizer, template_fn=template_fn)
        if mode not in ("solution", "final_answer"):
            raise ValueError(f"unknown privileged mode {mode!r}")
        self.field = field
        self.mode = mode
        self.template = template

    def privileged_text(self, row) -> Optional[str]:
        value = row_field(row, self.field)
        if value is None or not str(value).strip():
            return None
        text = str(value)
        return final_answer_only(text) if self.mode == "final_answer" else text.strip()

    def teacher_observation(self, episode, group) -> Optional[str]:
        info = _episode_info(episode)
        observation = info.get("observation")
        privileged = self.privileged_text(info.get("prompt"))
        if observation is None or privileged is None:
            return None
        return self.template.format(observation=observation, privileged=privileged)


#: SDPO teacher prompt, following the reprompting scheme of Hubotter et al.
#: (2026): optional successful solution, optional feedback, then the
#: instruction. The student's own attempt is deliberately not included.
SDPO_TEMPLATE = "{observation}\n\n{solution_section}{feedback_section}Correctly solve the original question."
SDPO_SOLUTION_SECTION = "Correct solution:\n\n{solution}\n\n"
SDPO_FEEDBACK_SECTION = (
    "The following is feedback from your unsuccessful earlier attempt:\n\n{feedback}\n\n"
)


def episode_succeeded(episode) -> bool:
    """Whether an episode solved its task: the env's ``passed`` flag, else a positive return."""
    if not episode:
        return False
    step_info = episode[-1].info.get("step_info") if episode[-1].info else None
    if isinstance(step_info, dict) and "passed" in step_info:
        return bool(step_info["passed"])
    return sum(float(t.reward) for t in episode) > 0.0


def sandbox_feedback(step_info: Optional[dict], max_chars: int = 2000) -> Optional[str]:
    """Readable feedback from a code_rl step's sandbox result, or ``None``.

    The LiveCodeBench harness prints the first failing test's metadata
    (``error_message``, inputs, expected vs. actual output) to stdout and exits
    non-zero; compile and runtime errors land in stderr. Sandbox-level errors
    (``{"error": ...}``) are passed through. Output is truncated from the end,
    keeping the head, where the failing test is described.
    """
    if not isinstance(step_info, dict):
        return None
    if step_info.get("parse_error"):
        return "No code block could be extracted from the answer."
    metrics = step_info.get("metrics") or {}
    details = metrics.get("details") if isinstance(metrics, dict) else None
    if not isinstance(details, dict) or not details:
        return None
    parts: list[str] = []
    for key in ("compile_result", "run_result"):
        result = details.get(key)
        if not isinstance(result, dict):
            continue
        for stream in ("stdout", "stderr"):
            text = (result.get(stream) or "").strip()
            if text:
                parts.append(text)
    if not parts and details.get("error"):
        parts.append(str(details["error"]))
    if not parts:
        return None
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text


def code_solution(tokenizer, episode) -> Optional[str]:
    """The fenced code of an episode's final answer (``None`` when there is none)."""
    from training_sessions_cookbook.rl.envs.lcb_utils import extract_code_from_model

    code = extract_code_from_model(decode_text(tokenizer, episode[-1].act_tokens))
    return f"```python\n{code}\n```" if code else None


def text_solution(tokenizer, episode) -> Optional[str]:
    """An episode's final response text."""
    text = decode_text(tokenizer, episode[-1].act_tokens).strip()
    return text or None


class FeedbackContext(TeacherContext):
    """SDPO: prompt + feedback on this attempt and/or a sibling's successful solution.

    For each episode in a group, the teacher sees:

    * ``solution_section``: the solution of a *different* episode in the same
      group that succeeded (if ``use_sibling_solutions`` and one exists);
    * ``feedback_section``: the environment's feedback on this attempt (if
      ``use_env_feedback``, the attempt failed, and ``feedback_fn`` returns text).

    An episode with neither (e.g. the only success in its group) is unscored.
    ``feedback_fn(step_info) -> str | None`` and ``solution_fn(tokenizer,
    episode) -> str | None`` are environment-specific; see
    :func:`sandbox_feedback` / :func:`code_solution` for code_rl.
    """

    def __init__(self, *, tokenizer, template_fn,
                 feedback_fn: Callable[[Optional[dict]], Optional[str]] = sandbox_feedback,
                 solution_fn: Callable[[Any, list], Optional[str]] = text_solution,
                 use_sibling_solutions: bool = True, use_env_feedback: bool = True,
                 template: str = SDPO_TEMPLATE) -> None:
        super().__init__(tokenizer=tokenizer, template_fn=template_fn)
        self.feedback_fn = feedback_fn
        self.solution_fn = solution_fn
        self.use_sibling_solutions = use_sibling_solutions
        self.use_env_feedback = use_env_feedback
        self.template = template

    def teacher_observation(self, episode, group) -> Optional[str]:
        observation = _episode_info(episode).get("observation")
        if observation is None:
            return None
        solution = None
        if self.use_sibling_solutions:
            for sibling in group or []:
                if sibling is episode or not sibling or not episode_succeeded(sibling):
                    continue
                solution = self.solution_fn(self.tokenizer, sibling)
                if solution:
                    break
        feedback = None
        if self.use_env_feedback and not episode_succeeded(episode):
            step_info = episode[-1].info.get("step_info") if episode[-1].info else None
            feedback = self.feedback_fn(step_info)
        if solution is None and feedback is None:
            return None
        return self.template.format(
            observation=observation,
            solution_section=SDPO_SOLUTION_SECTION.format(solution=solution) if solution else "",
            feedback_section=SDPO_FEEDBACK_SECTION.format(feedback=feedback) if feedback else "",
        )
