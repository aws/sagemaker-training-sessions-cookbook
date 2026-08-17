"""Code RL environment: code-execution results verification and reward.

Ties together DeepCoder tasks, LiveCodeBench (LCB) prompts, code extraction, SandboxFusion grading, and the ``deepcoder_reward`` formula. ``CodeRLEnv`` supports both a single-turn core and a multi-turn agent-tool loop.

Single-turn core:
  - ``deepcoder_reward``: the reward formula.
  - ``CodeRLEnv``: LCB prompt -> sample -> extract -> grade -> done, with grading on an async path because the SandboxFusion client is async.

Multi-turn agent-tool loop:
  - the initial observation is the LCB prompt plus the ``check_solution`` tool schema (rendered by ``envs.tool_render.render_tool_schema``);
  - each turn parses the assistant text for tool calls; a ``check_solution`` call grades the submitted code in the sandbox and feeds the result back as a tool-result message for the next turn;
  - the episode ends when the model emits no tool call (a final answer), a tool signals stop, ``max_turns`` is reached, the output fails to parse (``failed_parse_reward``), or the trajectory would overflow the context window (``context_overflow_reward``);
  - the terminal reward is graded ONCE from the final assistant message via ``deepcoder_reward``.

Grading is async because ``sandbox_check_correctness`` is a coroutine and the synchronous ``step`` cannot safely block on the event loop:
  - ``async def grade(self, action)``: extract + await the sandbox check + compute reward.
  - ``async def step_async(self, action)``: the full env 5-tuple, dispatching to the single-turn or multi-turn path based on ``max_turns``.
  - ``def step(self, action)`` raises ``NotImplementedError`` pointing to the async path.

Token accounting for the context-overflow guard is approximate: this env does not tokenize directly (the training loop owns the tokenizer/SamplingClient), so a ``count_tokens`` callable may be injected, e.g. ``lambda text: len(tokenizer.encode(text))``. When none is provided a cheap ``len(text) // 4`` character heuristic is used. The guard terminates before the next generation when ``trajectory_tokens + max_generation_tokens > max_trajectory_tokens``.

The async grading function defaults to ``envs.sandbox.sandbox_check_correctness``, resolved lazily on first use so tests can inject a mock.
"""

from __future__ import annotations

import json
import random
from typing import Any, Awaitable, Callable, Optional

from .base import Message, ToolCall  # ToolCall re-exported for callers
from .deepcoder import DeepcoderTask, load_deepcoder_tasks
from .lcb_utils import extract_code_from_model
from .sandbox import sandbox_check_correctness
from .tool_render import (
    check_solution_tool_spec,
    family_for_model,
    parse_tool_calls,
    render_tool_call, # re-exported for symmetry / tests
    render_tool_result,
    render_tool_schema,
)

# Type alias for an injectable async grading function. It mirrors ``envs.sandbox.sandbox_check_correctness(sample, generation, timeout)`` and returns ``(passed, details)``.
SandboxCheckFn = Callable[..., Awaitable[tuple[bool, dict[str, Any]]]]


def deepcoder_reward(
    final_text: str,
    passed: bool,
    format_coef: float = 0.1,
) -> tuple[float, dict[str, float]]:
    """Compute the DeepCoder episode reward.

    The reward rewards correctness and applies a small formatting penalty when the final response contains no fenced code block::

        reward = format_coef * (has_code - 1.0) + correct

    where ``has_code`` is 1.0 when ``extract_code_from_model(final_text)`` finds a fenced block (else 0.0) and ``correct`` is 1.0 when ``passed`` is truthy (else 0.0). When there IS a code block, the format term is zero and the reward equals ``correct``; when there is NO code block, ``-format_coef`` is applied on top of ``correct``.

    Args:
        final_text: The final assistant message to grade.
        passed: Whether the graded solution passed all test cases.
        format_coef: Magnitude of the no-code formatting penalty (default 0.1).

    Returns:
        ``(reward, metrics)`` where ``metrics`` is ``{"format": float(has_code), "correct": float(passed)}``.
    """
    has_code = extract_code_from_model(final_text) is not None
    correct = float(bool(passed))
    reward = format_coef * (float(has_code) - 1.0) + correct
    return reward, {"format": float(has_code), "correct": correct}


class CodeRLEnv:
    """Single-turn Code RL environment over one ``DeepcoderTask`` per episode.

    Conforms to the Env protocol's ``assign_problem`` / ``reset`` shape. Because grading is async, the single-turn ``step`` is exposed via ``step_async`` (with ``grade`` as the core); the synchronous ``step`` raises ``NotImplementedError`` and points callers at the async path.

    A central sampler pre-assigns the next task via ``assign_problem`` (matching the GSM8K pattern); when nothing is assigned, ``reset`` draws a random task from ``dataset``.
    """

    def __init__(
        self,
        dataset: list[DeepcoderTask],
        timeout: int = 6,
        format_coef: float = 0.1,
        max_turns: int = 1,
        sandbox_check: Optional[SandboxCheckFn] = None,
        seed: int = 42,
        *,
        family: str = "gpt_oss",
        model_name: Optional[str] = None,
        failed_parse_reward: float = -0.1,
        context_overflow_reward: float = -0.1,
        max_trajectory_tokens: Optional[int] = 32768,
        max_generation_tokens: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ):
        """Construct the env.

        Args:
            dataset: Tasks to draw from when no problem is pre-assigned.
            timeout: Per-test sandbox timeout in seconds.
            format_coef: No-code formatting penalty magnitude.
            max_turns: Number of turns. ``max_turns == 1`` runs the single-turn path; ``max_turns > 1`` runs the multi-turn agent-tool loop.
            sandbox_check: Injectable async grading fn. Defaults to ``envs.sandbox.sandbox_check_correctness``, resolved lazily on first use so tests can inject a mock without a live sandbox.
            seed: RNG seed for random task selection.
            family: Tool-renderer family for the multi-turn loop (default ``"gpt_oss"``). Ignored when ``model_name`` is given.
            model_name: If provided, the renderer family is derived from it via ``envs.tool_render.family_for_model`` (overrides ``family``).
            failed_parse_reward: Terminal reward when a turn's output cannot be parsed into either a tool call or a usable final answer (default ``-0.1``).
            context_overflow_reward: Terminal reward when continuing would exceed ``max_trajectory_tokens`` (default ``-0.1``).
            max_trajectory_tokens: Approximate trajectory token budget for the context-overflow guard (default ``32768``). ``None`` disables it.
            max_generation_tokens: Reserved generation budget added to the running trajectory count when checking for overflow.
            count_tokens: Optional ``str -> int`` token counter (e.g. ``lambda t: len(tokenizer.encode(t))``). When ``None`` a cheap ``len(text) // 4`` character heuristic is used.
        """
        self.dataset = dataset
        self.timeout = timeout
        self.format_coef = format_coef
        self.max_turns = max_turns
        self._sandbox_check = sandbox_check
        self.rng = random.Random(seed)

        self.family = family_for_model(model_name) if model_name else family
        self.failed_parse_reward = failed_parse_reward
        self.context_overflow_reward = context_overflow_reward
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self._count_tokens = count_tokens

        self._assigned_task: Optional[DeepcoderTask] = None
        self.task: Optional[DeepcoderTask] = None
        self.turns: int = 0
        # Multi-turn conversation state.
        self._history: list[Message] = []
        self._traj_tokens: int = 0

    # --- problem assignment / reset ----------------------------------------

    def assign_problem(self, task: DeepcoderTask) -> None:
        """Pre-assign the ``DeepcoderTask`` for the next ``reset()``.

        Accepts either a ``DeepcoderTask`` or a dict-like with ``problem`` / ``tests`` / ``starter_code`` keys (e.g. a row fed by a central sampler).
        """
        self._assigned_task = self._coerce_task(task)

    @staticmethod
    def _coerce_task(task: Any) -> DeepcoderTask:
        """Normalize an assigned object into a ``DeepcoderTask``."""
        if isinstance(task, DeepcoderTask):
            return task
        if isinstance(task, dict):
            return DeepcoderTask(
                problem=task["problem"],
                tests=task.get("tests", []),
                starter_code=task.get("starter_code"),
            )
        # Tuple/list fallback: (problem, tests[, starter_code]).
        if isinstance(task, (tuple, list)):
            problem = task[0]
            tests = task[1] if len(task) > 1 else []
            starter = task[2] if len(task) > 2 else None
            return DeepcoderTask(problem=problem, tests=tests, starter_code=starter)
        raise TypeError(f"Cannot coerce {type(task)!r} into a DeepcoderTask")

    def _next_task(self) -> DeepcoderTask:
        if self._assigned_task is not None:
            task = self._assigned_task
            self._assigned_task = None
            return task
        if not self.dataset:
            raise ValueError("CodeRLEnv has no assigned problem and an empty dataset")
        return self.rng.choice(self.dataset)

    def reset(self, seed: Optional[int] = None) -> tuple[str, dict]:
        """Start a new episode.

        Returns ``(observation, info)`` with ``info["env_type"] == "code_rl"``.

        For the single-turn case (``max_turns == 1``) the observation is the rendered LCB prompt for the task. For the multi-turn case (``max_turns > 1``) the observation is the LCB prompt followed by the ``check_solution`` tool schema (rendered via ``render_tool_schema``) so the model can see the tool, and the conversation history / approximate trajectory token count are initialized from it.
        """
        if seed is not None:
            self.rng = random.Random(seed)
        self.task = self._next_task()
        self.turns = 0
        self._history = []
        self._traj_tokens = 0

        if self.max_turns <= 1:
            return self.task.problem, {"env_type": "code_rl"}

        schema = render_tool_schema([check_solution_tool_spec()], self.family)
        observation = f"{self.task.problem}\n\n{schema}"
        self._history.append(Message(role="user", content=observation))
        self._traj_tokens = self._count(observation)
        return observation, {"env_type": "code_rl"}

    # --- token accounting / context-overflow guard -------------------------

    def _count(self, text: str) -> int:
        """Approximate token count for ``text``.

        Uses the injected ``count_tokens`` callable when available, otherwise a cheap ``len(text) // 4`` character heuristic (minimum 1 for non-empty text). The env does not tokenize directly because the training loop owns the tokenizer/SamplingClient.
        """
        if self._count_tokens is not None:
            return self._count_tokens(text)
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _exceeds_context_limit(self) -> bool:
        """Whether trajectory + next-generation budget exceeds the limit.

        Mirrors the reference ``EnvFromMessageEnv._exceeds_context_limit``: returns ``trajectory_tokens + max_generation_tokens > max_trajectory_tokens``. Always ``False`` when ``max_trajectory_tokens`` is ``None``.
        """
        if self.max_trajectory_tokens is None:
            return False
        reserve = self.max_generation_tokens or 0
        return self._traj_tokens + reserve > self.max_trajectory_tokens

    # --- grading (async core) ----------------------------------------------

    def _resolve_sandbox_check(self) -> SandboxCheckFn:
        """Resolve the grading fn, lazily importing the default on first use."""
        if self._sandbox_check is None:
            self._sandbox_check = sandbox_check_correctness
        return self._sandbox_check

    async def _run_sandbox(self, code: str) -> tuple[bool, dict]:
        """Run explicit ``code`` through the (async) sandbox check.

        Returns ``(passed, details)``. Used both by terminal grading (on code extracted from the final message) and by the multi-turn tool loop (on the explicit ``code`` argument of a ``check_solution`` call).
        """
        if self.task is None:
            raise RuntimeError("_run_sandbox() called before reset()")
        sandbox_check = self._resolve_sandbox_check()
        return await sandbox_check(self.task.tests, code, self.timeout)

    async def grade(self, action: str) -> tuple[float, bool, dict]:
        """Grade a single action.

        Extracts the last fenced code block, runs it through the (async) sandbox check, and computes the reward via ``deepcoder_reward``.

        Returns ``(reward, passed, metrics)`` where ``metrics`` extends the reward metrics with sandbox ``details``.
        """
        if self.task is None:
            raise RuntimeError("grade() called before reset()")

        code = extract_code_from_model(action)
        passed = False
        details: dict[str, Any] = {}
        if code is not None:
            passed, details = await self._run_sandbox(code)

        reward, metrics = deepcoder_reward(action, passed, self.format_coef)
        metrics = {**metrics, "details": details}
        return reward, bool(passed), metrics

    async def step_async(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """Async step, dispatching by ``max_turns``.

        Returns the env 5-tuple ``(obs, reward, terminated, truncated, info)``. Single-turn (``max_turns == 1``) grades ``action`` and ends the episode; multi-turn (``max_turns > 1``) drives the agent-tool loop one turn at a time (see :meth:`_step_multi`).
        """
        if self.max_turns <= 1:
            return await self._step_single(action)
        return await self._step_multi(action)

    async def _step_single(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """Single-turn step: grade ``action`` and end the episode."""
        self.turns += 1
        reward, passed, metrics = await self.grade(action)
        info = {"env_type": "code_rl", "passed": passed, "metrics": metrics}
        return "", reward, True, True, info

    async def _step_multi(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """Multi-turn agent-tool loop step (one turn per call).

        Faithful native port of ``AgentToolMessageEnv.step`` + ``build_agent_tool_env`` semantics. The end-condition logic is:

        1. **Continue** when the assistant emits a tool call AND ``max_turns`` has not been reached: grade the submitted code in the sandbox and return the rendered tool-result observation with intermediate reward ``0.0`` (episode not done). Before returning, if the running trajectory + generation budget would exceed ``max_trajectory_tokens`` the episode terminates with ``context_overflow_reward`` instead.
        2. **Failed parse** when the assistant emits no tool call AND no usable final answer (no fenced code block): terminate with ``failed_parse_reward``.
        3. **Terminal grade** otherwise, i.e. no tool call (final answer), OR ``max_turns`` reached, OR a tool signalled stop: grade the FINAL assistant message via ``deepcoder_reward`` (re-extracting and running its code, matching ``DeepcoderReward``) and end the episode.
        """
        self.turns += 1
        self._history.append(Message(role="assistant", content=action))
        self._traj_tokens += self._count(action)

        tool_calls = parse_tool_calls(action, self.family)
        no_tool_calls = len(tool_calls) == 0
        max_turns_reached = self.turns >= self.max_turns

        # (1) Continue: a tool call is present and we have turns left.
        if tool_calls and not max_turns_reached:
            tc = tool_calls[0]
            if tc.name == "check_solution":
                code = tc.arguments.get("code", "")
                if code:
                    passed, details = await self._run_sandbox(code)
                else:
                    passed, details = False, {"error": "missing 'code' argument"}
            else:
                passed, details = False, {"error": f"unknown tool: {tc.name}"}

            result_content = json.dumps(
                {"passed": bool(passed), "details": details},
                ensure_ascii=False,
            )
            self._history.append(Message(role="tool", content=result_content))
            next_obs = render_tool_result("check_solution", result_content, self.family)
            self._traj_tokens += self._count(next_obs)

            # Context-overflow guard before the next generation.
            if self._exceeds_context_limit():
                info = {
                    "env_type": "code_rl",
                    "context_overflow": True,
                    "turn": self.turns,
                    "metrics": {"context_overflow": 1.0},
                }
                return "", self.context_overflow_reward, True, True, info

            info = {
                "env_type": "code_rl",
                "tool_call": tc.name,
                "passed": bool(passed),
                "turn": self.turns,
            }
            return next_obs, 0.0, False, False, info

        # (2) Failed parse: neither a tool call nor a usable final answer.
        if no_tool_calls and extract_code_from_model(action) is None:
            info = {
                "env_type": "code_rl",
                "parse_error": True,
                "turn": self.turns,
                "metrics": {"parse_error": 1.0},
            }
            return "", self.failed_parse_reward, True, True, info

        # (3) Terminal grade of the final assistant message.
        reward, passed, metrics = await self.grade(action)
        if max_turns_reached and not no_tool_calls:
            metrics = {**metrics, "max_turns": 1.0}
        info = {
            "env_type": "code_rl",
            "passed": passed,
            "turn": self.turns,
            "metrics": metrics,
        }
        return "", reward, True, True, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """Synchronous step is not supported: grading is async.

        The SandboxFusion grading call is a coroutine and cannot be safely run from a synchronous ``step`` inside the training loop's event loop. Use ``await env.step_async(action)`` (single-turn) or ``await env.grade(...)``.
        """
        raise NotImplementedError(
            "CodeRLEnv grading is async; call `await env.step_async(action)` "
            "(single-turn) or `await env.grade(action)`."
        )


def build_code_rl_env_factory(args: Any) -> Callable[[], "CodeRLEnv"]:
    """Build a zero-arg ``CodeRLEnv`` factory and wire DeepCoder into ``args``.

    This is the registry entry point for ``--env_type code_rl``. It mirrors the GSM8K factory contract: a *zero-arg* callable producing a fresh ``CodeRLEnv`` per concurrent rollout (envs are stateful). Two extra effects make it drop into the existing training loop unchanged:

    1. **Load DeepCoder tasks once** via :func:`envs.deepcoder.load_deepcoder_tasks` (lazy ``datasets`` import happens there). The same task list is shared by every env instance the factory produces.
    2. **Publish the tasks to ``args._dataset``** so the training loop's central sampler iterates DeepCoder tasks (popping one and handing it to each env via ``assign_problem``) exactly as it iterates GSM8K dicts. ``main()`` reads ``args._dataset`` back after building the factory so ``next_batch`` / ``run_training`` see DeepCoder tasks rather than the GSM8K dataset.

    All Code-RL-specific knobs (``--format_coef`` / ``--context_overflow_reward`` / ``--max_trajectory_tokens``) are read defensively with ``getattr`` so absent ones fall back to the documented defaults. This function does NOT register new CLI args.

    Args:
        args: Parsed CLI args namespace. Required attributes: ``max_turns`` and ``base_model``. Optional (with defaults): ``seed`` (0), ``max_prompts`` (None), ``code_timeout`` (6), ``format_coef`` (0.1), ``max_trajectory_tokens`` (32768), ``max_tokens`` (None → ``max_generation_tokens``), ``context_overflow_reward`` (-0.1).

    Returns:
        A zero-arg callable returning a fresh ``CodeRLEnv`` bound to the shared DeepCoder task list.
    """
    tasks = load_deepcoder_tasks(
        "train",
        seed=getattr(args, "seed", 0),
        max_prompts=getattr(args, "max_prompts", None),
    )

    # Publish to the central sampler's dataset slot so next_batch iterates the DeepCoder tasks (the sampler pops a task and assigns it to each env, exactly like GSM8K dicts).
    args._dataset = tasks

    return lambda: CodeRLEnv(
        dataset=tasks,
        timeout=getattr(args, "code_timeout", 6),
        format_coef=getattr(args, "format_coef", 0.1),
        max_turns=args.max_turns,
        model_name=args.base_model,
        max_trajectory_tokens=getattr(args, "max_trajectory_tokens", 32768),
        max_generation_tokens=getattr(args, "max_tokens", None),
        context_overflow_reward=getattr(args, "context_overflow_reward", -0.1),
    )
