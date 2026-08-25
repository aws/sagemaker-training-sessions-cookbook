"""Rollout collection: sampling episodes from the current policy.

``Transition`` is the unit of training data (one prompt/response turn with
its reward, advantage, and sampling logprobs). Collection is asynchronous so
many rollouts are in flight against the sampler at once.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from typing import Optional

from .backends import TrainingBackend
from .prompts import encode_prompt
from .envs.tool_render import generation_opener
from training_sessions_cookbook.service.session_watchdog import SessionDeadError
from .config import cfg


# === GSM8K env + builtin dataset ===
#
# The 20-problem GSM8K subset, the scoring helpers, and `GSM8KMultiTurnEnv` live in `envs/gsm8k.py` and are re-exported here so callers can reference them from this module.

from .envs.gsm8k import (  # noqa: F401  # re-exported for callers
    BUILTIN_GSM8K_20,
    GSM8KMultiTurnEnv,
    _gsm8k_compute_score,
    _gsm8k_extract_answer,
)


# === Async rollout helpers + Transition dataclass ===
#
# Defines the `Transition` dataclass that flows through advantage computation and datum building, plus three async coroutines that drive rollouts against the live `SamplingClient`:
#
#   - rollout_one: runs one episode for one prompt, returning the per-turn transitions. On any sampling failure we log a WARNING and return `[]` so the caller can drop the bad rollout without aborting the batch.
#   - collect_batch_async: one episode per prompt, gated by an `asyncio.Semaphore(args.num_rollout_workers)`.
#   - collect_grouped_batch_async: K episodes per prompt (GRPO-style); prompt x K rollout tasks are flattened for `asyncio.gather` then re-grouped into K episodes per prompt on return.
#
# The sample call is two-step:
#
#     fut = await sampling_client.sample_async(
#         prompt=ModelInput.from_ints(obs_tokens),
#         num_samples=1,
#         sampling_params=SamplingParams(
#             max_tokens=args.max_tokens, temperature=args.temperature,
#         ),
#     )
#     result = await fut.result_async()
#
# `tokenizer`, `env_factory`, and `template_fn` are passed through from `main()` / `run_training` (tasks 10 / 9): `GSM8KMultiTurnEnv` is stateful so each concurrent rollout needs its own fresh instance via `env_factory()`, and the tokenizer obtained via `training_client.get_tokenizer()` is needed for both `tokenizer.encode(prompt)` (build `obs_tokens` each turn) and `tokenizer.decode(act_tokens)` (turn the sampled tokens back into text the env can score).


@dataclass
class Transition:
    """One turn of one rollout episode.

    `obs_tokens` captures the running prompt + history tokens at this turn; `act_tokens` / `act_logprobs` are the sampled assistant tokens and the per-token logprobs returned by the sampler. `advantage` is populated later by the advantage helpers (`compute_monte_carlo_advantages` / `compute_grouped_advantages`); it defaults to `0.0` so freshly-built transitions are immediately consumable by `build_training_datum` even before advantages are filled in.
    """

    obs_tokens: list[int]
    act_tokens: list[int]
    act_logprobs: list[float]
    reward: float
    done: bool
    env_type: str
    advantage: float = 0.0


@dataclass
class WrappedGroup:
    """One prompt's K rollout episodes, stamped with its sampler version.

    Only used by the off-by-k async loop (``run_training_async``,
    ``--max_steps_off_policy > 0``). The strict serial ``run_training`` never
    constructs these; it trains on the group as soon as it is collected, so
    there is nothing to stamp.

    ``birth_step`` is the trainer step the sampler was at **when this group's
    rollouts started**. The consumer's staleness filter compares
    ``current_step - birth_step`` against ``--max_steps_off_policy``: a group is
    at most that many gradient steps off-policy when trained, keeping the
    loss's importance-sampling ratio (behavior policy = the sampler at
    ``birth_step``) in a correctable range.
    """

    #: K episodes for one prompt; each episode is a list of ``Transition`` in
    #: time order: the same shape ``collect_grouped_batch_async`` yields per
    #: prompt, so downstream advantage/datum code is unchanged.
    episodes: "list[list[Transition]]"
    #: Sampler version (trainer step) the rollouts in this group started on.
    birth_step: int
    #: The prompt this group was sampled for. Retained so the requeue
    #: staleness policy (``--stale_policy requeue``) can put the prompt back
    #: onto ``prompt_queue`` when the group is too stale to train. ``None``
    #: is accepted for backward compatibility with any caller that still
    #: constructs WrappedGroup positionally with two args.
    prompt: object = None


async def rollout_one(
    prompt,
    env: "GSM8KMultiTurnEnv",
    template_fn,
    sampling_client,
    sampling_params,
    max_turns: int,
    tokenizer,
    backend: "TrainingBackend",
) -> list[Transition]:
    """Run one multi-turn rollout against `sampling_client`.

    On any sampling exception we log a WARNING with a short prompt id and return an empty list; the caller (`collect_batch_async` / `collect_grouped_batch_async`) keeps going so a single bad rollout does not poison the batch.

    Args:
        prompt: A dataset entry shaped as `{"question": ..., "answer": ...}` (builtin / jsonl / parquet shape) or a `(question, answer)` tuple/list. The pair is fed into the env via `env.assign_problem` so `env.reset()` deterministically starts on this problem.
        env: A fresh `GSM8KMultiTurnEnv` (one per concurrent rollout, since the env is stateful).
        template_fn: Unary `(observation: str) -> str` template; main() resolves `--template` to one of the entries in `TEMPLATE_FACTORY` and binds `apply_auto` against `args.base_model` before passing it through.
        sampling_client: The live `SamplingClient` for this step.
        sampling_params: A pre-built `SamplingParams` (constructed once by the caller from `args.max_tokens` / `args.temperature`).
        max_turns: Hard cap on conversation turns; the env may terminate earlier when a `#### <answer>` is detected.
        tokenizer: `training_client.get_tokenizer()`-style tokenizer with `.encode(text) -> list[int]` and `.decode(list[int]) -> str`.

    Returns:
        A list of `Transition` objects, one per completed turn (advantages unset; populated by `compute_monte_carlo_advantages` / `compute_grouped_advantages`). Empty list on sampling failure.
    """
    log = logging.getLogger(__name__)

    # Resolve a short prompt id for log lines and pre-assign the problem so env.reset() starts on it deterministically.
    if isinstance(prompt, dict):
        prompt_id = prompt.get("id") or (prompt.get("question") or "<unknown>")[:80]
        env.assign_problem(prompt["question"], prompt["answer"])
    elif isinstance(prompt, (tuple, list)) and len(prompt) >= 2:
        prompt_id = (str(prompt[0]) or "<unknown>")[:80]
        env.assign_problem(prompt[0], prompt[1])
    else:
        prompt_id = "<unknown>"

    observation, info = env.reset()
    env_type = info.get("env_type", "gsm8k")

    # Running templated conversation, accumulated as token IDs (NOT a string). Storing tokens directly avoids two pitfalls:
    #   1. Gemma's `apply_gemma` returns `list[int]` because Gemma's tokenizer can't faithfully encode `<start_of_turn>` markers from a string; we'd corrupt the prompt if we round-tripped through str.
    #   2. Re-tokenizing the entire running string on each turn (the old pattern) can produce different boundary tokens than appending newly-encoded segments, especially near merged BPE chunks.
    running_tokens: list[int] = encode_prompt(template_fn(observation), tokenizer)

    transitions: list[Transition] = []
    for _turn in range(max_turns):
        obs_tokens = list(running_tokens)
        try:
            # Concurrency-critical: `backend.sample_async` submits the request
            # off the event loop's critical path (SDK `sample_async` fires it
            # immediately) and awaits the result on a worker thread, so many
            # rollouts are in-flight at once and the service batch-fuses them.
            result = await backend.sample_async(
                sampling_client,
                backend.wrap_prompt(obs_tokens),
                num_samples=1,
                sampling_params=sampling_params,
            )
            # SageMaker training-session SDK response shape: `result.samples[i].tokens` / `.logprobs`. Extraction is inside the try so a shape mismatch or empty-samples response is caught as a warning rather than crashing the training loop.
            sequence = result.samples[0]
            act_tokens = list(sequence.tokens)
            act_logprobs = list(sequence.logprobs)
        except SessionDeadError:
            # The session is terminal server-side: no retry can ever succeed.
            # Swallowing this like an ordinary sample failure turns a dead
            # session into an infinite warn-requeue-resubmit churn.
            raise
        except Exception as exc:  # sampling is best-effort
            log.warning(
                "rollout_one: sample failure for prompt id=%r: %r",
                prompt_id,
                exc,
            )
            return []

        # Env speaks plain text, not tokens. Decode for env.step().
        action_text = tokenizer.decode(act_tokens)
        next_obs, reward, terminated, truncated, _step_info = env.step(action_text)
        done = bool(terminated or truncated)

        transitions.append(
            Transition(
                obs_tokens=obs_tokens,
                act_tokens=act_tokens,
                act_logprobs=act_logprobs,
                reward=float(reward),
                done=done,
                env_type=env_type,
            )
        )

        if done:
            break

        # Append the sampled assistant action tokens + the next user observation (encoded as plain text: `next_obs` carries no chat-format markers for GSM8K).
        running_tokens = running_tokens + act_tokens + list(
            tokenizer.encode(next_obs)
        )

    return transitions


async def rollout_one_code(
    prompt,
    env,
    template_fn,
    sampling_client,
    sampling_params,
    max_turns: int,
    tokenizer,
    backend: "TrainingBackend",
) -> list[Transition]:
    """Run one (async, multi-turn-capable) Code RL rollout.

    Mirrors `rollout_one` but drives the async `CodeRLEnv` whose grading is a coroutine (SandboxFusion is async). The differences from the GSM8K path:

    * `prompt` is a `DeepcoderTask` (or a dict/tuple the env's `assign_problem` / `_coerce_task` accepts) rather than a `{"question", "answer"}` dict.
    * Token counting for the env's context-overflow guard is injected from the training tokenizer (`env._count_tokens = lambda t: len(tokenizer.encode(t))`) before `reset()`; pragmatic, since the env does not own a tokenizer.
    * Stepping uses `await env.step_async(action_text)` instead of the sync `env.step` (which `CodeRLEnv` intentionally does not support).

    The per-turn `Transition` records have the same shape GSM8K produces, so the GRPO grouping and datum building downstream are unchanged. On any sampling exception we log a WARNING and return an empty list, matching `rollout_one`'s best-effort behavior.

    Args:
        prompt: A `DeepcoderTask` (or dict/tuple accepted by `env.assign_problem`).
        env: A fresh `CodeRLEnv` (one per concurrent rollout; stateful).
        template_fn: Unary `(observation: str) -> str` template, applied ONLY to the first observation (the LCB prompt plus, in multi-turn, the tool schema). Later turns are already chat-framed by the tool renderer and are appended unwrapped.
        sampling_client: The live `SamplingClient` for this step.
        sampling_params: Pre-built `SamplingParams`.
        max_turns: Hard cap on conversation turns; the env may end earlier.
        tokenizer: Tokenizer with `.encode` / `.decode`.

    Returns:
        A list of `Transition` objects (advantages unset). Empty list on sampling failure.
    """
    log = logging.getLogger(__name__)

    # Resolve a short prompt id for log lines. `prompt` is a DeepcoderTask (has `.problem`) or a dict/tuple the env can coerce.
    problem_text = getattr(prompt, "problem", None)
    if problem_text is None and isinstance(prompt, dict):
        problem_text = prompt.get("problem")
    prompt_id = (str(problem_text) if problem_text else "<unknown>")[:80]

    env.assign_problem(prompt)

    # Inject token counting for the env's context-overflow guard (the env does not tokenize directly; the training loop owns the tokenizer). Done before reset() so the initial observation token count uses the real tokenizer.
    env._count_tokens = lambda t: len(tokenizer.encode(t))

    observation, info = env.reset()
    env_type = info.get("env_type", "code_rl")

    # Running templated conversation, accumulated as token IDs (NOT a string) for the same reasons as `rollout_one`: see comments there.
    #
    # When the env advertises tools (multi-turn), they are passed to the
    # template as data so each family's template can place the declarations
    # where its model expects them (gpt-oss: developer frame; qwen3: system
    # turn; gemma: the model's own jinja via `tools=`). A template bound
    # without tools support falls back to the plain path.
    # No fallback here on purpose: a template that cannot take `tools` must
    # fail loudly rather than silently render a tool-less prompt (the model
    # would then be trained/evaluated on episodes where it cannot see the tool
    # it is graded on using).
    tools = info.get("tools")
    first = template_fn(observation, tools=tools) if tools else template_fn(observation)
    running_tokens: list[int] = encode_prompt(first, tokenizer)

    transitions: list[Transition] = []
    for _turn in range(max_turns):
        obs_tokens = list(running_tokens)
        try:
            # See rollout_one: `backend.sample_async` submits off the loop's
            # critical path and awaits the result off-loop, so sibling rollouts
            # stay in-flight and the service batch-fuses them.
            result = await backend.sample_async(
                sampling_client,
                backend.wrap_prompt(obs_tokens),
                num_samples=1,
                sampling_params=sampling_params,
            )
            # See rollout_one: SageMaker SDK uses `.samples[i]`, extraction kept inside try so shape/empty errors surface as warnings.
            sequence = result.samples[0]
            act_tokens = list(sequence.tokens)
            act_logprobs = list(sequence.logprobs)
        except SessionDeadError:
            # The session is terminal server-side: no retry can ever succeed.
            # Swallowing this like an ordinary sample failure turns a dead
            # session into an infinite warn-requeue-resubmit churn.
            raise
        except Exception as exc:  # sampling is best-effort
            log.warning(
                "rollout_one_code: sample failure for prompt id=%r: %r",
                prompt_id,
                exc,
            )
            return []

        action_text = tokenizer.decode(act_tokens)
        next_obs, reward, terminated, truncated, _step_info = await env.step_async(
            action_text
        )
        done = bool(terminated or truncated)

        transitions.append(
            Transition(
                obs_tokens=obs_tokens,
                act_tokens=act_tokens,
                act_logprobs=act_logprobs,
                reward=float(reward),
                done=done,
                env_type=env_type,
            )
        )

        if done:
            break

        # Append the sampled assistant action tokens + the tool result.
        #
        # `next_obs` from CodeRLEnv is ALREADY a complete chat frame produced by
        # `envs.tool_render.render_tool_result` (e.g. for harmony:
        # `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>...<|end|>`).
        # It must not pass through `template_fn`: that would open a second
        # system block and nest the tool frame inside a fresh user message, so
        # every turn from the second onward would be malformed for the model.
        # This mirrors `rollout_one`, which likewise encodes its next
        # observation directly. Note the sampled `act_tokens` are appended
        # verbatim rather than re-encoded from text, because the training datums
        # and their logprobs refer to exactly those token ids.
        # After the tool result, open a fresh reply frame the way each model's
        # own template does with `add_generation_prompt=True` after a tool
        # message (gpt-oss: `<|start|>assistant`; qwen3: `<|im_start|>assistant`;
        # gemma: nothing - it continues the same model turn). Without it the
        # model is asked to continue right after the result's end token, a
        # position it never generated from in training.
        opener = generation_opener(getattr(env, "family", "gpt_oss"))
        running_tokens = (
            running_tokens
            + act_tokens
            + encode_prompt(next_obs, tokenizer)
            + (encode_prompt(opener, tokenizer) if opener else [])
        )

    return transitions


async def collect_batch_async(
    prompts: list,
    sampling_client,
    *,
    args,
    tokenizer,
    env_factory,
    template_fn,
    backend: "TrainingBackend",
) -> list[list[Transition]]:
    """Run one episode per prompt, capped by `args.num_rollout_workers`.

    Returns episodes in the same order as `prompts`. A failed rollout is an empty list (per `rollout_one`); the caller is responsible for filtering if needed.
    """
    sem = asyncio.Semaphore(args.num_rollout_workers)
    sampling_params = backend.make_sampling_params(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    rollout_fn = (
        rollout_one_code
        if cfg(args, "env_type") == "code_rl"
        else rollout_one
    )

    async def _one(prompt):
        async with sem:
            env = env_factory()
            return await rollout_fn(
                prompt=prompt,
                env=env,
                template_fn=template_fn,
                sampling_client=sampling_client,
                sampling_params=sampling_params,
                max_turns=args.max_turns,
                tokenizer=tokenizer,
                backend=backend,
            )

    return await asyncio.gather(*[_one(prompt) for prompt in prompts])


async def collect_grouped_batch_async(
    prompts: list,
    K: int,
    sampling_client,
    *,
    args,
    tokenizer,
    env_factory,
    template_fn,
    backend: "TrainingBackend",
) -> list[list[list[Transition]]]:
    """Run K episodes per prompt and re-group the flattened results.

    `len(prompts) * K` rollout coroutines are scheduled flat under the same `asyncio.Semaphore(args.num_rollout_workers)` as `collect_batch_async`, then the gathered result list is sliced into K episodes per prompt preserving prompt order. Each rollout coroutine constructs its own env via `env_factory()` so concurrent rollouts do not share state.
    """
    sem = asyncio.Semaphore(args.num_rollout_workers)
    sampling_params = backend.make_sampling_params(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    rollout_fn = (
        rollout_one_code
        if cfg(args, "env_type") == "code_rl"
        else rollout_one
    )

    async def _one(prompt):
        async with sem:
            env = env_factory()
            return await rollout_fn(
                prompt=prompt,
                env=env,
                template_fn=template_fn,
                sampling_client=sampling_client,
                sampling_params=sampling_params,
                max_turns=args.max_turns,
                tokenizer=tokenizer,
                backend=backend,
            )

    tasks = []
    for prompt in prompts:
        for _ in range(K):
            tasks.append(_one(prompt))
    flat = await asyncio.gather(*tasks)
    return [flat[i * K : (i + 1) * K] for i in range(len(prompts))]


async def collect_one_group_async(
    prompt,
    K: int,
    sampling_client,
    *,
    args,
    tokenizer,
    env_factory,
    template_fn,
    backend: "TrainingBackend",
    seed_base: "Optional[int]" = None,
) -> list[list["Transition"]]:
    """Run K episodes for a SINGLE prompt against ``sampling_client``.

    The off-by-k analogue of one prompt's slice of ``collect_grouped_batch_async``.
    Whereas that helper owns a semaphore and fans out over a whole batch of
    prompts at once (the sync loop's model), here concurrency is bounded by the
    number of rollout-worker tasks in ``run_training_async``, each worker pulls
    one prompt and calls this, so there is no internal semaphore. The K rollouts
    of this one group still run concurrently via ``asyncio.gather`` so a group
    is not itself serialized.

    Returns K episodes (each a list of ``Transition``) for the prompt; a failed
    rollout is an empty episode (per ``rollout_one``). ``seed_base`` (when set)
    gives rollout ``i`` ``seed = seed_base + i``, distinct within the group so
    responses differ, deterministic across runs sharing the flag.
    """
    rollout_fn = (
        rollout_one_code
        if cfg(args, "env_type") == "code_rl"
        else rollout_one
    )
    shared_params = backend.make_sampling_params(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    async def _one(rollout_idx: int):
        env = env_factory()
        params = (
            shared_params
            if seed_base is None
            else backend.make_sampling_params(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=seed_base + rollout_idx,
            )
        )
        return await rollout_fn(
            prompt=prompt,
            env=env,
            template_fn=template_fn,
            sampling_client=sampling_client,
            sampling_params=params,
            max_turns=args.max_turns,
            tokenizer=tokenizer,
            backend=backend,
        )

    return await asyncio.gather(*[_one(i) for i in range(K)])

