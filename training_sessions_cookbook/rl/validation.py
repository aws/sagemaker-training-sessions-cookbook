"""Held-out validation for the RL loop.

``ValidationRunner`` samples the current policy greedily over a fixed
validation set on a cadence and scores it with the environment's reward
function, without touching training state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import numpy as np
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from .backends import TrainingBackend
from .envs.gsm8k import _gsm8k_compute_score, make_gsm8k_observation
from training_sessions_cookbook.service.console import console_print, console_rule

from .prompts import encode_prompt
from .config import cfg
from training_sessions_cookbook.service.session_watchdog import SessionDeadError


# === Validation utilities ===
#
# Periodic evaluation on a held-out dataset with qualitative example logging.

# Minimal environment for subprocesses that execute model-generated code.
# Never pass the full os.environ through: it may carry cloud credentials
# (AWS_ACCESS_KEY_ID etc.) that generated code could exfiltrate.
_SAFE_ENV_KEYS = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"}


def compute_validation_accuracy(rewards: list[float]) -> float:
    """Fraction of problems scoring exactly 1.0.

    Returns 0.0 when the rewards list is empty.
    """
    if not rewards:
        return 0.0
    return sum(1 for r in rewards if r == 1.0) / len(rewards)


@dataclass
class ValidationExample:
    """One validation problem evaluated at a given step.

    `improved` is set by `ValidationRunner._check_improvements` when this prompt scored 0.0 at the previous validation and 1.0 now. It is used only for qualitative terminal output (the "IMPROVED" label) and is NOT part of the serialized JSONL schema.
    """

    step: int
    prompt: str
    response: str
    expected_answer: str
    reward: float
    improved: bool = False

    def to_record(self) -> dict:
        """Serialize to the validation_examples.jsonl record schema.

        Includes step, prompt, response, expected_answer, reward.
        """
        return {
            "step": self.step,
            "prompt": self.prompt,
            "response": self.response,
            "expected_answer": self.expected_answer,
            "reward": self.reward,
        }


class ValidationRunner:
    """Runs periodic validation and logs qualitative examples.

    Tracks per-prompt reward history to detect improvements (0.0 → 1.0) across validation checkpoints.
    """

    def __init__(
        self,
        val_dataset: list, # list[dict] (GSM8K / toy code jsonl) or list[DeepcoderTask]
        val_examples: int,
        run_folder: str,
        env_type: str,
    ):
        self.val_dataset = val_dataset
        self.val_examples = val_examples
        self.run_folder = run_folder
        self.env_type = env_type
        self.history: dict[str, list[tuple[int, float]]] = {}  # prompt_key -> [(step, reward)]

    async def run_validation(
        self,
        step: int,
        sampling_client,
        tokenizer,
        template_fn,
        args,
        backend: "TrainingBackend",
    ) -> dict:
        """Sample responses for val dataset, compute metrics, log examples.

        `backend` shapes the prompt and sampling params (`backend.wrap_prompt` / `backend.make_sampling_params`), so this method stays independent of the concrete SDK.

        Returns validation metrics dict with keys:
          val/reward_mean, val/reward_std, val/accuracy
        """
        log = logging.getLogger(__name__)

        sampling_params = backend.make_sampling_params(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Validation prompts are evaluated concurrently (same rationale as the training rollouts): sample via the sync `sample()` + `asyncio.to_thread` so the event loop stays free and the service batch-fuses the requests, instead of awaiting each prompt serially. Concurrency is bounded by `--num_rollout_workers`. `asyncio.gather` preserves dataset order, which `_check_improvements` and the example printing below rely on.
        sem = asyncio.Semaphore(cfg(args, "num_rollout_workers"))

        async def _evaluate_one(item) -> ValidationExample:
            # A DeepcoderTask (the "true" code-RL val set from --val_code_split)
            # is a dataclass, not a dict; it is graded below with the SAME
            # SandboxFusion harness training uses. Dict items keep the legacy
            # paths: GSM8K, or the toy {prompt,test_code,reference_solution}
            # code jsonl passed via --val_dataset_path.
            is_deepcoder_task = self.env_type == "code_rl" and not isinstance(item, dict)
            test_code = ""
            if is_deepcoder_task:
                prompt_text = item.problem
                expected_answer = item.starter_code or ""
                observation = prompt_text
            elif self.env_type == "code_rl":
                prompt_text = item.get("prompt", "")
                expected_answer = item.get("reference_solution", "")
                test_code = item.get("test_code", "")
                observation = prompt_text
            else:
                prompt_text = item.get("question", "")
                expected_answer = item.get("answer", "")
                # Use the shared GSM8K observation builder so the validation path emits the same instruction text as the training env for the chosen `--answer_format`. Without this, val and train would silently diverge when `--answer_format` changes (and val/accuracy would no longer be comparable to rollout/reward_mean).
                observation = make_gsm8k_observation(
                    prompt_text, cfg(args, "answer_format")
                )

            templated_prompt = template_fn(observation)
            obs_tokens = encode_prompt(templated_prompt, tokenizer)

            async with sem:
                # Sample a response from the model via the backend's non-blocking
                # submit/await-split path, same as the training rollouts.
                try:
                    result = await backend.sample_async(
                        sampling_client,
                        backend.wrap_prompt(obs_tokens),
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    # SageMaker training-session SDK response shape: `result.samples[i].tokens` / `.logprobs`.
                    sequence = result.samples[0]
                    response_text = tokenizer.decode(list(sequence.tokens))
                except SessionDeadError:
                    raise  # dead session: fail validation loudly, not per-prompt
                except Exception as exc:
                    log.warning(
                        "validation: sample failure for prompt=%r: %r",
                        prompt_text[:80],
                        exc,
                    )
                    response_text = ""

            # Compute reward.
            if is_deepcoder_task:
                # True code-RL validation: extract code, grade against the
                # task's real test cases in SandboxFusion, and score with the
                # same deepcoder_reward as training, so val/reward_mean lines up
                # with rollout/reward_mean and val/accuracy = fraction solved.
                reward = await self._grade_deepcoder(item, response_text, args)
            elif self.env_type == "code_rl":
                # Legacy toy path: local subprocess over the item's test_code.
                # Code grading runs a subprocess, so offload it to a thread to
                # avoid blocking the event loop while other prompts sample.
                reward = await asyncio.to_thread(
                    self._evaluate_code, response_text, test_code, args
                )
            else:
                reward = _gsm8k_compute_score(
                    response_text,
                    expected_answer,
                    format_score=0.0,
                    answer_format=cfg(args, "answer_format"),
                )

            return ValidationExample(
                step=step,
                prompt=prompt_text,
                response=response_text,
                expected_answer=expected_answer,
                reward=float(reward),
            )

        results: list[ValidationExample] = list(
            await asyncio.gather(*[_evaluate_one(item) for item in self.val_dataset])
        )

        # Check for improvements and tag them
        results = self._check_improvements(step, results)

        # Compute aggregate metrics
        rewards = [r.reward for r in results]
        reward_arr = np.array(rewards) if rewards else np.array([0.0])
        metrics = {
            "val/reward_mean": float(reward_arr.mean()),
            "val/reward_std": float(reward_arr.std()),
            "val/accuracy": compute_validation_accuracy(rewards),
        }

        # Qualitative examples, framed so they cannot blur into the metrics
        # table (everything flows through the shared console in order).
        examples_to_print = results[: self.val_examples]
        console_rule(f"Validation after {step} steps")
        log.info(
            "val/reward_mean=%.4f  val/reward_std=%.4f  val/accuracy=%.4f",
            metrics["val/reward_mean"],
            metrics["val/reward_std"],
            metrics["val/accuracy"],
        )
        for i, ex in enumerate(examples_to_print):
            improved_tag = " [IMPROVED]" if ex.improved else ""
            console_print(f"\n  Example {i + 1}{improved_tag}")
            console_print(f"  Prompt:   {ex.prompt[:200]}")
            console_print(f"  Response: {ex.response[:500]}")
            console_print(f"  Expected: {ex.expected_answer[:500]}")
            console_print(f"  Reward:   {ex.reward:.2f}")
        if examples_to_print:
            console_rule(f"end validation after {step} steps")

        # Write all examples (untruncated) to validation_examples.jsonl
        val_jsonl_path = os.path.join(self.run_folder, "validation_examples.jsonl")
        with open(val_jsonl_path, "a") as f:
            for ex in results:
                f.write(json.dumps(ex.to_record()) + "\n")

        return metrics

    def _check_improvements(
        self, step: int, results: list[ValidationExample],
    ) -> list[ValidationExample]:
        """Tag results that improved from 0.0 to 1.0 since last validation."""
        for result in results:
            prompt_key = result.prompt
            reward = result.reward

            # Check if this prompt had a previous reward of 0.0 and now scores 1.0
            if prompt_key in self.history:
                prev_entries = self.history[prompt_key]
                if prev_entries:
                    last_reward = prev_entries[-1][1]
                    if last_reward == 0.0 and reward == 1.0:
                        result.improved = True

            # Update history
            if prompt_key not in self.history:
                self.history[prompt_key] = []
            self.history[prompt_key].append((step, reward))

        return results

    async def _grade_deepcoder(self, task, response: str, args) -> float:
        """Grade a response against a DeepcoderTask's real tests via SandboxFusion.

        This is the true-validation path: it mirrors training's grading exactly
, extract the fenced code block, run it against ``task.tests`` in the
        SandboxFusion LCB harness, then score with ``deepcoder_reward``, so
        ``val/reward_mean`` is directly comparable to ``rollout/reward_mean``
        and ``val/accuracy`` is the fraction of held-out problems fully solved.

        Imports are local so the GSM8K validation path never pulls in
        ``datasets``/``aiohttp``. ``sandbox_check_correctness`` is async, so we
        await it directly rather than offloading to a thread.
        """
        from .envs.code_rl import deepcoder_reward
        from .envs.lcb_utils import extract_code_from_model
        from .envs.sandbox import sandbox_check_correctness

        code = extract_code_from_model(response)
        passed = False
        if code is not None:
            try:
                passed, _ = await sandbox_check_correctness(
                    task.tests, code, cfg(args, "code_timeout")
                )
            except Exception as exc:  # grading failure -> treat as not solved
                logging.getLogger(__name__).warning(
                    "validation: sandbox grade failed for prompt=%r: %r",
                    task.problem[:80],
                    exc,
                )
                passed = False

        reward, _ = deepcoder_reward(
            response, bool(passed), cfg(args, "format_coef")
        )
        return float(reward)

    def _evaluate_code(self, response: str, test_code: str, args) -> float:
        """Evaluate a code response by executing it with test code."""
        if not response or not test_code:
            return 0.0
        full_code = response + "\n" + test_code
        timeout = cfg(args, "code_timeout")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = os.path.join(tmpdir, "solution.py")
                with open(script_path, "w") as f:
                    f.write(full_code)
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    timeout=timeout,
                    cwd=tmpdir,
                    env={k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS},
                )
                return 1.0 if result.returncode == 0 else 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            # Grading must never crash validation, but a harness error (bad
            # tmpdir, interpreter missing) must not silently masquerade as
            # "the model got it wrong" - log it so a systematic failure is
            # visible instead of appearing as a uniformly-zero val score.
            logging.getLogger(__name__).warning(
                "validation: toy-code grader errored; scoring 0.0", exc_info=True
            )
            return 0.0

