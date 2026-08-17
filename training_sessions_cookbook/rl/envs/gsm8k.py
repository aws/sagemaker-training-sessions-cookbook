"""GSM8K multi-turn environment + builtin dataset.

The env returns plain tuples (not ``Transition``), so it does not import from ``.base``.

Each builtin item is shaped as ``{"question": str, "answer": str}`` so the loader (which returns ``list[dict]`` for builtin / jsonl / parquet) converges on a single in-memory shape. The env accepts both that dict shape and raw ``(question, answer)`` tuples.
"""

from __future__ import annotations

import random  # used only by GSM8KMultiTurnEnv
import re  # used only by _gsm8k_extract_answer
from typing import Optional


# ---------------------------------------------------------------------------
# Answer-format configuration
# ---------------------------------------------------------------------------
#
# Choose how the model is asked to deliver its final numeric answer. Each
# format ships with:
#   - a prompt instruction (appended to the user observation) that tells the model where to put the answer
#   - an ordered list of extraction regexes the reward function tries when scoring a response
#
# `auto` is the "no format hint" mode: the prompt does not pin a format and the reward extractor tries every known format in priority order. Useful for observing what format the model picks on its own, and for comparing fluency across formats without rewriting the env each time.
#
# Numbers can carry an optional leading `$`, a sign, decimals, and thousands commas: all of which are stripped at compare time inside `_gsm8k_extract_answer`. Regexes therefore stay permissive on the numeric portion and strict on the surrounding marker.

_FORMAT_INSTRUCTION: dict[str, Optional[str]] = {
    "hash":          'output the final answer after "####"',
    "boxed":         "output the final answer in \\boxed{}",
    "tagged":        "output the final answer wrapped in <answer></answer>",
    "the_answer_is": "output the final answer as 'The answer is: <number>'",
    "auto":          None, # no format hint; let the model choose
}

_FORMAT_PATTERNS: dict[str, list[str]] = {
    # Each pattern allows optional bold (`**`), optional `$`, and optional whitespace around the number so the extractor stays robust to common model formatting quirks (`#### $125`, `\boxed{ $ 60 }`, `**$57,500**`). Numbers are stripped of `,` and `$` by `_gsm8k_extract_answer`.
    "hash":          [r"####\s*\*{0,2}\s*\$?\s*(-?[0-9\.\,]+)"],
    "boxed":         [r"\\boxed\{\s*\*{0,2}\s*\$?\s*(-?[0-9\.\,]+)\s*\*{0,2}\s*\}"],
    "tagged":        [r"<answer>\s*\*{0,2}\s*\$?\s*(-?[0-9\.\,]+)\s*\*{0,2}\s*</answer>"],
    "the_answer_is": [r"[Tt]he\s+(?:final\s+)?answer\s+is:?\s*\*{0,2}\s*\$?\s*(-?[0-9\.\,]+)"],
}
# `auto` tries every known format. Order matters only for tie-breaking on the rare case a response contains multiple distinct markers; we prefer the most explicit (hash > boxed > tagged > prose).
_FORMAT_PATTERNS["auto"] = (
    _FORMAT_PATTERNS["hash"]
    + _FORMAT_PATTERNS["boxed"]
    + _FORMAT_PATTERNS["tagged"]
    + _FORMAT_PATTERNS["the_answer_is"]
)


def make_gsm8k_observation(question: str, answer_format: str = "hash") -> str:
    """Build the user observation for a GSM8K problem.

    Shared between the training env (`GSM8KMultiTurnEnv.reset`) and the validation runner so both paths emit identical prompts for the same `answer_format`. Falls back silently to `hash` for unknown formats so older configs / typos don't crash the run.
    """
    instruction = _FORMAT_INSTRUCTION.get(answer_format, _FORMAT_INSTRUCTION["hash"])
    if instruction is None:
        return f"{question} Let's think step by step."
    return f"{question} Let's think step by step and {instruction}."


BUILTIN_GSM8K_20: list[dict] = [
    {
        "question": (
            "Natalia sold clips to 48 of her friends in April, and then she "
            "sold half as many clips in May. How many clips did Natalia sell "
            "altogether in April and May?"
        ),
        "answer": "72",
    },
    {
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just did "
            "50 minutes of babysitting. How much did she earn?"
        ),
        "answer": "10",
    },
    {
        "question": (
            "Betty is saving money for a new wallet which costs $100. Betty "
            "has only half of the money she needs. Her parents decided to "
            "give her $15 for that purpose, and her grandparents twice as "
            "much as her parents. How much more money does Betty need to "
            "make to buy the wallet?"
        ),
        "answer": "5",
    },
    {
        "question": (
            "Julie is reading a 120-page book. Yesterday, she was able to "
            "read 12 pages and today, she read twice as many pages as "
            "yesterday. If she wants to read half of the remaining pages "
            "tomorrow, how many pages should she read tomorrow?"
        ),
        "answer": "42",
    },
    {
        "question": (
            "James writes a 3-page letter to 2 different friends twice a "
            "week. How many pages does he write a year?"
        ),
        "answer": "624",
    },
    {
        "question": (
            "Mark has a garden with flowers. He planted plants of three "
            "different colors in it. Ten of them are yellow, and there are "
            "80% more of those in purple. There are only 25% as many green "
            "flowers as there are yellow and purple flowers. How many "
            "flowers does Mark have in his garden?"
        ),
        "answer": "35",
    },
    {
        "question": (
            "Albert is wondering how much pizza he can eat in one day. He "
            "buys 2 large pizzas and 2 small pizzas. A large pizza has 16 "
            "slices and a small pizza has 8 slices. If he eats it all, how "
            "many pieces does he eat that day?"
        ),
        "answer": "48",
    },
    {
        "question": (
            "Ken created a care package to send to his brother, who was "
            "away at boarding school. Ken placed a box on a scale, and then "
            "he poured into the box enough jelly beans to bring the weight "
            "to 2 pounds. Then, he added enough brownies to cause the "
            "weight to triple. Next, he added another 2 pounds of jelly "
            "beans. And finally, he added enough gummy worms to double the "
            "weight once again. What was the final weight of the box of "
            "goodies, in pounds?"
        ),
        "answer": "16",
    },
    {
        "question": (
            "Alexis is applying for a new job and bought a new set of "
            "business clothes to wear to the interview. She went to a "
            "department store with a budget of $200 and spent $30 on a pair "
            "of slacks and $46 on a blouse. She also purchased a pair of "
            "shoes and a belt. If the belt cost 1/4 as much as the shoes, "
            "and together they cost $64, how much money does Alexis have "
            "left from her budget?"
        ),
        "answer": "60",
    },
    {
        "question": (
            "Tina makes $18.00 an hour. If she works more than 8 hours per "
            "shift, she is eligible for overtime, which is paid by your "
            "hourly wage + 1/2 your hourly wage. If she works 10 hours every "
            "day for 5 days, how much money does she make?"
        ),
        "answer": "990",
    },
    {
        "question": (
            "A robe takes 2 bolts of blue fiber and half that much white "
            "fiber. How many bolts in total does it take?"
        ),
        "answer": "3",
    },
    {
        "question": (
            "Josh decides to try flipping a house. He buys a house for "
            "$80,000 and then puts in $50,000 in repairs. This increased "
            "the value of the house by 150%. How much profit did he make?"
        ),
        "answer": "70000",
    },
    {
        "question": (
            "James decides to run 3 sprints 3 times a week. He runs 60 "
            "meters each sprint. How many total meters does he run a week?"
        ),
        "answer": "540",
    },
    {
        "question": (
            "Every day, Wendi feeds each of her chickens three cups of "
            "mixed chicken feed, containing seeds, mealworms and vegetables "
            "to help keep them healthy. She gives the chickens their feed "
            "in three separate meals. In the morning, she gives her flock "
            "of chickens 15 cups of feed. In the afternoon, she gives her "
            "chickens another 25 cups of feed. If the carry-over from prior "
            "feedings is 35 cups, how many cups of feed does she need to "
            "give her chickens in the final meal of the day?"
        ),
        "answer": "20",
    },
    {
        "question": (
            "Kylar went to the store to get water and some apples. He spent "
            "$3 on water and $5 on apples. If he had $20 in his pocket, how "
            "much money did he have left?"
        ),
        "answer": "12",
    },
    {
        "question": (
            "Toulouse has twice as many sheep as Charleston. Charleston has "
            "4 times as many sheep as Seattle. How many sheep do Toulouse, "
            "Charleston, and Seattle have together if Seattle has 20 sheep?"
        ),
        "answer": "260",
    },
    {
        "question": (
            "Carla is downloading a 200 GB file. Normally she can download "
            "2 GB/minute, but 40% of the way through the download, Windows "
            "forces a restart to install updates, which takes 20 minutes. "
            "Then Carla has to restart the download from the beginning. How "
            "long does it take to download the file?"
        ),
        "answer": "160",
    },
    {
        "question": (
            "John drives for 3 hours at a speed of 60 mph and then turns "
            "around because he realizes he forgot something very important "
            "at home. He tries to get home in 4 hours but spends the first "
            "2 hours in standstill traffic. He spends the rest of the time "
            "driving at 30 mph. How far is he from home when he runs out "
            "of time?"
        ),
        "answer": "120",
    },
    {
        "question": (
            "Eliza's rate per hour for the first 40 hours she works each "
            "week is $10. She also receives an overtime pay of 1.2 times "
            "her regular hourly rate. If Eliza worked for 45 hours this "
            "week, how much are her earnings for this week?"
        ),
        "answer": "460",
    },
    {
        "question": (
            "A new program had 60 downloads in the first month. The number "
            "of downloads in the second month was three times as many as "
            "the downloads in the first month, but then reduced by 30% in "
            "the third month. How many downloads did the program have "
            "total over the three months?"
        ),
        "answer": "366",
    },
]


def _gsm8k_extract_answer(text: str, answer_format: str = "hash") -> Optional[str]:
    """Extract the final GSM8K answer from a model response.

    Clips to the trailing 300 chars, runs every regex registered for `answer_format` in order, and returns the last match across all matched patterns. `,` and `$` are stripped so numeric comparison is normalization-safe. Returns `None` when no pattern matches.

    `auto` tries every known format; the priority order in `_FORMAT_PATTERNS` breaks ties when a single response contains multiple distinct markers (rare; usually the model commits to one shape).
    """
    _SOLUTION_CLIP_CHARS = 300
    if len(text) > _SOLUTION_CLIP_CHARS:
        text = text[-_SOLUTION_CLIP_CHARS:]
    patterns = _FORMAT_PATTERNS.get(answer_format, _FORMAT_PATTERNS["hash"])
    last_match: Optional[str] = None
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            # Prefer the last match within the most-specific pattern that
            # produced a hit. Don't keep scanning lower-priority patterns
            # once we've found one: they would only catch sub-strings of
            # the same answer.
            last_match = matches[-1]
            break
    if last_match is None:
        return None
    return last_match.replace(",", "").replace("$", "")


def _gsm8k_normalize_gt(ground_truth: str) -> str:
    """Normalize a GSM8K ground-truth answer to the bare numeric string.

    The canonical HF `openai/gsm8k` `answer` field is the full chain-of-thought solution ending in `#### <number>`. When the `####` marker is present we take the segment after the last one; otherwise we use the string as-is (e.g. the bare `"72"` used by the builtin subset). Commas and `$` are then stripped so the result matches the format produced by `_gsm8k_extract_answer`.
    """
    gt = ground_truth or ""
    if "####" in gt:
        gt = gt.rsplit("####", 1)[-1]
    return gt.strip().replace(",", "").replace("$", "")


def _gsm8k_compute_score(
    action: str,
    ground_truth: str,
    format_score: float = 0.0,
    answer_format: str = "hash",
) -> float:
    """Score a GSM8K response.

    Returns 1.0 on a correct extracted answer, `format_score` on a well-formatted but wrong answer, and 0.0 when no answer could be extracted at all. The ground truth is normalized via `_gsm8k_normalize_gt` so both the bare-number builtin format and the canonical `#### <number>` HF format compare correctly.

    `answer_format` selects which extraction regex set runs. Defaults to `hash` for backward compatibility with pre-existing runs and metrics.
    """
    answer = _gsm8k_extract_answer(action, answer_format=answer_format)
    if answer is None:
        return 0.0
    return 1.0 if answer == _gsm8k_normalize_gt(ground_truth) else format_score


class GSM8KMultiTurnEnv:
    """Multi-turn GSM8K environment with per-turn format scoring.

    Each instance holds its own RNG and cursor over `dataset`, so multiple instances can run rollouts in parallel. `assign_problem` lets a central sampler feed a specific (question, answer) pair to the next `reset()`, matching the GRPO grouping pattern in the reference.

    State per episode:
      - `ground_truth`: target answer for scoring this episode.
      - `turns`: number of completed action turns.
      - `history`: running concatenation of observations and actions, useful for inspection / logging. The rollout loop builds its own templated/tokenized string for sampling; this buffer is not used as the prompt source.
    """

    def __init__(
        self,
        seed: int = 42,
        max_turns: int = 5,
        dataset: Optional[list] = None,
        answer_format: str = "hash",
    ):
        self.rng = random.Random(seed)
        self.max_turns = max_turns
        self.dataset = dataset if dataset is not None else BUILTIN_GSM8K_20
        self.format_score_per_turn: float = 0.0
        # Answer-format knob: controls both the instruction text appended to the user observation (so the model knows where to put the answer) and the extraction regex used at scoring time. Set once at env construction so every turn within an episode stays consistent.
        self.answer_format: str = answer_format
        self._indices: list[int] = []
        self._pos: int = 0
        self.epoch: int = 0
        self.ground_truth: Optional[str] = None
        self.turns: int = 0
        self.history: str = ""
        self._assigned_problem: Optional[tuple] = None

    def assign_problem(self, question: str, answer: str) -> None:
        """Pre-assign a specific (question, answer) for the next `reset()`."""
        self._assigned_problem = (question, answer)

    @staticmethod
    def _problem_pair(item) -> tuple:
        """Normalize a dataset entry to a `(question, answer)` tuple.

        Accepts both dict items (`{"question": ..., "answer": ...}`, the builtin / jsonl / parquet shape) and raw `(question, answer)` tuples for compatibility with reference fixtures.
        """
        if isinstance(item, dict):
            return item["question"], item["answer"]
        return item[0], item[1]

    def _next_problem(self) -> tuple:
        if self._assigned_problem is not None:
            problem = self._assigned_problem
            self._assigned_problem = None
            return problem
        if self._pos >= len(self._indices):
            self._indices = list(range(len(self.dataset)))
            self.rng.shuffle(self._indices)
            self._pos = 0
            self.epoch += 1
        idx = self._indices[self._pos]
        self._pos += 1
        return self._problem_pair(self.dataset[idx])

    def reset(self, seed: Optional[int] = None) -> tuple:
        """Start a new episode.

        Returns `(observation, info)` where `info["env_type"] == "gsm8k"`.
        """
        if seed is not None:
            self.rng = random.Random(seed)
        question, answer = self._next_problem()
        self.ground_truth = answer
        self.turns = 0
        self.history = ""
        obs = make_gsm8k_observation(question, self.answer_format)
        self.history += obs
        return obs, {"env_type": "gsm8k"}

    def step(self, action: str) -> tuple:
        """Apply one action turn.

        Returns `(observation, reward, terminated, truncated, info)`. Episode ends (`terminated=truncated=True`) when either the model produces a correct answer in the configured format or `max_turns` is reached.
        """
        self.turns += 1
        self.history += action
        reward = _gsm8k_compute_score(
            action,
            self.ground_truth or "",
            format_score=self.format_score_per_turn,
            answer_format=self.answer_format,
        )
        done = self.turns >= self.max_turns or reward == 1.0
        if done:
            return "", reward, True, True, {"env_type": "gsm8k"}
        remaining = self.max_turns - self.turns
        instruction = _FORMAT_INSTRUCTION.get(
            self.answer_format, _FORMAT_INSTRUCTION["hash"]
        )
        if instruction is None:
            # `auto` mode: no format pin in the continuation either.
            if remaining > 1:
                obs = "Please continue your step-by-step reasoning."
            else:
                obs = "Now output the final answer."
        else:
            if remaining > 1:
                obs = (
                    f"Please continue your step-by-step reasoning, and "
                    f"{instruction}."
                )
            else:
                obs = f"Now {instruction}."
        self.history += obs
        return obs, reward, False, False, {"env_type": "gsm8k"}
