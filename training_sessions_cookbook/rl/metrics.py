"""Per-step metrics: aggregation, JSONL/MLflow logging, terminal tables.

Pure aggregation functions over rollout transitions/episodes, the
``MetricsLogger`` that writes one JSON line per training step (optionally
mirrored to MLflow), rollout capture sinks, and terminal table formatting.
"""

from __future__ import annotations

import json
import numpy as np
import os
from typing import Optional

from training_sessions_cookbook.service import append_metrics
from training_sessions_cookbook.service.console import console_print, format_metric_values, print_kv_table
from .config import cfg
from .runstate import run_artifacts


# === Metrics aggregation and logging ===
#
# Pure aggregation functions for per-step metrics computation. These operate on lists of Transition objects and episode lists to produce summary statistics for the JSONL metrics log.


def compute_reward_stats(episodes: list) -> dict:
    """Compute reward_mean, reward_std, reward_min, reward_max from episodes.

    Each episode is a list of Transition objects. The per-episode return is the sum of rewards across all transitions in that episode. Empty episodes (failed rollouts) are excluded from the computation.

    Returns a dict with keys: reward_mean, reward_std, reward_min, reward_max. All values are 0.0 when no non-empty episodes are provided.
    """
    rewards = [sum(t.reward for t in ep) for ep in episodes if ep]
    if not rewards:
        return {"reward_mean": 0.0, "reward_std": 0.0, "reward_min": 0.0, "reward_max": 0.0}
    arr = np.array(rewards)
    return {
        "reward_mean": float(arr.mean()),
        "reward_std": float(arr.std()),
        "reward_min": float(arr.min()),
        "reward_max": float(arr.max()),
    }


def compute_advantage_stats(transitions: list) -> dict:
    """Compute advantage_mean, advantage_std, advantage_min, advantage_max.

    Operates on a flat list of Transition objects. Returns zeros when the list is empty.
    """
    if not transitions:
        return {"advantage_mean": 0.0, "advantage_std": 0.0, "advantage_min": 0.0, "advantage_max": 0.0}
    arr = np.array([t.advantage for t in transitions])
    return {
        "advantage_mean": float(arr.mean()),
        "advantage_std": float(arr.std()),
        "advantage_min": float(arr.min()),
        "advantage_max": float(arr.max()),
    }


def compute_episode_length_mean(episodes: list) -> float:
    """Mean number of turns per non-empty episode.

    Returns 0.0 when no non-empty episodes are provided.
    """
    non_empty = [ep for ep in episodes if ep]
    if not non_empty:
        return 0.0
    return sum(len(ep) for ep in non_empty) / len(non_empty)


def count_episodes(episodes: list) -> tuple:
    """Returns (num_episodes, num_failed_rollouts).

    num_episodes is the count of non-empty episodes (successful rollouts). num_failed_rollouts is the count of empty episodes (sampling failures).
    """
    non_empty = sum(1 for ep in episodes if ep)
    failed = sum(1 for ep in episodes if not ep)
    return non_empty, failed


def compute_token_turn_stats(episodes: list, transitions: list) -> dict:
    """Token-volume and turn-count stats from the rollout structures.

    Backend-agnostic: computed purely from ``Transition.obs_tokens`` /
    ``act_tokens`` and the episode grouping, which are shared rollout code. All
    keys are namespaced under ``rollout/`` by the caller. Empty inputs yield
    zeros so the JSONL schema stays stable across steps.

    These drive the ``tokens`` figure family in ``rl/analysis/plot_runs.py``
    (token budget and turn distribution are what ``--max_turns`` /
    ``--max_tokens`` are tuned against). The function was lost when the metrics
    helpers moved into this module, which left that figure family silently
    unpopulated for every run after the move.
    """
    non_empty = [ep for ep in episodes if ep]
    flat = [t for ep in non_empty for t in ep]
    if not flat:
        return {
            "ac_tokens_per_turn": 0.0, "ob_tokens_per_turn": 0.0,
            "response_tokens_min": 0.0, "response_tokens_max": 0.0,
            "response_tokens_std": 0.0, "turns_per_episode": 0.0,
            "total_turns": 0, "total_episodes": 0,
            "total_ac_tokens": 0, "total_ob_tokens": 0,
            "frac_max_turns_reached": 0.0,
        }
    ac = [len(t.act_tokens) for t in flat]
    ob = [len(t.obs_tokens) for t in flat]
    turns_per_ep = [len(ep) for ep in non_empty]
    # An episode "reached max turns" if it ran out of turns rather than ending
    # early on a reward-driven done: >1 turn and a done final transition.
    # Single-turn envs report 0 by construction.
    max_turns_hits = sum(1 for ep in non_empty if len(ep) > 1 and ep[-1].done)
    n = len(ac)
    mean_ac = sum(ac) / n
    var_ac = sum((x - mean_ac) ** 2 for x in ac) / n
    return {
        "ac_tokens_per_turn": mean_ac,
        "ob_tokens_per_turn": sum(ob) / n,
        "response_tokens_min": float(min(ac)),
        "response_tokens_max": float(max(ac)),
        "response_tokens_std": var_ac ** 0.5,
        "turns_per_episode": sum(turns_per_ep) / len(turns_per_ep),
        "total_turns": int(n),
        "total_episodes": int(len(non_empty)),
        "total_ac_tokens": int(sum(ac)),
        "total_ob_tokens": int(sum(ob)),
        "frac_max_turns_reached": max_turns_hits / len(non_empty),
    }


def compute_group_health_stats(groups: list) -> dict:
    """Per-GRPO-group success composition (grouped advantage mode only).

    A group is the K rollouts for one prompt. GRPO centers advantages within a group, so groups where all K rollouts succeed (or all fail) give ~zero training signal. Tracking the fractions surfaces that. "Good" = the episode's summed reward is positive. `groups` is the list-of-groups structure from `collect_grouped_batch_async`; pass None/empty for non-grouped runs to skip.
    """
    if not groups:
        return {}
    n_all_good = n_all_bad = n_mixed = 0
    n = 0
    for group in groups:
        eps = [ep for ep in group if ep]
        if not eps:
            continue
        n += 1
        goods = [sum(t.reward for t in ep) > 0 for ep in eps]
        if all(goods):
            n_all_good += 1
        elif not any(goods):
            n_all_bad += 1
        else:
            n_mixed += 1
    if n == 0:
        return {}
    return {
        "by_group/frac_all_good": n_all_good / n,
        "by_group/frac_all_bad": n_all_bad / n,
        "by_group/frac_mixed": n_mixed / n,
    }


def record_trajectories(
    path: str,
    step: int,
    episodes: "list[list]",
    tokenizer,
) -> None:
    """Append one JSON object per transition to ``{run_folder}/trajectories.jsonl``.

    Opt-in via ``--record_trajectories``. Each line captures the decoded prompt/response for one turn of one rollout plus its reward and (already computed) advantage, so a run's rollouts can be inspected turn-by-turn after the fact. ``episodes`` is the list-of-episodes view (each episode a list of ``Transition``), so ``turn`` indexes within an episode and ``episode`` is a flat episode index for the step. Decoding is best-effort: a tokenizer that chokes on a token list degrades to the raw ids rather than aborting the run.
    """
    def _decode(toks):
        try:
            return tokenizer.decode(list(toks))
        except Exception:
            return {"raw_token_ids": list(toks)}

    with open(path, "a") as f:
        for ep_idx, episode in enumerate(episodes):
            for turn_idx, t in enumerate(episode):
                rec = {
                    "step": step,
                    "episode": ep_idx,
                    "turn": turn_idx,
                    "env_type": getattr(t, "env_type", None),
                    "prompt": _decode(t.obs_tokens),
                    "response": _decode(t.act_tokens),
                    "reward": float(t.reward),
                    "advantage": float(t.advantage),
                    "done": bool(t.done),
                    "num_prompt_tokens": len(t.obs_tokens),
                    "num_response_tokens": len(t.act_tokens),
                }
                f.write(json.dumps(rec) + "\n")


def episodes_to_rollout_dicts(episodes: list, tokenizer, cap: int) -> list:
    """Decode the first ``cap`` episodes into plain dicts for MLflow tracing.

    ``cap=-1`` (or any negative) means ALL episodes. Feeds
    :meth:`service.mlflow_log.MlflowPublisher.log_rollout_traces`, which is
    deliberately tokenizer-free; decoding happens here. One dict per
    episode: ``{"episode": i, "turns": [{"prompt", "response", "reward",
    "advantage", "done"}, ...]}``. Decoding is best-effort (same policy as
    ``record_trajectories``): a tokenizer that chokes degrades to the raw ids.
    """
    def _decode(toks):
        try:
            return tokenizer.decode(list(toks))
        except Exception:
            return str(list(toks))

    selected = episodes if cap < 0 else episodes[:cap]
    rollouts = []
    for ep_idx, episode in enumerate(selected):
        turns = [
            {
                "prompt": _decode(t.obs_tokens),
                "response": _decode(t.act_tokens),
                "reward": float(t.reward),
                "advantage": float(t.advantage),
                "done": bool(t.done),
            }
            for t in episode
        ]
        rollouts.append({"episode": ep_idx, "turns": turns})
    return rollouts


def capture_rollouts(args, step: int, episodes: list, tokenizer, metrics_path: str) -> None:
    """Shared per-step rollout capture for both training loops.

    Two opt-in sinks, independent of each other:

    * ``--record_trajectories``: append every episode's decoded turns to
      ``{run_folder}/trajectories.jsonl`` (uploaded as an MLflow artifact at
      the end of the run when publishing is enabled).
    * ``--mlflow_log_rollouts N``: publish the first N episodes of this step
      as MLflow traces (visible live in the experiment's Traces view).
    """
    if cfg(args, "record_trajectories"):
        traj_path = os.path.join(
            run_artifacts(args).run_folder or os.path.dirname(metrics_path),
            "trajectories.jsonl",
        )
        record_trajectories(traj_path, step, episodes, tokenizer)

    n_traces = int(cfg(args, "mlflow_log_rollouts") or 0)
    publisher = run_artifacts(args).mlflow
    if n_traces != 0 and publisher is not None:
        publisher.log_rollout_traces(
            step, episodes_to_rollout_dicts(episodes, tokenizer, n_traces)
        )


class MetricsLogger:
    """Accumulates and writes per-step metrics to JSONL.

    Computes rollout statistics, namespaces the backend's training metrics under train/, records timing information, and writes one JSON line per training step. When constructed with a ``publisher`` (an :class:`service.mlflow_log.MlflowPublisher`), every record is also mirrored to MLflow (numeric values only), keyed by the same step.
    """

    def __init__(self, metrics_path: str, publisher=None):
        self.metrics_path = metrics_path
        self.publisher = publisher

    def log_step(
        self,
        step: int,
        episodes: list,
        transitions: list,
        fb_result,
        sample_seconds: float,
        train_seconds: float,
        step_total_seconds: float,
        effective_lr: float,
        kl_coef: float,
        extra_metrics: Optional[dict] = None,
        groups: Optional[list] = None,
    ) -> dict:
        """Compute all metrics for one step and append to JSONL.

        `extra_metrics`, when provided, is merged into the record before it is written. This is how periodic validation metrics (`val/reward_mean`, `val/reward_std`, `val/accuracy`) are appended to the step's JSONL record on validation steps.

        `groups`, when provided (grouped advantage mode), is the list-of-groups rollout structure used only for the GRPO per-group health stats; None on non-grouped runs, which simply omit those keys.

        Returns the metrics dict for use by the terminal table printer.
        """
        record = {"step": step}

        # Reward stats
        reward_stats = compute_reward_stats(episodes)
        record.update({f"rollout/{k}": v for k, v in reward_stats.items()})

        # Advantage stats
        adv_stats = compute_advantage_stats(transitions)
        record.update({f"rollout/{k}": v for k, v in adv_stats.items()})

        # Episode length
        record["rollout/episode_length_mean"] = compute_episode_length_mean(episodes)

        # Episode counts
        num_eps, num_failed = count_episodes(episodes)
        record["rollout/num_episodes"] = num_eps
        record["rollout/num_failed_rollouts"] = num_failed

        # Token volume + turn distribution (drives the `tokens` plot family).
        for k, v in compute_token_turn_stats(episodes, transitions).items():
            record[f"rollout/{k}"] = v

        # Per-GRPO-group success composition (grouped advantage mode only).
        for k, v in compute_group_health_stats(groups).items():
            record[f"rollout/{k}"] = v

        # Training-step metrics namespaced under train/
        if fb_result and hasattr(fb_result, "metrics") and fb_result.metrics:
            for k, v in fb_result.metrics.items():
                record[f"train/{k}"] = v

        # Timing
        record["time/sample_seconds"] = sample_seconds
        record["time/train_seconds"] = train_seconds
        record["time/step_total_seconds"] = step_total_seconds

        # Config
        record["config/learning_rate"] = effective_lr

        # Conditional KL divergence
        if kl_coef > 0 and transitions:
            # KL divergence computed from transition logprobs when available. Placeholder: actual KL computation requires reference logprobs which will be added when the KL penalty path is fully wired.
            pass

        # Merge any extra metrics (e.g. validation metrics on validation steps) into the step's record.
        if extra_metrics:
            record.update(extra_metrics)

        append_metrics(self.metrics_path, record)
        if self.publisher is not None:
            self.publisher.log_metrics(record, step=step)

        return record


# === Terminal table output ===
#
# Formatted stdout output during training: a one-line header at the start of the run, then an aligned metrics row every --log_every steps. The header and data rows share one column layout so the columns line up beneath each other.

# Shared column layout. Each column is fixed-width and right-aligned; two spaces separate columns. `format_metrics_table` mirrors these widths exactly.
_METRICS_TABLE_HEADER = (
    f"{'Step':>6}  {'Reward':>8}  {'EpLen':>7}  {'Loss':>10}  "
    f"{'LR':>11}  {'SampleT':>9}  {'TrainT':>8}"
)


def format_metrics_table(metrics: dict, step: int) -> str:
    """Format a metrics dict into an aligned single-line table row.

    Columns: Step | Reward | EpLen | Loss | LR | SampleT | TrainT

    Uses fixed-width formatting so successive rows (and the header printed by `print_training_header`) line up. Missing keys fall back to sensible defaults; the Loss column shows `train/loss` or `train/actor/pg_loss` (whichever is present), else "n/a". Floats use reasonable precision: reward to 3 decimals, episode length / times to 2 decimals, and the learning rate in scientific notation.
    """
    reward = metrics.get("rollout/reward_mean", 0.0)
    ep_len = metrics.get("rollout/episode_length_mean", 0.0)
    lr = metrics.get("config/learning_rate", 0.0)
    sample_t = metrics.get("time/sample_seconds", 0.0)
    train_t = metrics.get("time/train_seconds", 0.0)

    # The policy-gradient losses report under train/actor/pg_loss; a plain
    # train/loss is used by non-PG runs. Show whichever is present.
    loss = metrics.get("train/loss", metrics.get("train/actor/pg_loss"))
    loss_str = f"{loss:>10.4f}" if loss is not None else f"{'n/a':>10}"

    # Steps are 0-based internally (metrics.jsonl, checkpoints); the console
    # shows the 1-based number a reader expects (step 1/N .. N/N).
    return (
        f"{step + 1:>6}  "
        f"{reward:>8.3f}  "
        f"{ep_len:>7.2f}  "
        f"{loss_str}  "
        f"{lr:>11.2e}  "
        f"{sample_t:>9.2f}  "
        f"{train_t:>8.2f}"
    )


def format_metrics_kv(record: dict) -> dict[str, str]:
    """Format a step record for the detailed metrics table.

    Shared formatting rule; see
    :func:`training_sessions_cookbook.service.console.format_metric_values`.
    """
    return format_metric_values(record)


class MetricsTablePrinter:
    """Prints metric-table rows, emitting the column header lazily.

    The header goes out immediately before the first row, never earlier, so
    the two are always adjacent in the output regardless of how much rollout
    or validation output the step produced in between. One instance per
    training run.
    """

    def __init__(self) -> None:
        self._header_printed = False

    def print_row(self, metrics: dict, step: int) -> None:
        if not self._header_printed:
            console_print(_METRICS_TABLE_HEADER)
            self._header_printed = True
        console_print(format_metrics_table(metrics, step))

    def print_details(self, metrics: dict, step: int) -> None:
        """Print every metric in the record as a key/value table.

        The compact row shows the headline numbers; this shows everything
        that went into ``metrics.jsonl`` for the step. The training loops
        call it on the ``--log_every`` cadence.
        """
        print_kv_table(f"Step {step + 1} metrics", format_metrics_kv(metrics))
        # The compact table continues below; reprint its header so the next
        # row does not appear unlabeled after the detail table.
        self._header_printed = False

