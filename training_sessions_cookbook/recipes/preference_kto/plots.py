#!/usr/bin/env python3
"""Plot the KTO recipe's metrics.jsonl.

    python -m training_sessions_cookbook.recipes.preference_kto.plots \
        --metrics_path results/preference_kto/metrics.jsonl \
        --output_dir results/preference_kto

Writes three figures next to the metrics (or to --output_dir):

- ``learning.png``  — KTO loss and the implicit reward margin (train + eval)
- ``accuracy.png``  — preference accuracy (train + eval)
- ``logps.png``     — mean completion log-probability per class, the
  degeneration guard: a sharp drop on the desirable class means the model is
  buying margin by degrading its own generations.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt

TRAIN_COLOR = "#1f77b4"
EVAL_COLOR = "#2ca02c"
DESIRABLE_COLOR = "#1f77b4"
UNDESIRABLE_COLOR = "#d62728"
RAW_ALPHA = 0.25


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Trailing mean over up to ``window`` points (no phase shift at the start)."""
    out, buf = [], []
    for v in values:
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def _style(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _train_eval_panel(ax, steps, train_raw, ev_steps, ev_vals, *, window, ylabel,
                      baseline=None, baseline_label=None, show_raw=True):
    """Train rolling mean (optionally over the faint raw series) and eval points.

    ``show_raw=False`` for series whose per-step variance spans the axis
    (e.g. the reward margin): the faint raw line would set the y-limits and
    flatten the trend it is meant to decorate.
    """
    if show_raw:
        ax.plot(steps, train_raw, color=TRAIN_COLOR, lw=0.7, alpha=RAW_ALPHA)
    ax.plot(steps, rolling_mean(train_raw, window), color=TRAIN_COLOR, lw=2.0,
            label=f"train ({window}-step mean)")
    if ev_steps:
        ax.plot(ev_steps, ev_vals, "o-", color=EVAL_COLOR, lw=2.0, ms=5,
                label="eval (held-out)")
    if baseline is not None:
        ax.axhline(baseline, color="gray", ls=":", lw=1.2, label=baseline_label)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    _style(ax)


def plot_kto_run(metrics_path: str, output_dir: str, window: int = 10) -> list[str]:
    records = [json.loads(line) for line in open(metrics_path)]
    steps = [r["step"] for r in records]
    evals = [r for r in records if "kto/eval_reward_margin" in r]
    ev_steps = [r["step"] for r in evals]

    def series(key: str) -> list[float]:
        return [r[key] for r in records]

    def eval_series(key: str) -> list[float]:
        return [r[key] for r in evals]

    os.makedirs(output_dir, exist_ok=True)
    written = []

    # --- learning.png: loss + reward margin -------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax1.plot(steps, series("kto/loss"), color=TRAIN_COLOR, lw=0.7, alpha=RAW_ALPHA)
    ax1.plot(steps, rolling_mean(series("kto/loss"), window), color=TRAIN_COLOR,
             lw=2.0, label=f"train ({window}-step mean)")
    ax1.axhline(0.5, color="gray", ls=":", lw=1.2, label="0.5 (untrained)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("KTO loss")
    ax1.set_title("Training loss")
    ax1.legend(frameon=False)
    _style(ax1)

    _train_eval_panel(
        ax2, steps, series("kto/reward_margin"),
        ev_steps, eval_series("kto/eval_reward_margin"),
        window=window, ylabel="reward margin (nats per completion)",
        baseline=0, baseline_label="0 (no separation)", show_raw=False)
    ax2.set_title("Implicit reward margin: desirable \u2212 undesirable")
    fig.tight_layout()
    path = os.path.join(output_dir, "learning.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- accuracy.png ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _train_eval_panel(
        ax, steps, series("kto/accuracy"),
        ev_steps, eval_series("kto/eval_accuracy"),
        window=window, ylabel="accuracy",
        baseline=0.5, baseline_label="0.5 (chance)")
    # Step 0 reports accuracy 0.0 (all margins exactly zero at init); keep the
    # point but do not let the artifact set the axis floor.
    ax.set_ylim(0.35, 0.9)
    ax.set_title("Preference accuracy")
    fig.tight_layout()
    path = os.path.join(output_dir, "accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # --- logps.png: per-class completion log-probs ------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(steps, rolling_mean(series("kto/logps_desirable"), window),
            color=DESIRABLE_COLOR, lw=2.0, label="desirable (train)")
    ax.plot(steps, rolling_mean(series("kto/logps_undesirable"), window),
            color=UNDESIRABLE_COLOR, lw=2.0, label="undesirable (train)")
    if ev_steps:
        ax.plot(ev_steps, eval_series("kto/eval_logps_desirable"), "o--",
                color=DESIRABLE_COLOR, lw=1.4, ms=5, alpha=0.7,
                label="desirable (eval)")
        ax.plot(ev_steps, eval_series("kto/eval_logps_undesirable"), "o--",
                color=UNDESIRABLE_COLOR, lw=1.4, ms=5, alpha=0.7,
                label="undesirable (eval)")
    ax.set_xlabel("step")
    ax.set_ylabel("mean completion log-probability")
    ax.set_title("Completion log-probability by preference class")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    path = os.path.join(output_dir, "logps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics_path", required=True,
                        help="Path to a KTO run's metrics.jsonl.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write the PNGs (default: next to the metrics).")
    parser.add_argument("--window", type=int, default=10,
                        help="Rolling-mean window over train steps (default 10).")
    args = parser.parse_args()
    out = args.output_dir or os.path.dirname(os.path.abspath(args.metrics_path))
    for path in plot_kto_run(args.metrics_path, out, window=args.window):
        print("wrote", path)


if __name__ == "__main__":
    main()
