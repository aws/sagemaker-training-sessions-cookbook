#!/usr/bin/env python3
"""Unified RL comparison plots: one script, config-driven per experiment.

Every RL comparison reads the same uniform layout (one ``metrics.jsonl`` per run
under ``results/<experiment>/<run>/``, the shape the SFT cookbooks use) and
draws the same panel families, so a single script covers all comparisons. Each
experiment is one entry in :data:`EXPERIMENTS` declaring its runs and which
figure families to draw:

* ``learning``: train reward, validation reward/accuracy, per-step wall clock,
  sample-vs-train time (drawn for every experiment).
* ``staleness``: off-by-k signals (stale groups dropped, steps-off-policy
  spread, failed rollouts, episodes/step); off-by-k runs only.
* ``tokens``: token generation & rollout shape (runs that log token stats).

All plots run fully offline off the committed ``results/`` tree; there is no
remote/EC2 fetch anywhere in this package. Every figure is truncated to the
minimum common step range across its runs, so a short run does not leave the
other lines trailing off alone past its last step.

Run from the repo root:

  python -m training_sessions_cookbook.rl.analysis.plot_runs                 # all experiments
  python -m training_sessions_cookbook.rl.analysis.plot_runs rl_coderl_offbyk   # one experiment
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from training_sessions_cookbook.rl.analysis._common import load_rows, plots_dir, series  # noqa: E402


# --- Experiment registry --------------------------------------------------
#
# Each run is (run_folder, legend_label, color, marker). ``figures`` lists
# which families to draw; ``title`` is the figure suptitle stem. Single-run
# experiments still draw their supported families (no comparison needed).
EXPERIMENTS: dict[str, dict] = {
    "rl_coderl_offbyk": {
        "title": "codeRL off-by-k requeue (k=3, 80 workers, 50s step), "
                 "gpt-oss-20b LoRA",
        "runs": [
            ("service_requeue", "Requeue", "#E41A1C", "o"),
        ],
        "figures": ["learning", "staleness", "tokens"],
    },
    "rl_gsm8k": {
        "title": "GSM8K on-policy, service backend (gpt-oss-20b LoRA)",
        "runs": [
            ("", "Service", "#377EB8", "o"),
        ],
        # The service run logs no token or staleness stats: only the learning
        # family (train/val reward, accuracy, per-step wall clock) is drawable.
        "figures": ["learning"],
    },
}


class _Run:
    """A loaded run: its rows plus display attributes and step ceiling."""

    def __init__(self, folder, label, color, marker, rows):
        self.label = label
        self.color = color
        self.marker = marker
        self.rows = rows
        self.max_step = max((r["step"] for r in rows if "step" in r), default=-1)


def _load(experiment: str) -> tuple[list[_Run], int]:
    """Load an experiment's runs; return (runs, min_common_max_step).

    The step ceiling is the *minimum* of each run's last step, so every figure
    can be truncated to a range all runs cover.
    """
    runs = [
        _Run(folder, label, color, marker, load_rows(experiment, folder))
        for folder, label, color, marker in EXPERIMENTS[experiment]["runs"]
    ]
    step_cap = min((r.max_step for r in runs), default=-1)
    return runs, step_cap


def _plot(ax, run: _Run, key: str, step_cap: int, *, label=None,
          lw=2.4, ls="-", markersize=5, alpha=1.0):
    """Draw one run's ``key`` series on ``ax``, truncated at ``step_cap``.

    Returns True if anything was drawn (so callers can decide on a legend).
    """
    s, v = series(run.rows, key)
    pairs = [(x, y) for x, y in zip(s, v) if step_cap < 0 or x <= step_cap]
    if not pairs:
        return False
    xs, ys = zip(*pairs)
    ax.plot(xs, ys, color=run.color, lw=lw, ls=ls, marker=run.marker,
            markersize=markersize, markevery=max(1, len(xs) // 12), alpha=alpha,
            label=label if label is not None else run.label)
    return True


# --- Figure families ------------------------------------------------------

def _fig_learning(runs, step_cap, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0][0]
    for r in runs:
        _plot(ax, r, "rollout/reward_mean", step_cap)
    ax.set_title("Train reward (rollout/reward_mean)", fontsize=14)
    ax.set_ylabel("reward", fontsize=12)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    ax = axes[0][1]
    for r in runs:
        _plot(ax, r, "val/reward_mean", step_cap, label=f"{r.label}, val reward",
              markersize=6)
        _plot(ax, r, "val/accuracy", step_cap, label=f"{r.label}, val accuracy",
              lw=1.8, ls="--", markersize=4, alpha=0.85)
    ax.set_title("Validation reward (solid) & accuracy (dashed)", fontsize=14)
    ax.set_ylabel("reward / accuracy", fontsize=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    ax = axes[1][0]
    for r in runs:
        _plot(ax, r, "time/step_total_seconds", step_cap)
    ax.set_title("Per-step wall clock (time/step_total_seconds)", fontsize=14)
    ax.set_ylabel("seconds", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    ax = axes[1][1]
    for r in runs:
        _plot(ax, r, "time/sample_seconds", step_cap, label=f"{r.label}, sample")
        _plot(ax, r, "time/train_seconds", step_cap, label=f"{r.label}, train",
              lw=1.8, ls="--", markersize=3, alpha=0.85)
    ax.set_title("Sample time (solid) & train time (dashed)", fontsize=14)
    ax.set_ylabel("seconds", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for ax in axes.flat:
        ax.set_xlabel("step", fontsize=12)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, "learning.png"


def _fig_staleness(runs, step_cap, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    def simple(ax, key, ptitle, ylab):
        # Materialize the draws in a list (not a generator) so every run is
        # plotted: ``any(generator)`` would short-circuit after the first
        # truthy result and skip _plot's draw side-effect for later runs.
        drew = any([_plot(ax, r, key, step_cap) for r in runs])
        ax.set_title(ptitle, fontsize=14)
        ax.set_ylabel(ylab, fontsize=12)
        if drew:
            ax.legend(loc="best", fontsize=9, framealpha=0.9)

    simple(axes[0][0], "rollout/num_stale_dropped",
           "Stale groups dropped (per step)", "groups")

    ax = axes[0][1]
    drew = False
    for r in runs:
        if _plot(ax, r, "sampler/step_mean", step_cap):
            drew = True
        band = [
            (row["step"], row.get("sampler/step_min"), row.get("sampler/step_max"))
            for row in r.rows
            if "step" in row and (step_cap < 0 or row["step"] <= step_cap)
            and row.get("sampler/step_min") is not None
            and row.get("sampler/step_max") is not None
        ]
        band.sort(key=lambda t: t[0])
        if band:
            ax.fill_between([t[0] for t in band], [t[1] for t in band],
                            [t[2] for t in band], color=r.color, alpha=0.12,
                            linewidth=0)
    ax.set_title("Staleness, mean (line) & min/max (band) steps off-policy",
                 fontsize=14)
    ax.set_ylabel("steps off-policy", fontsize=12)
    if drew:
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    simple(axes[1][0], "rollout/num_failed_rollouts",
           "Failed rollouts (per step)", "rollouts")
    simple(axes[1][1], "rollout/num_episodes", "Episodes per step", "episodes")

    for ax in axes.flat:
        ax.set_xlabel("step", fontsize=12)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{title}, staleness & failure signals", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, "staleness.png"


def _fig_tokens(runs, step_cap, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0][0]
    drew = False
    for r in runs:
        pts = [
            (row["step"],
             row["rollout/total_ac_tokens"] + row["rollout/total_ob_tokens"])
            for row in r.rows
            if "step" in row and (step_cap < 0 or row["step"] <= step_cap)
            and row.get("rollout/total_ac_tokens") is not None
            and row.get("rollout/total_ob_tokens") is not None
        ]
        pts.sort(key=lambda t: t[0])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=r.color,
                    lw=2.4, marker=r.marker, markersize=5,
                    markevery=max(1, len(pts) // 12), label=r.label)
            drew = True
    ax.set_title("Total tokens per batch (action + observation)", fontsize=14)
    ax.set_ylabel("tokens", fontsize=12)
    if drew:
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    def simple(ax, key, ptitle, ylab):
        # List, not generator: ``any(gen)`` short-circuits and would skip
        # _plot's draw side-effect for runs after the first truthy one.
        d = any([_plot(ax, r, key, step_cap) for r in runs])
        ax.set_title(ptitle, fontsize=14)
        ax.set_ylabel(ylab, fontsize=12)
        if d:
            ax.legend(loc="best", fontsize=9, framealpha=0.9)

    simple(axes[0][1], "rollout/ac_tokens_per_turn",
           "Model-generated (action) tokens per turn", "tokens")
    simple(axes[1][0], "rollout/turns_per_episode", "Turns per episode", "turns")
    simple(axes[1][1], "rollout/frac_max_turns_reached",
           "Fraction of rollouts hitting max turns", "fraction")

    for ax in axes.flat:
        ax.set_xlabel("step", fontsize=12)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{title}, token generation & rollout shape",
                 fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, "tokens.png"


_FIGURES = {
    "learning": _fig_learning,
    "staleness": _fig_staleness,
    "tokens": _fig_tokens,
}


def render(experiment: str) -> None:
    """Render every declared figure family for one experiment."""
    spec = EXPERIMENTS[experiment]
    runs, step_cap = _load(experiment)
    out_dir = plots_dir(experiment)
    for family in spec["figures"]:
        fig, fname = _FIGURES[family](runs, step_cap, spec["title"])
        out = out_dir / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}  (steps 0..{step_cap})")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    targets = argv if argv else list(EXPERIMENTS)
    for exp in targets:
        if exp not in EXPERIMENTS:
            raise SystemExit(
                f"unknown experiment {exp!r}; choices: {', '.join(EXPERIMENTS)}"
            )
        render(exp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
