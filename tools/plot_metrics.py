#!/usr/bin/env python3
"""Generate training metric plots from one or more metrics.jsonl files.

Single run (unchanged):
    python tools/plot_metrics.py --metrics_path runs/my-run/metrics.jsonl
    python tools/plot_metrics.py --metrics_path metrics.jsonl --smoothing 0.9 --output_dir ./plots

Multi-run overlay (compare two or more runs):
    python tools/plot_metrics.py \
        --compare run_a=runs/gsm8k-a/metrics.jsonl \
                  run_b=runs/gsm8k-b/metrics.jsonl \
        --smoothing 0.9 --output_dir ./plots

Each `--compare label=path` run is overlaid on every panel with its own color and a legend, so the curves land on the same axes. Runs that emit the same metrics.jsonl schema (e.g. two training runs that differ only in hyperparameters or model) can be compared apples-to-apples.
"""
import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script works headless (no display).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # must come after matplotlib.use

# Distinct colors for overlaid runs. Cycled if more runs than colors are compared.
_SERIES_COLORS = [
    "#1f77b4",  # blue   (single-run default)
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
]


# Panels that are always attempted. Each entry: (metric_key, title, y_label).
BASE_PANELS = [
    ("rollout/reward_mean", "Reward", "reward"),
    ("rollout/episode_length_mean", "Episode Length", "turns"),
    ("train/loss", "Training Loss", "loss"),
    ("config/learning_rate", "Learning Rate", "lr"),
    ("time/sample_seconds", "Sample Time", "seconds"),
    ("time/train_seconds", "Train Time", "seconds"),
]

# Panels that are only rendered when the key is present in at least one record.
CONDITIONAL_PANELS = [
    ("rollout/advantage_mean", "Advantage Mean", "advantage"),
    ("rollout/kl_divergence_mean", "KL Divergence", "kl"),
]


def ema_smooth(values: list[float], alpha: float) -> list[float]:
    """Apply exponential moving average smoothing.

    Args:
        values: Raw time series.
        alpha: Smoothing factor in [0, 1]. 0 = no smoothing, 1 = full smoothing.

    Returns:
        Smoothed series of same length.
    """
    if alpha <= 0.0 or not values:
        return list(values)
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
    return smoothed


def load_metrics(path: str) -> list[dict]:
    """Read all JSON lines from a metrics file."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _extract_series(records: list[dict], key: str) -> tuple[list, list]:
    """Extract (steps, values) for a metric key, skipping records lacking it.

    The x-axis is taken from each record's ``step`` field. Records that are missing either the metric ``key`` or the ``step`` field are skipped so a metric that only appears in some records (e.g. conditional KL) still plots cleanly.
    """
    steps = []
    values = []
    for i, rec in enumerate(records):
        if key not in rec:
            continue
        value = rec[key]
        if value is None:
            continue
        # Fall back to record index if a step field is unavailable.
        step = rec.get("step", i)
        steps.append(step)
        values.append(value)
    return steps, values


def _select_panels(runs: list[tuple[str, list[dict]]]) -> list[tuple]:
    """Return the panels that have data in at least one run.

    ``runs`` is a list of ``(label, records)``. A panel is included when its metric key appears in any run. Returns a list of ``(key, title, y_label)`` in BASE then CONDITIONAL order.
    """
    def _any_has(key: str) -> bool:
        return any(_extract_series(recs, key)[1] for _, recs in runs)

    panels = []
    for key, title, y_label in BASE_PANELS:
        if _any_has(key):
            panels.append((key, title, y_label))
    for key, title, y_label in CONDITIONAL_PANELS:
        if _any_has(key):
            panels.append((key, title, y_label))
    return panels


def plot_metrics(records: list[dict], smoothing: float, output_dir: str) -> None:
    """Single-run plot (unchanged public behavior).

    Thin wrapper over :func:`plot_runs` with one unlabeled run, preserving the original API and output filenames.
    """
    plot_runs([("run", records)], smoothing, output_dir, overlay=False)


def plot_runs(
    runs: list[tuple[str, list[dict]]],
    smoothing: float,
    output_dir: str,
    *,
    overlay: bool,
) -> None:
    """Generate a grid of time-series plots for one or more runs.

    Args:
        runs: List of ``(label, records)``. With one run this reproduces the original single-run plot; with several, each run is overlaid on every panel with its own color + a legend.
        smoothing: EMA factor in [0, 1]. 0 disables smoothing.
        output_dir: Directory for the PNG/PDF output.
        overlay: When True (multiple runs), draw a legend keyed by run label and label each curve by run; when False, keep the original raw/smoothed single-series legend.

    Panels whose metric keys are absent from every run are skipped (graceful degradation). Each panel plots value vs. the ``step`` field.
    """
    panels = _select_panels(runs)
    os.makedirs(output_dir, exist_ok=True)

    if not panels:
        fig = plt.figure(figsize=(8, 6))
        fig.text(0.5, 0.5, "No plottable metrics found", ha="center", va="center")
    else:
        n = len(panels)
        ncols = 2 if n > 1 else 1
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False
        )
        flat_axes = [ax for row in axes for ax in row]

        for ax, (key, title, y_label) in zip(flat_axes, panels):
            has_legend = False
            for i, (label, records) in enumerate(runs):
                steps, values = _extract_series(records, key)
                if not values:
                    continue
                color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
                if overlay:
                    # One curve per run, labeled by run. Raw is faded when smoothing is on; the smoothed curve carries the label.
                    if smoothing > 0:
                        ax.plot(steps, values, color=color, alpha=0.25)
                        ax.plot(steps, ema_smooth(values, smoothing),
                                color=color, label=label)
                    else:
                        ax.plot(steps, values, color=color, label=label)
                    has_legend = True
                else:
                    ax.plot(steps, values, color=color,
                            alpha=0.4 if smoothing > 0 else 1.0,
                            label="raw" if smoothing > 0 else None)
                    if smoothing > 0:
                        ax.plot(steps, ema_smooth(values, smoothing),
                                color=color, label="smoothed")
                        has_legend = True
            if has_legend:
                ax.legend(loc="best", fontsize="small")
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)

        for ax in flat_axes[len(panels):]:
            ax.set_visible(False)

    fig.tight_layout()

    # Overlay runs write a distinct filename so a comparison doesn't clobber a single-run plot sitting in the same output dir.
    stem = "comparison_plots" if overlay else "training_plots"
    png_path = os.path.join(output_dir, f"{stem}.png")
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved plots to:\n  {png_path}\n  {pdf_path}")


def parse_compare_specs(specs: list[str]) -> list[tuple[str, str]]:
    """Parse ``label=path`` overlay specs into ``(label, path)`` pairs.

    A spec without ``=`` uses the file's parent directory name as the label. Order is preserved so colors/legend follow the order given on the CLI.
    """
    runs = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = os.path.basename(os.path.dirname(os.path.abspath(path))) or path
        runs.append((label, path))
    return runs


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics (single run or overlay).")
    parser.add_argument("--metrics_path", default=None,
                        help="Single metrics.jsonl to plot.")
    parser.add_argument("--compare", nargs="+", default=None, metavar="LABEL=PATH",
                        help="Overlay multiple runs on the same axes, e.g. "
                             "--compare run_a=a/metrics.jsonl run_b=b/metrics.jsonl")
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.compare:
        runs = parse_compare_specs(args.compare)
        loaded = [(label, load_metrics(path)) for label, path in runs]
        # Default the output dir to the first run's parent when not given.
        output_dir = args.output_dir or str(Path(runs[0][1]).parent)
        plot_runs(loaded, args.smoothing, output_dir, overlay=True)
    elif args.metrics_path:
        output_dir = args.output_dir or str(Path(args.metrics_path).parent)
        plot_metrics(load_metrics(args.metrics_path), args.smoothing, output_dir)
    else:
        parser.error("provide --metrics_path (single run) or --compare (overlay)")


if __name__ == "__main__":
    main()
