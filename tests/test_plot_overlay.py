"""Tests for the multi-run overlay in plot_metrics.py.

Covers spec parsing, the overlay output path, single-run behavior preservation, and graceful handling of runs with partially-overlapping metric keys.
"""
import json
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"
    ),
)

import plot_metrics as pm


def _write_run(dirpath, name, rewards, extra_keys=True):
    run_dir = os.path.join(dirpath, name)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path, "w") as f:
        for i, r in enumerate(rewards):
            rec = {"step": i, "rollout/reward_mean": r}
            if extra_keys:
                rec.update({
                    "rollout/episode_length_mean": 1.0,
                    "train/loss": 0.5 - 0.01 * i,
                    "config/learning_rate": 4e-5,
                    "time/sample_seconds": 2.0,
                    "time/train_seconds": 1.0,
                })
            f.write(json.dumps(rec) + "\n")
    return path


def test_parse_compare_specs_label_and_bare(tmp_path):
    p = _write_run(str(tmp_path), "run_a", [0.1, 0.2])
    labeled = pm.parse_compare_specs([f"svc={p}"])
    assert labeled == [("svc", p)]
    # Bare path -> parent dir name as label.
    bare = pm.parse_compare_specs([p])
    assert bare[0][0] == "run_a"
    assert bare[0][1] == p


def test_overlay_writes_comparison_plot(tmp_path):
    run_a = _write_run(str(tmp_path), "run_a", [0.1, 0.3, 0.5, 0.7])
    run_b = _write_run(str(tmp_path), "run_b", [0.1, 0.28, 0.52, 0.66])
    out = str(tmp_path / "plots")
    runs = [("run_a", pm.load_metrics(run_a)), ("run_b", pm.load_metrics(run_b))]
    pm.plot_runs(runs, smoothing=0.6, output_dir=out, overlay=True)
    assert os.path.exists(os.path.join(out, "comparison_plots.png"))
    assert os.path.exists(os.path.join(out, "comparison_plots.pdf"))
    # Must NOT clobber the single-run filename.
    assert not os.path.exists(os.path.join(out, "training_plots.png"))


def test_single_run_still_writes_training_plots(tmp_path):
    run_a = _write_run(str(tmp_path), "run_a", [0.1, 0.3, 0.5])
    out = str(tmp_path / "plots")
    pm.plot_metrics(pm.load_metrics(run_a), smoothing=0.0, output_dir=out)
    assert os.path.exists(os.path.join(out, "training_plots.png"))


def test_overlay_tolerates_missing_keys_in_one_run(tmp_path):
    # run_a has full schema; run_b has only reward_mean. The panel selection should still include reward (present in both) and the loss/time panels (present in run_a) without crashing on the sparse run.
    run_a = _write_run(str(tmp_path), "run_a", [0.1, 0.3, 0.5], extra_keys=True)
    run_b = _write_run(str(tmp_path), "run_b", [0.2, 0.4, 0.6], extra_keys=False)
    out = str(tmp_path / "plots")
    runs = [("run_a", pm.load_metrics(run_a)), ("run_b", pm.load_metrics(run_b))]
    pm.plot_runs(runs, smoothing=0.0, output_dir=out, overlay=True)
    assert os.path.exists(os.path.join(out, "comparison_plots.png"))


def test_empty_runs_emit_placeholder_figure(tmp_path):
    empty = _write_run(str(tmp_path), "empty", [], extra_keys=False)
    out = str(tmp_path / "plots")
    pm.plot_runs([("empty", pm.load_metrics(empty))], smoothing=0.0,
                 output_dir=out, overlay=True)
    # No plottable metrics -> still writes a (placeholder) figure.
    assert os.path.exists(os.path.join(out, "comparison_plots.png"))
