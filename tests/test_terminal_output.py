"""Unit tests for terminal table output functions.

- format_metrics_table produces a row with the required columns
- MetricsTablePrinter emits the aligned column header once, adjacent to the first row
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_sessions_cookbook.rl.train import (
    _METRICS_TABLE_HEADER,
    format_metrics_table,
    MetricsTablePrinter,
)


def _full_metrics() -> dict:
    return {
        "step": 42,
        "rollout/reward_mean": 0.357,
        "rollout/episode_length_mean": 2.4,
        "config/learning_rate": 4e-5,
        "time/sample_seconds": 12.34,
        "time/train_seconds": 0.81,
        "train/loss": 0.4231,
    }


def test_table_row_contains_all_values():
    """Row reflects reward, episode length, loss, LR, and timings (Req 5.2)."""
    row = format_metrics_table(_full_metrics(), step=42)
    assert "42" in row
    assert "0.357" in row  # reward to 3 decimals
    assert "2.40" in row  # episode length to 2 decimals
    assert "0.4231" in row  # loss to 4 decimals
    assert "4.00e-05" in row  # LR in scientific notation
    assert "12.34" in row  # sample time
    assert "0.81" in row  # train time


def test_header_lists_required_columns():
    """The shared header names every required column (Req 5.2)."""
    for col in ["Step", "Reward", "EpLen", "Loss", "LR", "SampleT", "TrainT"]:
        assert col in _METRICS_TABLE_HEADER


def test_row_aligns_with_header():
    """Each data row has the same width as the header (Req 5.4)."""
    row = format_metrics_table(_full_metrics(), step=42)
    assert len(row) == len(_METRICS_TABLE_HEADER)


def test_missing_loss_shows_na():
    """Loss may be absent depending on the backend; show n/a, not a crash."""
    metrics = _full_metrics()
    del metrics["train/loss"]
    row = format_metrics_table(metrics, step=1)
    assert "n/a" in row
    # Still aligned even with the n/a placeholder.
    assert len(row) == len(_METRICS_TABLE_HEADER)


def test_missing_keys_use_defaults():
    """An empty metrics dict still produces an aligned row (Req 5.1)."""
    row = format_metrics_table({}, step=0)
    assert "0" in row
    assert len(row) == len(_METRICS_TABLE_HEADER)


def test_table_printer_header_once_adjacent_to_first_row(capsys):
    """The column header prints exactly once, immediately before the first
    row, so the two are always adjacent regardless of intervening output."""
    printer = MetricsTablePrinter()
    printer.print_row(_full_metrics(), step=0)
    printer.print_row(_full_metrics(), step=1)
    out = capsys.readouterr().out
    assert out.count(_METRICS_TABLE_HEADER.strip()) == 1
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines[0] == _METRICS_TABLE_HEADER
    assert len(lines) == 3  # header + two rows


def test_table_printer_no_output_before_first_row(capsys):
    printer = MetricsTablePrinter()
    assert capsys.readouterr().out == ""


def test_format_metrics_kv_formatting_rules():
    """Ints bare; floats six-decimal (zero included); sub-1e-4 floats switch
    to scientific notation instead of collapsing to 0.000000; step omitted;
    sorted."""
    from training_sessions_cookbook.rl.metrics import format_metrics_kv

    out = format_metrics_kv({
        "step": 3,
        "rollout/num_episodes": 8,
        "rollout/reward_mean": 0.0,
        "train/loss": 0.5,
        "config/learning_rate": 5e-13,
        "phase": "baseline",
    })
    assert "step" not in out
    assert out["rollout/num_episodes"] == "8"
    assert out["rollout/reward_mean"] == "0.000000"
    assert out["train/loss"] == "0.500000"
    assert out["config/learning_rate"] == "5.00e-13"
    assert out["phase"] == "baseline"
    assert list(out) == sorted(out)


def test_print_details_reprints_header_before_next_row(capsys):
    """After a detail table, the next compact row is re-labeled with the
    column header so it never appears unlabeled."""
    printer = MetricsTablePrinter()
    printer.print_row(_full_metrics(), step=0)
    printer.print_details(_full_metrics(), step=0)
    printer.print_row(_full_metrics(), step=1)
    out = capsys.readouterr().out
    assert out.count(_METRICS_TABLE_HEADER.strip()) == 2
