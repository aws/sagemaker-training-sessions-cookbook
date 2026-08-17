"""Unit tests for run folder management utilities.

Tests: create_run_folder, _get_git_hash, dump_config, setup_file_logging.
"""
import json
import os
import re
import sys
import tempfile

import pytest

# Add the repo root to path so ``rl`` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_sessions_cookbook.rl.train import create_run_folder, dump_config, _get_git_hash, setup_file_logging


class TestCreateRunFolder:
    """Tests for create_run_folder."""

    def test_creates_auto_named_folder_with_timestamp(self, tmp_path, monkeypatch):
        """When output_dir is None, creates runs/{run_name}_{YYYYMMDD_HHMMSS}/."""
        monkeypatch.chdir(tmp_path)
        result = create_run_folder("my-run")
        # Verify the path matches the expected pattern
        basename = os.path.basename(result)
        assert re.match(r"my-run_\d{8}_\d{6}$", basename), f"Unexpected folder name: {basename}"
        assert os.path.isdir(result)
        # Verify it's under runs/
        parent = os.path.basename(os.path.dirname(result))
        assert parent == "runs"

    def test_uses_output_dir_when_provided(self, tmp_path):
        """When output_dir is provided, uses it directly."""
        output_dir = str(tmp_path / "custom_output")
        result = create_run_folder("ignored-name", output_dir=output_dir)
        assert result == os.path.abspath(output_dir)
        assert os.path.isdir(result)

    def test_exist_ok_when_folder_exists(self, tmp_path):
        """Does not error when the folder already exists."""
        output_dir = str(tmp_path / "existing")
        os.makedirs(output_dir)
        result = create_run_folder("ignored", output_dir=output_dir)
        assert os.path.isdir(result)

    def test_returns_absolute_path(self, tmp_path, monkeypatch):
        """Always returns an absolute path."""
        monkeypatch.chdir(tmp_path)
        result = create_run_folder("test-run")
        assert os.path.isabs(result)


class TestGetGitHash:
    """Tests for _get_git_hash."""

    def test_returns_string(self):
        """Should always return a string."""
        result = _get_git_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_hash_or_unknown(self):
        """Returns either a hex hash or 'unknown'."""
        result = _get_git_hash()
        assert result == "unknown" or re.match(r"^[0-9a-f]{40}$", result)


class TestDumpConfig:
    """Tests for dump_config."""

    def test_writes_config_json(self, tmp_path):
        """Creates config.json with all args plus git_hash."""
        import argparse
        args = argparse.Namespace(
            base_model="Qwen/Qwen3-8B",
            learning_rate=4e-5,
            batch_size=8,
            max_steps=100,
        )
        run_folder = str(tmp_path)
        dump_config(args, run_folder)

        config_path = os.path.join(run_folder, "config.json")
        assert os.path.exists(config_path)

        with open(config_path) as f:
            config = json.load(f)

        assert config["base_model"] == "Qwen/Qwen3-8B"
        assert config["learning_rate"] == 4e-5
        assert config["batch_size"] == 8
        assert config["max_steps"] == 100
        assert "git_hash" in config
        assert isinstance(config["git_hash"], str)

    def test_config_json_has_all_args(self, tmp_path):
        """Every key in the namespace appears in config.json."""
        import argparse
        args = argparse.Namespace(a=1, b="hello", c=True, d=None)
        dump_config(args, str(tmp_path))

        with open(os.path.join(str(tmp_path), "config.json")) as f:
            config = json.load(f)

        for key in vars(args):
            assert key in config


class TestSetupFileLogging:
    """Tests for setup_file_logging."""

    def test_creates_training_log(self, tmp_path):
        """Creates training.log file in the run folder."""
        import logging

        # Reset logging to avoid interference from prior basicConfig calls
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        setup_file_logging(str(tmp_path))

        # Log something
        logging.info("test message")

        # Verify the file was created and has content
        log_path = os.path.join(str(tmp_path), "training.log")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            content = f.read()
        assert "test message" in content

        # Cleanup handlers to avoid affecting other tests
        for handler in root.handlers[:]:
            root.removeHandler(handler)
