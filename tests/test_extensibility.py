"""Tests for the extensibility seams: env registry, per-recipe parsers,
programmatic run construction, and the validation/dataset injection points.

No network / no SDK -- backends and clients are faked.
"""

from __future__ import annotations

import argparse
import unittest
from unittest import mock

from training_sessions_cookbook.rl import envs
from training_sessions_cookbook.rl.train import build_parser, parse_with_config


class _DummyEnv:
    def __init__(self, tag):
        self.tag = tag


class EnvRegistryTest(unittest.TestCase):
    def tearDown(self):
        envs._REGISTRY.pop("custom_test", None)

    def test_builtins_registered(self):
        self.assertIn("gsm8k", envs.registered_envs())
        self.assertIn("code_rl", envs.registered_envs())

    def test_register_and_resolve_custom_env(self):
        envs.register_env(
            "custom_test",
            lambda args: (lambda: _DummyEnv(args.dataset_path)),
        )
        args = mock.Mock(dataset_path="s3://my/data")
        factory = envs.get_env_factory("custom_test", args)
        env = factory()
        self.assertIsInstance(env, _DummyEnv)
        self.assertEqual(env.tag, "s3://my/data")

    def test_custom_env_cli_args_surface_in_parser(self):
        def my_flags(parser):
            parser.add_argument("--my_special_knob", type=int, default=7)

        envs.register_env(
            "custom_test", lambda args: (lambda: _DummyEnv(None)),
            add_cli_args=my_flags,
        )
        parser = build_parser(envs=("custom_test",))
        args = parser.parse_args(["--base_model", "Qwen/Qwen3-8B"])
        self.assertEqual(args.env_type, "custom_test")
        self.assertEqual(args.my_special_knob, 7)

    def test_unknown_env_raises_with_registered_names(self):
        with self.assertRaises(ValueError) as ctx:
            envs.get_env_factory("nope", mock.Mock())
        self.assertIn("gsm8k", str(ctx.exception))


class PerRecipeParserTest(unittest.TestCase):
    def _flags(self, parser):
        return {a.dest for a in parser._actions}

    def test_gsm8k_parser_excludes_code_rl_flags(self):
        flags = self._flags(build_parser(envs=("gsm8k",)))
        self.assertIn("answer_format", flags)
        self.assertNotIn("code_timeout", flags)
        self.assertNotIn("format_coef", flags)

    def test_code_rl_parser_excludes_gsm8k_flags(self):
        flags = self._flags(build_parser(envs=("code_rl",)))
        self.assertIn("code_timeout", flags)
        self.assertNotIn("answer_format", flags)

    def test_default_parser_includes_all_registered(self):
        flags = self._flags(build_parser())
        self.assertIn("answer_format", flags)
        self.assertIn("code_timeout", flags)

    def test_env_type_choices_follow_selection(self):
        parser = build_parser(envs=("gsm8k",))
        with self.assertRaises(SystemExit):
            parser.parse_args(["--env_type", "code_rl"])

    def test_programmatic_parse_with_config_minimal_argv(self):
        args = parse_with_config(
            build_parser(envs=("gsm8k",)),
            argv=["--base_model", "Qwen/Qwen3-8B"],
        )
        self.assertEqual(args.env_type, "gsm8k")
        self.assertTrue(args.run_name.startswith("rl-train-"))   # post-parse defaulting ran
        self.assertEqual(args.save_every, args.max_steps)
        self.assertGreater(args.max_inflight_rollouts, 0)


class ValidationRunnerSeamTest(unittest.TestCase):
    """run_training uses an injected validation runner instead of building one."""

    def test_injected_runner_is_called(self):
        import asyncio
        import pathlib
        import tempfile

        from tests.test_backend_loop import MockBackend, _args
        from training_sessions_cookbook.rl import train as rl

        calls = []

        class FakeRunner:
            async def run_validation(self, step, sampling_client, tokenizer,
                                     template_fn, args, backend):
                calls.append(step)
                return {"val/accuracy": 1.0}

        tmp = tempfile.mkdtemp()
        args = _args(pathlib.Path(tmp), max_steps=1, val_every=1)
        backend = MockBackend()
        # two prompts: batch_size=2 with drop_last needs a full batch
        dataset = [
            {"question": "1+1?", "answer": "2"},
            {"question": "2+2?", "answer": "4"},
        ]
        asyncio.run(rl.run_training(
            args, backend, dataset,
            tokenizer=backend.tokenizer,
            env_factory=rl.get_env_factory("gsm8k", args),
            template_fn=rl.apply_qwen3_general,
            validation_runner=FakeRunner(),
        ))
        self.assertEqual(calls, [0])


class SftBatchesSeamTest(unittest.TestCase):
    """Injected batches bypass dataset loading; callable form gets the tokenizer."""

    def test_callable_batches_receive_tokenizer_and_loader_not_called(self):
        import asyncio

        import training_sessions_cookbook.sft.train as sft

        seen = {}

        def make_batches(tokenizer):
            seen["tokenizer"] = tokenizer
            # One batch; max_steps=0 keeps the loop from consuming it (empty
            # batches would correctly trip the no-training-data guard).
            return [[{"modelInput": {"chunks": [{"tokens": [1, 2]}]}}]]

        args = mock.Mock(
            num_epochs=1, max_steps=0, eval_every=0, save_every=0,
            rolling_save_every=0, checkpoint_ttl_seconds=None,
            learning_rate=1e-4, lr_schedule="linear",
            adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
            batch_size=4,
        )
        runlog = mock.Mock(run_folder="/tmp", metrics_path="/tmp/m.jsonl")
        svc, tc, tok = mock.Mock(), mock.Mock(), mock.Mock(name="tokenizer")

        with mock.patch.object(sft, "build_sft_batches") as loader, \
             mock.patch.object(sft.checkpoint_utils, "CheckpointManager") as mgr:
            mgr.return_value.save_final_async = mock.AsyncMock(return_value={})
            asyncio.run(sft._train(args, "s", runlog, svc, tc, tok,
                                   train_batches=make_batches, eval_batches=[]))
        loader.assert_not_called()
        self.assertIs(seen["tokenizer"], tok)


class AsyncEntryPointTest(unittest.TestCase):
    """Both loops expose awaitable entry points (usable inside Jupyter's loop)."""

    def test_run_async_are_coroutine_functions(self):
        import asyncio as aio

        import training_sessions_cookbook.sft.train as sft
        from training_sessions_cookbook.rl import train as rl

        self.assertTrue(aio.iscoroutinefunction(rl.run_async))
        self.assertTrue(aio.iscoroutinefunction(sft.run_async))
        # the sync wrappers stay plain callables for scripts/CLI
        self.assertFalse(aio.iscoroutinefunction(rl.run))
        self.assertFalse(aio.iscoroutinefunction(sft.run))


if __name__ == "__main__":
    unittest.main()


class SyncWrapperMisuseTest(unittest.TestCase):
    """Calling sync run() inside a running loop fails with guidance, not a
    cryptic asyncio error."""

    def test_run_inside_loop_points_to_run_async(self):
        import asyncio as aio

        import training_sessions_cookbook.sft.train as sft
        from training_sessions_cookbook.rl import train as rl

        async def call(fn):
            with self.assertRaises(RuntimeError) as ctx:
                fn(mock.Mock())
            self.assertIn("run_async", str(ctx.exception))

        aio.run(call(rl.run))
        aio.run(call(sft.run))


def test_single_env_parser_derives_default_env():
    """With a single-entry ``envs``, ``default_env`` is implied."""
    from training_sessions_cookbook.rl.train import build_parser

    args = build_parser(envs=("code_rl",)).parse_args(
        ["--base_model", "m"])
    assert args.env_type == "code_rl"
