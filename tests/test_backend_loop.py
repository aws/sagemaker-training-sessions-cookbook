"""End-to-end loop test over a mock backend (no SDK required).

The other unit tests cover the pure helpers (metrics, advantages, dataset, validation math) but never drive `run_training` itself, so they cannot catch a regression in how the loop calls the backend. This test closes that gap: it runs the real `run_training` against a fully in-memory `MockBackend` that implements the `TrainingBackend` protocol, and asserts

  * the backend hooks fire in the expected order (refresh -> build_datum -> train_step, refresh cadence honoring --refresh_every);
  * `metrics.jsonl` is written with the schema the overlay plotter and metrics reader depend on (rollout/*, train/*, time/*, config/*);
  * a real GSM8K reward signal flows through (the mock returns the correct "#### <answer>" completion, so rollout/reward_mean == 1.0).

Because the loop is backend-agnostic, exercising it with a mock is a meaningful proxy for the service backend: they differ only in the bodies of the hooks tested here, not in how the loop calls them.
"""

import asyncio
import pytest
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# The driver imports `from backends import make_backend` at module load. That package import is SDK-free (sagemaker is imported lazily inside the backend's create()), so we import the driver directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_sessions_cookbook.rl import train as rl


# --- Mock backend ---------------------------------------------------------


class _Seq:
    def __init__(self, tokens, logprobs):
        self.tokens = tokens
        self.logprobs = logprobs


class _SampleResult:
    def __init__(self, samples):
        self.samples = samples


class _FbResult:
    def __init__(self, metrics):
        self.metrics = metrics


class _MockTokenizer:
    """Round-trippable toy tokenizer: token id == ord(char)."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class MockBackend:
    """In-memory `TrainingBackend` used to drive `run_training` in tests.

    Records the sequence of hook calls so tests can assert ordering/cadence, and returns a canned "#### <answer>" completion so GSM8K scores 1.0.
    """

    def __init__(self, answer_text="#### 42"):
        self.tokenizer = _MockTokenizer()
        self._answer_text = answer_text
        self.calls = []  # ordered list of (hook_name, arg) tuples
        self.refresh_steps = []
        self.datum_count = 0
        self.train_steps = 0
        self.deployed = False
        self.closed = False

    # prompt / params shaping
    def wrap_prompt(self, tokens):
        return list(tokens)  # service-style pass-through

    def make_sampling_params(self, *, max_tokens, temperature, seed=None):
        return SimpleNamespace(max_tokens=max_tokens, temperature=temperature, seed=seed)

    # sampler lifecycle
    def refresh_sampler(self, step):
        self.calls.append(("refresh", step))
        self.refresh_steps.append(step)
        backend = self

        class _Client:
            def sample(self_inner, prompt, num_samples=1, sampling_params=None):
                toks = backend.tokenizer.encode(backend._answer_text)
                seq = _Seq(tokens=toks, logprobs=[-0.1] * len(toks))
                return _SampleResult([seq])

        return _Client()

    async def refresh_sampler_async(self, step):
        # Off-by-k loop path: just offload the sync refresh (nothing blocking to
        # split in the mock). Recorded like the sync hook so ordering asserts hold.
        return self.refresh_sampler(step)

    async def sample_async(self, sampling_client, prompt, *, num_samples=1, sampling_params=None):
        # The loop samples via backend.sample_async; the mock client exposes the
        # blocking `sample()`, so mirror ServiceBackend's fallback path.
        return await asyncio.to_thread(
            sampling_client.sample,
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )

    # training
    def build_datum(self, transition, *, temperature):
        self.calls.append(("build_datum", None))
        self.datum_count += 1
        # Return an opaque object; the loop only forwards it to train_step.
        return {"datum": True, "temperature": temperature}

    async def train_step(self, datums, *, loss_name, learning_rate, beta1, beta2, eps):
        self.calls.append(("train_step", len(datums)))
        self.train_steps += 1
        return _FbResult({"loss": 0.5, "grad_norm": 0.01})

    # checkpoint / deploy
    def deploy_and_invoke(self):
        self.deployed = True
        return {"endpoint": "mock", "generations": ["#### 42"]}

    def close(self):
        self.closed = True


# --- Fixtures -------------------------------------------------------------


def _args(tmp_path, **overrides):
    """Minimal argparse-like namespace matching what run_training reads."""
    base = dict(
        base_model="Qwen/Qwen3-8B",
        run_name="mock-run",
        max_steps=3,
        batch_size=2,
        responses_per_prompt=2,
        advantage_mode="grouped",
        num_rollout_workers=4,
        max_turns=1,
        max_tokens=32,
        temperature=1.0,
        learning_rate=4e-5,
        loss="ppo",
        gamma=0.9,
        use_rebn=True,
        refresh_every=1,
        epochs=1,
        drop_last=True,
        env_type="gsm8k",
        answer_format="hash",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        warmup_steps=0,
        kl_coef=0.0,
        log_every=1,
        val_every=50,
        val_examples=2,
        val_dataset_path=None,
        val_max_prompts=None,
        val_at_start=False,
        dataset_path=None,
        log_dir=str(tmp_path / "logs"),
    )
    base.update(overrides)
    ns = SimpleNamespace(**base)
    ns._run_folder = str(tmp_path)
    ns._metrics_path = os.path.join(str(tmp_path), "metrics.jsonl")
    return ns


def _gsm8k_dataset():
    # Answer must match the mock completion ("#### 42") so reward == 1.0.
    return [{"question": "What is 6*7?", "answer": "42"} for _ in range(4)]


def _run(args, backend):
    env_factory = rl.get_env_factory(args.env_type, args)
    asyncio.run(
        rl.run_training(
            args,
            backend,
            _gsm8k_dataset(),
            tokenizer=backend.tokenizer,
            env_factory=env_factory,
            template_fn=rl.apply_qwen3_general,
        )
    )


# --- Tests ----------------------------------------------------------------


def test_loop_writes_expected_metrics_schema(tmp_path):
    args = _args(tmp_path)
    backend = MockBackend()
    _run(args, backend)

    metrics_path = args._metrics_path
    assert os.path.exists(metrics_path)
    with open(metrics_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # max_steps=3, batch_size=2 over 4 prompts, drop_last -> 2 steps/epoch, capped at max_steps=3 but dataset exhausts at 2 -> expect 2 step records.
    assert len(records) >= 1
    r = records[0]
    # Schema the overlay plotter + beta metrics reader rely on.
    for key in (
        "step",
        "rollout/reward_mean",
        "rollout/episode_length_mean",
        "rollout/num_episodes",
        "train/loss",
        "time/sample_seconds",
        "time/train_seconds",
        "config/learning_rate",
    ):
        assert key in r, f"missing metric key {key!r} in {sorted(r)}"


def test_loop_reward_is_one_when_answer_matches(tmp_path):
    args = _args(tmp_path)
    backend = MockBackend(answer_text="#### 42")
    _run(args, backend)

    with open(args._metrics_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert records, "no metrics written"
    # The mock always emits the correct answer, so every episode scores 1.0.
    assert all(rec["rollout/reward_mean"] == 1.0 for rec in records)


def test_backend_hooks_fire_in_order(tmp_path):
    args = _args(tmp_path, refresh_every=1)
    backend = MockBackend()
    _run(args, backend)

    hook_names = [c[0] for c in backend.calls]
    # Each training step: at least one refresh, then build_datum(s), then a train_step. Assert the first train_step is preceded by a refresh and a build_datum.
    assert "refresh" in hook_names
    assert "build_datum" in hook_names
    assert "train_step" in hook_names
    first_train = hook_names.index("train_step")
    assert "refresh" in hook_names[:first_train]
    assert "build_datum" in hook_names[:first_train]
    # The final checkpoint is saved by run_async through the checkpoint
    # manager, not by the loop; the loop must not invoke any save hook.
    assert "final_save" not in hook_names


def test_refresh_cadence_honors_refresh_every(tmp_path):
    # With refresh_every=2, steps 0 and 2 refresh; step 1 does not.
    args = _args(tmp_path, refresh_every=2, max_steps=3, batch_size=1)
    backend = MockBackend()
    _run(args, backend)
    # Steps run until dataset (4 prompts, batch_size=1, drop_last) exhausts or max_steps hit. Refresh should only happen on even steps.
    assert all(s % 2 == 0 for s in backend.refresh_steps), backend.refresh_steps


def test_all_rollouts_failing_raises_instead_of_silent_success(tmp_path):
    """A sampling outage (every rollout fails, zero transitions every step)
    must fail the run, not end 'successfully' with an untrained final
    checkpoint. Regression test for the flag-rl-a live incident."""

    class DeadSamplerBackend(MockBackend):
        def refresh_sampler(self, step):
            self.calls.append(("refresh", step))

            class _Client:
                def sample(self_inner, prompt, num_samples=1, sampling_params=None):
                    raise RuntimeError("engine sampling outage")

            return _Client()

    args = _args(tmp_path)
    with pytest.raises(RuntimeError, match="no training occurred"):
        _run(args, DeadSamplerBackend())


def test_partial_step_skip_still_succeeds(tmp_path):
    """One bad step among good ones is skipped with a warning, not fatal."""

    class FlakyFirstStepBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self._first = True

        def refresh_sampler(self, step):
            self.calls.append(("refresh", step))
            outer = self

            class _Client:
                def sample(self_inner, prompt, num_samples=1, sampling_params=None):
                    if outer._first:
                        outer._first = False
                        raise RuntimeError("transient sampling failure")
                    toks = outer.tokenizer.encode(outer._answer_text)
                    seq = _Seq(tokens=toks, logprobs=[-0.1] * len(toks))
                    return _SampleResult([seq])

            return _Client()

    args = _args(tmp_path, max_steps=2, refresh_every=1)
    backend = FlakyFirstStepBackend()
    _run(args, backend)  # must not raise
    assert any(c[0] == "train_step" for c in backend.calls)
