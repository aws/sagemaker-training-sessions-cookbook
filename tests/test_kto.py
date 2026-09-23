"""KTO correctness: the pure loss/gradient math, datum shaping, and one step.

The service is never contacted. The pure arithmetic
(:mod:`training_sessions_cookbook.preference.kto`) is asserted directly; the
datum builder and one full ``_kto_step`` run against a toy tokenizer and a mock
training client, so the surrogate-gradient wiring (advantage signs, ratio-1 old
log-probs, z_ref) is pinned without the SDK.
"""

import asyncio
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_sessions_cookbook.preference import data as kdata
from training_sessions_cookbook.preference import kto
from training_sessions_cookbook.preference import train as ktrain
from training_sessions_cookbook.service.session_watchdog import NullWatchdog


class _CharTokenizer:
    """Toy tokenizer: one token per character. add_special_tokens ignored."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


# === Pure KTO math ===================================================


def test_sigmoid_matches_reference():
    for x in (-8.0, -1.0, 0.0, 0.5, 3.0, 20.0):
        assert abs(kto.sigmoid(x) - 1.0 / (1.0 + math.exp(-x))) < 1e-9


def test_sigmoid_is_stable_for_large_magnitudes():
    # No overflow either direction.
    assert kto.sigmoid(1000.0) == 1.0
    assert kto.sigmoid(-1000.0) == 0.0


def test_desirable_gradient_pushes_logprob_up():
    # s_i < 0 for desirable, so advantage = -s_i > 0 (increase log-prob).
    term = kto.kto_term(logp_policy=-2.0, logp_ref=-3.0, desirable=True, z_ref=0.0,
                        beta=0.1, desirable_weight=1.0, undesirable_weight=1.0)
    assert term.grad_wrt_logp < 0.0
    assert kto.advantage_from_grad(term.grad_wrt_logp) > 0.0
    assert 0.0 <= term.value <= 1.0
    assert term.reward == -2.0 - (-3.0)


def test_undesirable_gradient_pushes_logprob_down():
    term = kto.kto_term(logp_policy=-2.0, logp_ref=-3.0, desirable=False, z_ref=0.0,
                        beta=0.1, desirable_weight=1.0, undesirable_weight=1.0)
    assert term.grad_wrt_logp > 0.0
    assert kto.advantage_from_grad(term.grad_wrt_logp) < 0.0


def test_gradient_matches_analytic_formula():
    beta, lam = 0.2, 1.5
    logp_policy, logp_ref, z_ref = -1.0, -1.4, 0.1
    term = kto.kto_term(logp_policy, logp_ref, desirable=True, z_ref=z_ref,
                        beta=beta, desirable_weight=lam, undesirable_weight=1.0)
    r = logp_policy - logp_ref
    v = kto.sigmoid(beta * (r - z_ref))
    assert abs(term.value - v) < 1e-12
    assert abs(term.loss - lam * (1.0 - v)) < 1e-12
    assert abs(term.grad_wrt_logp - (-lam * beta * v * (1.0 - v))) < 1e-12


def test_class_weights_scale_gradient():
    kw = dict(logp_policy=-2.0, logp_ref=-2.5, z_ref=0.0, beta=0.1)
    base = kto.kto_term(desirable=True, desirable_weight=1.0, undesirable_weight=1.0, **kw)
    heavy = kto.kto_term(desirable=True, desirable_weight=3.0, undesirable_weight=1.0, **kw)
    assert abs(heavy.grad_wrt_logp - 3.0 * base.grad_wrt_logp) < 1e-12


def test_kl_reference_point_is_clamped_mean():
    assert kto.kl_reference_point([1.0, 2.0], [0.0, 0.0]) == 1.5  # mean of (1,2)
    assert kto.kl_reference_point([-5.0, -1.0], [0.0, 0.0]) == 0.0  # clamped at 0
    assert kto.kl_reference_point([], []) == 0.0


def test_kl_reference_point_length_mismatch_raises():
    try:
        kto.kl_reference_point([1.0], [1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


def test_sum_completion_logprob_masks_prompt():
    lp = [-1.0, -2.0, -3.0, -4.0]
    mask = [0.0, 0.0, 1.0, 1.0]
    assert kto.sum_completion_logprob(lp, mask) == -7.0


# === Datum shaping ===================================================


def _example(prompt="hi", completion="ok", label=True, kl="zz"):
    tok = _CharTokenizer()
    return kdata.KtoExample(
        prompt_tokens=tok.encode("<|im_start|>user\n" + prompt),
        completion_tokens=tok.encode(completion + "<|im_end|>"),
        label=label,
        kl_completion_tokens=tok.encode(kl + "<|im_end|>"),
    )


def test_matched_slice_masks_prompt_and_marks_completion():
    ex = _example()
    inp, tgt, mask = kdata.matched_slice(ex)
    p, c = len(ex.prompt_tokens), len(ex.completion_tokens)
    assert len(inp) == len(tgt) == len(mask) == p + c - 1
    assert mask[: p - 1] == [0.0] * (p - 1)      # prompt masked
    assert mask[p - 1 :] == [1.0] * c            # completion active
    assert sum(mask) == c


def test_train_datum_broadcasts_advantage_over_completion_only():
    ex = _example()
    inp, tgt, mask = kdata.matched_slice(ex)
    old = [-0.5] * len(tgt)
    d = kdata.train_datum(inp, tgt, mask, advantage=0.7, old_logprobs=old)
    adv = d["lossFnInputs"]["advantages"]["data"]
    lp = d["lossFnInputs"]["logprobs"]["data"]
    assert len(adv) == len(tgt)
    assert all(a == 0.0 for a, m in zip(adv, mask) if m == 0.0)      # prompt: 0
    assert all(abs(a - 0.7) < 1e-12 for a, m in zip(adv, mask) if m == 1.0)  # completion: advantage
    assert lp == old


def test_build_kto_batches_from_jsonl(tmp_path):
    rows = [
        {"prompt": "what is 2+2", "completion": "4", "label": True},
        {"prompt": "what is 2+2", "completion": "5", "label": False},
        {"prompt": "capital of france", "completion": "paris", "label": True},
        {"prompt": "capital of france", "completion": "berlin", "label": False},
    ]
    path = tmp_path / "prefs.jsonl"
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows))

    batches = kdata.build_kto_batches(
        dataset_path=str(path), tokenizer=_CharTokenizer(),
        batch_size=2, max_length=4096, split="all",
    )
    examples = [e for b in batches for e in b]
    assert len(examples) == 4
    assert sum(e.label for e in examples) == 2
    # KL partner is a genuine mismatch: its completion differs from the matched one.
    for e in examples:
        assert e.kl_completion_tokens != e.completion_tokens


# === One full step against a mock client =============================


class _FakeAdamParams:
    def __init__(self, **kw):
        self.kw = kw


class _FakeSdk:
    fwd_loss = "CROSS_ENTROPY"
    is_loss = "IMPORTANCE_SAMPLING"
    AdamParams = _FakeAdamParams


class _Result:
    def __init__(self, loss_fn_outputs=None, metrics=None):
        self.loss_fn_outputs = loss_fn_outputs
        self.metrics = metrics or {}


class _Op:
    def __init__(self, result):
        self._result = result

    async def result_async(self):
        return self._result


class _MockClient:
    """Records calls; returns a constant per-token log-prob on every forward."""

    def __init__(self, token_logprob=-0.5):
        self.token_logprob = token_logprob
        self.forward_calls = []
        self.fb_datums = None
        self.optim_calls = 0

    async def forward_async(self, datums, loss_fn=None):
        self.forward_calls.append(len(datums))
        outputs = [
            {"logprobs": {"data": [self.token_logprob] * len(d["lossFnInputs"]["target_tokens"]["data"]),
                          "dtype": "float32",
                          "shape": [len(d["lossFnInputs"]["target_tokens"]["data"])]}}
            for d in datums
        ]
        return _Op(_Result(loss_fn_outputs=outputs))

    async def forward_backward_async(self, datums, loss_fn=None):
        self.fb_datums = datums
        return _Op(_Result(metrics={"actor/loss": 0.1}))

    async def optim_step_async(self, adam_params):
        self.optim_calls += 1
        return _Op(_Result(metrics={"grad_norm": 0.01}))


class _Args:
    beta = 0.1
    desirable_weight = 1.0
    undesirable_weight = 1.0
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    adam_eps = 1e-8


def _dm(n):
    """A datum stub with an n-token model input."""
    return {"modelInput": {"chunks": [{"tokens": list(range(n))}]}}


def test_extract_logprobs_reads_per_token_and_validates_shape():
    # Per-datum shape (future engines): one entry per datum, used directly.
    result = _Result(loss_fn_outputs=[{"logprobs": {"data": [-1.0, -2.0]}}])
    assert ktrain._extract_logprobs(result, [_dm(2)]) == [[-1.0, -2.0]]

    # Flat shape (current engines): one entry, len(input)+1 values per datum,
    # split by input length and trimmed front-aligned.
    flat = _Result(loss_fn_outputs=[
        {"logprobs": {"data": [-1.0, -2.0, -99.0, -3.0, -4.0, -5.0, -98.0]}}])
    assert ktrain._extract_logprobs(flat, [_dm(2), _dm(3)]) == [
        [-1.0, -2.0], [-3.0, -4.0, -5.0]]

    # Flat shape without the appended extra position also partitions.
    flat0 = _Result(loss_fn_outputs=[{"logprobs": {"data": [-1.0, -2.0, -3.0]}}])
    assert ktrain._extract_logprobs(flat0, [_dm(2), _dm(1)]) == [[-1.0, -2.0], [-3.0]]

    # A flat tensor that fits neither partition is rejected.
    bad = _Result(loss_fn_outputs=[{"logprobs": {"data": [-1.0, -2.0]}}])
    try:
        ktrain._extract_logprobs(bad, [_dm(2), _dm(3)])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for non-partitionable flat tensor")

    missing = _Result(loss_fn_outputs=[{"actor/loss": 0.5}])
    try:
        ktrain._extract_logprobs(missing, [_dm(1)])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when logprobs are absent")


def test_kto_step_sets_advantage_sign_by_label_and_runs_fb_optim():
    # Two examples: one desirable, one undesirable, both with policy reward
    # above the (zero) reference -> desirable advantage > 0, undesirable < 0.
    batch = [_example(label=True), _example(label=False)]
    # Reference sums well below the policy sum (policy token lp = -0.5 each) so
    # reward = policy - ref > 0 for both.
    ref_matched = [-100.0, -100.0]
    client = _MockClient(token_logprob=-0.5)

    metrics = asyncio.run(ktrain._kto_step(
        _Args(), client, batch, ref_matched, None,
        lr=1e-6, use_kl=False, sdk=_FakeSdk(), watchdog=NullWatchdog(),
    ))

    # forward_backward + optim_step both ran exactly once.
    assert client.optim_calls == 1
    assert client.fb_datums is not None and len(client.fb_datums) == 2

    # Advantage sign follows the label: desirable > 0, undesirable < 0.
    def _active_adv(datum):
        adv = datum["lossFnInputs"]["advantages"]["data"]
        return next(a for a in adv if a != 0.0)

    assert _active_adv(client.fb_datums[0]) > 0.0   # desirable
    assert _active_adv(client.fb_datums[1]) < 0.0   # undesirable

    # old_log_probs equal the forward log-probs (ratio 1 at evaluation).
    assert all(x == -0.5 for x in client.fb_datums[0]["lossFnInputs"]["logprobs"]["data"])

    # Metrics surface the KTO diagnostics, including per-class log-probs.
    assert metrics["kto/frac_desirable"] == 0.5
    assert metrics["kto/z_ref"] == 0.0
    assert "kto/loss" in metrics and "kto/reward_margin" in metrics
    assert "kto/logps_desirable" in metrics and "kto/logps_undesirable" in metrics
    # KL disabled -> only the matched forward ran (one call), plus fb.
    assert client.forward_calls == [2]


def test_kto_step_with_kl_runs_extra_forward_and_positive_z_ref():
    batch = [_example(label=True), _example(label=False)]
    ref_matched = [-1.0, -1.0]
    ref_kl = [-100.0, -100.0]  # policy KL sum (>-100) - ref -> large positive mean
    client = _MockClient(token_logprob=-0.5)

    metrics = asyncio.run(ktrain._kto_step(
        _Args(), client, batch, ref_matched, ref_kl,
        lr=1e-6, use_kl=True, sdk=_FakeSdk(), watchdog=NullWatchdog(),
    ))
    # Matched forward + KL forward = two forward calls.
    assert client.forward_calls == [2, 2]
    assert metrics["kto/z_ref"] > 0.0


class _MockBaseRefClient:
    """ARN-less base client stub: returns a constant per-token log-prob.

    ``compute_logprobs_async`` scores a full token sequence and returns one
    value per token with the first element None (matching the real SDK).
    """

    def __init__(self, token_logprob=-1.0):
        self.token_logprob = token_logprob
        self.calls = 0

    async def compute_logprobs_async(self, prompt=None):
        self.calls += 1
        toks = list(prompt)
        return [None] + [self.token_logprob] * (len(toks) - 1)


def test_score_reference_sums_masks_prompt_and_handles_leading_none():
    ref = _MockBaseRefClient(token_logprob=-1.0)
    # prompt len 3, completion len 2 -> sum of the 2 completion-position logprobs.
    sums = asyncio.run(ktrain._score_reference_sums(
        ref, [([10, 11, 12], [20, 21])], watchdog=NullWatchdog(), what="test"))
    assert sums == [-2.0]  # two completion tokens at -1.0 each


def test_kto_step_compute_logprobs_source_scores_base_and_rerandomizes_kl():
    batch = [_example(label=True), _example(label=False)]
    ref_matched = [-1.0, -1.0]
    client = _MockClient(token_logprob=-0.5)      # policy: training forward
    ref_client = _MockBaseRefClient(token_logprob=-3.0)  # reference: base client

    metrics = asyncio.run(ktrain._kto_step(
        _Args(), client, batch, ref_matched, None,
        lr=1e-6, use_kl=True, sdk=_FakeSdk(), watchdog=NullWatchdog(),
        reference_source="compute_logprobs", ref_client=ref_client, step_seed=1,
    ))
    # Policy matched + policy KL = two training forwards; the reference KL was
    # scored on the base client (one call per example).
    assert client.forward_calls == [2, 2]
    assert ref_client.calls == 2
    # Policy KL sum (-0.5/token) is well above the reference (-3.0/token), so
    # z_ref = clamp(mean(policy - ref), 0) is strongly positive.
    assert metrics["kto/z_ref"] > 0.0
    # forward_backward + optim still run once with correct advantage signs.
    assert client.optim_calls == 1

    def _active_adv(datum):
        return next(a for a in datum["lossFnInputs"]["advantages"]["data"] if a != 0.0)

    assert _active_adv(client.fb_datums[0]) > 0.0   # desirable
    assert _active_adv(client.fb_datums[1]) < 0.0   # undesirable


def test_evaluate_kto_returns_held_out_metrics():
    # One eval batch: a desirable and an undesirable example. Reference sums are
    # set so the desirable reward is large-positive and the undesirable small,
    # giving a positive margin and perfect accuracy at z_ref = 0.
    eval_batches = [[_example(label=True), _example(label=False)]]
    eval_ref_matched = [[-100.0, -1.0]]
    client = _MockClient(token_logprob=-0.5)

    metrics = asyncio.run(ktrain._evaluate_kto(
        client, eval_batches, eval_ref_matched,
        z_ref=0.0, sdk=_FakeSdk(), watchdog=NullWatchdog()))

    assert metrics["kto/eval_num_examples"] == 2
    assert metrics["kto/eval_reward_margin"] > 0.0
    assert metrics["kto/eval_accuracy"] == 1.0
    assert "kto/eval_logps_desirable" in metrics
    # Forward-only: no optimizer step was taken during eval.
    assert client.optim_calls == 0
