"""On-policy distillation: the advantage math, teacher contexts, and loop wiring.

The service is never contacted. The pure math (:mod:`distill.losses`) and the
teacher-context splicing (:mod:`distill.context`) are asserted directly; the
teacher-scoring stage runs against a fake ``compute_logprobs`` client, and the
full ``run_training`` loop runs over the mock backend from
``test_backend_loop`` with the distillation ``advantage_fn`` plugged in.
"""

import asyncio
import json
import math
import os
from types import SimpleNamespace

import pytest

from tests.test_backend_loop import MockBackend, _args, _gsm8k_dataset
from tests.test_fb_stream_loop import StreamingMockBackend
from training_sessions_cookbook.distill import context as dctx
from training_sessions_cookbook.distill import losses
from training_sessions_cookbook.distill import teacher as dteacher
from training_sessions_cookbook.distill import train as dtrain
from training_sessions_cookbook.distill.advantages import DistillAdvantageFn
from training_sessions_cookbook.rl import train as rl
from training_sessions_cookbook.rl.rollouts import Transition
from training_sessions_cookbook.rl.runstate import run_artifacts
from training_sessions_cookbook.service.session_watchdog import SessionDeadError


class _CharTokenizer:
    """Toy tokenizer: one token per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(t) for t in tokens)

    def __len__(self):
        return 0x110000


def _template(observation, tools=None):
    return f"<user>{observation}</user><assistant>"


def _transition(obs, act, *, logprob=-0.5, reward=0.0, info=None):
    tok = _CharTokenizer()
    return Transition(
        obs_tokens=tok.encode(obs), act_tokens=tok.encode(act),
        act_logprobs=[logprob] * len(act), reward=reward, done=True,
        env_type="gsm8k", info=info,
    )


class _FakeClient:
    """compute_logprobs_async returning ``value`` per token (None first), recording calls."""

    def __init__(self, value=-1.0, fn=None):
        self.value = value
        self.fn = fn
        self.prompts = []

    async def compute_logprobs_async(self, prompt):
        self.prompts.append(list(prompt))
        if self.fn is not None:
            return self.fn(list(prompt))
        return [None] + [self.value] * (len(prompt) - 1)


# === Pure math ========================================================


def test_sampled_token_advantage_is_unbiased_reverse_kl():
    """The design doc's toy check: E_q[-A] over the student's samples == KL(q||p)."""
    q = [0.7, 0.2, 0.1]
    p = [0.5, 0.4, 0.1]
    true_kl = sum(qi * math.log(qi / pi) for qi, pi in zip(q, p))
    advs = losses.reverse_kl_advantages([math.log(x) for x in q], [math.log(x) for x in p])
    estimate = sum(qi * -a for qi, a in zip(q, advs))
    assert abs(true_kl - 0.0969) < 1e-3
    assert abs(estimate - true_kl) < 1e-12
    # The per-sample estimates from the doc: +0.336, -0.693, 0.
    assert [round(-a, 3) for a in advs] == [0.336, -0.693, 0.0]


def test_advantage_sign_pushes_toward_teacher():
    advs = losses.reverse_kl_advantages([-2.0, -0.5], [-0.5, -2.0])
    assert advs[0] > 0  # teacher likes it more -> push up
    assert advs[1] < 0  # teacher likes it less -> push down


def test_unscored_positions_get_zero_and_clip_bounds():
    advs = losses.reverse_kl_advantages([-1.0, -1.0, -1.0, -1.0],
                                        [None, float("-inf"), -30.0, 5.0], clip=3.0)
    assert advs == [0.0, 0.0, -3.0, 3.0]
    with pytest.raises(ValueError):
        losses.reverse_kl_advantages([-1.0], [-1.0, -2.0])


def test_mix_task_advantage():
    assert losses.mix_task_advantage([0.1, -0.2], 2.0, 0.0) == [0.1, -0.2]
    assert losses.mix_task_advantage([0.1, -0.2], 2.0, 0.5) == pytest.approx([1.1, 0.8])


def test_distill_stats_summary():
    stats = losses.DistillStats()
    stats.add_sequence([-1.0, -1.0, -1.0], [-2.0, None, -1.0], clip=0.5)
    stats.add_unscored_sequence(4)
    s = stats.summary()
    assert s["distill/reverse_kl"] == pytest.approx(0.5)
    assert s["distill/num_scored_tokens"] == 2
    assert s["distill/num_unscored_tokens"] == 5
    assert s["distill/clip_frac"] == pytest.approx(0.5)
    assert s["distill/frac_sequences_scored"] == pytest.approx(0.5)
    stats.reset()
    assert stats.summary()["distill/num_scored_tokens"] == 0


# === Teacher contexts =================================================


def test_splice_keeps_student_token_ids_and_later_turns():
    """Multi-turn: the teacher sees its own first prompt + the student's later turns verbatim."""
    tok = _CharTokenizer()
    first = _transition("PROMPT", "a1")
    second = Transition(
        obs_tokens=first.obs_tokens + first.act_tokens + tok.encode("obs2"),
        act_tokens=tok.encode("a2"), act_logprobs=[-0.1, -0.1], reward=1.0,
        done=True, env_type="gsm8k")
    q = dctx.splice(tok.encode("TEACHER"), len(first.obs_tokens), second)
    assert tok.decode(q.tokens) == "TEACHERa1obs2a2"
    assert q.n_tail == 2
    assert q.tokens[-q.n_tail:] == second.act_tokens


def test_same_context_scores_student_sequence_unchanged():
    tok = _CharTokenizer()
    t = _transition("<user>q</user><assistant>", "ans", info={"observation": "q"})
    ctx = dctx.SameContext(tokenizer=tok, template_fn=_template)
    [q] = ctx.queries([t], [[t]])
    assert q.tokens == t.obs_tokens + t.act_tokens and q.n_tail == 3


def test_privileged_context_inserts_solution_and_reuses_continuation():
    tok = _CharTokenizer()
    row = {"question": "6*7?", "answer": "6 times 7 is 42.\n#### 42"}
    obs = "6*7? think"
    t = _transition(_template(obs), "It is 42", info={"prompt": row, "observation": obs})
    ctx = dctx.PrivilegedContext(tokenizer=tok, template_fn=_template,
                                 template="{observation} || REF: {privileged}")
    [q] = ctx.queries([t], [[t]])
    text = tok.decode(q.tokens)
    assert text == _template("6*7? think || REF: 6 times 7 is 42.\n#### 42") + "It is 42"
    assert q.tokens[-q.n_tail:] == t.act_tokens

    final = dctx.PrivilegedContext(tokenizer=tok, template_fn=_template, mode="final_answer",
                                   template="{observation} || {privileged}")
    assert final.teacher_observation([t], [[t]]) == "6*7? think || 42"

    missing = _transition(_template(obs), "x", info={"prompt": {"question": "q"}, "observation": obs})
    assert ctx.queries([missing], [[missing]]) is None


def test_default_opsd_template_mentions_reference_solution():
    tok = _CharTokenizer()
    ctx = dctx.PrivilegedContext(tokenizer=tok, template_fn=_template)
    t = _transition("x", "y", info={"prompt": {"answer": "SOL"}, "observation": "PROB"})
    obs = ctx.teacher_observation([t], [[t]])
    assert obs.startswith("PROB\n\n") and "reference solution" in obs and "SOL" in obs


def _code_episode(act, *, passed, stdout="", observation="PROB"):
    info = {
        "prompt": None, "observation": observation, "tools": None,
        "step_info": {"env_type": "code_rl", "passed": passed,
                      "metrics": {"details": {"run_result": {"stdout": stdout, "stderr": ""}}}},
    }
    return [_transition(_template(observation), act, info=info, reward=1.0 if passed else 0.0)]


def test_sandbox_feedback_formats_and_truncates():
    info = {"metrics": {"details": {"run_result": {"stdout": "Wrong Answer " * 50, "stderr": "Trace"}}}}
    fb = dctx.sandbox_feedback(info, max_chars=40)
    assert fb.startswith("Wrong Answer") and fb.endswith("[truncated]")
    assert dctx.sandbox_feedback({"metrics": {"details": {"error": "timeout"}}}) == "timeout"
    assert dctx.sandbox_feedback({"parse_error": True}).startswith("No code block")
    assert dctx.sandbox_feedback({"metrics": {"details": {}}}) is None
    assert dctx.sandbox_feedback(None) is None


def test_feedback_context_uses_sibling_success_and_env_feedback():
    tok = _CharTokenizer()
    good = _code_episode("```python\nprint(42)\n```", passed=True)
    bad = _code_episode("```python\nprint(1)\n```", passed=False, stdout="Wrong Answer: expected 42")
    ctx = dctx.FeedbackContext(tokenizer=tok, template_fn=_template,
                               solution_fn=dctx.code_solution)
    group = [good, bad]

    bad_obs = ctx.teacher_observation(bad, group)
    assert "print(42)" in bad_obs and "Wrong Answer: expected 42" in bad_obs
    assert "print(1)" not in bad_obs  # never the student's own attempt

    # The only success in its group: no sibling to show, no feedback on a pass.
    assert ctx.teacher_observation(good, group) is None
    assert ctx.queries(good, group) is None

    # Two successes: each sees the other's solution.
    good2 = _code_episode("```python\nprint(6*7)\n```", passed=True)
    assert "print(6*7)" in ctx.teacher_observation(good, [good, good2])

    no_fb = dctx.FeedbackContext(tokenizer=tok, template_fn=_template, use_env_feedback=False)
    assert "Wrong Answer" not in no_fb.teacher_observation(bad, group)


def test_episode_succeeded_falls_back_to_reward():
    assert dctx.episode_succeeded([_transition("q", "a", reward=1.0, info={})])
    assert not dctx.episode_succeeded([_transition("q", "a", reward=0.0, info={})])
    assert not dctx.episode_succeeded([])


# === Teacher clients ===================================================


def test_teacher_score_returns_tail_and_validates_length():
    client = _FakeClient(fn=lambda p: [None] + [-float(i) for i in range(1, len(p))])
    t = dteacher.Teacher(client, name="t")
    assert asyncio.run(t.score([1, 2, 3, 4, 5], 2)) == [-3.0, -4.0]
    assert asyncio.run(t.score([1, 2], 0)) == []
    with pytest.raises(ValueError):
        asyncio.run(t.score([1, 2], 2))

    bad = dteacher.Teacher(_FakeClient(fn=lambda p: [None]), name="bad")
    with pytest.raises(RuntimeError):
        asyncio.run(bad.score([1, 2, 3], 1))


class _TeacherTc:
    """Teacher-session training client: records sampling-path binds."""

    def __init__(self):
        self.loaded = []

    def load_weights_for_sampling(self, arn):
        self.loaded.append(arn)

    def create_sampling_client(self, model_package_arn=""):
        return SimpleNamespace(model_package_arn=model_package_arn)


def test_lagged_teacher_rebinds_its_own_session_on_cadence():
    tc = _TeacherTc()
    base = tc.create_sampling_client()
    t = dteacher.LaggedSelfTeacher(base, refresh_every=5, name="lagged", svc=None,
                                   session_name="t", training_client=tc)
    student = lambda arn: SimpleNamespace(model_package_arn=arn)
    asyncio.run(t.update(0, student("arn:s0")))
    asyncio.run(t.update(3, student("arn:s3")))
    assert t.client is base and tc.loaded == []
    asyncio.run(t.update(5, student("arn:s5")))
    assert tc.loaded == ["arn:s5"] and t.client.model_package_arn == "arn:s5"
    asyncio.run(t.update(5, student("arn:s5")))  # repeated within a step: no reload
    assert tc.loaded == ["arn:s5"]
    asyncio.run(t.update(10, student("")))       # no checkpoint ARN: keep snapshot
    assert t.client.model_package_arn == "arn:s5"
    asyncio.run(t.update(15, student("arn:s15")))
    assert tc.loaded == ["arn:s5", "arn:s15"] and t.snapshot_step == 15


def test_compute_logprobs_accepts_sdk_result_shapes():
    seq = [None, -1.0, -2.0]

    class Direct:
        async def compute_logprobs_async(self, prompt):
            return list(seq)

    class Wrapped:                                   # ComputeLogprobsResult(.logprobs)
        async def compute_logprobs_async(self, prompt):
            return SimpleNamespace(logprobs=list(seq))

    class Future:                                    # APIFuture with result_async()
        def compute_logprobs_async(self, prompt):
            async def result_async():
                return SimpleNamespace(logprobs=list(seq))
            return SimpleNamespace(result_async=result_async)

    for client in (Direct(), Wrapped(), Future()):
        assert asyncio.run(dteacher.compute_logprobs(client, [1, 2, 3])) == seq


def test_check_tokenizers_match():
    tok = _CharTokenizer()
    assert dteacher.check_tokenizers_match(tok, _CharTokenizer()) is None

    class Shifted(_CharTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [ord(c) + 1 for c in text]

    assert "encodes differently" in dteacher.check_tokenizers_match(tok, Shifted())

    class Small(_CharTokenizer):
        def __len__(self):
            return 10

    assert "vocabulary sizes differ" in dteacher.check_tokenizers_match(tok, Small())


def test_session_teacher_close_deletes_session_once():
    calls = []
    svc = SimpleNamespace(delete_session=lambda name, stop_if_active: calls.append((name, stop_if_active)))
    t = dteacher.SessionTeacher(_FakeClient(), svc=svc, session_name="run-teacher", name="session")
    t.close()
    t.close()
    assert calls == [("run-teacher", True)]


def test_make_teacher_sources(monkeypatch):
    base_client = _FakeClient()
    backend = SimpleNamespace(
        training_client=SimpleNamespace(create_sampling_client=lambda: base_client))
    opened = []

    def fake_open(args, tok, *, base_model_arn, model_package_arn=None):
        opened.append((base_model_arn, model_package_arn))
        tc = _TeacherTc()
        if model_package_arn:
            tc.load_weights_for_sampling(model_package_arn)
        client = tc.create_sampling_client(model_package_arn=model_package_arn or "")
        return SimpleNamespace(delete_session=lambda *a, **k: None), "teacher", tc, client

    monkeypatch.setattr(dteacher, "_open_teacher_session", fake_open)
    args = SimpleNamespace(teacher="base", teacher_max_concurrency=4, teacher_refresh_every=3,
                           teacher_model_package_arn="arn:pkg", base_model_arn="arn:student",
                           teacher_base_model_arn="arn:other")
    # base shares the student's session: no teacher session opened.
    assert asyncio.run(dteacher.make_teacher(args, backend, None)).client is base_client
    assert opened == []
    # checkpoint: own session on the STUDENT's base, adapter bound before sampling.
    args.teacher = "checkpoint"
    ckpt = asyncio.run(dteacher.make_teacher(args, backend, None))
    assert isinstance(ckpt, dteacher.SessionTeacher)
    assert ckpt.client.model_package_arn == "arn:pkg" and opened[-1] == ("arn:student", "arn:pkg")
    # lagged: own session on the student's base, starting ARN-less.
    args.teacher = "lagged"
    lagged = asyncio.run(dteacher.make_teacher(args, backend, None))
    assert isinstance(lagged, dteacher.LaggedSelfTeacher) and lagged.refresh_every == 3
    assert opened[-1] == ("arn:student", None)
    # session: the other base model.
    args.teacher = "session"
    asyncio.run(dteacher.make_teacher(args, backend, None))
    assert opened[-1] == ("arn:other", None)
    for t in (ckpt, lagged):
        t.close()
    args.teacher = "nope"
    with pytest.raises(ValueError):
        asyncio.run(dteacher.make_teacher(args, backend, None))


# === Teacher-scoring stage =============================================


def _gsm8k_group(n=2, logprob=-0.5):
    obs = "Q?"
    row = {"question": "Q?", "answer": "worked.\n#### 42"}
    return [[_transition(_template(obs), "#### 42", logprob=logprob, reward=float(i % 2),
                         info={"prompt": row, "observation": obs, "step_info": {}})]
            for i in range(n)]


def test_advantage_fn_writes_per_token_advantages_and_metrics():
    tok = _CharTokenizer()
    client = _FakeClient(value=-0.2)
    fn = DistillAdvantageFn(
        teacher=dteacher.Teacher(client, name="t"),
        context=dctx.PrivilegedContext(tokenizer=tok, template_fn=_template),
        logprob_check_samples=0,
    )
    groups = [_gsm8k_group()]
    flat = asyncio.run(fn(groups, step=0, sampling_client=None))
    assert len(flat) == 2
    for t in flat:
        assert t.advantage == pytest.approx([0.3] * len(t.act_tokens))  # -0.2 - (-0.5)
    # The teacher saw the reference solution; the student's prompt did not have it.
    assert "worked." in tok.decode(client.prompts[0])
    assert "worked." not in tok.decode(flat[0].obs_tokens)
    metrics = fn.pop_step_metrics()
    assert metrics["distill/reverse_kl"] == pytest.approx(-0.3)
    assert metrics["distill/frac_sequences_scored"] == 1.0
    assert fn.pop_step_metrics()["distill/num_scored_tokens"] == 0  # reset after pop


def test_advantage_fn_mixes_grpo_task_advantage():
    tok = _CharTokenizer()
    fn = DistillAdvantageFn(
        teacher=dteacher.Teacher(_FakeClient(value=-0.5), name="t"),
        context=dctx.SameContext(tokenizer=tok, template_fn=_template),
        alpha_task=0.5, logprob_check_samples=0,
    )
    flat = asyncio.run(fn([_gsm8k_group()], step=0))
    # rewards [0, 1] -> GRPO advantages [-1, +1]; distillation part is 0.
    assert flat[0].advantage == pytest.approx([-0.5] * len(flat[0].act_tokens), abs=1e-6)
    assert flat[1].advantage == pytest.approx([0.5] * len(flat[1].act_tokens), abs=1e-6)


def test_advantage_fn_leaves_failed_or_contextless_sequences_unscored():
    tok = _CharTokenizer()

    class Flaky(_FakeClient):
        async def compute_logprobs_async(self, prompt):
            raise RuntimeError("engine hiccup")

    fn = DistillAdvantageFn(
        teacher=dteacher.Teacher(Flaky(), name="t"),
        context=dctx.SameContext(tokenizer=tok, template_fn=_template),
        logprob_check_samples=0,
    )
    flat = asyncio.run(fn([_gsm8k_group()], step=0))
    assert all(a == 0.0 for t in flat for a in t.advantage)
    assert fn.pop_step_metrics()["distill/frac_sequences_scored"] == 0.0

    class Dead(_FakeClient):
        async def compute_logprobs_async(self, prompt):
            raise SessionDeadError("gone")

    dead = DistillAdvantageFn(
        teacher=dteacher.Teacher(Dead(), name="t"),
        context=dctx.SameContext(tokenizer=tok, template_fn=_template),
        logprob_check_samples=0,
    )
    with pytest.raises(SessionDeadError):
        asyncio.run(dead([_gsm8k_group()], step=0))


def test_logprob_check_reports_sampler_vs_compute_logprobs_gap():
    tok = _CharTokenizer()
    fn = DistillAdvantageFn(
        teacher=dteacher.Teacher(_FakeClient(value=-0.5), name="t"),
        context=dctx.SameContext(tokenizer=tok, template_fn=_template),
        logprob_check_samples=4,
    )
    student_sampler = _FakeClient(value=-0.6)  # sampled logprobs are -0.5
    asyncio.run(fn([_gsm8k_group()], step=0, sampling_client=student_sampler))
    assert fn.pop_step_metrics()["distill/logprob_check_abs_diff"] == pytest.approx(0.1)
    # Only once per run.
    asyncio.run(fn([_gsm8k_group()], step=1, sampling_client=student_sampler))
    assert "distill/logprob_check_abs_diff" not in fn.pop_step_metrics()


# === Parser / validation ===============================================


_REQUIRED = ["--base_model", "Qwen/Qwen3-8B"]


def test_parser_method_defaults():
    opd = dtrain.parse_with_config(dtrain.build_parser("opd", envs=("gsm8k",)),
                                   _REQUIRED + ["--teacher_base_model_arn", "arn:teacher"])
    assert (opd.distill_method, opd.teacher, opd.loss) == ("opd", "session", "importance_sampling")
    opsd = dtrain.parse_with_config(dtrain.build_parser("opsd", envs=("gsm8k",)), _REQUIRED)
    assert (opsd.teacher, opsd.privileged_field) == ("base", "answer")
    sdpo = dtrain.parse_with_config(dtrain.build_parser("sdpo", envs=("code_rl",)), _REQUIRED)
    assert (sdpo.teacher, sdpo.env_type) == ("base", "code_rl")


def test_parser_rejects_missing_teacher_arns():
    with pytest.raises(SystemExit):
        dtrain.parse_with_config(dtrain.build_parser("opd", envs=("gsm8k",)), _REQUIRED)
    with pytest.raises(SystemExit):
        dtrain.parse_with_config(dtrain.build_parser("opsd", envs=("gsm8k",)),
                                 _REQUIRED + ["--teacher", "checkpoint"])


def test_build_context_per_method():
    tok = _CharTokenizer()
    for method, envs, kind in (("opd", ("gsm8k",), dctx.SameContext),
                               ("opsd", ("gsm8k",), dctx.PrivilegedContext),
                               ("sdpo", ("code_rl",), dctx.FeedbackContext)):
        argv = _REQUIRED + (["--teacher_base_model_arn", "arn:t"] if method == "opd" else [])
        args = dtrain.parse_with_config(dtrain.build_parser(method, envs=envs), argv)
        ctx = dtrain.build_context(args, tokenizer=tok, template_fn=_template)
        assert isinstance(ctx, kind)
    assert ctx.solution_fn is dctx.code_solution


# === Datum shaping ======================================================


def test_service_build_datum_places_per_token_advantages():
    pytest.importorskip("sagemaker.train.training_session")
    from training_sessions_cookbook.rl.backends.service import ServiceBackend

    backend = ServiceBackend.__new__(ServiceBackend)
    t = Transition(obs_tokens=[1, 2, 3], act_tokens=[4, 5], act_logprobs=[-0.1, -0.2],
                   reward=0.0, done=True, env_type="gsm8k", advantage=[0.5, -0.25])
    datum = backend.build_datum(t, temperature=1.0)
    adv = datum["lossFnInputs"]["advantages"]["data"]
    weights = datum["lossFnInputs"]["weights"]["data"]
    assert adv == [0.0, 0.0, 0.5, -0.25]
    assert weights == [0.0, 0.0, 1.0, 1.0]

    t.advantage = 0.7  # scalar still broadcasts
    assert backend.build_datum(t, temperature=1.0)["lossFnInputs"]["advantages"]["data"] == [0.0, 0.0, 0.7, 0.7]

    t.advantage = [0.5]
    with pytest.raises(AssertionError):
        backend.build_datum(t, temperature=1.0)


# === Loop integration ===================================================


class _RecordingMixin:
    def build_datum(self, transition, *, temperature):
        self.advantages = getattr(self, "advantages", [])
        self.advantages.append(transition.advantage)
        return super().build_datum(transition, temperature=temperature)


class _RecordingBackend(_RecordingMixin, MockBackend):
    pass


class _RecordingStreamingBackend(_RecordingMixin, StreamingMockBackend):
    pass


def _run_with(args, backend, advantage_fn):
    asyncio.run(rl.run_training(
        args, backend, _gsm8k_dataset(), tokenizer=backend.tokenizer,
        env_factory=rl.get_env_factory(args.env_type, args),
        template_fn=rl.apply_qwen3_general, advantage_fn=advantage_fn,
    ))


@pytest.mark.parametrize("backend_cls,chunks", [(_RecordingBackend, 0),
                                                (_RecordingStreamingBackend, 2)])
def test_run_training_uses_distillation_advantages(tmp_path, backend_cls, chunks):
    backend = backend_cls()
    args = _args(tmp_path, fb_stream_chunks=chunks)
    # Mock sampler logprobs are -0.1 per token; the teacher scores -0.4.
    fn = DistillAdvantageFn(
        teacher=dteacher.Teacher(_FakeClient(value=-0.4), name="t"),
        context=dctx.SameContext(tokenizer=backend.tokenizer, template_fn=_template),
    )
    _run_with(args, backend, fn)

    assert backend.advantages, "no datums were built"
    for adv in backend.advantages:
        assert isinstance(adv, list) and adv == pytest.approx([-0.3] * len(adv))
    with open(run_artifacts(args).metrics_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert records and all(r["distill/reverse_kl"] == pytest.approx(0.3) for r in records)
    assert all(r["distill/frac_sequences_scored"] == 1.0 for r in records)


def test_custom_advantage_fn_requires_grouped_mode(tmp_path):
    args = _args(tmp_path, advantage_mode="monte_carlo")

    async def fn(groups, *, step, sampling_client):  # pragma: no cover - never called
        return []

    with pytest.raises(SystemExit):
        _run_with(args, MockBackend(), fn)


def test_run_async_builds_advantage_fn_and_closes_it_on_failure(tmp_path, monkeypatch):
    """The factory runs once the session is open, its result reaches the loop,
    and close() runs at teardown (before the student session is stopped)."""
    events = []

    class _Backend(MockBackend):
        training_client = service_client = None

        def close(self):
            events.append("backend.close")

    class _Fn:
        def close(self):
            events.append("advantage_fn.close")

    fn = _Fn()

    async def factory(*, args, backend, tokenizer, template_fn):
        events.append("factory")
        assert tokenizer is backend.tokenizer and callable(template_fn)
        return fn

    async def fake_loop(*a, advantage_fn=None, **kw):
        events.append("loop")
        assert advantage_fn is fn
        raise RuntimeError("stop here")

    monkeypatch.setattr(rl, "make_backend", lambda name, args: _Backend())
    monkeypatch.setattr(rl, "run_training", fake_loop)
    args = rl.parse_with_config(rl.build_parser(envs=("gsm8k",)), [
        "--base_model", "Qwen/Qwen3-8B", "--output_dir", str(tmp_path / "run"),
        "--log_dir", str(tmp_path / "logs"),
    ])
    with pytest.raises(RuntimeError, match="stop here"):
        asyncio.run(rl.run_async(args, advantage_fn_factory=factory))
    assert events == ["factory", "loop", "advantage_fn.close", "backend.close"]


# === Tempered teacher ==================================================


def test_tempered_teacher_dropped_normalizer_leaves_expected_gradient_unchanged():
    """E_q[A * dlog q/dz] is identical with the exact tempered log p_T and with
    log p / T (the normalizer log Z is a per-position baseline)."""
    z = [0.3, -0.2, 1.1]                      # student logits
    q = [math.exp(v) for v in z]
    q = [v / sum(q) for v in q]
    p = [0.5, 0.4, 0.1]
    T = 0.5
    log_z = math.log(sum(pi ** (1 / T) for pi in p))
    exact = [math.log(pi) / T - log_z for pi in p]
    student = [math.log(v) for v in q]

    def expected_grad(teacher_lp, temperature):
        adv = losses.reverse_kl_advantages(student, teacher_lp, teacher_temperature=temperature)
        # d log q_i / d z_j = 1[i=j] - q_j
        return [sum(q[i] * adv[i] * ((1.0 if i == j else 0.0) - q[j]) for i in range(3))
                for j in range(3)]

    with_normalizer = expected_grad(exact, 1.0)
    dropped = expected_grad([math.log(pi) for pi in p], T)
    assert dropped == pytest.approx(with_normalizer, abs=1e-12)


def test_same_model_teacher_needs_temperature_below_one():
    lp = [-0.1, -2.0, -5.0]
    assert losses.reverse_kl_advantages(lp, lp) == [0.0, 0.0, 0.0]
    sharpened = losses.reverse_kl_advantages(lp, lp, teacher_temperature=0.5)
    assert sharpened == pytest.approx(lp)     # (1/T - 1) * log p with T = 0.5
    with pytest.raises(ValueError):
        losses.reverse_kl_advantages(lp, lp, teacher_temperature=0.0)


def test_advantage_fn_applies_teacher_temperature():
    tok = _CharTokenizer()
    fn = DistillAdvantageFn(
        teacher=dteacher.Teacher(_FakeClient(value=-0.5), name="t"),
        context=dctx.SameContext(tokenizer=tok, template_fn=_template),
        teacher_temperature=0.5, logprob_check_samples=0,
    )
    flat = asyncio.run(fn([_gsm8k_group(logprob=-0.5)], step=0))
    assert flat[0].advantage == pytest.approx([-0.5] * len(flat[0].act_tokens))  # -1.0 + 0.5
    assert fn.pop_step_metrics()["distill/teacher_logprob"] == pytest.approx(-1.0)


def test_parser_rejects_zero_teacher_temperature_and_opd_preset_parses():
    with pytest.raises(SystemExit):
        dtrain.parse_with_config(dtrain.build_parser("opd", envs=("gsm8k",)),
                                 _REQUIRED + ["--teacher", "base", "--teacher_temperature", "0"])
    preset = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "training_sessions_cookbook/recipes/opd_math/configs/opd_gsm8k_qwen3_6_27b.json")
    args = dtrain.parse_with_config(dtrain.build_parser("opd", envs=("gsm8k",)), ["--config", preset])
    assert (args.teacher, args.teacher_temperature, args.base_model) == ("base", 0.5, "Qwen/Qwen3.6-27B")


def test_open_teacher_session_binds_adapter_and_cleans_up_on_mismatch(monkeypatch):
    deleted = []
    tc = _TeacherTc()
    svc = SimpleNamespace(delete_session=lambda name, stop_if_active: deleted.append(name))
    teacher_tok = _CharTokenizer()
    monkeypatch.setattr(dteacher, "open_session", lambda **kw: (svc, tc, teacher_tok))
    args = SimpleNamespace(run_name="run", model_package_group_arn="g", role_arn="r",
                           output_s3_path="s3://b/", teacher_lora_rank=8, region="us-west-2",
                           allow_tokenizer_mismatch=False)
    _, name, _, client = dteacher._open_teacher_session(
        args, _CharTokenizer(), base_model_arn="arn:base", model_package_arn="arn:pkg")
    assert tc.loaded == ["arn:pkg"] and client.model_package_arn == "arn:pkg"
    assert name.startswith("run-teacher-") and deleted == []

    class Other(_CharTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [0]

    monkeypatch.setattr(dteacher, "open_session", lambda **kw: (svc, _TeacherTc(), Other()))
    with pytest.raises(SystemExit):
        dteacher._open_teacher_session(args, _CharTokenizer(), base_model_arn="arn:x")
    assert len(deleted) == 1   # the mismatched session is not left running
