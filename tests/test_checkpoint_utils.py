"""Unit tests for training_sessions_cookbook.service.checkpoint_utils.

All tests run against fake clients, no SDK or network required.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from training_sessions_cookbook.service.checkpoint_utils import (
    CHECKPOINTS_BASE_NAME,
    CheckpointManager,
    CheckpointRecord,
    get_last_checkpoint,
    load_checkpoints_file,
    resume_from_last,
    save_checkpoint,
    save_checkpoint_async,
)

ARN = "arn:aws:sagemaker:us-west-2:111111111111:model-package/group/{}"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, arn):
        self.model_package_arn = arn


class _PathResult:
    """Duck-typed result exposing .path instead of .model_package_arn."""

    def __init__(self, path):
        self.path = path


class _Future:
    def __init__(self, result):
        self._result = result

    def __await__(self):
        async def _get():
            return self._result

        return _get().__await__()


class FakeTrainingClient:
    def __init__(self, result_cls=_Result):
        self.calls: list[tuple[str, str, int | None]] = []  # (op, name, ttl)
        self._result_cls = result_cls
        self._n = 0

    def _next(self, op, name, ttl):
        self.calls.append((op, name, ttl))
        self._n += 1
        return _Future(self._result_cls(ARN.format(self._n)))

    async def save_state_async(self, description="", ttl_seconds=None):
        return self._next("state", description, ttl_seconds)

    async def save_weights_for_sampler_async(self, description="", ttl_seconds=None):
        return self._next("sampler", description, ttl_seconds)


class FakeServiceClient:
    def __init__(self):
        self.deleted: list[str] = []
        self.from_state_calls: list[tuple[str, bool]] = []

    def delete_model_package(self, arn, *, delete_artifacts=False):
        self.deleted.append(arn)

    def create_training_client_from_state(self, *, model_package_arn, **kw):
        self.from_state_calls.append((model_package_arn, False))
        return FakeTrainingClient()

    def create_training_client_from_state_with_optimizer(self, *, model_package_arn, **kw):
        self.from_state_calls.append((model_package_arn, True))
        return FakeTrainingClient()


def _read_records(log_dir):
    path = Path(log_dir) / CHECKPOINTS_BASE_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# CheckpointRecord
# ---------------------------------------------------------------------------


def test_record_round_trip_preserves_unknown_keys():
    d = {"name": "000010", "step": 10, "state_path": "arn:x", "reward_mean": 0.75}
    rec = CheckpointRecord.from_dict(d)
    assert rec.step == 10
    assert rec.get("reward_mean") == 0.75
    assert rec.has("state_path") and not rec.has("sampler_path")
    assert rec.to_dict() == d


def test_record_none_fields_omitted_and_extra_collision_dropped():
    rec = CheckpointRecord(name="final", final=True, extra={"name": "evil", "k": 1})
    d = rec.to_dict()
    assert d == {"name": "final", "final": True, "k": 1}
    assert "step" not in d


# ---------------------------------------------------------------------------
# load / get_last_checkpoint
# ---------------------------------------------------------------------------


def test_get_last_checkpoint_filters_on_required_key(tmp_path):
    rows = [
        {"name": "000001", "step": 1, "state_path": "arn:1"},
        {"name": "000002", "step": 2, "sampler_path": "arn:s2"},  # sampler-only
        {"name": "000003", "step": 3, "state_path": "arn:3"},
    ]
    with open(tmp_path / CHECKPOINTS_BASE_NAME, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    assert len(load_checkpoints_file(str(tmp_path))) == 3
    last = get_last_checkpoint(str(tmp_path))
    assert last is not None and last.state_path == "arn:3"
    last_sampler = get_last_checkpoint(str(tmp_path), required_key="sampler_path")
    assert last_sampler is not None and last_sampler.name == "000002"


def test_get_last_checkpoint_empty_dir(tmp_path):
    assert get_last_checkpoint(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


def test_save_checkpoint_both_writes_record_and_returns_arns(tmp_path):
    tc = FakeTrainingClient()
    paths = save_checkpoint(
        tc, name="000005", log_path=str(tmp_path),
        loop_state={"step": 5, "epoch": 0}, kind="both", ttl_seconds=3600,
    )
    assert set(paths) == {"state_path", "sampler_path"}
    assert all(p.startswith("arn:aws:sagemaker") for p in paths.values())
    # both saves carried the ttl and name
    assert tc.calls == [("state", "000005", 3600), ("sampler", "000005", 3600)]

    (rec,) = _read_records(tmp_path)
    assert rec["name"] == "000005" and rec["step"] == 5
    assert rec["state_path"] == paths["state_path"]
    assert rec["sampler_path"] == paths["sampler_path"]


def test_save_checkpoint_state_only(tmp_path):
    tc = FakeTrainingClient()
    paths = save_checkpoint(tc, "r1", str(tmp_path), {"step": 1}, kind="state")
    assert "sampler_path" not in paths
    assert [op for op, _, _ in tc.calls] == ["state"]


def test_save_checkpoint_accepts_path_shaped_results(tmp_path):
    """Duck-typed clients whose results expose .path work identically."""
    tc = FakeTrainingClient(result_cls=_PathResult)
    paths = save_checkpoint(tc, "000001", str(tmp_path), {"step": 1}, kind="both")
    assert set(paths) == {"state_path", "sampler_path"}
    (rec,) = _read_records(tmp_path)
    assert rec["state_path"].startswith("arn:")


# ---------------------------------------------------------------------------
# resume_from_last
# ---------------------------------------------------------------------------


def test_resume_from_last_uses_newest_state_and_optimizer_flag(tmp_path):
    svc = FakeServiceClient()
    with open(tmp_path / CHECKPOINTS_BASE_NAME, "w") as f:
        f.write(json.dumps({"name": "000001", "step": 1, "state_path": "arn:old"}) + "\n")
        f.write(json.dumps({"name": "000002", "step": 2, "state_path": "arn:new"}) + "\n")

    out = resume_from_last(svc, str(tmp_path))
    assert out is not None
    _, rec = out
    assert rec.step == 2
    assert svc.from_state_calls == [("arn:new", True)]

    resume_from_last(svc, str(tmp_path), with_optimizer=False)
    assert svc.from_state_calls[-1] == ("arn:new", False)


def test_resume_from_last_none_when_no_checkpoints(tmp_path):
    assert resume_from_last(FakeServiceClient(), str(tmp_path)) is None


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


def _mgr(tc, svc, tmp_path, **kw):
    return CheckpointManager(tc, svc, str(tmp_path), **kw)


def test_periodic_cadence(tmp_path):
    mgr = _mgr(FakeTrainingClient(), FakeServiceClient(), tmp_path, save_every=5)
    assert [s for s in range(0, 12) if mgr.should_save_periodic(s)] == [5, 10]
    # disabled
    mgr0 = _mgr(FakeTrainingClient(), FakeServiceClient(), tmp_path, save_every=0)
    assert not any(mgr0.should_save_periodic(s) for s in range(12))


def test_maybe_save_periodic_and_rolling_interaction(tmp_path):
    async def run():
        tc, svc = FakeTrainingClient(), FakeServiceClient()
        mgr = _mgr(tc, svc, tmp_path, save_every=4, rolling_save_every=2)
        for step in range(1, 5):
            await mgr.maybe_save_async(step, {"step": step})
        await mgr.finalize_async()
        return tc, svc

    tc, svc = asyncio.run(run())
    records = _read_records(tmp_path)
    # step 2 -> rolling (state only); step 4 -> periodic (both), NOT rolling
    names = [(r["name"], r.get("rolling", False)) for r in records]
    assert names == [("000002", True), ("000004", False)]
    ops = [op for op, _, _ in tc.calls]
    assert ops == ["state", "state", "sampler"]  # rolling, then periodic both
    # finalize deleted the last rolling checkpoint
    assert len(svc.deleted) == 1


def test_rolling_deletes_previous_after_new_save(tmp_path):
    async def run():
        tc, svc = FakeTrainingClient(), FakeServiceClient()
        mgr = _mgr(tc, svc, tmp_path, rolling_save_every=1)
        for step in (1, 2, 3):
            await mgr.maybe_save_async(step, {"step": step})
        await mgr.finalize_async()
        return svc

    svc = asyncio.run(run())
    records = _read_records(tmp_path)
    arns = [r["state_path"] for r in records]
    # All three rolling checkpoints are deleted eventually (two rotations,
    # then finalize), in save order.
    assert svc.deleted == arns


def test_save_final_no_ttl_and_marks_final(tmp_path):
    async def run():
        tc, svc = FakeTrainingClient(), FakeServiceClient()
        mgr = _mgr(tc, svc, tmp_path, save_every=2, ttl_seconds=3600)
        await mgr.maybe_save_async(2, {"step": 2})
        return tc, await mgr.save_final_async({"step": 3})

    tc, paths = asyncio.run(run())
    assert set(paths) == {"state_path", "sampler_path"}
    final_calls = [c for c in tc.calls if c[1] == "final"]
    assert final_calls == [("state", "final", None), ("sampler", "final", None)]
    last = _read_records(tmp_path)[-1]
    assert last["name"] == "final" and last["final"] is True and last["step"] == 3


def test_async_periodic_saves_record_lands_before_final(tmp_path):
    async def run():
        tc, svc = FakeTrainingClient(), FakeServiceClient()
        mgr = _mgr(tc, svc, tmp_path, save_every=1, async_periodic_saves=True)
        assert await mgr.maybe_save_async(1, {"step": 1}) is None  # backgrounded
        await mgr.save_final_async({"step": 1})

    asyncio.run(run())
    names = [r["name"] for r in _read_records(tmp_path)]
    assert names == ["000001", "final"]


def test_rolling_deletion_failure_is_swallowed(tmp_path):
    class ExplodingServiceClient(FakeServiceClient):
        def delete_model_package(self, arn, *, delete_artifacts=False):
            raise RuntimeError("deletion unavailable")

    async def run():
        mgr = _mgr(FakeTrainingClient(), ExplodingServiceClient(), tmp_path, rolling_save_every=1)
        await mgr.maybe_save_async(1, {"step": 1})
        await mgr.maybe_save_async(2, {"step": 2})
        await mgr.finalize_async()  # must not raise

    asyncio.run(run())
    assert len(_read_records(tmp_path)) == 2


def test_save_checkpoint_async_inside_running_loop(tmp_path):
    """The async API is usable directly inside a training loop's event loop."""

    async def run():
        return await save_checkpoint_async(
            FakeTrainingClient(), "000001", str(tmp_path), {"step": 1}, kind="state"
        )

    paths = asyncio.run(run())
    assert "state_path" in paths


def test_sync_manager_wrappers(tmp_path):
    """maybe_save/save_final work without an event loop (sync training loops)."""
    tc, svc = FakeTrainingClient(), FakeServiceClient()
    mgr = CheckpointManager(tc, svc, str(tmp_path), save_every=2, rolling_save_every=1)
    for step in (1, 2, 3):
        mgr.maybe_save(step, {"step": step})
    paths = mgr.save_final({"step": 3})
    assert set(paths) == {"state_path", "sampler_path"}
    names = [r["name"] for r in _read_records(tmp_path)]
    # step1 rolling, step2 periodic, step3 rolling, final
    assert names == ["000001", "000002", "000003", "final"]
    assert len(svc.deleted) == 2  # step1 rolled over by step3, step3 by finalize


def test_final_record_is_last_even_with_inflight_rolling_save(tmp_path):
    """An in-flight background rolling save must be drained BEFORE the final
    save writes its record: otherwise the rolling save (whose model package
    gets deleted right after) becomes the last checkpoints.jsonl record and
    resume_from_last would target a deleted checkpoint. Regression test for
    the flag-sft-b live incident."""

    async def run():
        tc, svc = FakeTrainingClient(), FakeServiceClient()
        mgr = _mgr(tc, svc, tmp_path, rolling_save_every=1)
        # Start a rolling save as a background task and IMMEDIATELY save
        # final, so the rolling task is still pending when final begins.
        await mgr.maybe_save_rolling_async(1, {"step": 1})
        await mgr.save_final_async({"step": 1})
        return tc, svc

    tc, svc = asyncio.run(run())
    records = _read_records(tmp_path)
    assert records[-1]["name"] == "final", [r["name"] for r in records]
    # the rolling record landed before final, and its package was deleted
    rolling = [r for r in records if r.get("rolling")]
    assert rolling and all(
        r["state_path"] in svc.deleted for r in rolling
    )
