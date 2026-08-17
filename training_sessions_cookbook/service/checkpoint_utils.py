"""Checkpoint management for training sessions.

Checkpoints on SageMaker Training Sessions are **model packages**: every
``save_state`` / ``save_weights_for_sampler`` call publishes a SageMaker model
package and returns its ARN. This module records those ARNs in a per-run
``checkpoints.jsonl`` file and layers a cadence policy on top:

* **Periodic** checkpoints: full saves (optimizer state + sampler weights)
  every ``save_every`` steps, with a configurable retention TTL.
* **Rolling** checkpoints: cheap resume-only saves (state, no sampler
  export) every ``rolling_save_every`` steps; each save deletes the previous
  rolling model package so storage stays bounded.
* **Final** checkpoint: a permanent save at the end of training (no TTL),
  after which any leftover rolling checkpoint is cleaned up.

``checkpoints.jsonl`` is append-only, one JSON object per line, and is the
resume contract: :func:`get_last_checkpoint` reads it back and
:func:`resume_from_last` turns the newest resumable record into a live
``TrainingClient``.

Record schema (``CheckpointRecord``): ``name`` plus optional ``step``,
``epoch``, ``final``, ``state_path``, ``sampler_path``; unknown keys are
preserved on read so external tools can add metadata without breaking
round-trips. The ``*_path`` fields hold model-package ARNs.

This module only talks to the client objects passed in, so importing it never
requires the SDK to be installed. It also accepts any client with the same
duck-typed surface (``save_state_async`` / ``save_weights_for_sampler_async``
returning awaitable futures whose result exposes ``model_package_arn`` or
``path``), so training loops written against other checkpoint conventions can
reuse it unchanged.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .data_io import read_jsonl

CHECKPOINTS_BASE_NAME = "checkpoints.jsonl"

logger = logging.getLogger(__name__)

_MISSING = object()  # sentinel distinguishing "not passed" from None


# ---------------------------------------------------------------------------
# Records and the checkpoints.jsonl file
# ---------------------------------------------------------------------------


@dataclass
class CheckpointRecord:
    """One line of ``checkpoints.jsonl``.

    Known fields are typed attributes; anything else found on disk is kept in
    :attr:`extra` so user metadata round-trips through load/save without loss.
    ``state_path`` and ``sampler_path`` hold model-package ARNs.
    """

    name: str
    step: int | None = None
    epoch: int | None = None
    final: bool | None = None
    state_path: str | None = None
    sampler_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Defensive: known keys must live on the attributes, never in extra,
        # so to_dict() cannot double-write them.
        overlap = set(self.extra) & _KNOWN_KEYS
        if overlap:
            logger.warning("CheckpointRecord: dropping known keys from extra: %s", overlap)
            self.extra = {k: v for k, v in self.extra.items() if k not in _KNOWN_KEYS}

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage; ``None`` optionals are omitted."""
        d: dict[str, Any] = {"name": self.name}
        for key in ("step", "epoch", "final", "state_path", "sampler_path"):
            value = getattr(self, key)
            if value is not None:
                d[key] = value
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointRecord":
        """Deserialize; unknown keys land in :attr:`extra`."""
        return cls(
            name=d["name"],
            step=d.get("step"),
            epoch=d.get("epoch"),
            final=d.get("final"),
            state_path=d.get("state_path"),
            sampler_path=d.get("sampler_path"),
            extra={k: v for k, v in d.items() if k not in _KNOWN_KEYS},
        )

    def has(self, key: str) -> bool:
        """True if *key* is present and not ``None`` (known field or extra)."""
        if key in _KNOWN_KEYS:
            return getattr(self, key) is not None
        return key in self.extra

    def get(self, key: str, default: Any = _MISSING) -> Any:
        """Uniform access across known fields and extra metadata."""
        if key in _KNOWN_KEYS:
            return getattr(self, key)
        if default is _MISSING:
            return self.extra.get(key)
        return self.extra.get(key, default)


_KNOWN_KEYS = frozenset(
    f.name for f in dataclasses.fields(CheckpointRecord) if f.name != "extra"
)


def load_checkpoints_file(log_dir: str) -> list[CheckpointRecord]:
    """Load every record from ``{log_dir}/checkpoints.jsonl`` (empty list if absent)."""
    path = Path(log_dir) / CHECKPOINTS_BASE_NAME
    if not path.exists():
        logger.info("No checkpoints found at %s", path)
        return []
    return [CheckpointRecord.from_dict(d) for d in read_jsonl(str(path))]


def get_last_checkpoint(
    log_dir: str, required_key: str = "state_path"
) -> CheckpointRecord | None:
    """Return the newest record that has *required_key*, or ``None``.

    Rolling and periodic saves share the file, and some rows may be partial
    (e.g. sampler-only), so resume callers filter on the field they need --
    ``state_path`` for full resume.
    """
    records = [c for c in load_checkpoints_file(log_dir) if c.has(required_key)]
    if not records:
        logger.info("No checkpoint with %r in %s", required_key, log_dir)
        return None
    logger.info("Using last checkpoint from %s: %s", log_dir, records[-1])
    return records[-1]


def append_checkpoint_record(log_path: str, record: CheckpointRecord) -> None:
    """Append one record to ``{log_path}/checkpoints.jsonl``.

    Normally called through :func:`save_checkpoint_async`; exposed for callers
    that produced a checkpoint through other means (e.g. a backend's final
    save) and need its record in the same file.
    """
    with open(Path(log_path) / CHECKPOINTS_BASE_NAME, "a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _arn_of(result: Any) -> str | None:
    """Extract the checkpoint reference from a save result.

    Native SDK responses expose ``model_package_arn``; clients following other
    checkpoint conventions expose ``path``. Reading both keeps this module
    usable behind either surface.
    """
    return getattr(result, "model_package_arn", None) or getattr(result, "path", None)


async def save_checkpoint_async(
    training_client: Any,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
    ttl_seconds: int | None = None,
) -> dict[str, str]:
    """Save checkpoint(s) and append one record to ``checkpoints.jsonl``.

    Args:
        training_client: Client to save from.
        name: Checkpoint name (also used as the model-package description).
        log_path: Run directory containing ``checkpoints.jsonl``.
        loop_state: Loop progress to record, ``step``, ``epoch``,
            ``final``, plus any custom keys (kept as extra metadata).
        kind: ``"state"`` (resumable, optimizer included), ``"sampler"``
            (inference weights only), or ``"both"``.
        ttl_seconds: Server-side retention; ``None`` keeps it indefinitely.

    Returns:
        Mapping with ``"state_path"`` and/or ``"sampler_path"``, the
        model-package ARNs of the saved checkpoint(s).
    """
    # The label goes positionally: it lands on the SDK's ``description``
    # parameter (checkpoints are identified by model-package ARN; the label is
    # a human-readable annotation).
    futures: dict[str, Any] = {}
    if kind in ("state", "both"):
        futures["state"] = await training_client.save_state_async(
            name, ttl_seconds=ttl_seconds
        )
    if kind in ("sampler", "both"):
        futures["sampler"] = await training_client.save_weights_for_sampler_async(
            name, ttl_seconds=ttl_seconds
        )

    paths: dict[str, str] = {}
    for key, fut in futures.items():
        arn = _arn_of(await fut)
        if arn:
            paths[key + "_path"] = arn
        else:
            logger.warning("save %s %r returned no model-package reference", key, name)
    logger.info("Saved checkpoint %r: %s", name, paths)

    append_checkpoint_record(log_path, CheckpointRecord.from_dict({"name": name, **loop_state, **paths}))
    return paths


def save_checkpoint(
    training_client: Any,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
    ttl_seconds: int | None = None,
) -> dict[str, str]:
    """Synchronous wrapper around :func:`save_checkpoint_async`."""
    return asyncio.run(
        save_checkpoint_async(
            training_client,
            name=name,
            log_path=log_path,
            loop_state=loop_state,
            kind=kind,
            ttl_seconds=ttl_seconds,
        )
    )


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def resume_from_last(
    service_client: Any,
    log_dir: str,
    *,
    with_optimizer: bool = True,
    **create_kwargs: Any,
) -> tuple[Any, CheckpointRecord] | None:
    """Resume training from the newest resumable checkpoint in *log_dir*.

    Creates a fresh session loaded from the recorded ``state_path`` model
    package. Returns ``(training_client, record)``, or ``None`` when the run
    directory has no resumable checkpoint (callers fall back to a cold start).

    Args:
        service_client: SDK ``ServiceClient``.
        log_dir: Run directory containing ``checkpoints.jsonl``.
        with_optimizer: Restore optimizer state too (full resume). Set False
            to warm-start from weights only.
        **create_kwargs: Extra arguments forwarded to session creation
            (e.g. ``training_session_name``).
    """
    record = get_last_checkpoint(log_dir)
    if record is None or not record.state_path:
        return None
    logger.info(
        "Resuming from checkpoint %r (step=%s): %s",
        record.name, record.step, record.state_path,
    )
    if with_optimizer:
        tc = service_client.create_training_client_from_state_with_optimizer(
            model_package_arn=record.state_path, **create_kwargs
        )
    else:
        tc = service_client.create_training_client_from_state(
            model_package_arn=record.state_path, **create_kwargs
        )
    return tc, record


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Periodic, rolling, and final checkpoints behind one interface.

    Usage::

        mgr = CheckpointManager(
            training_client=tc,
            service_client=svc,
            log_path=run_folder,
            save_every=20,
            rolling_save_every=1,
        )
        for step in range(1, num_steps + 1):
            ...train...
            await mgr.maybe_save_async(step, {"step": step})
        await mgr.save_final_async({"step": num_steps})

    Periodic saves are full (``kind="both"``); pass ``ttl_seconds`` for
    server-side retention where the SDK supports it. Rolling saves are
    state-only, run as background tasks so they overlap the next step, and
    each one deletes the previous rolling model package
    (``rolling_ttl_seconds`` adds a server-side safety net on SDKs that
    support TTLs). The final save has no TTL and always blocks.
    """

    def __init__(
        self,
        training_client: Any,
        service_client: Any,
        log_path: str,
        *,
        save_every: int = 0,
        ttl_seconds: int | None = None,
        rolling_save_every: int = 0,
        rolling_ttl_seconds: int | None = None,
        async_periodic_saves: bool = False,
    ) -> None:
        self._training_client = training_client
        self._service_client = service_client
        self._log_path = log_path
        self._save_every = save_every
        self._ttl_seconds = ttl_seconds
        self._rolling_save_every = rolling_save_every
        self._rolling_ttl_seconds = rolling_ttl_seconds
        self._async_periodic_saves = async_periodic_saves

        self._pending_rolling: asyncio.Task[None] | None = None
        self._pending_periodic: asyncio.Task[dict[str, str]] | None = None
        self._prev_rolling_arn: str | None = None

    # --- periodic ----------------------------------------------------------

    def should_save_periodic(self, step: int) -> bool:
        """True when *step* is on the ``save_every`` cadence."""
        return step > 0 and self._save_every > 0 and step % self._save_every == 0

    async def save_periodic_async(self, step: int, loop_state: dict[str, Any]) -> dict[str, str]:
        """Save a full periodic checkpoint (state + sampler); returns the paths."""
        return await save_checkpoint_async(
            self._training_client,
            name=f"{step:06d}",
            log_path=self._log_path,
            loop_state=loop_state,
            kind="both",
            ttl_seconds=self._ttl_seconds,
        )

    # --- rolling -----------------------------------------------------------

    def _should_save_rolling(self, step: int) -> bool:
        if self._rolling_save_every <= 0 or step <= 0:
            return False
        if step % self._rolling_save_every != 0:
            return False
        # A periodic save on the same step already produces a resumable state.
        return not self.should_save_periodic(step)

    async def maybe_save_rolling_async(self, step: int, loop_state: dict[str, Any]) -> None:
        """Resolve any in-flight rolling save, then start a new one if due."""
        await self._resolve_pending_rolling()
        if not self._should_save_rolling(step):
            return
        self._pending_rolling = asyncio.create_task(
            self._do_rolling_save(step, loop_state), name=f"rolling_checkpoint_{step:06d}"
        )

    # --- combined ----------------------------------------------------------

    async def maybe_save_async(
        self, step: int, loop_state: dict[str, Any]
    ) -> dict[str, str] | None:
        """Save periodic and/or rolling checkpoints if *step* is on cadence.

        Returns the periodic paths when a periodic save ran synchronously,
        else ``None`` (including when ``async_periodic_saves`` moved the save
        to a background task).
        """
        result: dict[str, str] | None = None
        if self.should_save_periodic(step):
            if self._async_periodic_saves:
                await self._resolve_pending_periodic()
                self._pending_periodic = asyncio.create_task(
                    self.save_periodic_async(step, loop_state),
                    name=f"periodic_checkpoint_{step:06d}",
                )
            else:
                result = await self.save_periodic_async(step, loop_state)
        await self.maybe_save_rolling_async(step, loop_state)
        return result

    def maybe_save(self, step: int, loop_state: dict[str, Any]) -> dict[str, str] | None:
        """Synchronous version of :meth:`maybe_save_async` (for sync loops).

        Rolling saves block instead of running in the background (a background
        task cannot outlive the per-call event loop), and
        ``async_periodic_saves`` is ignored for the same reason.
        """
        result: dict[str, str] | None = None
        if self.should_save_periodic(step):
            result = asyncio.run(self.save_periodic_async(step, loop_state))
        if self._should_save_rolling(step):
            try:
                asyncio.run(self._do_rolling_save(step, loop_state))
            except Exception:
                logger.warning("Rolling checkpoint save failed", exc_info=True)
        return result

    # --- final -------------------------------------------------------------

    async def save_final_async(self, loop_state: dict[str, Any]) -> dict[str, str]:
        """Save the permanent final checkpoint, then clean up rolling state.

        Drains any in-flight periodic AND rolling save first, so every other
        record lands in ``checkpoints.jsonl`` before the final one. The final
        record must be last: ``resume_from_last`` targets the last record,
        and the rolling package is deleted right after this save.
        """
        await self._resolve_pending_periodic()
        await self._resolve_pending_rolling()
        paths = await save_checkpoint_async(
            self._training_client,
            name="final",
            log_path=self._log_path,
            loop_state={**loop_state, "final": True},
            kind="both",
            ttl_seconds=None,
        )
        await self.finalize_async()
        return paths

    def save_final(self, loop_state: dict[str, Any]) -> dict[str, str]:
        """Synchronous version of :meth:`save_final_async`."""
        return asyncio.run(self.save_final_async(loop_state))

    async def finalize_async(self) -> None:
        """Drain pending saves and delete the last rolling checkpoint."""
        await self._resolve_pending_periodic()
        await self._resolve_pending_rolling()
        await self._delete_prev_rolling()

    # --- internals ---------------------------------------------------------

    async def _resolve_pending_rolling(self) -> None:
        if self._pending_rolling is None:
            return
        try:
            await self._pending_rolling
        except Exception:
            logger.warning("Rolling checkpoint save failed", exc_info=True)
        self._pending_rolling = None

    async def _resolve_pending_periodic(self) -> None:
        if self._pending_periodic is None:
            return
        try:
            await self._pending_periodic
        except Exception:
            logger.warning("Background periodic checkpoint save failed", exc_info=True)
        self._pending_periodic = None

    async def _do_rolling_save(self, step: int, loop_state: dict[str, Any]) -> None:
        paths = await save_checkpoint_async(
            self._training_client,
            name=f"{step:06d}",
            log_path=self._log_path,
            loop_state={**loop_state, "rolling": True},
            kind="state",
            ttl_seconds=self._rolling_ttl_seconds,
        )
        # Only after the new save exists is the previous one safe to delete.
        await self._delete_prev_rolling()
        self._prev_rolling_arn = paths.get("state_path")

    async def _delete_prev_rolling(self) -> None:
        arn = self._prev_rolling_arn
        if arn is None:
            return
        self._prev_rolling_arn = None
        try:
            await asyncio.to_thread(self._service_client.delete_model_package, arn)
            logger.info("Deleted rolling checkpoint %s", arn)
        except Exception:
            logger.warning(
                "Could not delete rolling checkpoint %s (retention TTL will apply)",
                arn,
                exc_info=True,
            )
