"""Engine startup/shutdown hooks, called from server.py's lifespan.

A no-op unless CLINICAL_ENGINE_ENABLED. Never raises: an engine problem must
not stop the rest of LitPortal from starting.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from clinical_engine.config import settings

logger = logging.getLogger("litportal.clinical_engine")

_stop: Optional[asyncio.Event] = None
_recovery: Optional[asyncio.Task[None]] = None


async def start() -> None:
    global _stop, _recovery
    if not settings.ENABLED:
        return
    try:
        from clinical_engine import store
        from clinical_engine.orchestrator.runner import recovery_loop

        await store.ensure_indexes()
        _stop = asyncio.Event()
        _recovery = asyncio.create_task(recovery_loop(_stop), name="engine-recovery")
        logger.info("clinical engine enabled (worker %s, adapter %s)", store.WORKER_ID, settings.RETRIEVAL_ADAPTER)
    except Exception:  # noqa: BLE001
        logger.exception("clinical engine failed to start (engine routes will still answer)")


async def stop() -> None:
    global _stop, _recovery
    if _stop is None:
        return
    try:
        from clinical_engine.orchestrator.runner import shutdown

        _stop.set()
        if _recovery is not None:
            try:
                await asyncio.wait_for(_recovery, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                _recovery.cancel()
        await shutdown()
    except Exception:  # noqa: BLE001
        logger.exception("clinical engine shutdown failed (non-fatal)")
    finally:
        _stop, _recovery = None, None
