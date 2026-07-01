"""Phase K4 — clinician + admin endpoints for branch-aware evidence packets.

Clinician-accessible:
  * ``GET /api/search-sessions/{id}/evidence-packet`` — owner only;
    builds the packet lazily on first GET (or returns the persisted
    copy if it already exists for the session).

Admin-accessible:
  * ``GET /api/admin/evidence-packets/{id}`` — admin-only; reads any
    persisted packet by its packet doc id.
  * ``POST /api/admin/evidence-packets/by-session/{session_id}/rebuild``
    — admin-only; force-rebuild a packet for a session (idempotent).

Closed-set responses. Leak-scanned writes via the builder. NO
clinician-visible ranking, default-selection, or summary change.
M3.2 dormant, K4A protected, R1A/B/C/C.1 unchanged.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from core.db import Collections, get_db
from core.security import get_current_user, require_admin
from services.k4_evidence_packet_builder import build_evidence_packet
from services.evidence_lens import preference_packet_metadata

logger = logging.getLogger("clinevidence.k4")

clinician_router = APIRouter(
    prefix="/search-sessions", tags=["k4-evidence-packets"],
)
admin_router = APIRouter(
    prefix="/admin/evidence-packets", tags=["k4-admin"],
)


async def _ensure_session_owned(
    *, session_id: str, current_user: dict,
) -> dict:
    db = get_db()
    doc = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "deleted_at": None}, {"_id": 0},
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="search_session_not_found",
        )
    if doc.get("user_id") != current_user.get("id"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="not_session_owner",
        )
    return doc


@clinician_router.get("/{session_id}/evidence-packet")
async def get_evidence_packet(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Clinician-accessible — returns the branch-aware evidence packet
    for ``session_id``. Builds the packet lazily if missing.
    """
    session_doc = await _ensure_session_owned(
        session_id=session_id, current_user=current_user,
    )
    # PHI-blocked sessions never produce a packet.
    if session_doc.get("status") == "blocked_phi":
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": "phi_blocked_session_no_packet",
                "search_session_id": session_id,
            },
        )
    db = get_db()
    persisted = await db[Collections.EVIDENCE_PACKETS].find_one(
        {"search_session_id": session_id}, {"_id": 0},
    )
    if not persisted:
        try:
            persisted = await build_evidence_packet(
                search_session_id=session_id,
            )
        except KeyError as exc:
            logger.exception("k4 packet build raised KeyError")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "r_series_label_unmapped",
                    "missing_label": str(exc).strip("'\""),
                },
            )
        except ValueError as exc:
            logger.exception("k4 packet build leak-scan failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "evidence_packet_leak_scan_failed",
                    "reason": str(exc),
                },
            )
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )
    # Phase L1 — overlay the LATEST applied evidence-lens / population
    # preference onto the packet's closed-set metadata on read, so the
    # clinician's currently-selected lens is always reflected even if the
    # packet was built before the most recent preference change.
    if isinstance(persisted, dict):
        persisted["preference_metadata"] = preference_packet_metadata(
            session_doc.get("l1_preference_diagnostics_json")
        )
    return persisted


@admin_router.get("/{packet_id}")
async def admin_get_evidence_packet(
    packet_id: str,
    _admin: dict = Depends(require_admin),
):
    db = get_db()
    doc = await db[Collections.EVIDENCE_PACKETS].find_one(
        {"id": packet_id}, {"_id": 0},
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evidence_packet_not_found",
        )
    return doc


@admin_router.post("/by-session/{session_id}/rebuild")
async def admin_rebuild_evidence_packet(
    session_id: str,
    _admin: dict = Depends(require_admin),
):
    try:
        packet = await build_evidence_packet(
            search_session_id=session_id,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "r_series_label_unmapped",
                "missing_label": str(exc).strip("'\""),
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "evidence_packet_leak_scan_failed",
                "reason": str(exc),
            },
        )
    return {
        "rebuilt": True,
        "packet_id": packet.get("id"),
        "search_session_id": session_id,
    }
