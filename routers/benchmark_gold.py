"""Phase 6E — Admin benchmark gold curation router.

All endpoints under ``/api/admin/benchmark-gold/`` require ``require_admin``.
Clinician → 403.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from core.security import require_admin
from models.benchmark_gold import (
    BenchmarkCaseActivateRequest,
    BenchmarkCaseRetireRequest,
    BenchmarkGoldCandidateCreate,
    BenchmarkGoldCandidateUpdate,
    BenchmarkGoldSeedImport,
)
from services import benchmark_gold as service
from services.trusted_domains import TRUSTED_OFFICIAL_DOMAINS


router = APIRouter(prefix="/admin/benchmark-gold", tags=["benchmark-gold"])


@router.get("/cases")
async def list_cases(current_admin: dict = Depends(require_admin)):
    """All cases (28 today) with their governance_status +
    completeness_score + missing requirements."""
    return await service.list_cases_with_governance()


@router.get("/trusted-domains")
async def list_trusted_domains(current_admin: dict = Depends(require_admin)):
    return {"trusted_domains": list(TRUSTED_OFFICIAL_DOMAINS)}


@router.get("/candidates")
async def list_candidates(
    benchmark_case_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    current_admin: dict = Depends(require_admin),
):
    return await service.list_candidates(benchmark_case_id, status)


@router.post("/candidates", status_code=201)
async def add_candidate(
    body: BenchmarkGoldCandidateCreate,
    current_admin: dict = Depends(require_admin),
):
    try:
        return await service.add_candidate(body, user_id=current_admin["id"])
    except service.CurationError as e:
        raise HTTPException(status_code=e.status, detail=e.code)


@router.patch("/candidates/{candidate_id}")
async def update_candidate(
    candidate_id: str,
    body: BenchmarkGoldCandidateUpdate,
    current_admin: dict = Depends(require_admin),
):
    if body.status is None:
        # We only accept status transitions on PATCH today.
        raise HTTPException(status_code=422, detail="status_required")
    try:
        return await service.set_candidate_status(
            candidate_id, body.status, current_admin["id"],
            reviewer_notes=body.reviewer_notes,
        )
    except service.CurationError as e:
        raise HTTPException(status_code=e.status, detail=e.code)


@router.post("/import")
async def import_seed(
    body: BenchmarkGoldSeedImport,
    dry_run: bool = Query(default=False),
    current_admin: dict = Depends(require_admin),
):
    return await service.import_seed(
        body, user_id=current_admin["id"], dry_run=dry_run,
    )


@router.post("/cases/{case_id}/activate")
async def activate_case(
    case_id: str,
    body: BenchmarkCaseActivateRequest,
    current_admin: dict = Depends(require_admin),
):
    try:
        return await service.activate_case(
            case_id, user_id=current_admin["id"],
            reason=body.activate_reason,
        )
    except service.CurationError as e:
        raise HTTPException(status_code=e.status, detail=e.code)


@router.post("/cases/{case_id}/retire")
async def retire_case(
    case_id: str,
    body: BenchmarkCaseRetireRequest,
    current_admin: dict = Depends(require_admin),
):
    try:
        return await service.retire_case(
            case_id, user_id=current_admin["id"],
            reason=body.retire_reason,
        )
    except service.CurationError as e:
        raise HTTPException(status_code=e.status, detail=e.code)
