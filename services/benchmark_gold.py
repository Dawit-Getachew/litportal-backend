"""Phase 6E — Benchmark gold-set curation service.

Pure-Python business logic; the router is a thin pass-through.

Key responsibilities:
- Validate identifier shapes (PMID / DOI / PMCID / official_source_url).
- Look up the case from the unified case roster
  (``SearchBenchmarkService.list_all()``).
- Persist candidates with closed-set audit events.
- Enforce idempotency on the natural key
  ``(benchmark_case_id, identifier_type, identifier_value)``.
- Compute the 0-5 completeness score per case.
- Enforce activation rules — a case can only transition to
  ``active_benchmark`` if it has ≥1 accepted gold (PMID/DOI/PMCID OR
  official_source_url) AND ``must_have_anchors`` non-empty AND
  ``expected_type_of_question`` non-empty.
- Persist a per-case ``governance_status`` override doc in
  ``BENCHMARK_CASE_OVERRIDES`` so the in-memory dataclass roster stays
  immutable.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

from core.db import Collections, get_db
from models.benchmark_gold import (
    BenchmarkGoldCandidateCreate,
    BenchmarkGoldCandidateDoc,
    BenchmarkGoldSeedImport,
    GovernanceStatus,
)
from services.audit import audit_service
from services.search_benchmark import search_benchmark_service
from services.trusted_domains import classify_official_url


_PMID_RE = re.compile(r"^\d{1,8}$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
_PMCID_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)


class CurationError(Exception):
    """Raised with a closed-set ``code`` + HTTP ``status`` for the
    router to translate. Never includes user-supplied raw text."""
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


# ============================================================
# Closed-set identifier validation
# ============================================================
def _validate_identifier(identifier_type: str, value: str) -> dict[str, Any]:
    """Returns ``{ok: bool, reason: str | None, domain_trust: ..., domain_host: ...}``.

    ``reason`` is a closed-set code; ``value`` is NEVER echoed.
    """
    v = (value or "").strip()
    if not v:
        return {"ok": False, "reason": "empty_value"}
    if identifier_type == "pmid":
        if _PMID_RE.match(v):
            return {"ok": True, "reason": None}
        return {"ok": False, "reason": "invalid_pmid_shape"}
    if identifier_type == "doi":
        if _DOI_RE.match(v.lower()):
            return {"ok": True, "reason": None}
        return {"ok": False, "reason": "invalid_doi_shape"}
    if identifier_type == "pmcid":
        if _PMCID_RE.match(v):
            return {"ok": True, "reason": None}
        return {"ok": False, "reason": "invalid_pmcid_shape"}
    if identifier_type == "title_exact":
        if 5 <= len(v) <= 500:
            return {"ok": True, "reason": None}
        return {"ok": False, "reason": "invalid_title_length"}
    if identifier_type == "source_record_id":
        if 1 <= len(v) <= 200:
            return {"ok": True, "reason": None}
        return {"ok": False, "reason": "invalid_source_record_id"}
    if identifier_type == "official_source_url":
        classified = classify_official_url(v)
        domain_trust = "trusted" if classified["is_trusted"] else "untrusted"
        return {
            "ok": True,  # accepted but tagged
            "reason": None if classified["is_trusted"]
                       else classified["reason"],
            "domain_trust": domain_trust,
            "domain_host": classified["host"],
        }
    return {"ok": False, "reason": "unknown_identifier_type"}


# ============================================================
# Completeness score (0-5)
# ============================================================
async def _accepted_candidate_count(case_id: str, *,
                                     gold_only: bool = False) -> int:
    db = get_db()
    q: dict[str, Any] = {
        "benchmark_case_id": case_id, "status": "accepted",
    }
    if gold_only:
        q["relevance_role"] = {
            "$in": ["primary_gold", "secondary_gold",
                     "official_guideline", "background_reference"],
        }
    return await db[Collections.BENCHMARK_GOLD_CANDIDATES].count_documents(q)


async def _negative_candidate_count(case_id: str) -> int:
    db = get_db()
    return await db[Collections.BENCHMARK_GOLD_CANDIDATES].count_documents({
        "benchmark_case_id": case_id, "status": "accepted",
        "relevance_role": "negative_example",
    })


async def compute_completeness_score(case) -> dict[str, Any]:
    """Returns ``{score: int, missing: list[str]}``.

    The integer score is 0-5; ``missing`` is a closed-set list of the
    requirements that didn't earn a point.
    """
    missing: list[str] = []
    score = 0
    if case.expected_type_of_question:
        score += 1
    else:
        missing.append("expected_type_of_question")
    if case.expected_specialty:
        score += 1
    else:
        missing.append("expected_specialty")
    if case.must_have_anchors:
        score += 1
    else:
        missing.append("must_have_anchors")
    # Gold present? Either built-in non-placeholder PMIDs OR an
    # accepted curation candidate.
    accepted_gold = await _accepted_candidate_count(case.id, gold_only=True)
    has_builtin_gold = (
        len(case.gold_pmids) > 0
        and not getattr(case, "placeholder_gold", False)
    )
    if accepted_gold > 0 or has_builtin_gold:
        score += 1
    else:
        missing.append("accepted_gold_or_real_pmids")
    # Negative examples (either built-in negative_concepts OR accepted
    # negative_example candidates).
    has_built_negatives = bool(case.negative_concepts) or bool(
        case.negative_title_patterns)
    accepted_negatives = await _negative_candidate_count(case.id)
    if has_built_negatives or accepted_negatives > 0:
        score += 1
    else:
        missing.append("negative_examples")
    return {"score": score, "missing": missing}


# ============================================================
# governance_status override storage
# ============================================================
async def _get_governance_override(case_id: str) -> Optional[dict[str, Any]]:
    db = get_db()
    return await db[Collections.BENCHMARK_CASE_OVERRIDES].find_one(
        {"benchmark_case_id": case_id}, {"_id": 0},
    )


def _derive_default_governance_status(case) -> str:
    """Map a case's static dataclass state to the default ``governance_status``.

    Migration rule (per Phase 6E Stage 0): cases with real gold PMIDs
    (placeholder_gold=False, gold_pmids non-empty) AND
    provisional=False → ``active_benchmark``. Everything else →
    ``needs_gold_set``. Cases explicitly marked ``status="retired"`` →
    ``retired``.
    """
    if getattr(case, "status", "") == "retired":
        return "retired"
    if (
        not getattr(case, "provisional", False)
        and not getattr(case, "placeholder_gold", False)
        and len(case.gold_pmids or []) > 0
    ):
        return "active_benchmark"
    return "needs_gold_set"


async def get_governance_status(case) -> str:
    override = await _get_governance_override(case.id)
    if override and override.get("governance_status"):
        return override["governance_status"]
    return _derive_default_governance_status(case)


async def _set_governance_status(case_id: str, new_status: GovernanceStatus,
                                  user_id: str, reason: str | None) -> None:
    db = get_db()
    now = datetime.now(timezone.utc)
    await db[Collections.BENCHMARK_CASE_OVERRIDES].update_one(
        {"benchmark_case_id": case_id},
        {"$set": {
            "benchmark_case_id": case_id,
            "governance_status": new_status,
            "updated_by_user_id": user_id,
            "updated_at": now,
            # Reason capped at 200 chars by the request schema.
            "reason": (reason or None),
        }, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )


# ============================================================
# CRUD on candidates
# ============================================================
async def _resolve_case(case_id: str):
    for c in search_benchmark_service.list_all():
        if c.id == case_id:
            return c
    raise CurationError("benchmark_case_not_found", status=404)


async def add_candidate(body: BenchmarkGoldCandidateCreate, *,
                        user_id: str) -> dict[str, Any]:
    """Idempotent add. Returns the persisted candidate doc.

    If a row already exists for the natural key
    ``(benchmark_case_id, identifier_type, identifier_value)``, the
    existing row is returned unchanged + audit event NOT re-fired
    (idempotency).
    """
    db = get_db()
    # Validate that the benchmark case exists (raises CurationError if not).
    await _resolve_case(body.benchmark_case_id)
    valid = _validate_identifier(body.identifier_type, body.identifier_value)
    if not valid["ok"]:
        raise CurationError(f"identifier_invalid:{valid['reason']}",
                             status=422)
    # Idempotency check.
    existing = await db[Collections.BENCHMARK_GOLD_CANDIDATES].find_one(
        {
            "benchmark_case_id": body.benchmark_case_id,
            "identifier_type": body.identifier_type,
            "identifier_value": body.identifier_value.strip(),
        },
        {"_id": 0},
    )
    if existing:
        return existing
    # Untrusted official URLs auto-flagged ``needs_review``.
    initial_status = "pending_review"
    domain_trust = valid.get("domain_trust")
    domain_host = valid.get("domain_host")
    reviewer_notes = None
    if (
        body.identifier_type == "official_source_url"
        and domain_trust == "untrusted"
    ):
        initial_status = "needs_review"
        reviewer_notes = "untrusted_domain"
    doc = BenchmarkGoldCandidateDoc(
        benchmark_case_id=body.benchmark_case_id,
        identifier_type=body.identifier_type,
        identifier_value=body.identifier_value.strip(),
        relevance_role=body.relevance_role,
        expected_lane=body.expected_lane,
        ranking_signal=list(body.ranking_signal or []),
        rationale=body.rationale,
        status=initial_status,
        domain_trust=domain_trust,
        domain_host=domain_host,
        reviewer_notes=reviewer_notes,
        created_by_user_id=user_id,
    ).model_dump()
    for k in ("created_at", "updated_at"):
        doc[k] = doc[k].astimezone(timezone.utc)
    await db[Collections.BENCHMARK_GOLD_CANDIDATES].insert_one(doc)
    payload = {
        "candidate_id": doc["id"],
        "benchmark_case_id": body.benchmark_case_id,
        "identifier_type": body.identifier_type,
        "relevance_role": body.relevance_role,
        "status": initial_status,
        "domain_trust": domain_trust,
        "domain_host": domain_host,
    }
    await audit_service.log(
        "benchmark_gold_candidate_added", user_id=user_id, payload=payload,
    )
    if body.identifier_type == "official_source_url":
        await audit_service.log(
            "benchmark_official_source_added", user_id=user_id,
            payload={
                "benchmark_case_id": body.benchmark_case_id,
                "candidate_id": doc["id"],
                "domain_trust": domain_trust,
                "domain_host": domain_host,
            },
        )
    return {k: v for k, v in doc.items() if k != "_id"}


async def list_candidates(case_id: str | None = None,
                           status: str | None = None) -> list[dict[str, Any]]:
    db = get_db()
    q: dict[str, Any] = {}
    if case_id:
        q["benchmark_case_id"] = case_id
    if status:
        q["status"] = status
    cursor = db[Collections.BENCHMARK_GOLD_CANDIDATES].find(
        q, {"_id": 0},
    ).sort("created_at", -1)
    return await cursor.to_list(length=500)


async def set_candidate_status(
    candidate_id: str, new_status: str, user_id: str,
    reviewer_notes: str | None = None,
) -> dict[str, Any]:
    db = get_db()
    cand = await db[Collections.BENCHMARK_GOLD_CANDIDATES].find_one(
        {"id": candidate_id}, {"_id": 0},
    )
    if not cand:
        raise CurationError("candidate_not_found", status=404)
    if new_status not in ("accepted", "rejected", "needs_review", "pending_review"):
        raise CurationError("invalid_status", status=422)
    now = datetime.now(timezone.utc)
    update: dict[str, Any] = {"status": new_status, "updated_at": now}
    if reviewer_notes is not None:
        update["reviewer_notes"] = reviewer_notes[:200]
    event = "benchmark_gold_candidate_added"
    if new_status == "accepted":
        update["accepted_by_user_id"] = user_id
        update["accepted_at"] = now
        event = "benchmark_gold_candidate_accepted"
    elif new_status == "rejected":
        update["rejected_by_user_id"] = user_id
        update["rejected_at"] = now
        event = "benchmark_gold_candidate_rejected"
    await db[Collections.BENCHMARK_GOLD_CANDIDATES].update_one(
        {"id": candidate_id}, {"$set": update},
    )
    await audit_service.log(
        event, user_id=user_id,
        payload={
            "candidate_id": candidate_id,
            "benchmark_case_id": cand["benchmark_case_id"],
            "identifier_type": cand["identifier_type"],
            "relevance_role": cand["relevance_role"],
            "status": new_status,
        },
    )
    return await db[Collections.BENCHMARK_GOLD_CANDIDATES].find_one(
        {"id": candidate_id}, {"_id": 0},
    )


# ============================================================
# Seed import (idempotent, dry-run supported)
# ============================================================
async def import_seed(body: BenchmarkGoldSeedImport, *, user_id: str,
                       dry_run: bool = False) -> dict[str, Any]:
    """Idempotent seed import.

    Per ``(benchmark_case_id, identifier_type, identifier_value)``:
    - already-present rows are reported as ``skipped_existing``;
    - new rows are persisted (unless ``dry_run`` is true) and reported
      as ``would_create`` (dry) or ``created`` (real).

    Validation failures are reported as ``would_skip_invalid``. No DB
    writes are issued in dry-run mode.
    """
    db = get_db()
    summary = {
        "schema_version": body.schema_version,
        "n_cases": len(body.cases),
        "n_proposed_total": 0,
        "n_skipped_unknown_case": 0,
        "n_skipped_existing": 0,
        "n_skipped_invalid": 0,
        "n_created": 0,
        "n_needs_review": 0,
        "per_case": [],
        "dry_run": bool(dry_run),
    }
    for case_seed in body.cases:
        case_summary = {
            "benchmark_case_id": case_seed.benchmark_case_id,
            "n_proposed": len(case_seed.candidates),
            "n_skipped_unknown_case": 0,
            "n_skipped_existing": 0,
            "n_skipped_invalid": 0,
            "n_created": 0,
            "n_needs_review": 0,
        }
        # Resolve case (skip whole block on unknown).
        try:
            await _resolve_case(case_seed.benchmark_case_id)
        except CurationError:
            case_summary["n_skipped_unknown_case"] = len(case_seed.candidates)
            summary["n_skipped_unknown_case"] += len(case_seed.candidates)
            summary["n_proposed_total"] += len(case_seed.candidates)
            summary["per_case"].append(case_summary)
            continue
        for cand in case_seed.candidates:
            summary["n_proposed_total"] += 1
            valid = _validate_identifier(cand.identifier_type, cand.identifier_value)
            if not valid["ok"]:
                case_summary["n_skipped_invalid"] += 1
                summary["n_skipped_invalid"] += 1
                continue
            existing = await db[Collections.BENCHMARK_GOLD_CANDIDATES].find_one(
                {
                    "benchmark_case_id": case_seed.benchmark_case_id,
                    "identifier_type": cand.identifier_type,
                    "identifier_value": cand.identifier_value.strip(),
                }, {"_id": 0, "id": 1},
            )
            if existing:
                case_summary["n_skipped_existing"] += 1
                summary["n_skipped_existing"] += 1
                continue
            if dry_run:
                case_summary["n_created"] += 1
                summary["n_created"] += 1
                continue
            # Persist via add_candidate to share the validation +
            # closed-set audit event path.
            res = await add_candidate(
                BenchmarkGoldCandidateCreate(
                    benchmark_case_id=case_seed.benchmark_case_id,
                    identifier_type=cand.identifier_type,
                    identifier_value=cand.identifier_value,
                    relevance_role=cand.relevance_role,
                    expected_lane=cand.expected_lane,
                    ranking_signal=list(cand.ranking_signal or []),
                    rationale=cand.rationale,
                ),
                user_id=user_id,
            )
            if res.get("status") == "needs_review":
                case_summary["n_needs_review"] += 1
                summary["n_needs_review"] += 1
            else:
                case_summary["n_created"] += 1
                summary["n_created"] += 1
        summary["per_case"].append(case_summary)
    if dry_run:
        await audit_service.log(
            "benchmark_gold_seed_dry_run", user_id=user_id,
            payload={
                "n_proposed_total": summary["n_proposed_total"],
                "n_cases": summary["n_cases"],
                "schema_version": body.schema_version,
            },
        )
    else:
        await audit_service.log(
            "benchmark_gold_seed_imported", user_id=user_id,
            payload={
                "n_created": summary["n_created"],
                "n_skipped_existing": summary["n_skipped_existing"],
                "n_skipped_invalid": summary["n_skipped_invalid"],
                "n_skipped_unknown_case": summary["n_skipped_unknown_case"],
                "n_needs_review": summary["n_needs_review"],
                "schema_version": body.schema_version,
            },
        )
    return summary


# ============================================================
# Activation rules (Task 2)
# ============================================================
async def activate_case(case_id: str, user_id: str,
                         reason: str | None) -> dict[str, Any]:
    """Activation rules (server-enforced):

    1. Case must have ``must_have_anchors`` non-empty.
    2. Case must have ``expected_type_of_question`` non-empty.
    3. Case must have at least ONE accepted gold candidate
       (relevance_role ∈ {primary_gold, secondary_gold,
       official_guideline, background_reference}) OR built-in real
       gold PMIDs.
    """
    case = await _resolve_case(case_id)
    failed: list[str] = []
    if not case.must_have_anchors:
        failed.append("missing_must_have_anchors")
    if not case.expected_type_of_question:
        failed.append("missing_expected_type_of_question")
    accepted_gold = await _accepted_candidate_count(case_id, gold_only=True)
    has_builtin_gold = (
        len(case.gold_pmids or []) > 0
        and not getattr(case, "placeholder_gold", False)
    )
    if accepted_gold == 0 and not has_builtin_gold:
        failed.append("no_accepted_gold")
    if failed:
        raise CurationError(
            "activation_blocked:" + ",".join(failed), status=409,
        )
    await _set_governance_status(case_id, "active_benchmark",
                                  user_id=user_id, reason=reason)
    await audit_service.log(
        "benchmark_case_activated", user_id=user_id,
        payload={
            "benchmark_case_id": case_id,
            "accepted_gold_count": accepted_gold,
            "has_builtin_gold": has_builtin_gold,
        },
    )
    return {"benchmark_case_id": case_id,
             "governance_status": "active_benchmark"}


async def retire_case(case_id: str, user_id: str,
                       reason: str | None) -> dict[str, Any]:
    await _resolve_case(case_id)
    await _set_governance_status(case_id, "retired",
                                  user_id=user_id, reason=reason)
    await audit_service.log(
        "benchmark_case_retired", user_id=user_id,
        payload={"benchmark_case_id": case_id},
    )
    return {"benchmark_case_id": case_id, "governance_status": "retired"}


# ============================================================
# List all cases with completeness + governance_status
# ============================================================
async def list_cases_with_governance() -> list[dict[str, Any]]:
    cases = search_benchmark_service.list_all()
    out: list[dict[str, Any]] = []
    for c in cases:
        gov = await get_governance_status(c)
        comp = await compute_completeness_score(c)
        accepted_count = await _accepted_candidate_count(c.id)
        pending_count = await _candidate_count_by_status(c.id, "pending_review")
        out.append({
            "id": c.id,
            "question": c.question,
            "expected_type_of_question": sorted(c.expected_type_of_question),
            "expected_specialty": c.expected_specialty,
            "must_have_anchors": list(c.must_have_anchors or []),
            "gold_pmid_count": len(c.gold_pmids or []),
            "placeholder_gold": bool(getattr(c, "placeholder_gold", False)),
            "provisional": bool(getattr(c, "provisional", False)),
            "governance_status": gov,
            "completeness_score": comp["score"],
            "missing_requirements": comp["missing"],
            "accepted_candidate_count": accepted_count,
            "pending_candidate_count": pending_count,
        })
    # Sort: needs_gold_set → ready_for_review → active_benchmark →
    # retired → parse_only.
    order = {
        "needs_gold_set": 0, "ready_for_review": 1,
        "active_benchmark": 2, "retired": 3, "parse_only": 4,
    }
    out.sort(key=lambda x: (order.get(x["governance_status"], 9), x["id"]))
    return out


async def _candidate_count_by_status(case_id: str, status: str) -> int:
    db = get_db()
    return await db[Collections.BENCHMARK_GOLD_CANDIDATES].count_documents({
        "benchmark_case_id": case_id, "status": status,
    })
