"""Phase K3B Wave 1 Step 1.5 — Soak collector (pre-positioned).

Walks ``search_sessions`` over a configurable retention window (1-30
days, default 30) and emits one JSONL row per per-article diagnostic
where the K3B Wave 1 shadow-enforcement gate flagged
``would_have_been_excluded_in_active == True``.

This is the K3B Wave 1 successor to ``scripts/k3a3_soak_collector.py``.
The K3A.2 / K3A.3 collectors filtered on ``cap_reason_codes``
membership (broad surface — every retracted / EoC candidate). The
K3B collector narrows to the EXACT post-gate decision the active
mode would have taken — only the (retracted ∩ stable-join) subset
the K3B Wave 1 Step 1 enforcement gate would have excluded.

The collector enforces every K3A.2 / K3A.3 hard invariant at the
file boundary:

- ``status == "blocked_phi"`` sessions are EXCLUDED at the Mongo
  filter level (and re-asserted in Python defense-in-depth).
- ``deleted_at != None`` sessions are excluded.
- The collector NEVER reads ``original_question`` or
  ``sanitized_question`` from any document. The Mongo projection
  forbids these fields, and an explicit assertion at the top of the
  per-document loop confirms they were not pulled.
- The output JSONL ONLY contains closed-set fields + closed-set
  identifiers (PMID / DOI / PMCID / article_id / session_id /
  enforcement_reason_codes / enforcement_explanation /
  publication_year / short truncated title / journal).
  No user IDs, no emails, no raw prompts/completions/evidence text.
- A defense-in-depth ``_FORBIDDEN_DIAG_SUBSTRINGS`` sweep is applied
  to every JSONL row before writing.
- Defense-in-depth ``article_join_status`` re-check on the live
  ``articles`` row: if the in-flight K3A diagnostic carried
  ``article_join_status == "stable"`` but the article is no longer
  joinable, the live row is downgraded to
  ``"article_removed_between_classification_and_review"`` and
  excluded from the soak output (the K3B gate would not fire on
  it today). This bucket is counted under
  ``orphan_or_join_unstable_total`` in the summary.

Hard invariant (defense-in-depth): the collector NEVER stamps
``cap_enforcement_mode == "active"``. It reads the env once at
start-up and asserts the value is in ``{"disabled", "shadow"}``.

Usage::

    python -m scripts.k3b_wave1_step1_soak_collector \\
        --window-days 30 \\
        --out /app/memory/k3b_wave1_step1_soak_candidates_<UTC>.jsonl

The collector NEVER mutates any document; it only reads.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from scripts.k3a2_soak_collector import (  # noqa: E402
    FORBIDDEN_SESSION_FIELDS, _clamp_window_days, _scrub_row, _truncate,
)
from services.article_lane_classifier import (  # noqa: E402
    _FORBIDDEN_DIAG_SUBSTRINGS,
)
from services.cap_enforcement_guard import (  # noqa: E402
    compute_article_join_status,
)


def _is_safe_str(s: Any) -> bool:
    """Return True iff ``s`` is a string and contains no forbidden
    substring (``system:``, ``sk-``, ``@gmail.com``, etc.).
    """
    if not isinstance(s, str):
        return False
    return not any(
        bad in s for bad in _FORBIDDEN_DIAG_SUBSTRINGS
    )


def _deep_scrub(value: Any) -> Any:
    """Recursively walk ``value`` and drop / redact any string that
    contains a forbidden substring. Dicts and lists are walked
    element-by-element; primitive non-string values pass through
    unchanged.

    Behavior:
      - Inside a ``list``: unsafe strings are DROPPED.
      - Anywhere else (e.g. dict value, top-level): unsafe strings
        are REDACTED to ``"[REDACTED]"`` so column / field alignment
        is preserved (CSV writers depend on this).

    This is the K3B Wave 1 leak guard — strictly tighter than the
    K3A.2 ``_scrub_row`` which only swept top-level string values.
    The K3A.2 helper is still used as the OUTER pass; this helper
    runs INSIDE every list / dict before ``_scrub_row`` runs.
    """
    if isinstance(value, dict):
        return {k: _deep_scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            if isinstance(item, str):
                if _is_safe_str(item):
                    out.append(item)
                continue
            out.append(_deep_scrub(item))
        return out
    if isinstance(value, str):
        return value if _is_safe_str(value) else "[REDACTED]"
    return value


# K3B Wave 1 Step 1 closed-set enforcement reason codes the gate may
# emit on a ``would_have_been_excluded_in_active=True`` diagnostic.
# Wave 1 scope = retracted only.
K3B_WAVE1_ENFORCEMENT_REASON_CODES: tuple[str, ...] = ("retracted",)


def _read_enforcement_mode() -> str:
    """Single source of truth for the enforcement mode. Asserts the
    value is in the K3B Wave 1 allowed set. Raises ``AssertionError``
    if someone has flipped the env to ``"active"`` without operator
    approval (K3B Wave 1 Step 2).
    """
    mode = (os.environ.get(
        "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "disabled",
    ) or "").strip().lower()
    assert mode in {"disabled", "shadow", "active"}, (
        "K3B Wave 1 Step 1.5 soak collector invariant: "
        "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE must be one of "
        "{'disabled', 'shadow', 'active'}. Got: " + repr(mode)
    )
    return mode


async def collect(
    *,
    mongo_url: str,
    db_name: str,
    window_days: int,
    out_path: Path,
    provenance: str = "organic",
) -> dict[str, Any]:
    """Run a single soak pass. Returns a per-bucket summary count and
    writes the JSONL artifact. Idempotent — overwrites ``out_path``.
    """
    enforcement_mode = _read_enforcement_mode()
    window_days = _clamp_window_days(window_days)
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)
    cli = AsyncIOMotorClient(mongo_url)
    db = cli[db_name]
    summary: dict[str, int] = {
        "sessions_scanned": 0,
        "sessions_with_lane_diagnostics": 0,
        "candidates_total": 0,
        "candidates_retracted": 0,
        "orphan_or_join_unstable_total": 0,
        "skipped_non_wave1_reason_total": 0,
        "skipped_phi_blocked_total": 0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written_keys: set[str] = set()

    # Strict projection — never includes the forbidden text fields.
    projection = {
        "_id": 0,
        "id": 1,
        "status": 1,
        "created_at": 1,
        "session_diagnostics_json.article_lane_diagnostics": 1,
        "clinical_question_map_json.primary_intent": 1,
    }
    query = {
        "created_at": {"$gte": window_start},
        "deleted_at": None,
        "status": {"$ne": "blocked_phi"},
    }

    with out_path.open("w", encoding="utf-8") as fh:
        async for sdoc in db.search_sessions.find(query, projection):
            summary["sessions_scanned"] += 1
            # Defense-in-depth: assert no forbidden fields surfaced.
            for f in FORBIDDEN_SESSION_FIELDS:
                assert f not in sdoc, (
                    f"projection leaked forbidden field {f!r}"
                )
            sid = str(sdoc.get("id") or "")
            if not sid:
                continue
            status = sdoc.get("status")
            if status == "blocked_phi":
                # Should never happen because of the query filter,
                # but assert + count anyway.
                summary["skipped_phi_blocked_total"] += 1
                continue
            sdj = sdoc.get("session_diagnostics_json") or {}
            ald = sdj.get("article_lane_diagnostics") if isinstance(
                sdj, dict
            ) else None
            if not isinstance(ald, dict):
                continue
            diags = ald.get("diagnostics") or []
            if not diags:
                continue
            summary["sessions_with_lane_diagnostics"] += 1
            primary_intent = (
                (sdoc.get("clinical_question_map_json") or {}).get(
                    "primary_intent"
                )
                if isinstance(sdoc.get("clinical_question_map_json"), dict)
                else None
            )
            for diag in diags:
                if not isinstance(diag, dict):
                    continue
                # Wave-1 gate inversion: we only output rows where the
                # pipeline-stamped ``would_have_been_excluded_in_active``
                # is True.
                if not bool(
                    diag.get("would_have_been_excluded_in_active")
                ):
                    # Track "skipped because reason set is out of
                    # Wave-1 scope" when the gate would have fired
                    # otherwise. This is informational only.
                    codes = diag.get(
                        "would_have_been_excluded_reason_codes"
                    ) or []
                    if codes:
                        summary["skipped_non_wave1_reason_total"] += 1
                    continue
                article_id = str(diag.get("article_id") or "")
                if not article_id:
                    continue
                # Wave-1 reason-code filter (defense-in-depth): the
                # row is only counted as a Wave-1 candidate if at
                # least one of its enforcement reason codes is in the
                # Wave-1 scope (currently: ``retracted``).
                wbe_codes_raw = diag.get(
                    "would_have_been_excluded_reason_codes"
                ) or []
                wbe_codes = [
                    str(c).lower() for c in wbe_codes_raw
                    if isinstance(c, str)
                ]
                wave1_hits = [
                    c for c in wbe_codes
                    if c in K3B_WAVE1_ENFORCEMENT_REASON_CODES
                ]
                if not wave1_hits:
                    summary["skipped_non_wave1_reason_total"] += 1
                    continue
                # Live article lookup (closed-set fields only).
                art = await db.articles.find_one(
                    {"id": article_id},
                    {
                        "_id": 0,
                        "id": 1, "pmid": 1, "doi": 1, "pmcid": 1,
                        "title": 1, "journal": 1,
                        "publication_year": 1, "publication_date": 1,
                        "publication_types_json": 1,
                        "is_retracted": 1,
                        "is_retraction_notice": 1,
                        "has_expression_of_concern": 1,
                        "source_ids_json": 1,
                        "source_urls_json": 1,
                        "canonical_key": 1,
                    },
                )
                if not art:
                    art = {}
                # Live SSA lookup (presence check only).
                ssa = await db.search_session_articles.find_one(
                    {
                        "search_session_id": sid,
                        "article_id": article_id,
                    },
                    {"_id": 0, "id": 1},
                )
                # Re-compute join status against the LIVE DB rather
                # than trusting the in-flight value. K3B Wave 1 Step
                # 1 already does this at gate time; we mirror it here
                # so the soak artifact stays correct even if rows
                # were dedup-shifted or cleaned up between
                # classification and review.
                persisted_join = str(
                    diag.get("article_join_status") or "stable"
                )
                live_status = compute_article_join_status(
                    article_doc=art if art else None,
                    search_session_article_doc=ssa,
                    classification_canonical_key=(
                        art.get("canonical_key") if isinstance(
                            art, dict
                        ) else None
                    ),
                )
                if live_status != "stable":
                    summary["orphan_or_join_unstable_total"] += 1
                    # Drop the row — gate would refuse to fire today.
                    continue
                # Normalize ``enforcement_explanation`` to List[str]
                # — Item 2 type-invariant.
                expl_raw = diag.get(
                    "enforcement_explanation_full"
                ) or diag.get("enforcement_explanation") or []
                expl = [
                    str(x) for x in expl_raw if isinstance(x, str)
                ][:64]
                for code in wave1_hits:
                    summary[f"candidates_{code}"] = (
                        summary.get(f"candidates_{code}", 0) + 1
                    )
                    summary["candidates_total"] += 1
                    key = f"{sid}|{article_id}|{code}"
                    if key in written_keys:
                        continue
                    written_keys.add(key)
                    # Closed-set lists for downstream adjudication. We
                    # explicitly carry the full WBE / cap reason-code
                    # lists alongside the per-row scalar so the
                    # Step 1.6 adjudicator can verify Wave-1
                    # invariants (retracted ∈ both lists, no
                    # non-Wave-1 codes).
                    full_cap_codes = [
                        str(c).lower() for c in (
                            diag.get("cap_reason_codes") or []
                        )
                        if isinstance(c, str)
                    ]
                    full_wbe_codes = wbe_codes
                    row = {
                        "session_id": sid,
                        "article_id": article_id,
                        "enforcement_reason_code": code,
                        "would_have_been_excluded_reason_codes": (
                            full_wbe_codes
                        ),
                        "cap_reason_codes": full_cap_codes,
                        "enforcement_explanation": expl,
                        "lane_label": str(diag.get("lane_label") or ""),
                        "directness_label": str(
                            diag.get("directness_label") or ""
                        ),
                        "cap_severity": str(
                            diag.get("cap_severity") or "low"
                        ),
                        "primary_intent": (
                            str(primary_intent) if primary_intent
                            else "unknown"
                        ),
                        "pmid": str(art.get("pmid") or ""),
                        "doi": str(art.get("doi") or ""),
                        "pmcid": str(art.get("pmcid") or ""),
                        "title": _truncate(art.get("title"), 180),
                        "journal": _truncate(art.get("journal"), 120),
                        "publication_year": art.get("publication_year"),
                        "publication_date": (
                            str(art.get("publication_date") or "")
                            if art.get("publication_date") else ""
                        ),
                        "publication_types": list(
                            art.get("publication_types_json") or []
                        )[:10],
                        "is_retracted":
                            bool(art.get("is_retracted")),
                        "is_retraction_notice":
                            bool(art.get("is_retraction_notice")),
                        "has_expression_of_concern":
                            bool(art.get("has_expression_of_concern")),
                        "source_ids": (
                            art.get("source_ids_json") or {}
                        ),
                        "source_urls": list(
                            art.get("source_urls_json") or []
                        )[:5],
                        "provenance": provenance,
                        # Closed-set join status surfaces (persisted
                        # vs live re-checked).
                        "article_join_status": live_status,
                        "persisted_article_join_status": persisted_join,
                        # Hard invariants stamped at every row.
                        "shadow_invariants": {
                            "cap_mode": "shadow",
                            "would_cap": False,
                            "would_exclude": False,
                            "cap_enforcement_mode": enforcement_mode,
                            "enforcement_candidate_type": (
                                "retracted_only"
                            ),
                        },
                    }
                    scrubbed = _scrub_row(_deep_scrub(row))
                    fh.write(
                        json.dumps(scrubbed, default=str) + "\n"
                    )
    cli.close()
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 60)
    print("K3B Wave 1 Step 1.5 soak collector — summary")
    print("=" * 60)
    for k in sorted(summary.keys()):
        print(f"  {k:48s} = {summary[k]}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument(
        "--out", type=Path, required=True,
        help="Output JSONL path",
    )
    ap.add_argument(
        "--provenance", type=str, default="organic",
        choices=(
            "organic", "k3a2_seed", "k3a3_seed", "k3b_seed",
            "mixed", "test",
        ),
    )
    args = ap.parse_args()
    mongo_url = os.environ.get("MONGO_URL")
    db_name = os.environ.get("DB_NAME")
    assert mongo_url and db_name, (
        "MONGO_URL / DB_NAME must be set"
    )
    summary = asyncio.run(collect(
        mongo_url=mongo_url, db_name=db_name,
        window_days=args.window_days, out_path=args.out,
        provenance=args.provenance,
    ))
    _print_summary(summary)


if __name__ == "__main__":
    main()
