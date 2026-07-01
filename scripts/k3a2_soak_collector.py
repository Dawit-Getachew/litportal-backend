"""Phase K3A.2 — Shadow soak candidate collector.

Walks ``search_sessions`` over a configurable retention window (1-30
days, default 30) and emits one JSONL row per article that the shadow
K3A classifier stamped with ``cap_reason_codes`` containing
``"retracted"`` or ``"expression_of_concern"``.

The collector enforces every K3A.2 hard invariant at the file boundary:

- ``status == "blocked_phi"`` sessions are EXCLUDED at the Mongo filter
  level (and re-asserted in Python defense-in-depth).
- ``deleted_at != None`` sessions are excluded.
- The collector NEVER reads ``original_question`` or
  ``sanitized_question`` from any document. The Mongo projection
  forbids these fields, and an explicit assertion at the top of the
  per-document loop confirms they were not pulled.
- The output JSONL ONLY contains closed-set fields + closed-set
  identifiers (PMID / DOI / PMCID / article_id / session_id /
  cap_reason_code / lane_label / directness_label / publication_year /
  short truncated title / journal). No user IDs, no emails, no raw
  prompts/completions/evidence text.
- A defense-in-depth ``_FORBIDDEN_DIAG_SUBSTRINGS`` sweep is applied
  to every JSONL row before writing.

Usage::

    python -m scripts.k3a2_soak_collector \\
        --window-days 30 \\
        --out /app/memory/k3a2_soak_candidates_<UTC>.jsonl

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

from services.article_lane_classifier import (  # noqa: E402
    _FORBIDDEN_DIAG_SUBSTRINGS,
)

# Closed-set cap reason codes this soak gate evaluates.
SOAK_REASON_CODES: tuple[str, ...] = (
    "retracted",
    "expression_of_concern",
)

# Banned projection fields — defense-in-depth.
FORBIDDEN_SESSION_FIELDS: tuple[str, ...] = (
    "original_question", "sanitized_question",
    "user_id", "user_email", "email",
)


def _clamp_window_days(window_days: int) -> int:
    return max(1, min(30, int(window_days)))


def _truncate(s: str | None, n: int) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    return (s if len(s) <= n else s[: n - 1] + "…")


def _scrub_row(row: dict[str, Any]) -> dict[str, Any]:
    """Reject any field whose value contains a forbidden substring."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, str) and any(
            s in v for s in _FORBIDDEN_DIAG_SUBSTRINGS
        ):
            continue
        out[k] = v
    return out


async def collect(
    *,
    mongo_url: str,
    db_name: str,
    window_days: int,
    out_path: Path,
    provenance: str = "organic",
) -> dict[str, Any]:
    """Run a single soak pass. Returns a per-reason summary count and
    writes the JSONL artifact. Idempotent — overwrites ``out_path``.
    """
    window_days = _clamp_window_days(window_days)
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)
    cli = AsyncIOMotorClient(mongo_url)
    db = cli[db_name]
    summary: dict[str, int] = {
        "sessions_scanned": 0,
        "sessions_with_lane_diagnostics": 0,
        "candidates_total": 0,
    }
    for code in SOAK_REASON_CODES:
        summary[f"candidates_{code}"] = 0
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
                # but assert anyway.
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
                codes = diag.get("cap_reason_codes") or []
                hits = [c for c in codes if c in SOAK_REASON_CODES]
                if not hits:
                    continue
                article_id = str(diag.get("article_id") or "")
                if not article_id:
                    continue
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
                    },
                )
                if not art:
                    art = {}
                for code in hits:
                    summary[f"candidates_{code}"] += 1
                    summary["candidates_total"] += 1
                    key = f"{sid}|{article_id}|{code}"
                    if key in written_keys:
                        continue
                    written_keys.add(key)
                    row = {
                        "session_id": sid,
                        "article_id": article_id,
                        "cap_reason_code": code,
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
                        "shadow_invariants": {
                            "cap_mode": "shadow",
                            "would_cap": False,
                            "would_exclude": False,
                            "cap_enforcement_mode": "disabled",
                        },
                    }
                    scrubbed = _scrub_row(row)
                    fh.write(json.dumps(scrubbed, default=str) + "\n")

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 60)
    print("K3A.2 soak collector — summary")
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
        choices=("organic", "k3a2_seed"),
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
