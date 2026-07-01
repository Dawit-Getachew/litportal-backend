"""Phase K3A.3 — Soak collector v2 (orphan/join bucketing).

Builds on ``scripts/k3a2_soak_collector.py`` but:

- Reads the new ``article_join_status`` field on every diagnostic
  (defaults to ``"stable"`` if missing — covers K3A.2 vintage rows).
- Buckets rows whose ``article_join_status != "stable"`` into a
  separate ``orphan_or_join_unstable`` summary count, so the manual
  review can route them to ``unclear_requires_source_review`` without
  conflating them with FP / TP.
- Preserves every K3A.2 hard invariant (no PHI projection, forbidden
  substring scrub, ``blocked_phi`` excluded at Mongo filter, etc.).

The output JSONL row now carries an additional ``article_join_status``
closed-set field at the top level so the adjudicator and the
markdown table can branch on it directly.

Usage::

    python -m scripts.k3a3_soak_collector \\
        --window-days 30 \\
        --out /app/memory/k3a3_soak_candidates_<UTC>.jsonl
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
    FORBIDDEN_SESSION_FIELDS, SOAK_REASON_CODES,
    _clamp_window_days, _scrub_row, _truncate,
)


async def collect(
    *,
    mongo_url: str, db_name: str,
    window_days: int, out_path: Path,
    provenance: str = "organic",
) -> dict[str, Any]:
    window_days = _clamp_window_days(window_days)
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)
    cli = AsyncIOMotorClient(mongo_url)
    db = cli[db_name]
    summary: dict[str, int] = {
        "sessions_scanned": 0,
        "sessions_with_lane_diagnostics": 0,
        "candidates_total": 0,
        "orphan_or_join_unstable_total": 0,
    }
    for code in SOAK_REASON_CODES:
        summary[f"candidates_{code}"] = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written_keys: set[str] = set()
    projection = {
        "_id": 0, "id": 1, "status": 1, "created_at": 1,
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
            for f in FORBIDDEN_SESSION_FIELDS:
                assert f not in sdoc, (
                    f"projection leaked forbidden field {f!r}"
                )
            sid = str(sdoc.get("id") or "")
            if not sid or sdoc.get("status") == "blocked_phi":
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
                join_status = str(
                    diag.get("article_join_status") or "stable"
                )
                art = await db.articles.find_one(
                    {"id": article_id},
                    {
                        "_id": 0, "id": 1, "pmid": 1, "doi": 1,
                        "pmcid": 1, "title": 1, "journal": 1,
                        "publication_year": 1, "publication_date": 1,
                        "publication_types_json": 1,
                        "is_retracted": 1,
                        "is_retraction_notice": 1,
                        "has_expression_of_concern": 1,
                        "source_ids_json": 1, "source_urls_json": 1,
                        "canonical_key": 1,
                    },
                ) or {}
                # If the persisted ``article_join_status`` is
                # ``"stable"`` but the article doc isn't joinable now,
                # downgrade the live status — the K3B Wave 1 guard
                # would refuse to enforce on the current snapshot.
                live_status = join_status
                if (
                    join_status == "stable"
                    and not art.get("id")
                ):
                    live_status = (
                        "article_removed_between_classification_and_review"
                    )
                for code in hits:
                    summary[f"candidates_{code}"] += 1
                    summary["candidates_total"] += 1
                    if live_status != "stable":
                        summary["orphan_or_join_unstable_total"] += 1
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
                        "is_retracted": bool(art.get("is_retracted")),
                        "is_retraction_notice": bool(
                            art.get("is_retraction_notice")
                        ),
                        "has_expression_of_concern": bool(
                            art.get("has_expression_of_concern")
                        ),
                        "source_ids": art.get("source_ids_json") or {},
                        "source_urls": list(
                            art.get("source_urls_json") or []
                        )[:5],
                        "provenance": provenance,
                        # Phase K3A.3 — closed-set join status.
                        "article_join_status": live_status,
                        "persisted_article_join_status": join_status,
                        "shadow_invariants": {
                            "cap_mode": "shadow",
                            "would_cap": False,
                            "would_exclude": False,
                            "cap_enforcement_mode": "disabled",
                        },
                    }
                    fh.write(json.dumps(_scrub_row(row), default=str) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--provenance", type=str, default="organic",
        choices=("organic", "k3a3_seed", "k3a2_seed", "mixed", "test"),
    )
    args = ap.parse_args()
    mongo_url = os.environ.get("MONGO_URL")
    db_name = os.environ.get("DB_NAME")
    assert mongo_url and db_name, "MONGO_URL / DB_NAME must be set"
    summary = asyncio.run(collect(
        mongo_url=mongo_url, db_name=db_name,
        window_days=args.window_days, out_path=args.out,
        provenance=args.provenance,
    ))
    print("=" * 60)
    print("K3A.3 soak collector — summary")
    print("=" * 60)
    for k in sorted(summary.keys()):
        print(f"  {k:48s} = {summary[k]}")


if __name__ == "__main__":
    main()
