"""Phase K3B Wave 1 Step 1.6 — Review-runner orchestrator.

End-to-end review pipeline:

1. Read a soak-collector JSONL artifact (produced by
   ``scripts/k3b_wave1_step1_soak_collector.py``).
2. For each WBE=true row, look up the live ``articles`` row to
   re-fetch authoritative metadata (publication_types,
   is_retracted, is_retraction_notice, has_expression_of_concern,
   pmid/doi/pmcid). Re-confirms `article_join_status` live.
3. Run the Wave-1 adjudicator
   (``scripts.k3b_wave1_step1_6_verdict_adjudicator.adjudicate_wave1``)
   on each row.
4. Write a closed-set CSV to a path of the caller's choice. Every
   string cell is scrubbed against the deep leak guard before write.

The runner does NOT mutate any DB row. It is read-only.

Usage::

    python -m scripts.k3b_wave1_step1_6_review_runner \\
        --jsonl /app/memory/k3b_wave1_step1_soak_<UTC>.jsonl \\
        --csv /app/memory/k3b_wave1_step1_6_review_<UTC>.csv

Hard invariant: ``CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE`` must be in
``{"disabled", "shadow"}`` at start-up. ``"active"`` raises
``AssertionError`` before any I/O.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from scripts.k3b_wave1_step1_6_verdict_adjudicator import (  # noqa: E402
    adjudicate_wave1,
)
from scripts.k3b_wave1_step1_soak_collector import (  # noqa: E402
    _deep_scrub, _is_safe_str, _read_enforcement_mode,
)
from services.cap_enforcement_guard import (  # noqa: E402
    compute_article_join_status,
)


# Closed-set CSV header — every column documented in the Step 1.6
# spec. No PHI / question text / emails / raw prompts / completions /
# evidence text / uploads / credentials / secrets.
CSV_HEADER: tuple[str, ...] = (
    "row_id",
    "session_id",
    "article_id",
    "pmid",
    "doi",
    "title",
    "journal",
    "publication_date",
    "source_provenance",
    "cap_reason_codes",
    "would_have_been_excluded_reason_codes",
    "article_join_status",
    "source_metadata_signal",
    "verdict",
    "verdict_basis",
    "detector_defect",
)


def _truncate(s: Any, n: int) -> str:
    """Truncate a string to ``n`` chars and scrub forbidden tokens."""
    if s is None:
        return ""
    t = str(s)
    if not _is_safe_str(t):
        return "[REDACTED]"
    return t if len(t) <= n else (t[: n - 1] + "…")


async def _enrich_row(db, row: dict[str, Any]) -> dict[str, Any]:
    """Re-fetch live article metadata + join status for one row."""
    article_id = str(row.get("article_id") or "")
    art = (await db.articles.find_one(
        {"id": article_id},
        {
            "_id": 0,
            "id": 1, "pmid": 1, "doi": 1, "pmcid": 1,
            "title": 1, "journal": 1,
            "publication_year": 1, "publication_date": 1,
            "publication_types_json": 1,
            "is_retracted": 1, "is_retraction_notice": 1,
            "has_expression_of_concern": 1,
            "canonical_key": 1,
        },
    )) or {}
    ssa = await db.search_session_articles.find_one(
        {
            "search_session_id": str(row.get("session_id") or ""),
            "article_id": article_id,
        },
        {"_id": 0, "id": 1},
    )
    live_join = compute_article_join_status(
        article_doc=art if art else None,
        search_session_article_doc=ssa,
        classification_canonical_key=art.get("canonical_key"),
    )
    enriched = dict(row)
    enriched["publication_types"] = list(
        art.get("publication_types_json") or row.get("publication_types") or []
    )
    enriched["is_retracted"] = bool(
        art.get("is_retracted")
        if "is_retracted" in art
        else row.get("is_retracted")
    )
    enriched["is_retraction_notice"] = bool(
        art.get("is_retraction_notice")
        if "is_retraction_notice" in art
        else row.get("is_retraction_notice")
    )
    enriched["has_expression_of_concern"] = bool(
        art.get("has_expression_of_concern")
        if "has_expression_of_concern" in art
        else row.get("has_expression_of_concern")
    )
    enriched["article_join_status"] = live_join
    # Live ``cap_reason_codes`` from the soak row are kept as-is. The
    # collector already populates them. We deliberately do NOT re-read
    # them from the diagnostic blob to keep the adjudication
    # reproducible against the JSONL.
    return enriched


async def run_review(
    *,
    mongo_url: str,
    db_name: str,
    jsonl_in: Path,
    csv_out: Path,
) -> dict[str, Any]:
    _read_enforcement_mode()  # asserts disabled|shadow only.
    cli = AsyncIOMotorClient(mongo_url)
    db = cli[db_name]
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    verdict_counter: Counter[str] = Counter()
    basis_counter: Counter[str] = Counter()
    defect_counter: Counter[str] = Counter()
    join_counter: Counter[str] = Counter()
    wbe_reason_codes: Counter[str] = Counter()
    provenance_counter: Counter[str] = Counter()

    rows_out: list[dict[str, Any]] = []
    row_id = 0
    with jsonl_in.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # The soak collector ONLY writes WBE=true rows. Sanity
            # assert anyway.
            sig = row.get("shadow_invariants") or {}
            assert sig.get("cap_mode") == "shadow"
            row_id += 1
            enriched = await _enrich_row(db, row)
            verdict = adjudicate_wave1(enriched)
            verdict_counter[verdict["verdict"]] += 1
            basis_counter[verdict["verdict_basis"]] += 1
            if verdict.get("detector_defect"):
                defect_counter[verdict["detector_defect"]] += 1
            join_counter[
                enriched.get("article_join_status", "stable")
            ] += 1
            for c in enriched.get(
                "would_have_been_excluded_reason_codes",
            ) or []:
                if isinstance(c, str):
                    wbe_reason_codes[c] += 1
            provenance_counter[str(row.get("provenance") or "organic")] += 1
            rows_out.append({
                "row_id": row_id,
                "session_id": str(row.get("session_id") or ""),
                "article_id": str(row.get("article_id") or ""),
                "pmid": str(enriched.get("pmid") or ""),
                "doi": str(enriched.get("doi") or ""),
                "title": _truncate(enriched.get("title"), 180),
                "journal": _truncate(enriched.get("journal"), 120),
                "publication_date": _truncate(
                    enriched.get("publication_date"), 32,
                ),
                "source_provenance": str(row.get("provenance") or "organic"),
                "cap_reason_codes": json.dumps(sorted({
                    str(c) for c in (row.get("cap_reason_codes") or [])
                    if isinstance(c, str)
                })),
                "would_have_been_excluded_reason_codes": json.dumps(sorted({
                    str(c) for c in (
                        row.get("would_have_been_excluded_reason_codes")
                        or []
                    )
                    if isinstance(c, str)
                })),
                "article_join_status": enriched.get(
                    "article_join_status", "stable",
                ),
                "source_metadata_signal": _truncate(
                    verdict.get("signal"), 240,
                ),
                "verdict": verdict["verdict"],
                "verdict_basis": verdict["verdict_basis"],
                "detector_defect": (
                    verdict.get("detector_defect") or ""
                ),
            })
    cli.close()

    # Deep-scrub every row before writing the CSV.
    rows_out_safe = [_deep_scrub(r) for r in rows_out]

    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_HEADER))
        w.writeheader()
        for r in rows_out_safe:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER})

    summary = {
        "rows_reviewed": row_id,
        "verdict_counts": dict(verdict_counter),
        "verdict_basis_counts": dict(basis_counter),
        "detector_defect_counts": dict(defect_counter),
        "article_join_status_counts": dict(join_counter),
        "wbe_reason_code_counts": dict(wbe_reason_codes),
        "provenance_counts": dict(provenance_counter),
        "csv_path": str(csv_out),
        "jsonl_path": str(jsonl_in),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()
    mongo_url = os.environ.get("MONGO_URL")
    db_name = os.environ.get("DB_NAME")
    assert mongo_url and db_name, (
        "MONGO_URL / DB_NAME must be set"
    )
    summary = asyncio.run(run_review(
        mongo_url=mongo_url, db_name=db_name,
        jsonl_in=args.jsonl, csv_out=args.csv,
    ))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
