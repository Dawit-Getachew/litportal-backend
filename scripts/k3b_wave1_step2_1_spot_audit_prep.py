"""Phase K3B Wave 1 Step 2.1 — Spot-audit prep script.

Samples ≤10 actively-excluded articles from the audit-trail export
and cross-checks each PMID against PubMed via the existing E-utils
connector for the ``"Retracted Publication"`` publication-type.
Emits a markdown table + CSV with a ``<TO BE FILLED BY OPERATOR>``
placeholder verdict column.

The script is read-only and respects all K3B Step 2.1 invariants:
  - No LLM judgment.
  - No publisher scraping.
  - PubMed via existing connector only.
  - No PHI / question text / emails / raw prompts / completions /
    evidence text in output. Closed-set columns only.

Usage::

    python -m scripts.k3b_wave1_step2_1_spot_audit_prep \\
        --api-url http://localhost:8001 \\
        --admin-email admin@test.com \\
        --admin-password AdminTest123! \\
        --window-days 7 \\
        --out-base /app/memory/k3b_wave1_step2_1_spot_audit_<UTC>
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Closed-set verdict_basis_candidate enum (mirror of the K3B Wave 1
# Step 1.6 adjudicator basis set, narrowed for spot-audit usage).
VERDICT_BASIS_CANDIDATES = (
    "pubmed_retracted_publication_type",
    "pubmed_retraction_in_link",
    "source_metadata_missing_or_ambiguous",
    "article_join_unstable_no_verdict_possible",
)


async def _login_admin(
    api: str, email: str, password: str,
) -> str:
    async with httpx.AsyncClient(timeout=20.0) as cli:
        r = await cli.post(
            f"{api}/api/auth/login",
            json={"email": email, "password": password},
        )
        r.raise_for_status()
        return r.json()["access_token"]


async def _fetch_audit_export(
    api: str, token: str, window_days: int,
) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=60.0) as cli:
        r = await cli.get(
            f"{api}/api/admin/k3b-wave1-active-exclusions/export"
            f"?window_days={window_days}&format=json",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()


_PUB_TYPE_RETRACTED_RE = re.compile(
    r"retracted publication", re.IGNORECASE,
)


async def _pubmed_verifies_retraction(pmid: str) -> tuple[bool, str]:
    """Cross-check the PMID against PubMed E-utils for the
    ``"Retracted Publication"`` publication-type. Returns
    ``(verifies, verdict_basis_candidate)``.

    No LLM, no scraping. Uses the EFetch endpoint with the public
    free-tier rate (no API key required for low-volume probes).
    """
    if not pmid:
        return False, "source_metadata_missing_or_ambiguous"
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={pmid}&retmode=xml"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as cli:
            r = await cli.get(url)
            if r.status_code != 200:
                return (
                    False, "source_metadata_missing_or_ambiguous",
                )
            text = r.text
    except Exception:  # noqa: BLE001
        return False, "source_metadata_missing_or_ambiguous"
    # Cheap regex scan — we don't need full XML parsing for the
    # closed-set check. "PublicationType" tags surface the
    # canonical PubMed strings.
    if _PUB_TYPE_RETRACTED_RE.search(text):
        return True, "pubmed_retracted_publication_type"
    # PubMed "Retraction Of Publication" / "Retraction of Publication"
    # would appear here too; we count it as the secondary basis.
    if re.search(r"retraction of publication", text, re.IGNORECASE):
        return True, "pubmed_retraction_in_link"
    return False, "source_metadata_missing_or_ambiguous"


def _truncate(s: Any, n: int) -> str:
    if s is None:
        return ""
    t = str(s)
    return t if len(t) <= n else (t[: n - 1] + "…")


async def _build_table(
    api: str, admin_email: str, admin_password: str,
    window_days: int, sample_cap: int,
) -> list[dict[str, Any]]:
    token = await _login_admin(api, admin_email, admin_password)
    records = await _fetch_audit_export(api, token, window_days)
    # Bias toward organic; otherwise label seeded explicitly. The
    # audit export already carries the ``provenance`` field; this
    # function just picks the first ``sample_cap`` entries with
    # organic preference.
    organic = [r for r in records if r.get("provenance") == "organic"]
    seeded = [r for r in records if r.get("provenance") != "organic"]
    sample = (organic + seeded)[:sample_cap]
    rows: list[dict[str, Any]] = []
    for idx, r in enumerate(sample, start=1):
        pmid = r.get("pmid") or ""
        verifies, basis_candidate = await _pubmed_verifies_retraction(
            str(pmid),
        )
        rows.append({
            "row_id": idx,
            "session_id": r.get("session_id") or "",
            "article_id": r.get("article_id") or "",
            "pmid": str(pmid) if pmid else "",
            "doi": r.get("doi") or "",
            "title": _truncate(r.get("title"), 180),
            "journal": _truncate(r.get("journal"), 120),
            "publication_year": r.get("publication_year") or "",
            "source_signal": r.get("source_signal") or "",
            "source_provenance": r.get("provenance") or "organic",
            "article_join_status": (
                r.get("article_join_status") or "stable"
            ),
            "verdict_basis_candidate": basis_candidate,
            "pubmed_verifies_retraction": verifies,
            "verdict": "<TO BE FILLED BY OPERATOR>",
        })
    return rows


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    header = [
        "row_id", "session_id", "article_id", "pmid", "doi",
        "title", "journal", "publication_year",
        "source_signal", "source_provenance",
        "article_join_status", "verdict_basis_candidate",
        "pubmed_verifies_retraction", "verdict",
    ]
    lines = [
        "# K3B Wave 1 Step 2.1 — Spot-Audit Table",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Sample size: {len(rows)}",
        "",
        (
            "Closed-set verdict (operator-filled): "
            "`true_positive_retracted` | `false_positive` | "
            "`unclear_requires_source_review`"
        ),
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in rows:
        cells = []
        for h in header:
            v = r.get(h, "")
            if isinstance(v, bool):
                v = "true" if v else "false"
            s = str(v).replace("|", "\\|").replace("\n", " ")
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    header = [
        "row_id", "session_id", "article_id", "pmid", "doi",
        "title", "journal", "publication_year",
        "source_signal", "source_provenance",
        "article_join_status", "verdict_basis_candidate",
        "pubmed_verifies_retraction", "verdict",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--api-url", default="http://localhost:8001",
    )
    ap.add_argument(
        "--admin-email",
        default=os.environ.get(
            "K3B_SPOT_AUDIT_ADMIN_EMAIL", "admin@test.com",
        ),
    )
    ap.add_argument(
        "--admin-password",
        default=os.environ.get(
            "K3B_SPOT_AUDIT_ADMIN_PASSWORD", "AdminTest123!",
        ),
    )
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--sample-cap", type=int, default=10)
    ap.add_argument(
        "--out-base", type=Path, required=True,
        help="Base path (no extension). Will write .md and .csv.",
    )
    args = ap.parse_args()

    rows = asyncio.run(
        _build_table(
            api=args.api_url.rstrip("/"),
            admin_email=args.admin_email,
            admin_password=args.admin_password,
            window_days=args.window_days,
            sample_cap=args.sample_cap,
        )
    )
    out_md = args.out_base.with_suffix(".md")
    out_csv = args.out_base.with_suffix(".csv")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(rows, out_md)
    _write_csv(rows, out_csv)
    summary = {
        "rows": len(rows),
        "out_md": str(out_md),
        "out_csv": str(out_csv),
        "organic": sum(
            1 for r in rows if r["source_provenance"] == "organic"
        ),
        "seeded": sum(
            1 for r in rows if r["source_provenance"] != "organic"
        ),
        "verifies_retraction_true": sum(
            1 for r in rows if r["pubmed_verifies_retraction"]
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
