"""ImpactRank-1.1 — fresh M4-enabled prostate diagnostic (READ-ONLY).

Creates a fresh preview session for the prostate stage-3 prognosis question,
polls to terminal, and dumps the M4 interpretation state + included-set size +
cap-reason distribution so we can decide whether the included set is non-empty
under the current M4-enabled path BEFORE running R2A/ImpactRank.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import requests

from core.db import Collections, get_db

API = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") + "/api"
H = {"User-Agent": "ClinEvidence-ImpactRank11/1.0", "Content-Type": "application/json"}
Q = os.environ.get("IMPACTRANK_PROSTATE_Q", "What is the prognosis of stage 3 prostate cancer?")
PROC = {"searching", "queued", "ranking", "brokering_access", "fetching",
        "fetching_oa_full_text", "planning", "retrieving", "scoring",
        "deduplicating", "running", None}


def login(e, p):
    return requests.post(f"{API}/auth/login", json={"email": e, "password": p}, headers=H, timeout=60).json()["access_token"]


def auth(t):
    return {**H, "Authorization": f"Bearer {t}"}


async def main():
    tok = login("clinician@test.com", "ClinicianTest123!")
    d = requests.post(f"{API}/search-sessions/", json={"question": Q}, headers=auth(tok), timeout=90).json()
    sid = d.get("session_id") or d.get("id")
    print("sid:", sid, "init:", d.get("status"), flush=True)
    deadline = time.time() + 240
    st = None
    while time.time() < deadline:
        time.sleep(5)
        st = requests.get(f"{API}/search-sessions/{sid}/status", headers=auth(tok), timeout=30).json().get("status")
        if st not in PROC:
            break
    print("terminal status:", st, flush=True)

    db = get_db()
    doc = await db[Collections.SEARCH_SESSIONS].find_one({"id": sid}, {"_id": 0})
    sdj = doc.get("session_diagnostics_json") or {}
    cqm = doc.get("clinical_question_map_json") or {}
    m4a = sdj.get("m4_active") or {}
    print("\n=== M4 ACTIVE ===")
    print(json.dumps({k: m4a.get(k) for k in (
        "mode", "triggered", "trigger_reasons", "accepted", "validation_status",
        "active_retrieval_changed", "decision", "rejection_reasons",
        "active_route_id", "timed_out", "final_strategy", "strategy_label")},
        default=str, indent=2), flush=True)
    print("\n=== CLINICAL QUESTION MAP ===")
    print(json.dumps({k: cqm.get(k) for k in (
        "type_of_question", "question_subtype", "clinical_problem", "stage",
        "population", "outcome", "outcomes", "specialty")}, default=str, indent=2), flush=True)
    print("interpreted_display:", json.dumps(doc.get("interpreted_question_display_json"), default=str)[:500], flush=True)

    rows = await db[Collections.SEARCH_SESSION_ARTICLES].find(
        {"search_session_id": sid}, {"_id": 0}).sort("overall_rank", -1).to_list(length=500)
    incl_default = [r for r in rows if r.get("included_by_default")]
    score_ge60 = [r for r in rows if (r.get("overall_rank") or 0) >= 60]
    cap_dist = {}
    for r in rows:
        for c in (r.get("cap_reasons") or []):
            cap_dist[c] = cap_dist.get(c, 0) + 1
    print("\n=== INCLUDED SET ===")
    print("total_rows:", len(rows), "| included_by_default:", len(incl_default),
          "| score>=60:", len(score_ge60), flush=True)
    print("cap_reason_distribution:", json.dumps(cap_dist, default=str), flush=True)
    print("top 8 by score:", flush=True)
    for r in rows[:8]:
        art = await db[Collections.ARTICLES].find_one({"id": r.get("article_id")}, {"_id": 0, "title": 1, "pmid": 1})
        print(f"  score={r.get('overall_rank')} incl={r.get('included_by_default')} caps={r.get('cap_reasons')} | {((art or {}).get('title') or '')[:80]}", flush=True)
    print("\nSID_FOR_REUSE:", sid, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
