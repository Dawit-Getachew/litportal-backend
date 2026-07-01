"""ImpactRank-1.1 — compact M4 timeout consistency probe (READ-ONLY)."""
from __future__ import annotations
import asyncio, os, time, json
import requests
from core.db import Collections, get_db

API = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") + "/api"
H = {"User-Agent": "ClinEvidence-IR11-Probe/1.0", "Content-Type": "application/json"}
Q = "What is the prognosis of stage 3 prostate cancer?"
PROC = {"searching", "queued", "ranking", "brokering_access", "fetching",
        "fetching_oa_full_text", "planning", "retrieving", "scoring",
        "deduplicating", "running", None}


def login():
    return requests.post(f"{API}/auth/login", json={"email": "clinician@test.com", "password": "ClinicianTest123!"}, headers=H, timeout=60).json()["access_token"]


async def probe(tok, n):
    d = requests.post(f"{API}/search-sessions/", json={"question": Q}, headers={**H, "Authorization": f"Bearer {tok}"}, timeout=90).json()
    sid = d.get("session_id") or d.get("id")
    dl = time.time() + 240
    st = None
    while time.time() < dl:
        time.sleep(5)
        st = requests.get(f"{API}/search-sessions/{sid}/status", headers={**H, "Authorization": f"Bearer {tok}"}, timeout=30).json().get("status")
        if st not in PROC:
            break
    db = get_db()
    doc = await db[Collections.SEARCH_SESSIONS].find_one({"id": sid}, {"_id": 0})
    sdj = doc.get("session_diagnostics_json") or {}
    m4a = sdj.get("m4_active") or {}
    pj = doc.get("picott_json") or {}
    rows = await db[Collections.SEARCH_SESSION_ARTICLES].find({"search_session_id": sid}, {"_id": 0}).to_list(length=500)
    incl = sum(1 for r in rows if r.get("included_by_default") or (r.get("overall_rank") or 0) >= 60)
    print(f"[probe {n}] sid={sid[:8]} status={st} | m4_timed_out={m4a.get('timed_out')} m4_total_ms={m4a.get('m4_total_latency_ms')} decision={m4a.get('decision')} baseline_degraded={m4a.get('baseline_degraded')} applied={m4a.get('applied')} | picott_type={pj.get('type_of_question')} | rows={len(rows)} included={incl} | picott_ms={(sdj.get('latency_by_stage_json') or {}).get('picott_parsing')}", flush=True)
    return {"timed_out": m4a.get("timed_out"), "included": incl, "picott_type": pj.get("type_of_question")}


async def main():
    tok = login()
    for n in (1, 2):
        await probe(tok, n)


if __name__ == "__main__":
    asyncio.run(main())
