"""Phase M4.6 — live regression: prognosis-cancer variants must get a clean
interpretation + non-empty included set; control questions must NOT false-fire
the rescue. READ-ONLY DB reads; preview only."""
from __future__ import annotations
import asyncio
import os
import time
import requests
from core.db import Collections, get_db

API = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") + "/api"
H = {"User-Agent": "ClinEvidence-M46-Regress/1.0", "Content-Type": "application/json"}
PROC = {"searching", "queued", "ranking", "brokering_access", "fetching",
        "fetching_oa_full_text", "planning", "retrieving", "scoring",
        "deduplicating", "running", None}

PROGNOSIS = [
    "What is the prognosis of stage 3 prostate cancer?",
    "What is the prognosis for stage III prostate cancer?",
    "What is the survival rate for stage II colon cancer?",
    "What is the recurrence risk after stage I colon cancer?",
    "What is the prognosis for HER2 positive stage 2 breast cancer?",
]
CONTROL = [
    "What is colon cancer?",
    "What is the dose?",
    "What are the guidelines for colorectal cancer screening in average-risk adults?",
    "What are the markers of severity in acute pancreatitis?",
]


def login():
    return requests.post(f"{API}/auth/login", json={"email": "clinician@test.com", "password": "ClinicianTest123!"}, headers=H, timeout=60).json()["access_token"]


def create(tok, q):
    d = requests.post(f"{API}/search-sessions/", json={"question": q}, headers={**H, "Authorization": f"Bearer {tok}"}, timeout=90).json()
    return d.get("session_id") or d.get("id")


def status(tok, sid):
    return requests.get(f"{API}/search-sessions/{sid}/status", headers={**H, "Authorization": f"Bearer {tok}"}, timeout=30).json().get("status")


async def main():
    tok = login()
    jobs = []  # (label, q, sid)
    for q in PROGNOSIS:
        jobs.append(("PROGNOSIS", q, create(tok, q)))
    for q in CONTROL:
        jobs.append(("CONTROL", q, create(tok, q)))
    print(f"created {len(jobs)} sessions", flush=True)

    # poll all to terminal
    deadline = time.time() + 300
    pending = {sid for _, _, sid in jobs if sid}
    final = {}
    while pending and time.time() < deadline:
        time.sleep(8)
        for sid in list(pending):
            st = status(tok, sid)
            if st not in PROC:
                final[sid] = st
                pending.discard(sid)
    for sid in pending:
        final[sid] = "TIMEOUT"

    db = get_db()
    print("\n=== RESULTS ===", flush=True)
    ok = True
    for label, q, sid in jobs:
        if not sid:
            print(f"[{label}] CREATE-FAIL | {q}")
            ok = False
            continue
        doc = await db[Collections.SEARCH_SESSIONS].find_one({"id": sid}, {"_id": 0, "picott_json": 1})
        pj = doc.get("picott_json") or {}
        rows = await db[Collections.SEARCH_SESSION_ARTICLES].find({"search_session_id": sid}, {"_id": 0, "overall_rank": 1}).to_list(length=600)
        incl = sum(1 for r in rows if (r.get("overall_rank") or 0) >= 60)
        conds = [m.get("concept") for m in (pj.get("must_have_concepts") or []) if m.get("role") == "condition"]
        toq = pj.get("type_of_question")
        sub = pj.get("question_subtype")
        applied = (sub == "stage_specific_prognosis")
        verdict = "ok"
        if label == "PROGNOSIS":
            if not applied or toq != "prognosis" or incl <= 0:
                verdict = "FAIL"
                ok = False
        else:  # CONTROL must NOT fire the rescue
            if applied:
                verdict = "FALSE-FIRE"
                ok = False
        print(f"[{label}:{verdict}] status={final.get(sid)} toq={toq} sub={sub} incl>=60={incl} cond={conds} | {q}", flush=True)
    print("\nREGRESSION_PASS:", ok, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
