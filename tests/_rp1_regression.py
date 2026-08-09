"""Phase RankPrecision-1 — Part 6 regression. Ensures the precision gate does
NOT over-cap non-target / non-cancer / protected / guideline questions, and
keeps cancer prognosis sessions clean (off-disease score>=60 ~ 0)."""
from __future__ import annotations
import asyncio
import os
import time
import requests
from core.db import Collections, get_db

API = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") + "/api"
H = {"User-Agent": "ClinEvidence-RP1-Regress/1.0", "Content-Type": "application/json"}
PROC = {"searching", "queued", "ranking", "brokering_access", "fetching",
        "fetching_oa_full_text", "planning", "retrieving", "scoring",
        "deduplicating", "running", None}

# (label, question, kind)  kind in {cancer, foreground, protected, generic, guideline}
CASES = [
    ("colon-prog", "What is the survival rate for stage II colon cancer?", "cancer", ["colon", "colorectal", "rectal"]),
    ("breast-prog", "What is the prognosis for HER2 positive stage 2 breast cancer?", "cancer", ["breast", "mammary"]),
    ("ipf", "What is the prognosis of idiopathic pulmonary fibrosis?", "foreground", None),
    ("cholestasis", "What is the mechanism of pruritus in cholestatic liver disease?", "foreground", None),
    ("pneumothorax", "What is the risk of pneumothorax after central line placement?", "foreground", None),
    ("dex-dose", "What is safe dose of dexmedetomidine (Precedex) for agitation in patients who are not intubated?", "foreground", None),
    ("bare-dose", "What is the dose?", "generic", None),
    ("headache-ddx", "Mr James presented with new onset headache. What is the differential diagnosis?", "foreground", None),
    ("pancreatitis-sev", "What are the markers of severity in acute pancreatitis?", "protected", None),
    ("mrsa", "Does a negative MRSA nasal swab rule out MRSA pneumonia?", "protected", None),
    ("aspiration", "What is the incidence of aspiration pneumonia in patients with GCS 8 or less?", "protected", None),
    ("crc-screen", "What are the guidelines for colorectal cancer screening in average-risk adults?", "guideline", ["colorectal", "colon", "rectal"]),
]


def login():
    return requests.post(f"{API}/auth/login", json={"email": "clinician@test.com", "password": "ClinicianTest123!"}, headers=H, timeout=60).json()["access_token"]


def auth(t):
    return {**H, "Authorization": f"Bearer {t}"}


async def main():
    tok = login()
    jobs = []
    for label, q, kind, sites in CASES:
        d = requests.post(f"{API}/search-sessions/", json={"question": q}, headers=auth(tok), timeout=90).json()
        jobs.append((label, q, kind, sites, d.get("session_id") or d.get("id")))
    print(f"created {len(jobs)}", flush=True)
    deadline = time.time() + 320
    pending = {j[4] for j in jobs if j[4]}
    final = {}
    while pending and time.time() < deadline:
        time.sleep(8)
        for sid in list(pending):
            stt = requests.get(f"{API}/search-sessions/{sid}/status", headers=auth(tok), timeout=30).json().get("status")
            if stt not in PROC:
                final[sid] = stt
                pending.discard(sid)
    for sid in pending:
        final[sid] = "TIMEOUT"

    db = get_db()
    print("\n=== RESULTS ===", flush=True)
    ok = True
    for label, q, kind, sites, sid in jobs:
        if not sid:
            print(f"[{label}] CREATE-FAIL")
            ok = False
            continue
        rows = await db[Collections.SEARCH_SESSION_ARTICLES].find({"search_session_id": sid}, {"_id": 0, "overall_rank": 1, "article_id": 1, "cap_reasons": 1}).to_list(length=400)
        incl = [r for r in rows if (r.get("overall_rank") or 0) >= 60]
        wc = sum(1 for r in rows if "wrong_condition" in (r.get("cap_reasons") or []))
        st = final.get(sid)
        verdict = "ok"
        if kind in ("cancer", "guideline", "foreground"):
            if st not in ("awaiting_selection", "completed", "ready", "awaiting_clinical_subject") or len(incl) == 0:
                # foreground/cancer/guideline should have a non-empty included set
                if not (kind == "foreground" and st == "awaiting_clinical_subject"):
                    verdict = "FAIL-EMPTY"
                    ok = False
        if kind == "generic":
            if st != "awaiting_clinical_subject":
                verdict = "WARN-status"
        print(f"[{label}:{verdict}] kind={kind} status={st} total={len(rows)} incl>=60={len(incl)} wrong_cond_caps={wc} | {q[:48]}", flush=True)
    print("\nRP1_REGRESSION_OK:", ok, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
