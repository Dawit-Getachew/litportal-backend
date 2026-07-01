"""Phase M4.4 — focused live B-arm verification (PREVIEW only).

Disengages the Mongo kill switch, runs the 6 ELIGIBLE + clean cases with
M4 ON, captures the m4_active.m4_specificity gate outcome per case, then
re-engages the kill switch (SAFE/OFF). NO production. Restores SAFE/OFF.
"""
from __future__ import annotations

import json
import os
import time

import requests
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

load_dotenv("/app/backend/.env")
API = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") + "/api"
H = {"User-Agent": "ClinEvidence-M4.4-Verify/1.0", "Content-Type": "application/json"}

CASES = [
    ("case1_schizophrenia_etiology", "What are the causes of schizophrenia in young adults?"),
    ("case2_aps_mechanism", "What is the mechanism of thrombosis in antiphospholipid syndrome?"),
    ("case4_her2_breast_prognosis", "What is the prognosis of HER2-positive stage 2 breast cancer?"),
    ("case5_ipf_prognosis", "What is the prognosis of idiopathic pulmonary fibrosis?"),
    ("case6_troponin_sepsis", "Does troponin elevation predict mortality in sepsis?"),
    ("case7_pruritus_cholestasis", "What is the mechanism of pruritus in cholestasis?"),
]


def login(e, p):
    return requests.post(f"{API}/auth/login", json={"email": e, "password": p}, headers=H, timeout=60).json()["access_token"]


def auth(t):
    return {**H, "Authorization": f"Bearer {t}"}


def set_kill(t, engaged):
    r = requests.post(f"{API}/admin/m4/active-kill-switch", json={"engaged": engaged}, headers=auth(t), timeout=60)
    r.raise_for_status()
    return r.json()["state"]


def create(t, q):
    j = requests.post(f"{API}/search-sessions/", json={"question": q}, headers=auth(t), timeout=90).json()
    return j.get("session_id") or j.get("id")


def poll(t, sids, timeout_s=200):
    terminal = {"awaiting_selection", "awaiting_clinical_subject", "error", "blocked_phi", "completed", "ready"}
    t0 = time.time()
    pend = dict(sids)
    while pend and time.time() - t0 < timeout_s:
        for k, sid in list(pend.items()):
            try:
                st = requests.get(f"{API}/search-sessions/{sid}/status", headers=auth(t), timeout=30).json().get("status")
                if st in terminal:
                    pend.pop(k)
            except Exception:
                pass
        if pend:
            time.sleep(5)


async def read(sids):
    c = AsyncIOMotorClient(os.environ["MONGO_URL"])
    db = c[os.environ["DB_NAME"]]
    out = {}
    for k, sid in sids.items():
        s = await db["search_sessions"].find_one({"id": sid}, {"_id": 0, "status": 1, "session_diagnostics_json": 1})
        m4 = (s.get("session_diagnostics_json") or {}).get("m4_active") or {}
        spec = m4.get("m4_specificity") or {}
        out[k] = {
            "status": s.get("status"),
            "applied": m4.get("applied"),
            "decision": m4.get("decision"),
            "final_decision": spec.get("final_decision"),
            "family": spec.get("question_family"),
            "mode": spec.get("m4_application_mode"),
            "strategy": spec.get("final_query_strategy"),
            "added_families": spec.get("added_m4_rescue_families"),
            "div_before": spec.get("query_family_diversity_before"),
            "div_after": spec.get("query_family_diversity_after"),
            "precision_veto": spec.get("baseline_precision_veto"),
            "block_reasons": spec.get("m4_specificity_block_reasons"),
        }
    c.close()
    return out


def main():
    clin = login("clinician@test.com", "ClinicianTest123!")
    admin = login("admin@test.com", "AdminTest123!")
    print("kill disengage:", json.dumps(set_kill(admin, False)))
    try:
        sids = {}
        for k, q in CASES:
            sids[k] = create(clin, q)
            time.sleep(2)
        poll(clin, sids)
        res = asyncio.run(read(sids))
        for k in [c[0] for c in CASES]:
            print(k, "->", json.dumps(res[k]))
    finally:
        print("kill RESTORE SAFE/OFF:", json.dumps(set_kill(admin, True)))


if __name__ == "__main__":
    main()
