"""Phase M4-Prod-1 — PREVIEW dress rehearsal (manual driver, not pytest).

Runs the operator's controlled M4 pilot ENTIRELY in PREVIEW. No production
access, no production credentials. Drives the real async pipeline via the
clinician API and the admin M4 runtime kill switch, then reads the compact
M4 decision from session_diagnostics_json.m4_active (+ the m4_shadow_decisions
sidecar) for each case. Restores the preview kill switch to SAFE/OFF (engaged)
at the end.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import requests
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv("/app/backend/.env")

BASE = os.environ["REACT_APP_BACKEND_URL"].rstrip("/")
API = f"{BASE}/api"
H = {"User-Agent": "ClinEvidence-M4-Rehearsal/1.0", "Content-Type": "application/json"}
CLIN = ("clinician@test.com", "ClinicianTest123!")
ADMIN = ("admin@test.com", "AdminTest123!")


def login(email, pw):
    r = requests.post(f"{API}/auth/login", json={"email": email, "password": pw}, headers=H, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def auth(tok):
    return {**H, "Authorization": f"Bearer {tok}"}


def set_kill(admin_tok, engaged: bool):
    r = requests.post(f"{API}/admin/m4/active-kill-switch", json={"engaged": engaged},
                      headers=auth(admin_tok), timeout=60)
    r.raise_for_status()
    return r.json()["state"]


def get_kill(admin_tok):
    r = requests.get(f"{API}/admin/m4/active-kill-switch", headers=auth(admin_tok), timeout=60)
    r.raise_for_status()
    return r.json()


def create(clin_tok, question):
    r = requests.post(f"{API}/search-sessions/", json={"question": question},
                      headers=auth(clin_tok), timeout=90)
    r.raise_for_status()
    j = r.json()
    return j.get("session_id") or j.get("id"), j.get("status")


def poll_status(clin_tok, sid, timeout_s=170):
    terminal = {"awaiting_selection", "awaiting_clinical_subject", "error", "blocked_phi", "completed", "ready"}
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        r = requests.get(f"{API}/search-sessions/{sid}/status", headers=auth(clin_tok), timeout=60)
        if r.status_code == 200:
            last = r.json().get("status")
            if last in terminal:
                return last
        time.sleep(4)
    return last or "timeout"


async def read_diag(sid):
    c = AsyncIOMotorClient(os.environ["MONGO_URL"])
    db = c[os.environ["DB_NAME"]]
    doc = await db["search_sessions"].find_one({"id": sid}, {"_id": 0, "status": 1, "session_diagnostics_json": 1})
    sidecar = await db["m4_shadow_decisions"].find_one(
        {"search_session_id": sid}, {"_id": 0}, sort=[("created_at", -1)])
    c.close()
    sdj = (doc or {}).get("session_diagnostics_json") or {}
    m4 = sdj.get("m4_active") or {}
    return {
        "status": (doc or {}).get("status"),
        "m4_decision": m4.get("decision"),
        "m4_applied": m4.get("applied"),
        "m4_chip": (m4.get("chip") or {}).get("primary_text") if m4.get("chip") else None,
        "baseline_deg_reasons": m4.get("baseline_degradation_reasons"),
        "sidecar": sidecar,
    }


def leak_scan(diag_blob: str) -> dict:
    b = diag_blob.lower()
    return {
        "system_prompt_phrase": ("you are a" in b),
        "emergent_key": ("sk-emergent" in b),
        "bearer": ("bearer " in b),
    }


CASES_ACTIVE = [
    ("A_etiology", "What are the causes of schizophrenia in young adults?", "applied"),
    ("F_mechanism", "What is the mechanism of thrombosis in antiphospholipid syndrome?", "applied"),
    ("G_proc_risk", "What is the risk of infection after pacemaker implantation?", "applied"),
    ("prognosis_biomarker", "What is the prognosis for HER2 positive stage 2 breast cancer?", "applied"),
    ("clean_mechanism", "What is the mechanism of pruritus in cholestatic liver disease?", "preserved"),
    ("generic_only", "What is the dose?", "clarify"),
    ("phi_block", "Mr James presented with new onset headache. What is the differential diagnosis?", "phi"),
    ("k4a2_severity", "What are the markers of severity in acute pancreatitis?", "k4a"),
    ("k4a3_ruleout", "Does a negative MRSA nasal swab rule out MRSA pneumonia?", "k4a"),
    ("k4a4_incidence", "What is the incidence of aspiration pneumonia in patients with GCS 8 or less?", "k4a"),
    ("medsafety_dexmed", "What is safe dose of dexmedetomidine (Precedex) for agitation in patients who are not intubated?", "medsafety"),
]
DORMANCY_CASE = ("dormancy_killon", "What is the prognosis for stage II prostate cancer?")


def main():
    clin = login(*CLIN)
    admin = login(*ADMIN)
    results = {}

    prior = get_kill(admin)
    print("PRIOR_KILL_STATE:", json.dumps(prior))

    # --- Phase 0/1: M4 OFF (engage Mongo kill) -> dormancy check ---
    st = set_kill(admin, True)
    print("KILL_ENGAGED (M4 OFF):", json.dumps(st))
    name, q = DORMANCY_CASE
    sid, cstatus = create(clin, q)
    final = poll_status(clin, sid, 170)
    diag = asyncio.run(read_diag(sid))
    results[name] = {"sid": sid, "create_status": cstatus, "final_status": final,
                     "m4_decision": diag["m4_decision"], "m4_applied": diag["m4_applied"],
                     "m4_chip": diag["m4_chip"]}
    print(f"[DORMANCY] {name}: status={final} m4_decision={diag['m4_decision']} applied={diag['m4_applied']} chip={diag['m4_chip']}")

    # --- Phase 2-4: temporarily activate (disengage Mongo kill) ---
    st = set_kill(admin, False)
    print("KILL_DISENGAGED (M4 ON, temporary preview activation):", json.dumps(st))

    # create all sessions, then poll
    created = {}
    for name, q, _exp in CASES_ACTIVE:
        try:
            sid, cstatus = create(clin, q)
            created[name] = (sid, cstatus, q)
            print(f"  created {name}: sid={sid} create_status={cstatus}")
        except Exception as e:
            created[name] = (None, f"create_error:{e}", q)
            print(f"  create FAILED {name}: {e}")
        time.sleep(1)

    leak_any = {}
    for name, q, exp in CASES_ACTIVE:
        sid, cstatus, _q = created[name]
        if not sid:
            results[name] = {"sid": None, "expected": exp, "final_status": cstatus}
            continue
        final = poll_status(clin, sid, 170)
        diag = asyncio.run(read_diag(sid))
        blob = json.dumps(diag, default=str)
        ls = leak_scan(blob)
        if any(ls.values()):
            leak_any[name] = ls
        results[name] = {
            "sid": sid, "expected": exp, "create_status": cstatus, "final_status": final,
            "m4_decision": diag["m4_decision"], "m4_applied": diag["m4_applied"],
            "m4_chip": diag["m4_chip"], "baseline_deg_reasons": diag["baseline_deg_reasons"],
            "leak": ls,
        }
        print(f"[ACTIVE] {name} (exp={exp}): status={final} m4_decision={diag['m4_decision']} "
              f"applied={diag['m4_applied']} chip={bool(diag['m4_chip'])} deg={diag['baseline_deg_reasons']} leak={any(ls.values())}")

    # --- Phase 6: restore SAFE/OFF (engage Mongo kill) ---
    st = set_kill(admin, True)
    print("FINAL_KILL_STATE_RESTORED (SAFE/OFF, engaged):", json.dumps(st))

    print("\n=== LEAK SUMMARY (must be empty):", json.dumps(leak_any), "===")
    print("\n=== FULL RESULTS JSON ===")
    print(json.dumps(results, indent=2, default=str)[:9000])


if __name__ == "__main__":
    main()
