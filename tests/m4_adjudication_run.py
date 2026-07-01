"""Phase M4 — Physician adjudication A/B run (PREVIEW ONLY, no UI).

Phase A: kill ENGAGED (M4 OFF) -> baseline A-arm for eligible+clean cases.
Phase B: kill DISENGAGED (M4 ON) -> B-arm for eligible+clean + protected
(K4A gate) + generic-only + synthetic-PHI safety cases.
Restores kill ENGAGED (SAFE/OFF) at the end. Stores session ids to JSON.
"""
from __future__ import annotations

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv("/app/backend/.env")
API = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") + "/api"
H = {"User-Agent": "ClinEvidence-M4-Adjudication/1.0", "Content-Type": "application/json"}
SID_FILE = "/tmp/m4_adj_sids.json"

ELIGIBLE = [
    ("case1_schizophrenia_etiology", "What are the causes of schizophrenia in young adults?"),
    ("case2_aps_mechanism", "What is the mechanism of thrombosis in antiphospholipid syndrome?"),
    ("case3_pacemaker_infection_risk", "What is the risk of infection after pacemaker implantation?"),
    ("case4_her2_breast_prognosis", "What is the prognosis of HER2-positive stage 2 breast cancer?"),
    ("case5_ipf_prognosis", "What is the prognosis of idiopathic pulmonary fibrosis?"),
    ("case6_troponin_sepsis", "Does troponin elevation predict mortality in sepsis?"),
]
CLEAN = [("case7_pruritus_cholestasis", "What is the mechanism of pruritus in cholestasis?")]
PROTECTED = [
    ("case8_k4a2_pancreatitis_severity", "What are the markers of severity in acute pancreatitis?"),
    ("case9_k4a3_mrsa_ruleout", "Does a negative MRSA nasal swab rule out MRSA pneumonia?"),
    ("case10_k4a4_aspiration_gcs", "What is the incidence of aspiration pneumonia in patients with GCS 8 or less?"),
]
GENERIC = [("case11_generic_dose", "What is the dose?")]
# Synthetic PHI fixture (fake identifiers; never reported verbatim).
PHI = [("case12_phi_synthetic", "Mr James (MRN 55512345, DOB 04/12/1961, contact james.fixture@example.com) presented with new onset headache. What is the differential diagnosis?")]


def login(e, p):
    return requests.post(f"{API}/auth/login", json={"email": e, "password": p}, headers=H, timeout=60).json()["access_token"]


def auth(t):
    return {**H, "Authorization": f"Bearer {t}"}


def set_kill(t, engaged):
    r = requests.post(f"{API}/admin/m4/active-kill-switch", json={"engaged": engaged}, headers=auth(t), timeout=60)
    r.raise_for_status()
    return r.json()["state"]


def create(t, q):
    r = requests.post(f"{API}/search-sessions/", json={"question": q}, headers=auth(t), timeout=90)
    j = r.json()
    return j.get("session_id") or j.get("id"), j.get("status")


def poll_all(t, sids, timeout_s=260):
    terminal = {"awaiting_selection", "awaiting_clinical_subject", "error", "blocked_phi", "completed", "ready"}
    t0 = time.time()
    done = {}
    pending = dict(sids)
    while pending and time.time() - t0 < timeout_s:
        for key, sid in list(pending.items()):
            if not sid:
                done[key] = "no_session_id"
                pending.pop(key)
                continue
            try:
                r = requests.get(f"{API}/search-sessions/{sid}/status", headers=auth(t), timeout=30)
                if r.status_code == 200:
                    st = r.json().get("status")
                    if st in terminal:
                        done[key] = st
                        pending.pop(key)
            except Exception:
                pass
        if pending:
            time.sleep(5)
    for key in pending:
        done[key] = "timeout_poll"
    return done


def run_phase(clin, cases, arm, store):
    created = {}
    for key, q in cases:
        sid, cs = create(clin, q)
        created[f"{key}"] = sid
        store.setdefault(key, {})[arm] = {"sid": sid, "create_status": cs}
        print(f"  [{arm}] created {key}: sid={sid} create_status={cs}")
        time.sleep(2)
    final = poll_all(clin, created)
    for key, st in final.items():
        store[key][arm]["final_status"] = st
        print(f"  [{arm}] {key}: final_status={st}")
    return store


def main():
    clin = login("clinician@test.com", "ClinicianTest123!")
    admin = login("admin@test.com", "AdminTest123!")
    store = {}

    print("=== PHASE A: M4 OFF (kill ENGAGED) ===")
    print("kill:", json.dumps(set_kill(admin, True)))
    run_phase(clin, ELIGIBLE + CLEAN, "A", store)
    json.dump(store, open(SID_FILE, "w"), indent=2)

    print("=== PHASE B: M4 ON (kill DISENGAGED) ===")
    print("kill:", json.dumps(set_kill(admin, False)))
    run_phase(clin, ELIGIBLE + CLEAN + PROTECTED + GENERIC + PHI, "B", store)
    json.dump(store, open(SID_FILE, "w"), indent=2)

    print("=== RESTORE SAFE/OFF (kill ENGAGED) ===")
    print("kill:", json.dumps(set_kill(admin, True)))
    json.dump(store, open(SID_FILE, "w"), indent=2)
    print("DONE. sids ->", SID_FILE)


if __name__ == "__main__":
    main()
