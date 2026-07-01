"""Phase M4.1 — LIVE preview smoke for the explicit K4A route-exclusion gate.

Drives the real async pipeline (clinician API) with M4 temporarily active
(kill disengaged) and asserts K4A.2/.3/.4 routes block M4 active rewriting
via the explicit gate, while safety + eligible cases behave as before.
Restores the preview kill switch to SAFE/OFF (engaged) at the end.
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
H = {"User-Agent": "ClinEvidence-M4.1-Smoke/1.0", "Content-Type": "application/json"}


def login(e, p):
    r = requests.post(f"{API}/auth/login", json={"email": e, "password": p}, headers=H, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def auth(t):
    return {**H, "Authorization": f"Bearer {t}"}


def set_kill(t, engaged):
    r = requests.post(f"{API}/admin/m4/active-kill-switch", json={"engaged": engaged}, headers=auth(t), timeout=60)
    r.raise_for_status()
    return r.json()["state"]


def create(t, q):
    r = requests.post(f"{API}/search-sessions/", json={"question": q}, headers=auth(t), timeout=90)
    r.raise_for_status()
    j = r.json()
    return j.get("session_id") or j.get("id"), j.get("status")


def poll(t, sid, timeout_s=180):
    terminal = {"awaiting_selection", "awaiting_clinical_subject", "error", "blocked_phi", "completed", "ready"}
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        r = requests.get(f"{API}/search-sessions/{sid}/status", headers=auth(t), timeout=60)
        if r.status_code == 200:
            last = r.json().get("status")
            if last in terminal:
                return last
        time.sleep(4)
    return last or "timeout"


async def read_m4(sid):
    c = AsyncIOMotorClient(os.environ["MONGO_URL"])
    db = c[os.environ["DB_NAME"]]
    d = await db["search_sessions"].find_one({"id": sid}, {"_id": 0, "status": 1, "session_diagnostics_json": 1})
    c.close()
    m4 = ((d or {}).get("session_diagnostics_json") or {}).get("m4_active") or {}
    return {
        "status": (d or {}).get("status"),
        "decision": m4.get("decision"),
        "applied": m4.get("applied"),
        "active_route_id": m4.get("active_route_id"),
        "route_owned": m4.get("route_owned_by_k4a_specialty"),
        "active_query_rewritten": m4.get("active_query_rewritten"),
        "m4_block_reasons": m4.get("m4_block_reasons"),
        "chip": bool(m4.get("chip")),
        "raw": m4,
    }


CASES = [
    ("k4a2", "What are the markers of severity in acute pancreatitis?"),
    ("k4a3", "Does a negative MRSA nasal swab rule out MRSA pneumonia?"),
    ("k4a4", "What is the incidence of aspiration pneumonia in patients with GCS 8 or less?"),
    ("generic_only", "What is the dose?"),
    ("phi", "Mr James (MRN 55512345, DOB 04/12/1961, james@mail.com) presented with new onset headache. What is the differential diagnosis?"),
    ("clean_baseline", "What is the mechanism of pruritus in cholestatic liver disease?"),
    ("eligible_degraded", "What is the prognosis for HER2 positive stage 2 breast cancer?"),
]


def main():
    clin = login("clinician@test.com", "ClinicianTest123!")
    admin = login("admin@test.com", "AdminTest123!")
    print("KILL_DISENGAGED:", json.dumps(set_kill(admin, False)))

    results = {}
    created = {}
    for name, q in CASES:
        sid, cs = create(clin, q)
        created[name] = (sid, cs, q)
        print(f"  created {name}: sid={sid} create_status={cs}")
        time.sleep(1)

    leaks = {}
    for name, q in CASES:
        sid, cs, _q = created[name]
        final = poll(clin, sid, 180)
        m4 = asyncio.run(read_m4(sid))
        blob = json.dumps(m4.get("raw"), default=str).lower()
        ls = {"prompt": "you are a" in blob, "key": "sk-emergent" in blob, "bearer": "bearer " in blob}
        if any(ls.values()):
            leaks[name] = ls
        results[name] = {"final_status": final, **{k: m4[k] for k in
                         ("decision", "applied", "active_route_id", "route_owned", "active_query_rewritten", "m4_block_reasons", "chip")}}
        print(f"[{name}] status={final} decision={m4['decision']} applied={m4['applied']} "
              f"route_id={m4['active_route_id']} route_owned={m4['route_owned']} "
              f"rewritten={m4['active_query_rewritten']} block={m4['m4_block_reasons']} chip={m4['chip']} leak={any(ls.values())}")

    print("RESTORE_SAFE_OFF:", json.dumps(set_kill(admin, True)))
    print("\nLEAKS (must be empty):", json.dumps(leaks))
    print("\nRESULTS:", json.dumps(results, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
