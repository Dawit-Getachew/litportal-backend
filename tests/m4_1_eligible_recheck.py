"""Re-check that non-K4A eligible degraded baselines still APPLY M4 live."""
from __future__ import annotations
import asyncio
import json
import os
import time
import requests
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv("/app/backend/.env")
API = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") + "/api"
H = {"User-Agent": "x", "Content-Type": "application/json"}


def login(e, p):
    return requests.post(f"{API}/auth/login", json={"email": e, "password": p}, headers=H, timeout=60).json()["access_token"]


def auth(t):
    return {**H, "Authorization": f"Bearer {t}"}


def main():
    clin = login("clinician@test.com", "ClinicianTest123!")
    admin = login("admin@test.com", "AdminTest123!")
    requests.post(f"{API}/admin/m4/active-kill-switch", json={"engaged": False}, headers=auth(admin), timeout=60)
    cases = [
        ("A_etiology", "What are the causes of schizophrenia in young adults?"),
        ("HER2_prognosis", "What is the prognosis for HER2 positive stage 2 breast cancer?"),
    ]
    out = {}
    for name, q in cases:
        sid = requests.post(f"{API}/search-sessions/", json={"question": q}, headers=auth(clin), timeout=90).json().get("session_id")
        t0 = time.time()
        st = None
        while time.time() - t0 < 200:
            st = requests.get(f"{API}/search-sessions/{sid}/status", headers=auth(clin), timeout=60).json().get("status")
            if st in {"awaiting_selection", "awaiting_clinical_subject", "blocked_phi", "error"}:
                break
            time.sleep(4)

        async def rd():
            c = AsyncIOMotorClient(os.environ["MONGO_URL"])
            db = c[os.environ["DB_NAME"]]
            d = await db["search_sessions"].find_one({"id": sid}, {"_id": 0, "session_diagnostics_json": 1})
            c.close()
            return ((d or {}).get("session_diagnostics_json") or {}).get("m4_active") or {}
        m4 = asyncio.run(rd())
        out[name] = {"status": st, "decision": m4.get("decision"), "applied": m4.get("applied"),
                     "rewritten": m4.get("active_query_rewritten"), "route_owned": m4.get("route_owned_by_k4a_specialty"),
                     "chip": bool(m4.get("chip"))}
        print(f"[{name}] {out[name]}")
    requests.post(f"{API}/admin/m4/active-kill-switch", json={"engaged": True}, headers=auth(admin), timeout=60)
    print("RESTORED_SAFE_OFF")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
