"""Phase M4.1.5 — live kill-switch verification.

Verifies: kill switch ENGAGED → M4 active is dormant (applied=False) on a
degraded baseline; then DISENGAGED → M4 applies on the same degraded
baseline. Leaves the switch DISENGAGED (preview default-on end state).
"""
import asyncio
import json
import time
import urllib.request
import urllib.error
from core.db import get_db, Collections
from services.m4_active_feature_flags import (
    set_m4_mongo_kill_switch_state, is_m4_mongo_kill_engaged,
)

API = None
for line in open("/app/frontend/.env").read().splitlines():
    if line.startswith("REACT_APP_BACKEND_URL"):
        API = line.split("=", 1)[1].strip().strip('"')
        break
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
PROC = {"searching","queued","ranking","brokering_access","fetching","fetching_oa_full_text",
        "planning","retrieving","scoring","deduplicating",None}
Q = "What is the prognosis of stage 1 colon cancer?"

def req(m,p,tok=None,body=None):
    r=urllib.request.Request(API+p,data=json.dumps(body).encode() if body is not None else None,method=m)
    r.add_header("Content-Type","application/json")
    r.add_header("User-Agent",UA)
    if tok:
        r.add_header("Authorization","Bearer "+tok)
    try:
        with urllib.request.urlopen(r,timeout=60) as x:
            raw=x.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw=e.read().decode()
        try:
            return json.loads(raw) if raw.strip() else {}
        except Exception:
            return {"_raw":raw[:200]}

async def run_one(tok, db):
    d=req("POST","/api/search-sessions/",tok,{"question":Q})
    sid=d.get("session_id") or d.get("id")
    deadline=time.time()+180
    while time.time()<deadline:
        time.sleep(5)
        if req("GET","/api/search-sessions/"+sid,tok).get("status") not in PROC:
            break
    doc=await db[Collections.SEARCH_SESSIONS].find_one({"id":sid},{"_id":0})
    return (doc.get("session_diagnostics_json") or {}).get("m4_active") or {}

async def main():
    tok=req("POST","/api/auth/login",None,{"email":"clinician@test.com","password":"ClinicianTest123!"}).get("access_token")
    db=get_db()
    # ENGAGED → dormant.
    await set_m4_mongo_kill_switch_state(engaged=True, actor_user_id="m4_1_5_killcheck")
    print("kill engaged=",await is_m4_mongo_kill_engaged(),flush=True)
    m4a_on=await run_one(tok,db)
    print("ENGAGED: applied=",m4a_on.get("applied"),"decision=",m4a_on.get("decision"),flush=True)
    # DISENGAGED → applies (degraded baseline).
    await set_m4_mongo_kill_switch_state(engaged=False, actor_user_id="m4_1_5_killcheck")
    print("kill engaged=",await is_m4_mongo_kill_engaged(),flush=True)
    m4a_off=await run_one(tok,db)
    print("DISENGAGED: applied=",m4a_off.get("applied"),"decision=",m4a_off.get("decision"),flush=True)
    ok = (m4a_on.get("applied") is not True
          and m4a_on.get("decision")=="ineligible_kill_switch_engaged"
          and m4a_off.get("applied") is True and m4a_off.get("decision")=="eligible")
    print("KILL_SWITCH_VERIFY_OK=",ok,flush=True)
    print("FINAL kill engaged=",await is_m4_mongo_kill_engaged(),"(should be False = preview default-on)",flush=True)

if __name__ == "__main__":
    asyncio.run(main())
