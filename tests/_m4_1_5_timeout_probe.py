"""Phase M4.1.5 — live timeout-fallback probe.

Runs ONE degraded-baseline question under a tiny M4 latency budget and
verifies M4 falls back to baseline retrieval (no partial promotion, no
crash) with the closed-set reason ``m4_latency_timeout``.
Preview only. Requires M4_ACTIVE_MAX_LATENCY_MS set low + backend restart.
"""
import asyncio
import json
import time
import urllib.request
import urllib.error
from core.db import get_db, Collections

API = None
for line in open("/app/frontend/.env").read().splitlines():
    if line.startswith("REACT_APP_BACKEND_URL"):
        API = line.split("=", 1)[1].strip().strip('"')
        break
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
PROC = {"searching","queued","ranking","brokering_access","fetching","fetching_oa_full_text",
        "planning","retrieving","scoring","deduplicating",None}

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

async def main():
    tok=req("POST","/api/auth/login",None,{"email":"clinician@test.com","password":"ClinicianTest123!"}).get("access_token")
    d=req("POST","/api/search-sessions/",tok,{"question":"What is the prognosis of stage 1 colon cancer?"})
    sid=d.get("session_id") or d.get("id")
    print("created sid=",sid,"init=",d.get("status"),flush=True)
    deadline=time.time()+180
    while time.time()<deadline:
        time.sleep(5)
        st=req("GET","/api/search-sessions/"+sid,tok).get("status")
        if st not in PROC:
            break
        print("polling status=",st,flush=True)
    db=get_db()
    doc=await db[Collections.SEARCH_SESSIONS].find_one({"id":sid},{"_id":0})
    m4a=(doc.get("session_diagnostics_json") or {}).get("m4_active") or {}
    nart=await db[Collections.SEARCH_SESSION_ARTICLES].count_documents({"search_session_id":sid})
    print("FINAL status=",doc.get("status"),flush=True)
    print("m4_active.timed_out=",m4a.get("timed_out"),flush=True)
    print("m4_active.decision=",m4a.get("decision"),flush=True)
    print("m4_active.applied=",m4a.get("applied"),flush=True)
    print("m4_active.fallback_reason=",m4a.get("fallback_reason"),flush=True)
    print("m4_active.chip=",m4a.get("chip"),flush=True)
    print("narticles=",nart,flush=True)
    ok = (m4a.get("timed_out") is True and m4a.get("decision")=="m4_latency_timeout"
          and not m4a.get("applied") and m4a.get("chip") in (None,{})
          and doc.get("status")=="awaiting_selection" and nart>0)
    print("TIMEOUT_FALLBACK_OK=",ok,flush=True)

if __name__ == "__main__":
    asyncio.run(main())
