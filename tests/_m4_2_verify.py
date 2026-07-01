"""Phase M4.2 — live preview consolidation verification.

Runs the M3/M4 semantic + safety cases through the live preview pipeline
(M4 preview-active) and proves:
  * M4 applies on degraded baselines (forward semantic path),
  * clean baselines are preserved, generic clarifies, PHI blocks,
  * MedSafety / K4A protected / K5 guideline lanes are NOT M4-applied,
  * legacy diag blocks are now DEPRECATED markers (no duplicate LLM
    semantic_repair_candidate),
  * M3.2 does NOT co-fire and writes NO new m3_2_session_decisions rows.
Preview only. Production untouched.
"""
import asyncio, json, time, urllib.request, urllib.error
from datetime import datetime, timezone
from core.db import get_db, Collections
from services.m4_active_feature_flags import is_m4_mongo_kill_engaged

API = None
for line in open("/app/frontend/.env").read().splitlines():
    if line.startswith("REACT_APP_BACKEND_URL"):
        API = line.split("=", 1)[1].strip().strip('"'); break
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
PROC = {"searching","queued","ranking","brokering_access","fetching","fetching_oa_full_text",
        "planning","retrieving","scoring","deduplicating",None}

def req(m,p,tok=None,body=None):
    r=urllib.request.Request(API+p,data=json.dumps(body).encode() if body is not None else None,method=m)
    r.add_header("Content-Type","application/json"); r.add_header("User-Agent",UA)
    if tok: r.add_header("Authorization","Bearer "+tok)
    try:
        with urllib.request.urlopen(r,timeout=60) as x:
            raw=x.read().decode(); return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw=e.read().decode()
        try: return json.loads(raw) if raw.strip() else {}
        except Exception: return {"_raw":raw[:200]}

# (tag, question, expectation)
CASES = [
    # M3/M4 semantic cases (degraded → M4 should engage; some are clean)
    ("S1","What are the causes of schizophrenia in young adults?","semantic"),
    ("S2","What is the mechanism of thrombosis in antiphospholipid syndrome?","semantic"),
    ("S3","What is the risk of infection after pacemaker implantation?","semantic"),
    ("S4","What is the prognosis of stage 1 colon cancer?","semantic"),
    ("S5","What is the survival rate for stage II colon cancer?","semantic"),
    ("S6","What is the recurrence risk after stage I colon cancer?","semantic"),
    ("S7","What is the prognosis for HER2 positive stage 2 breast cancer?","semantic"),
    ("S8","What is the prognosis of idiopathic pulmonary fibrosis?","semantic"),
    ("S9","What is the mechanism of pruritus in cholestatic liver disease?","semantic"),
    ("S10","What is the risk of pneumothorax after central line placement?","semantic"),
    # Safety / regression
    ("G1","What is the dose?","generic_clarify"),
    ("P1","Mr James presented with new onset headache. What is the differential diagnosis?","phi"),
    ("M1","What is safe dose of dexmedetomidine (Precedex) for agitation in patients who are not intubated?","medsafety"),
    ("K1","What are the markers of severity in acute pancreatitis?","k4a"),
    ("K2","Does a negative MRSA nasal swab rule out MRSA pneumonia?","k4a"),
    ("K3","What is the incidence of aspiration pneumonia in patients with GCS 8 or less?","k4a"),
    ("K5","What are the guidelines for colorectal cancer screening in average-risk adults?","k5"),
]

async def collect(db, sid):
    doc=await db[Collections.SEARCH_SESSIONS].find_one({"id":sid},{"_id":0})
    if not doc: return {"_missing":True}
    sdj=doc.get("session_diagnostics_json") or {}
    m4a=sdj.get("m4_active") or {}
    sar=sdj.get("semantic_anchor_repair") or {}
    m32=sdj.get("m3_2_active_repair") or {}
    nart=await db[Collections.SEARCH_SESSION_ARTICLES].count_documents({"search_session_id":sid})
    return {
        "status":doc.get("status"),
        "m4_applied":m4a.get("applied"),"m4_decision":m4a.get("decision"),
        "m4_rescue":m4a.get("rescue_path"),"m4_timed_out":m4a.get("timed_out"),
        "m4_total_latency_ms":m4a.get("m4_total_latency_ms"),
        # consolidation invariants:
        "sar_deprecated":sar.get("deprecated"),
        "sar_has_candidate":bool(sar.get("semantic_repair_candidate")),
        "sar_superseded_by":sar.get("superseded_by"),
        "m3_2_deprecated":m32.get("deprecated"),
        "m3_2_applied":m32.get("applied"),
        "m3_2_decision":m32.get("decision"),
        "narticles":nart,
    }

async def main():
    print("kill engaged (should be False = preview default-on):", await is_m4_mongo_kill_engaged(), flush=True)
    db=get_db()
    n_m3_2_before=await db[Collections.M3_2_SESSION_DECISIONS].count_documents({})
    n_sem_before=await db[Collections.M4_SHADOW_DECISIONS].count_documents({"phase":"M"}) \
        if hasattr(Collections,"M4_SHADOW_DECISIONS") else None
    print("m3_2_session_decisions BEFORE:", n_m3_2_before, flush=True)
    tok=req("POST","/api/auth/login",None,{"email":"clinician@test.com","password":"ClinicianTest123!"}).get("access_token")
    created=[]
    for tag,q,exp in CASES:
        d=req("POST","/api/search-sessions/",tok,{"question":q})
        sid=d.get("session_id") or d.get("id")
        created.append({"tag":tag,"q":q,"exp":exp,"sid":sid,"init":d.get("status")})
        print(f"create {tag} ({exp}) status={d.get('status')}",flush=True)
        time.sleep(0.8)
    deadline=time.time()+420
    pending=[c for c in created if c["sid"] and c["init"] in PROC]
    while pending and time.time()<deadline:
        time.sleep(6)
        pending=[c for c in pending if req("GET","/api/search-sessions/"+c["sid"],tok).get("status") in PROC]
        print(f"polling... {len(pending)} left",flush=True)
    results=[]
    for c in created:
        rec=dict(c)
        if c["sid"]: rec["data"]=await collect(db,c["sid"])
        results.append(rec)
    n_m3_2_after=await db[Collections.M3_2_SESSION_DECISIONS].count_documents({})
    ts=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out={"m3_2_before":n_m3_2_before,"m3_2_after":n_m3_2_after,"results":results}
    json.dump(out,open(f"/tmp/m4_2_verify_{ts}.json","w"),indent=2,default=str)
    print("m3_2_session_decisions AFTER:", n_m3_2_after, "(delta:", n_m3_2_after-n_m3_2_before, ")", flush=True)
    print("RESULTS_FILE=/tmp/m4_2_verify_%s.json"%ts,flush=True); print("DONE",flush=True)

if __name__ == "__main__":
    asyncio.run(main())
