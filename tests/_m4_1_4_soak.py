"""Phase M4.1.4 — short preview ACTIVE M4 soak (after no-primary rescue)."""
import asyncio, json, time, urllib.request, urllib.error
from datetime import datetime, timezone
from core.db import get_db, Collections

API = None
for line in open("/app/frontend/.env").read().splitlines():
    if line.startswith("REACT_APP_BACKEND_URL"):
        API = line.split("=", 1)[1].strip().strip('"'); break
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
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

def login(e,p):
    for _ in range(4):
        t=req("POST","/api/auth/login",None,{"email":e,"password":p}).get("access_token")
        if t: return t
        time.sleep(2)
    return None

QUESTIONS = [
    ("A1","What is the prognosis of stage 1 colon cancer?"),
    ("A2","What is the survival rate for stage II colon cancer?"),
    ("A3","What is the recurrence risk after stage I colon cancer?"),
    ("A4","What is the prognosis for HER2 positive stage 2 breast cancer?"),
    ("A5","What is the prognosis of idiopathic pulmonary fibrosis?"),
    ("A6","What are the causes of schizophrenia in young adults?"),
    ("A7","What causes recurrent miscarriage?"),
    ("A8","What are causes of new onset schizophrenia in young adults?"),
    ("A9","What is the prognosis of stage 2 pancreatic cancer?"),
    ("A10","What is the survival rate for localized prostate cancer?"),
    ("B11","What are risk factors for pancreatic cancer in smokers?"),
    ("B12","What is the mechanism of pruritus in cholestatic liver disease?"),
    ("B13","What is the risk of pneumothorax after central line placement?"),
    ("B14","What are adverse effects of dexmedetomidine in ICU patients?"),
    ("B15","What are side effects of Seroquel?"),
    ("C16","What is colon cancer?"),
    ("C17","What is the dose?"),
    ("C18","Mr James presented with new onset headache. What is the differential diagnosis?"),
    ("C19","What is safe dose of dexmedetomidine (Precedex) for agitation in patients who are not intubated?"),
    ("C20","What are the markers of severity in acute pancreatitis?"),
    ("C21","Does a negative MRSA nasal swab rule out MRSA pneumonia?"),
    ("C22","What is the incidence of aspiration pneumonia in patients with GCS 8 or less?"),
    ("C23","What are the guidelines for colorectal cancer screening in average-risk adults?"),
    ("C24","How often should colonoscopy be repeated after polypectomy?"),
]

async def collect(db, sid):
    doc=await db[Collections.SEARCH_SESSIONS].find_one({"id":sid},{"_id":0})
    if not doc: return {"_missing":True}
    cqm=doc.get("clinical_question_map_json") or {}
    anchors=(doc.get("primary_clinical_anchors_json") or {}).get("primary_anchors") or []
    sdj=doc.get("session_diagnostics_json") or {}
    m4a=sdj.get("m4_active") or {}
    sh=await db[Collections.M4_SHADOW_DECISIONS].find_one({"search_session_id":sid,"phase":"M4"},{"_id":0})
    cand=(sh or {}).get("candidate") or {}
    act=await db[Collections.M4_SHADOW_DECISIONS].find_one({"search_session_id":sid,"phase":{"$in":["M4.1.1","M4.1.3"]}},{"_id":0})
    nart=await db[Collections.SEARCH_SESSION_ARTICLES].count_documents({"search_session_id":sid})
    psrc={}; fams=set(); qstr=[]
    cur=db[Collections.SEARCH_QUERIES].find({"search_session_id":sid},{"_id":0,"planner_source":1,"query_family":1,"query_type":1,"query_string":1})
    async for q in cur:
        ps=q.get("planner_source") or "?"; psrc[ps]=psrc.get(ps,0)+1
        fams.add(q.get("query_family") or q.get("query_type") or "?")
        if q.get("query_string"): qstr.append(q["query_string"])
    return {
        "status":doc.get("status"),
        "baseline":{"intent":cqm.get("type_of_question"),"framework":cqm.get("framework_list_json") or cqm.get("frameworks"),
                    "anchors":[a.get("canonical_concept") or a.get("text") for a in anchors if a.get("required_for_search")]},
        "triggered":(sh or {}).get("triggered"),"accepted":(sh or {}).get("accepted"),
        "cand":{"intent":cand.get("primary_intent"),"frameworks":cand.get("frameworks"),
                "required_anchors":cand.get("required_anchors"),"clinical_problem":cand.get("clinical_problem"),
                "stage":cand.get("stage"),"biomarker":cand.get("biomarker"),"population":cand.get("population"),
                "outcome":cand.get("outcome"),"mechanism":cand.get("mechanism"),"medication":cand.get("medication")} if cand else None,
        "applied":m4a.get("applied"),"decision":m4a.get("decision"),"rescue_path":m4a.get("rescue_path"),
        "baseline_degraded":m4a.get("baseline_degraded"),"reasons":m4a.get("baseline_degradation_reasons"),
        "chip":(m4a.get("chip") or {}).get("primary_text") if m4a.get("chip") else None,
        "active_changed":(act or {}).get("active_retrieval_changed"),
        "leak_ok":((act or {}).get("safety_flags") or {}).get("leak_scan_passed"),
        "latency_ms":(act or {}).get("latency_ms"),
        "m4_fams":(act or {}).get("m4_query_family_count"),"base_fams":(act or {}).get("baseline_query_family_count"),
        "narticles":nart,"psrc":psrc,"families":sorted(fams),
        "leak_markers":[q[:50] for q in qstr if "mr " in q.lower() or "prompt" in q.lower() or "completion" in q.lower()],
    }

async def main():
    clin=login("clinician@test.com","ClinicianTest123!")
    db=get_db()
    created=[]
    for tag,q in QUESTIONS:
        d=req("POST","/api/search-sessions/",clin,{"question":q})
        sid=d.get("session_id") or d.get("id")
        created.append({"tag":tag,"q":q,"sid":sid,"init":d.get("status"),"rc":d.get("reason_codes")})
        print(f"create {tag} status={d.get('status')} rc={d.get('reason_codes')}",flush=True)
        time.sleep(1.0)
    deadline=time.time()+540
    pending=[c for c in created if c["sid"] and c["init"] in PROC]
    while pending and time.time()<deadline:
        time.sleep(6)
        pending=[c for c in pending if req("GET","/api/search-sessions/"+c["sid"],clin).get("status") in PROC]
        print(f"polling... {len(pending)} left",flush=True)
    results=[]
    for c in created:
        rec=dict(c)
        if c["sid"]: rec["data"]=await collect(db,c["sid"])
        results.append(rec)
    ts=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json.dump(results,open(f"/tmp/m4_1_4_soak_{ts}.json","w"),indent=2,default=str)
    print("RESULTS_FILE=/tmp/m4_1_4_soak_%s.json"%ts,flush=True); print("DONE",flush=True)

if __name__ == "__main__":
    asyncio.run(main())
