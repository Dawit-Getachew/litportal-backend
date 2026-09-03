"""M4 mismatch diagnosis — single query, full decision-chain dump. READ-ONLY."""
import asyncio
import json
import time
import urllib.request
import urllib.error
from core.db import get_db, Collections

API = "https://med-evidence-qa.preview.emergentagent.com"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
PROC = {"searching","queued","ranking","brokering_access","fetching","fetching_oa_full_text",
        "planning","retrieving","scoring","deduplicating",None}
Q = "What is the prognosis for stage II prostate cancer?"

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
    d=req("POST","/api/search-sessions/",tok,{"question":Q})
    sid=d.get("session_id") or d.get("id")
    print("sid:", sid, "| init status:", d.get("status"), flush=True)
    deadline=time.time()+180
    while time.time()<deadline:
        time.sleep(5)
        if req("GET","/api/search-sessions/"+sid,tok).get("status") not in PROC:
            break
    db=get_db()
    doc=await db[Collections.SEARCH_SESSIONS].find_one({"id":sid},{"_id":0})
    sdj=doc.get("session_diagnostics_json") or {}
    cqm=doc.get("clinical_question_map_json") or {}
    anchors=(doc.get("primary_clinical_anchors_json") or {})
    print("\n===== BASELINE PARSE =====")
    print("status:", doc.get("status"))
    print("type_of_question:", cqm.get("type_of_question"), "| subtype:", cqm.get("question_subtype"))
    print("frameworks:", cqm.get("framework_list_json") or cqm.get("frameworks"))
    print("primary_anchors:", [(a.get("canonical_concept") or a.get("text"), a.get("required_for_search")) for a in (anchors.get("primary_anchors") or [])])
    print("anchor_failure_reason:", anchors.get("anchor_failure_reason"))
    print("generic_modifiers:", [ (a.get('canonical_concept') or a.get('text')) for a in (anchors.get('generic_modifiers') or [])])

    print("\n===== M4 SHADOW INTERPRETER BLOCK (m4_full_question_interpreter) =====")
    m4b = sdj.get("m4_full_question_interpreter") or {}
    print("mode:", m4b.get("mode"))
    print("triggered:", m4b.get("triggered"))
    print("trigger_reasons:", m4b.get("trigger_reasons"))
    print("validation_status:", m4b.get("validation_status"))
    print("accepted:", m4b.get("accepted"))
    print("rejection_reasons:", m4b.get("rejection_reasons"))
    print("timed_out:", m4b.get("timed_out"))
    cand = m4b.get("candidate")
    if cand:
        print("CANDIDATE:", json.dumps({k:cand.get(k) for k in ("primary_intent","frameworks","clinical_problem","stage","population","outcome","required_anchors","optional_or_ranking_only_terms","clarification_required","repair_reason")}, default=str))
    else:
        print("CANDIDATE: None")

    print("\n===== M4 ACTIVE DECISION BLOCK (m4_active) =====")
    m4a = sdj.get("m4_active") or {}
    print(json.dumps(m4a, default=str)[:1200])

    print("\n===== M4 SHADOW SIDECAR (M4_SHADOW_DECISIONS) =====")
    for phase in ["M4","M4.1.1","M4.1.3"]:
        s=await db[Collections.M4_SHADOW_DECISIONS].find_one({"search_session_id":sid,"phase":phase},{"_id":0})
        if s:
            print(f"[{phase}] keys:", sorted(s.keys()))
            print(f"[{phase}] triggered={s.get('triggered')} accepted={s.get('accepted')} decision={s.get('decision')} active_retrieval_changed={s.get('active_retrieval_changed')} rejection_reasons={s.get('rejection_reasons')} trigger_reasons={s.get('trigger_reasons')}")

    print("\n===== UI PROJECTION (what clinician sees) =====")
    print("interpreted_question_display_json:", json.dumps(doc.get("interpreted_question_display_json"), default=str)[:400])
    print("clinician m4_active (trimmed GET):", json.dumps(req("GET","/api/search-sessions/"+sid,tok).get("session_diagnostics_json",{}).get("m4_active"), default=str))

if __name__ == "__main__":
    asyncio.run(main())
