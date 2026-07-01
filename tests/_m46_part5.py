"""Phase M4.6 — Part 5 R2A + ImpactRank-1.1 rerun on a clean (non-empty
included set) prostate prognosis session. READ-ONLY DB joins; shadow/admin."""
from __future__ import annotations
import asyncio, json, os, re, sys
import requests
from core.db import Collections, get_db

API = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") + "/api"
H = {"User-Agent": "ClinEvidence-M46-Part5/1.0", "Content-Type": "application/json"}
SID = sys.argv[1] if len(sys.argv) > 1 else None
LEAK = [r"sk-emergent", r"Bearer ", r"system prompt", r"You are an", r"MRN\s*\d",
        r"mailto:", r"api_key", r"password"]


def login(e, p):
    return requests.post(f"{API}/auth/login", json={"email": e, "password": p}, headers=H, timeout=60).json()["access_token"]


def auth(t):
    return {**H, "Authorization": f"Bearer {t}"}


def leak_scan(o):
    blob = json.dumps(o, default=str)
    return [p for p in LEAK if re.search(p, blob)]


async def main():
    admin = login("admin@test.com", "AdminTest123!")
    assert SID, "pass session id as argv[1]"

    # R2A shadow
    r = requests.post(f"{API}/admin/search-sessions/{SID}/rescue-classifier-shadow",
                      headers=auth(admin), timeout=180)
    r2a = r.json()
    print("R2A http:", r.status_code, "| leaks:", leak_scan(r2a), flush=True)
    print("R2A counts:", json.dumps({k: r2a.get(k) for k in (
        "total_rejected_reviewed", "direct_rescue_candidates", "borderline_candidates",
        "indirect_candidates", "keep_rejected_count", "enabled", "mode")}, default=str), flush=True)

    # ImpactRank-1.1 shadow
    r = requests.post(f"{API}/admin/search-sessions/{SID}/impact-enrichment-shadow",
                      headers=auth(admin), timeout=180)
    imp = r.json()
    print("\nImpact http:", r.status_code, "| leaks:", leak_scan(imp), flush=True)
    sim = imp.get("shadow_rank_simulation") or {}
    print("Impact headline:", json.dumps({
        "articles_total": imp.get("articles_total"),
        "articles_enriched": imp.get("articles_enriched"),
        "boost_eligible_count": imp.get("boost_eligible_count"),
        "max_impact_boost": imp.get("max_impact_boost"),
        "impact_reason_counts": imp.get("impact_reason_counts"),
        "icite_stats": imp.get("icite_stats"),
        "included_count": sim.get("included_count"),
        "moved_up": len(sim.get("clinically_direct_high_impact_moved_up") or []),
        "wrong_lane_capped": len(sim.get("wrong_lane_high_citation_remain_capped") or []),
        "no_clinician_visible_change": imp.get("no_clinician_visible_change"),
    }, default=str, indent=2), flush=True)

    print("\n=== TOP 10 impact-enriched candidates ===")
    for r_ in (imp.get("impact_enriched_articles") or [])[:10]:
        print(f"  boost={r_.get('impact_boost_simulated')} applied={r_.get('impact_applied')} "
              f"reason={r_.get('impact_reason')} rcr={r_.get('icite_rcr')} apt={r_.get('icite_apt')} "
              f"oa={r_.get('openalex_cited_by_count')} score={r_.get('current_relevance_score')} "
              f"r2a={r_.get('r2a_rescue_label')} | {(r_.get('title') or '')[:64]}", flush=True)

    print("\n=== wrong-lane high-citation correctly BLOCKED (boost 0) ===")
    for r_ in (sim.get("wrong_lane_high_citation_remain_capped") or [])[:8]:
        print(f"  reason={r_.get('impact_reason')} cited={r_.get('citation_count') or r_.get('openalex_cited_by_count')} "
              f"boost={r_.get('impact_boost_simulated')} | {(r_.get('title') or '')[:64]}", flush=True)

    # DB: included vs rejected counts
    db = get_db()
    rows = await db[Collections.SEARCH_SESSION_ARTICLES].find(
        {"search_session_id": SID}, {"_id": 0, "overall_rank": 1, "included_by_default": 1,
         "exclusion_reason": 1, "cap_reasons": 1}).to_list(length=600)
    incl60 = sum(1 for r_ in rows if (r_.get("overall_rank") or 0) >= 60)
    rej = sum(1 for r_ in rows if (r_.get("overall_rank") or 0) < 60 or r_.get("exclusion_reason"))
    inclbd = sum(1 for r_ in rows if r_.get("included_by_default"))
    print("\n=== DB included/rejected ===")
    print(f"total={len(rows)} score>=60={incl60} included_by_default={inclbd} rejected/low={rej}", flush=True)

    doc = await db[Collections.SEARCH_SESSIONS].find_one({"id": SID}, {"_id": 0, "picott_json": 1})
    pj = doc.get("picott_json") or {}
    print("picott type_of_question:", pj.get("type_of_question"), "| subtype:", pj.get("question_subtype"),
          "| scope:", pj.get("question_scope"), flush=True)
    print("picott condition must_haves:", [m.get("concept") for m in (pj.get("must_have_concepts") or [])
          if m.get("role") == "condition"], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
