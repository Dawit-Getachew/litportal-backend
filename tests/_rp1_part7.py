"""Phase RankPrecision-1 — Part 7: fresh prostate prognosis session, off-disease
score>=60 count (after gate), wrong_condition caps, R2A + ImpactRank rerun."""
from __future__ import annotations
import asyncio
import json
import os
import re
import time
import requests
from core.db import Collections, get_db

API = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") + "/api"
H = {"User-Agent": "ClinEvidence-RP1-Part7/1.0", "Content-Type": "application/json"}
Q = "What is the prognosis of stage 3 prostate cancer?"
TARGET = ["prostate cancer", "prostate carcinoma", "prostate neoplasm", "prostatic", "prostate"]
OTHER = ["colorectal", "colon ", "breast", "lung ", "pancrea", "gastric", "ovarian",
         "cervical", "bladder", "renal", "melanoma", "lymphoma", "leukemia", "esophag",
         "thyroid", "hepatocellular", "glioma", "glioblastoma"]
PROC = {"searching", "queued", "ranking", "brokering_access", "fetching",
        "fetching_oa_full_text", "planning", "retrieving", "scoring",
        "deduplicating", "running", None}
LEAK = [r"sk-emergent", r"Bearer ", r"system prompt", r"You are an", r"MRN\s*\d", r"api_key", r"password"]


def login(e, p):
    return requests.post(f"{API}/auth/login", json={"email": e, "password": p}, headers=H, timeout=60).json()["access_token"]


def auth(t):
    return {**H, "Authorization": f"Bearer {t}"}


def leak(o):
    return [p for p in LEAK if re.search(p, json.dumps(o, default=str))]


async def main():
    clin = login("clinician@test.com", "ClinicianTest123!")
    admin = login("admin@test.com", "AdminTest123!")
    d = requests.post(f"{API}/search-sessions/", json={"question": Q}, headers=auth(clin), timeout=90).json()
    sid = d.get("session_id") or d.get("id")
    print("sid:", sid, flush=True)
    dl = time.time() + 260
    st = None
    while time.time() < dl:
        time.sleep(8)
        st = requests.get(f"{API}/search-sessions/{sid}/status", headers=auth(clin), timeout=30).json().get("status")
        if st not in PROC:
            break
    print("status:", st, flush=True)

    db = get_db()
    rows = await db[Collections.SEARCH_SESSION_ARTICLES].find({"search_session_id": sid}, {"_id": 0}).sort("overall_rank", -1).to_list(length=400)
    arts = {}
    async for a in db[Collections.ARTICLES].find({"id": {"$in": [r["article_id"] for r in rows]}}, {"_id": 0, "id": 1, "title": 1, "abstract": 1}):
        arts[a["id"]] = a
    incl60 = [r for r in rows if (r.get("overall_rank") or 0) >= 60]
    wrong_cond = [r for r in rows if "wrong_condition" in (r.get("cap_reasons") or [])]
    offdz60 = []
    for r in incl60:
        a = arts.get(r["article_id"]) or {}
        tl = (a.get("title") or "").lower()
        target_in_title = any(re.search(rf"\b{re.escape(t)}\b", tl) for t in TARGET)
        other_in_title = [c for c in OTHER if c.strip() in tl]
        if other_in_title and not target_in_title:
            offdz60.append((r.get("overall_rank"), (a.get("title") or "")[:70], other_in_title, r.get("cap_reasons")))
    print("\n=== RANKER PRECISION ===")
    print(f"total_rows={len(rows)} included>=60={len(incl60)} wrong_condition_caps={len(wrong_cond)}", flush=True)
    print(f"OFF-DISEASE score>=60 (other cancer in title, prostate NOT in title) = {len(offdz60)} (was 7 before gate)", flush=True)
    for s, t, oth, caps in offdz60[:12]:
        print(f"  STILL-HIGH score={s} other={oth} caps={caps} | {t}", flush=True)

    # R2A + ImpactRank
    r2a = requests.post(f"{API}/admin/search-sessions/{sid}/rescue-classifier-shadow", headers=auth(admin), timeout=180).json()
    print("\nR2A:", json.dumps({k: r2a.get(k) for k in (
        "total_rejected_reviewed", "direct_rescue_candidates", "borderline_candidates",
        "indirect_candidates")}, default=str), "leaks:", leak(r2a), flush=True)
    imp = requests.post(f"{API}/admin/search-sessions/{sid}/impact-enrichment-shadow", headers=auth(admin), timeout=180).json()
    sim = imp.get("shadow_rank_simulation") or {}
    print("\nImpactRank:", json.dumps({
        "articles_total": imp.get("articles_total"), "articles_enriched": imp.get("articles_enriched"),
        "boost_eligible_count": imp.get("boost_eligible_count"), "max_impact_boost": imp.get("max_impact_boost"),
        "impact_reason_counts": imp.get("impact_reason_counts"),
        "included_count": sim.get("included_count"),
        "moved_up": len(sim.get("clinically_direct_high_impact_moved_up") or []),
        "wrong_lane_capped": len(sim.get("wrong_lane_high_citation_remain_capped") or []),
        "no_clinician_visible_change": imp.get("no_clinician_visible_change"),
    }, default=str, indent=2), "leaks:", leak(imp), flush=True)
    print("\n=== TOP 10 impact-enriched ===")
    for r_ in (imp.get("impact_enriched_articles") or [])[:10]:
        print(f"  boost={r_.get('impact_boost_simulated')} applied={r_.get('impact_applied')} reason={r_.get('impact_reason')} "
              f"score={r_.get('current_relevance_score')} r2a={r_.get('r2a_rescue_label')} | {(r_.get('title') or '')[:58]}", flush=True)
    print("\nSID:", sid, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
