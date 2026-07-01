"""Phase ImpactRank-1 — live PREVIEW verification (admin-only, SHADOW).

Fresh live verification focused on the PROSTATE STAGE III PROGNOSIS session,
plus PHI-refusal and generic-only safety cases. Drives the admin
impact-enrichment-shadow endpoint, then joins (READ-ONLY) the persisted
search_session_articles cap/rejection fields + the icite_metrics_cache to
produce the full 8-section closeout data (top-15 table, metric availability,
boost distribution, wrong-lane safety scan, leak scan).

Strictly shadow/admin-only. No production deploy/env. No clinician-visible
change (the admin endpoint never bumps updated_at). No LLM calls of any kind
(the R2A rescue classifier is intentionally NOT invoked). No connector change.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import time

import requests

from core.db import Collections, get_db
from services.impact_enrichment import (
    compute_impact_for_article,
    max_boost,
)

API = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") + "/api"
H = {"User-Agent": "ClinEvidence-ImpactRank-Verify/1.0", "Content-Type": "application/json"}
OUT = "/tmp/impactrank_report_data.json"
INCLUDED_FLOOR = 60.0

PROSTATE_Q = os.environ.get(
    "IMPACTRANK_PROSTATE_Q", "What is the prognosis of stage III prostate cancer?")
GENERIC_Q = "What is the dose?"
PHI_Q = "Mr. John Smith, MRN 12345, age 67 — prognosis of his stage III prostate cancer?"

LEAK_PATTERNS = [
    r"sk-emergent", r"Bearer ", r"system prompt", r"You are an", r"MRN\s*\d",
    r"mailto:", r"api_key", r"password", r"John Smith",
]

# Title-keyword classes for the wrong-lane safety scan (reporting only).
LANE_KEYWORDS = {
    "screening_only": ["screening", "psa screen", "early detection", "psa test"],
    "imaging_only": ["mri", "pet", "pet/ct", "psma pet", "imaging", "ct scan",
                     "ultrasound", "radiomic", "scintigraphy"],
    "molecular_mechanism_only": ["molecular", "genomic", "biomarker", "pathway",
                                 "in vitro", "cell line", "expression of", "mrna",
                                 "signaling", "mechanism", "proteomic", "mutation"],
}


def login(e, p):
    r = requests.post(f"{API}/auth/login", json={"email": e, "password": p}, headers=H, timeout=60)
    return r.json()["access_token"]


def auth(t):
    return {**H, "Authorization": f"Bearer {t}"}


def create(t, q):
    j = requests.post(f"{API}/search-sessions/", json={"question": q}, headers=auth(t), timeout=90).json()
    return j.get("session_id") or j.get("id"), j.get("status")


def poll(t, sid, timeout_s=240):
    terminal = {"awaiting_selection", "awaiting_clinical_subject", "error",
                "blocked_phi", "completed", "ready"}
    t0 = time.time()
    st = None
    while time.time() - t0 < timeout_s:
        st = requests.get(f"{API}/search-sessions/{sid}/status", headers=auth(t), timeout=30).json().get("status")
        if st in terminal:
            return st
        time.sleep(5)
    return st


def impact(t, sid):
    r = requests.post(f"{API}/admin/search-sessions/{sid}/impact-enrichment-shadow",
                      headers=auth(t), timeout=180)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text[:300]}
    return r.status_code, body


def leak_scan(obj):
    blob = json.dumps(obj, default=str)
    return [p for p in LEAK_PATTERNS if re.search(p, blob)]


def _lane_class(title: str) -> str:
    t = (title or "").lower()
    for cls, kws in LANE_KEYWORDS.items():
        if any(k in t for k in kws):
            return cls
    return "other"


async def deep_prostate_analysis(sid: str, shadow_json: dict) -> dict:
    """READ-ONLY DB join: full metric availability, boost distribution, the
    top-15 candidate table (with cap/rejection reasons), and a wrong-lane
    high-citation safety scan. Recomputes per-article impact deterministically
    (no network) to verify the endpoint output + derive full distributions."""
    db = get_db()
    sa_rows = await (
        db[Collections.SEARCH_SESSION_ARTICLES]
        .find({"search_session_id": sid}, {"_id": 0})
        .sort("overall_rank", -1).limit(200).to_list(length=200)
    )
    art_ids = [r.get("article_id") for r in sa_rows if r.get("article_id")]
    articles_by_id = {}
    async for art in db[Collections.ARTICLES].find({"id": {"$in": art_ids}}, {"_id": 0}):
        articles_by_id[art.get("id") or ""] = art

    pmids = sorted({str((articles_by_id.get(r.get("article_id")) or {}).get("pmid") or "").strip()
                    for r in sa_rows} - {""})
    cache = {}
    async for c in db["icite_metrics_cache"].find({"pmid": {"$in": pmids}}, {"_id": 0}):
        cache[str(c["pmid"])] = c.get("metrics") or {}

    # --- recompute per-article impact deterministically (no network) ---
    recomputed = []
    for r in sa_rows:
        art = articles_by_id.get(r.get("article_id"))
        if not art:
            continue
        pmid = str(art.get("pmid") or "").strip()
        m = cache.get(pmid, {}) if pmid else {}
        row = compute_impact_for_article(article_doc=art, sa_row=r, icite_metrics=m)
        # join the clinician-side cap / rejection / inclusion fields (READ ONLY)
        row["_score"] = r.get("overall_rank")
        row["_included_by_default"] = bool(r.get("included_by_default"))
        row["_exclusion_reason"] = r.get("exclusion_reason")
        row["_cap_reasons"] = r.get("cap_reasons") or []
        row["_flags"] = r.get("flags") or []
        row["_default_selection_rationale"] = r.get("default_selection_rationale")
        row["_lane_class"] = _lane_class(row.get("title"))
        recomputed.append(row)

    # --- iCite metric availability across ALL pmid-bearing articles ---
    pmid_rows = [r for r in recomputed if str((articles_by_id.get(r["article_id"]) or {}).get("pmid") or "").strip()]
    rcr_avail = sum(1 for r in recomputed if r.get("icite_rcr") is not None)
    apt_avail = sum(1 for r in recomputed if r.get("icite_apt") is not None)
    cbc_avail = sum(1 for r in recomputed if (r.get("icite_cited_by_clin") or 0) and r["icite_cited_by_clin"] > 0)
    citcount_avail = sum(1 for r in recomputed if r.get("icite_citation_count") is not None)

    # --- OpenAlex availability ---
    oa_avail = sum(1 for r in recomputed if r.get("openalex_cited_by_count") not in (None, 0))
    oa_present_any = sum(1 for r in recomputed if r.get("openalex_cited_by_count") is not None)
    doi_present = sum(1 for r in recomputed
                      if (articles_by_id.get(r["article_id"]) or {}).get("doi"))
    pmid_present = len(pmid_rows)

    # --- boost distribution ---
    boosts = [r["impact_boost_simulated"] for r in recomputed]
    eligible_boosts = [b for b in boosts if b > 0]
    blocked_reasons = {}
    for r in recomputed:
        if not r["impact_applied"]:
            blocked_reasons[r["impact_reason"]] = blocked_reasons.get(r["impact_reason"], 0) + 1

    # --- included-set membership + simulated rank delta ---
    moved_up = {m.get("article_id"): m.get("positions_gained")
                for m in ((shadow_json.get("shadow_rank_simulation") or {}).get("clinically_direct_high_impact_moved_up") or [])}

    def rank_delta(r):
        score = r.get("_score")
        if score is None or score < INCLUDED_FLOOR:
            return "n/a (not in clinician-visible included set)"
        return moved_up.get(r["article_id"], 0)

    def cap_reason_label(r):
        if r.get("_included_by_default"):
            return "included_by_default (no cap)"
        bits = []
        if r.get("_exclusion_reason"):
            bits.append(f"excl:{r['_exclusion_reason']}")
        if r.get("_cap_reasons"):
            bits.append("caps:" + ",".join(map(str, r["_cap_reasons"]))[:80])
        if r.get("_default_selection_rationale"):
            bits.append(f"sel:{r['_default_selection_rationale']}")
        if r.get("_score") is not None and r["_score"] < INCLUDED_FLOOR and not bits:
            bits.append(f"below_include_floor(score={r['_score']})")
        return "; ".join(bits) or "n/a"

    # --- top-15 impact-enriched candidates (sorted by simulated boost) ---
    top = sorted(recomputed, key=lambda r: r["impact_boost_simulated"], reverse=True)[:15]
    top15 = [{
        "pmid": r.get("pmid"),
        "title": (r.get("title") or "")[:120],
        "current_score": r.get("_score"),
        "current_cap_or_rejection_reason": cap_reason_label(r),
        "r2a_rescue_label": r.get("r2a_rescue_label"),  # None (R2A not invoked)
        "rcr": r.get("icite_rcr"),
        "apt": r.get("icite_apt"),
        "nih_percentile": r.get("icite_nih_percentile"),
        "openalex_cited_by_count": r.get("openalex_cited_by_count"),
        "icite_citation_count": r.get("icite_citation_count"),
        "impact_score_simulated": r.get("impact_score_simulated"),
        "impact_boost_simulated": r.get("impact_boost_simulated"),
        "boost_allowed": r.get("impact_applied"),
        "boost_reason": r.get("impact_reason"),
        "simulated_rank_delta": rank_delta(r),
        "lane_class": r.get("_lane_class"),
    } for r in top]

    # --- wrong-lane high-citation safety scan ---
    # High-citation rejected papers (NOT in included set) that must stay out.
    def cited(r):
        return (r.get("openalex_cited_by_count") or r.get("icite_citation_count") or 0)

    high_cite_rejected = sorted(
        [r for r in recomputed
         if (r.get("_score") is None or r["_score"] < INCLUDED_FLOOR) and cited(r) >= 50],
        key=lambda r: cited(r), reverse=True,
    )[:25]
    safety_rejected = [{
        "pmid": r.get("pmid"),
        "title": (r.get("title") or "")[:110],
        "lane_class": r.get("_lane_class"),
        "current_score": r.get("_score"),
        "cited_by": cited(r),
        "rcr": r.get("icite_rcr"),
        "impact_boost_simulated": r.get("impact_boost_simulated"),
        "boost_allowed": r.get("impact_applied"),
        "in_clinician_included_set": (r.get("_score") is not None and r["_score"] >= INCLUDED_FLOOR),
        "in_simulated_included_set": False,  # structurally impossible (see note)
        "impact_reason": r.get("impact_reason"),
    } for r in high_cite_rejected]

    # Policy-excluded / non-article in the pool
    policy_excluded = [{
        "pmid": r.get("pmid"), "title": (r.get("title") or "")[:90],
        "reason": r.get("impact_reason"), "boost": r.get("impact_boost_simulated"),
    } for r in recomputed if r.get("impact_reason") == "not_applied_policy_excluded"][:15]

    # by lane_class, count how many high-cite rejected stayed out (all of them)
    lane_safety = {}
    for r in safety_rejected:
        lane_safety.setdefault(r["lane_class"], {"count": 0, "max_boost_any": 0.0, "any_in_included": False})
        lane_safety[r["lane_class"]]["count"] += 1
        lane_safety[r["lane_class"]]["max_boost_any"] = max(
            lane_safety[r["lane_class"]]["max_boost_any"], r["impact_boost_simulated"])
        lane_safety[r["lane_class"]]["any_in_included"] |= r["in_clinician_included_set"]

    return {
        "articles_total": len(recomputed),
        "pmid_bearing_articles": pmid_present,
        "doi_present": doi_present,
        "icite": {
            "pmids_in_pool": len(pmids),
            "rcr_available": rcr_avail,
            "apt_available": apt_avail,
            "cited_by_clin_available": cbc_avail,
            "citation_count_available": citcount_avail,
        },
        "openalex": {
            "cited_by_count_present": oa_present_any,
            "cited_by_count_nonzero": oa_avail,
        },
        "boost_distribution": {
            "max_boost_ceiling": max_boost(),
            "observed_max_boost": max(boosts) if boosts else 0.0,
            "median_boost_all": round(statistics.median(boosts), 3) if boosts else 0.0,
            "median_boost_eligible": round(statistics.median(eligible_boosts), 3) if eligible_boosts else 0.0,
            "nonzero_boost_count": len(eligible_boosts),
            "zero_boost_count": len(boosts) - len(eligible_boosts),
            "blocked_reason_counts": blocked_reasons,
            "ceiling_respected": (max(boosts) if boosts else 0.0) <= max_boost() + 1e-9,
        },
        "top15": top15,
        "safety_high_cite_rejected": safety_rejected,
        "lane_safety_summary": lane_safety,
        "policy_excluded": policy_excluded,
    }


def main():
    clin = login("clinician@test.com", "ClinicianTest123!")
    admin = login("admin@test.com", "AdminTest123!")
    results = {"max_impact_boost_env": max_boost()}

    # ---- PHI refusal ----
    phi_sid, phi_status = create(clin, PHI_Q)
    if phi_sid:
        poll(clin, phi_sid, timeout_s=60)
        code, body = impact(admin, phi_sid)
        results["phi"] = {"create_status": phi_status, "impact_http": code,
                          "detail": body.get("detail"), "leaks": leak_scan(body)}
        print("PHI ->", results["phi"], flush=True)

    # ---- generic-only (no clinical subject) ----
    g_sid, g_status = create(clin, GENERIC_Q)
    if g_sid:
        st = poll(clin, g_sid, timeout_s=120)
        code, body = impact(admin, g_sid)
        results["generic"] = {"status": st, "impact_http": code,
                              "articles_total": body.get("articles_total"),
                              "articles_enriched": body.get("articles_enriched"),
                              "leaks": leak_scan(body)}
        print("GENERIC ->", results["generic"], flush=True)

    # ---- prostate stage III prognosis (main) ----
    sid, status0 = create(clin, PROSTATE_Q)
    print("prostate sid:", sid, "init:", status0, flush=True)
    st = poll(clin, sid)
    code, shadow = impact(admin, sid)
    leaks = leak_scan(shadow)
    print("prostate impact_http:", code, "| status:", st, "| leaks:", leaks, flush=True)

    headline = {
        "session_id": sid, "status": st, "impact_http": code,
        "mode": shadow.get("mode"),
        "no_clinician_visible_change": shadow.get("no_clinician_visible_change"),
        "max_impact_boost": shadow.get("max_impact_boost"),
        "articles_total": shadow.get("articles_total"),
        "articles_enriched": shadow.get("articles_enriched"),
        "boost_eligible_count": shadow.get("boost_eligible_count"),
        "impact_reason_counts": shadow.get("impact_reason_counts"),
        "icite_stats": shadow.get("icite_stats"),
        "shadow_rank_simulation_summary": {
            "included_count": (shadow.get("shadow_rank_simulation") or {}).get("included_count"),
            "moved_up_count": len((shadow.get("shadow_rank_simulation") or {}).get("clinically_direct_high_impact_moved_up") or []),
            "wrong_lane_capped_count": len((shadow.get("shadow_rank_simulation") or {}).get("wrong_lane_high_citation_remain_capped") or []),
        },
        "leaks": leaks,
    }
    deep = asyncio.get_event_loop().run_until_complete(deep_prostate_analysis(sid, shadow))
    results["prostate"] = {"headline": headline, "deep": deep}

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nWROTE", OUT, flush=True)
    all_leaks = []
    for k, v in results.items():
        if isinstance(v, dict) and v.get("leaks"):
            all_leaks.append(k)
        if isinstance(v, dict) and isinstance(v.get("headline"), dict) and v["headline"].get("leaks"):
            all_leaks.append(k)
    print("LEAK_SCAN_CLEAN:", not all_leaks, "| sessions_with_leaks:", all_leaks, flush=True)
    print("\n=== PROSTATE HEADLINE ===")
    print(json.dumps(headline, indent=2, default=str), flush=True)
    print("\n=== PROSTATE DEEP (icite/openalex/boost/safety) ===")
    print(json.dumps({k: deep[k] for k in ("articles_total", "pmid_bearing_articles",
          "doi_present", "icite", "openalex", "boost_distribution", "lane_safety_summary")},
          indent=2, default=str), flush=True)
    print("\n=== TOP 15 ===")
    print(json.dumps(deep["top15"], indent=2, default=str), flush=True)
    print("\n=== HIGH-CITE REJECTED (safety) [first 12] ===")
    print(json.dumps(deep["safety_high_cite_rejected"][:12], indent=2, default=str), flush=True)
    print("\n=== POLICY EXCLUDED ===")
    print(json.dumps(deep["policy_excluded"], indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
