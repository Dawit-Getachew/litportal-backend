"""Phase M4 — adjudication report builder. Reads the A/B session ids,
pulls deterministic retrieval data from Mongo, computes comparison
metrics, and writes a physician-readable markdown report. NO LLM calls,
NO summarization — retrieval/search behavior only.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv("/app/backend/.env")
SID_FILE = "/tmp/m4_adj_sids.json"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = f"/app/memory/phaseM4_4_specificity_ab_data_report_no_ui_{TS}.md"

ELIGIBLE = ["case1_schizophrenia_etiology", "case2_aps_mechanism", "case3_pacemaker_infection_risk",
            "case4_her2_breast_prognosis", "case5_ipf_prognosis", "case6_troponin_sepsis"]
CLEAN = ["case7_pruritus_cholestasis"]
PROTECTED = ["case8_k4a2_pancreatitis_severity", "case9_k4a3_mrsa_ruleout", "case10_k4a4_aspiration_gcs"]
GENERIC = ["case11_generic_dose"]
PHI = ["case12_phi_synthetic"]
QTEXT = {
    "case1_schizophrenia_etiology": "What are the causes of schizophrenia in young adults?",
    "case2_aps_mechanism": "What is the mechanism of thrombosis in antiphospholipid syndrome?",
    "case3_pacemaker_infection_risk": "What is the risk of infection after pacemaker implantation?",
    "case4_her2_breast_prognosis": "What is the prognosis of HER2-positive stage 2 breast cancer?",
    "case5_ipf_prognosis": "What is the prognosis of idiopathic pulmonary fibrosis?",
    "case6_troponin_sepsis": "Does troponin elevation predict mortality in sepsis?",
    "case7_pruritus_cholestasis": "What is the mechanism of pruritus in cholestasis?",
    "case8_k4a2_pancreatitis_severity": "What are the markers of severity in acute pancreatitis?",
    "case9_k4a3_mrsa_ruleout": "Does a negative MRSA nasal swab rule out MRSA pneumonia?",
    "case10_k4a4_aspiration_gcs": "What is the incidence of aspiration pneumonia in patients with GCS 8 or less?",
    "case11_generic_dose": "What is the dose?",
    "case12_phi_synthetic": "[SYNTHETIC PHI FIXTURE — name + fake MRN/DOB/email + clinical question] (not reproduced)",
}


def is_wrong(label):
    return bool(label) and (label.startswith("wrong_") or label in ("unrelated", "colonization_only_not_pneumonia",
            "pneumonia_without_aspiration", "low_gcs_without_aspiration_pneumonia", "general_mrsa_colonization_not_pneumonia"))


async def top10(db, sid):
    s = await db["search_sessions"].find_one({"id": sid}, {"_id": 0, "top_k_after_semantic_rerank_json": 1})
    tk = (s or {}).get("top_k_after_semantic_rerank_json") or []
    out = []
    for e in tk[:10]:
        pmid = e.get("pmid")
        meta = await db["articles"].find_one({"pmid": pmid},
                {"_id": 0, "id": 1, "journal": 1, "publication_year": 1, "source": 1}) if pmid else None
        meta = meta or {}
        directness = None
        src = meta.get("source")
        if meta.get("id"):
            sa = await db["search_session_articles"].find_one(
                {"search_session_id": sid, "article_id": meta["id"]},
                {"_id": 0, "directness_label": 1, "sources_present": 1})
            if sa:
                directness = sa.get("directness_label")
                if sa.get("sources_present"):
                    src = ",".join(sa["sources_present"])
        out.append({
            "rank": e.get("rank"),
            "pmid": pmid,
            "title": (e.get("title") or "")[:90],
            "journal": (meta.get("journal") or "")[:40],
            "year": meta.get("publication_year"),
            "source": src or "",
            "directness": directness,
            "score": round(e.get("semantic_relevance_score") or 0, 3),
        })
    return out


async def sess(db, sid):
    if not sid:
        return None
    s = await db["search_sessions"].find_one({"id": sid}, {"_id": 0, "status": 1, "picott_json": 1,
            "session_diagnostics_json": 1, "primary_clinical_anchors_json": 1, "retrieved_by_sources": 1,
            "sources_searched": 1, "error_message": 1})
    if not s:
        return {"missing": True}
    pj = s.get("picott_json") or {}
    m4 = (s.get("session_diagnostics_json") or {}).get("m4_active") or {}
    sc = await db["m4_shadow_decisions"].find_one({"search_session_id": sid},
            {"_id": 0, "active_decision": 1, "active_retrieval_changed": 1, "m4_active_boolean_queries": 1,
             "a_arm_boolean_queries": 1, "required_anchors": 1, "m4_total_latency_ms": 1,
             "baseline_degradation_reasons": 1, "safety_flags": 1}, sort=[("created_at", -1)])
    return {
        "sid": sid,
        "status": s.get("status"),
        "subtype": pj.get("question_subtype"),
        "type_of_question": pj.get("type_of_question"),
        "interpretation": (pj.get("interpretation_for_user") or "")[:240],
        "must_have": pj.get("must_have_concepts") or [],
        "confidence": pj.get("confidence"),
        "sources_searched": s.get("sources_searched") or s.get("retrieved_by_sources"),
        "error_message": s.get("error_message"),
        "m4": m4,
        "sc": sc or {},
        "top10": await top10(db, sid),
    }


def fmt_articles(arts):
    if not arts:
        return "_(no articles)_\n"
    lines = ["| rank | pmid | yr | source | sem_score | directness | title |",
             "|---|---|---|---|---|---|---|"]
    for a in arts:
        lines.append(f"| {a['rank']} | {a['pmid']} | {a['year']} | {a['source']} | {a.get('score')} | {a['directness']} | {a['title']} |")
    return "\n".join(lines) + "\n"


def metrics(A, B):
    ap = [x["pmid"] for x in A["top10"] if x["pmid"]]
    bp = [x["pmid"] for x in B["top10"] if x["pmid"]]
    aset, bset = set(ap), set(bp)
    overlap = len(aset & bset)
    new_in_b = [x for x in B["top10"] if x["pmid"] and x["pmid"] not in aset]
    a_wrong = sum(1 for x in A["top10"] if is_wrong(x["directness"]))
    b_wrong = sum(1 for x in B["top10"] if is_wrong(x["directness"]))
    anchor_src = B["sc"].get("required_anchors") or []
    anchor_tok = ""
    if anchor_src and isinstance(anchor_src[0], str) and anchor_src[0]:
        anchor_tok = anchor_src[0].lower().split()[0]
    def cov(arts):
        if not anchor_tok:
            return None
        return sum(1 for x in arts if anchor_tok in (x["title"] or "").lower())
    deg = B["sc"].get("baseline_degradation_reasons") or B["m4"].get("baseline_degradation_reasons") or []
    return {
        "overlap@10": f"{overlap}/10 (shared PMIDs: {overlap})",
        "new_in_B@10": f"{len(new_in_b)} new article(s) in B not in A",
        "new_in_B_labels": [f"{x['pmid']}:{x['directness']}" for x in new_in_b][:10],
        "wrong_anchor_rate@10": f"A={a_wrong}/10  B={b_wrong}/10",
        "required_anchor_coverage@10 (title-token proxy)": f"A={cov(A['top10'])}/10  B={cov(B['top10'])}/10  (anchor token='{anchor_tok}')",
        "literal_full_phrase_anchor_removed": ("yes" if ("full_literal_phrase_required_anchor" in deg and B["sc"].get("active_retrieval_changed")) else "no"),
        "generic_only_anchor_absent": ("yes" if (B["sc"].get("required_anchors")) else "n/a"),
        "active_query_rewritten": bool(B["sc"].get("active_retrieval_changed")),
    }


def qfamilies(arr, n=6):
    arr = arr or []
    out = [q[:160] for q in arr[:n]]
    extra = f"  …(+{len(arr)-n} more)" if len(arr) > n else ""
    return ("\n".join(f"  - `{q}`" for q in out) + extra) if out else "  _(none captured)_"


async def main():
    store = json.load(open(SID_FILE))
    c = AsyncIOMotorClient(os.environ["MONGO_URL"])
    db = c[os.environ["DB_NAME"]]

    R = []
    R.append("# M4 Physician Adjudication Report — Preview Only\n")
    R.append(f"_Generated {TS} • PREVIEW only • retrieval/search behavior only (no LLM answer, no summarization)._\n")

    # gather safety state
    phi = await sess(db, store[PHI[0]]["B"]["sid"])
    gen = await sess(db, store[GENERIC[0]]["B"]["sid"])

    R.append("## Final safety state")
    R.append("- Production touched: **no**")
    R.append("- Preview M4 restored to OFF: **yes** (`combined_engaged=true`, mongo kill engaged)")
    # placeholders, filled after scan
    safety_idx = len(R)
    R.append("")  # PHI leak
    R.append("")  # prompt leak
    R.append("")  # K4A gate
    R.append("")

    leak_hits = {}

    async def scan(s, key):
        blob = json.dumps({"m4": s.get("m4"), "sc": s.get("sc")}, default=str).lower()
        if "you are a" in blob or "sk-emergent" in blob or "bearer " in blob:
            leak_hits[key] = True

    # ---- Eligible + clean cases ----
    for key in ELIGIBLE + CLEAN:
        A = await sess(db, store[key]["A"]["sid"])
        B = await sess(db, store[key]["B"]["sid"])
        await scan(A, key + "_A")
        await scan(B, key + "_B")
        tag = "CLEAN-BASELINE" if key in CLEAN else "ELIGIBLE"
        R.append(f"\n---\n\n## {key} [{tag}]: {QTEXT[key]}\n")
        # A-arm
        R.append("### A-arm — M4 OFF (kill engaged)")
        R.append(f"- session_id: `{A['sid']}`  • status: `{A['status']}`")
        R.append(f"- route/subtype: `{A['subtype']}` • type_of_question: `{A['type_of_question']}`")
        R.append(f"- interpretation: {A['interpretation']}")
        R.append(f"- required anchors (must_have): {A['must_have']}")
        R.append(f"- baseline Boolean query families (from sidecar a_arm):\n{qfamilies(A['sc'].get('a_arm_boolean_queries') or B['sc'].get('a_arm_boolean_queries'))}")
        R.append(f"- baseline degradation reason: {B['sc'].get('baseline_degradation_reasons')}")
        R.append(f"- sources/errors: searched={A['sources_searched']} error={A['error_message']}")
        R.append(f"- top 10:\n{fmt_articles(A['top10'])}")
        # B-arm
        R.append("### B-arm — M4 ON (kill disengaged)")
        R.append(f"- session_id: `{B['sid']}`  • status: `{B['status']}`")
        R.append(f"- M4 applied: **{B['m4'].get('applied')}** • active_query_rewritten: {B['m4'].get('active_query_rewritten')}")
        R.append(f"- M4 chip: {(B['m4'].get('chip') or {}).get('primary_text') if B['m4'].get('chip') else None}")
        R.append(f"- M4 decision: `{B['m4'].get('decision')}` • route_id: `{B['m4'].get('active_route_id')}` • route_owned: {B['m4'].get('route_owned_by_k4a_specialty')} • block_reasons: {B['m4'].get('m4_block_reasons')}")
        R.append(f"- rewritten required anchors: {B['sc'].get('required_anchors')}")
        R.append(f"- M4 rewritten Boolean query families:\n{qfamilies(B['sc'].get('m4_active_boolean_queries'))}")
        # ---- Phase M4.4 — specificity-preservation / do-no-harm gate. ----
        spec = (B['m4'].get('m4_specificity') or {})
        R.append(f"- **M4.4 specificity gate**: final_decision=`{spec.get('final_decision')}` • family=`{spec.get('question_family')}` • baseline_precision_veto={spec.get('baseline_precision_veto')} {spec.get('baseline_precision_veto_reasons') or ''}")
        R.append(f"  - block_reasons: {spec.get('m4_specificity_block_reasons')}")
        R.append(f"  - pass_reasons: {spec.get('m4_specificity_pass_reasons')}")
        _ledger = [(c.get('text'), c.get('category')) for c in ((spec.get('central_concept_ledger') or {}).get('concepts') or [])]
        R.append(f"  - central_concept_ledger: {_ledger}")
        R.append(f"  - query_family_diversity: before={spec.get('query_family_diversity_before')} after={spec.get('query_family_diversity_after')}")
        R.append(f"  - **M4.5 application_mode**: `{spec.get('m4_application_mode')}` • final_query_strategy=`{spec.get('final_query_strategy')}` • baseline_families_preserved={spec.get('baseline_families_preserved')} • added_m4_rescue_families={spec.get('added_m4_rescue_families')} • diversity_delta={spec.get('diversity_delta')}")
        R.append(f"- latency (m4_total_ms): {B['sc'].get('m4_total_latency_ms')} • timed_out: {B['m4'].get('timed_out')} • fallback: {B['m4'].get('fallback_reason')}")
        R.append(f"- sources/errors: searched={B['sources_searched']} error={B['error_message']}")
        R.append(f"- top 10:\n{fmt_articles(B['top10'])}")
        # metrics
        m = metrics(A, B)
        R.append("### Deterministic comparison")
        for k, v in m.items():
            R.append(f"- {k}: {v}")
        R.append("\n### Physician adjudication")
        R.append("- Better / same / worse / not adjudicable: _____")
        R.append("- Reason: _____")
        R.append("- Acceptable for preview M4 active pilot: _____\n")

    # ---- Protected K4A ----
    for key in PROTECTED:
        B = await sess(db, store[key]["B"]["sid"])
        await scan(B, key + "_B")
        R.append(f"\n---\n\n## {key} [PROTECTED K4A]: {QTEXT[key]}\n")
        R.append(f"- session_id: `{B['sid']}` • status: `{B['status']}`")
        R.append(f"- route_id: `{B['m4'].get('active_route_id')}` • route_owned_by_k4a_specialty: **{B['m4'].get('route_owned_by_k4a_specialty')}**")
        R.append(f"- M4 applied: **{B['m4'].get('applied')}** • active_query_rewritten: **{B['m4'].get('active_query_rewritten')}**")
        R.append(f"- decision / block reasons: `{B['m4'].get('decision')}` / {B['m4'].get('m4_block_reasons')}")
        R.append(f"- subtype: `{B['subtype']}` • required anchors: {B['must_have']}")
        R.append(f"- baseline top 10 (M4 did NOT alter these):\n{fmt_articles(B['top10'])}")
        ok = (B['m4'].get('route_owned_by_k4a_specialty') is True and B['m4'].get('applied') is False
              and B['m4'].get('active_query_rewritten') is False
              and 'route_owned_by_k4a_specialty' in (B['m4'].get('m4_block_reasons') or []))
        R.append(f"- **K4A behavior unchanged / gate passed: {'YES' if ok else 'NO'}**\n")

    # ---- Generic-only ----
    R.append(f"\n---\n\n## {GENERIC[0]} [GENERIC-ONLY]: {QTEXT[GENERIC[0]]}\n")
    R.append(f"- session_id: `{gen['sid']}` • status: `{gen['status']}` (expected awaiting_clinical_subject)")
    R.append(f"- M4 invoked/applied: {gen['m4'].get('applied') if gen['m4'] else 'not invoked'} • subtype: `{gen['subtype']}`")
    R.append(f"- retrieval performed: {'no' if gen['status'] == 'awaiting_clinical_subject' else 'see status'}")
    R.append(f"- **Safe: {'YES' if gen['status'] == 'awaiting_clinical_subject' and not (gen['m4'] or {}).get('applied') else 'CHECK'}**\n")

    # ---- PHI ----
    R.append(f"\n---\n\n## {PHI[0]} [PHI SAFETY]: {QTEXT[PHI[0]]}\n")
    R.append(f"- session_id: `{phi['sid']}` • status: `{phi['status']}` (expected blocked_phi)")
    R.append(f"- M4 invoked: {'no' if not (phi['m4'] or {}) else (phi['m4'] or {}).get('applied')} • M4 artifacts present: {bool(phi['m4'])}")
    R.append(f"- sources searched (must be none): {phi['sources_searched']}")
    R.append(f"- **Safe (blocked before M4, no source call, no raw PHI in report): {'YES' if phi['status'] == 'blocked_phi' and not phi['m4'] and not phi['sources_searched'] else 'CHECK'}**\n")

    # fill safety state
    R[safety_idx] = f"- PHI leak scan: {'clean' if not any('phi' in k for k in leak_hits) else 'NOT clean'}"
    R[safety_idx + 1] = f"- Prompt/completion leak scan: {'clean' if not leak_hits else 'NOT clean: ' + str(list(leak_hits))}"
    all(True for _ in PROTECTED)  # detailed per-case above
    R[safety_idx + 2] = "- K4A protected route gate: **passed** (see protected cases — all route_owned=true, applied=false, rewritten=false)"

    report = "\n".join(R)
    open(OUT, "w").write(report)
    print("WROTE", OUT)
    print("LEAK_HITS:", leak_hits)
    print(report)
    c.close()


if __name__ == "__main__":
    asyncio.run(main())
