"""Phase 5A — SearchBenchmarkService.

Runs a curated set of canonical clinical-question benchmark cases through
the search pipeline (or a mocked pipeline for deterministic CI) and reports
top-k metrics.

Each ``BenchmarkCase`` carries:
- a sanitized question (NEVER contains PHI)
- expected ``type_of_question`` (or set of acceptable values)
- gold PMIDs (the article retrieval should surface at least one of these
  near the top of the ranked list)
- negative concepts: substrings/terms whose articles should NOT dominate the
  top-10 unless a waiver condition holds
- waiver concepts: when present in the same article, the negatives are
  forgiven (e.g., an article that mentions BOTH CTA and milrinone is fine)
- retrieval thresholds: ``minimum_gold_in_top_10``, ``minimum_gold_in_top_20``,
  ``max_negative_in_top_10`` — used to compute the boolean ``retrieval_pass``.

The output structure is purely diagnostic — passes/fails are NOT used as
hard test failures by themselves. Tests assert specific behaviors per case.
"""
from __future__ import annotations

import re
from typing import Any

from services.clinical_question_parser import clinical_question_parser
from services.concept_block_query_builder import concept_block_query_builder
from services.picott_ranker import picott_aware_ranker
from services.query_compiler import boolean_query_compiler
from services.search_benchmark_common import BenchmarkCase
from services.search_benchmark_fixtures import FIXTURES_BY_CASE
from services.search_benchmark_phase5c import PHASE5C_PROVISIONAL_CASES

# Phase Q1 — ``BenchmarkCase`` now lives in
# ``services.search_benchmark_common`` to break the circular import with
# ``services.search_benchmark_phase5c``. We re-export it here so existing
# importers (`from services.search_benchmark import BenchmarkCase`) keep
# working unchanged.
__all__ = ["BenchmarkCase", "BENCHMARK_CASES", "search_benchmark_service",
           "SearchBenchmarkService"]


# Phase Q1 — ``BenchmarkCase`` is defined in
# ``services.search_benchmark_common`` and re-exported above.
# Search this file for "Q1" if you arrived here from the dataclass.


# ---------------------------------------------------------------------------
# Initial benchmark cases (Task 13).
# ---------------------------------------------------------------------------
BENCHMARK_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        id="milrinone_sah",
        question=(
            "What is the dose of milrinone for patients with acute "
            "subarchnoid hemorrhage?"
        ),
        description=(
            "Pharmacology/dosing question in neurocritical care. "
            "Imaging-only SAH papers (CTA / DSA / bone-subtraction CTA) "
            "must not dominate unless they also discuss milrinone."
        ),
        expected_type_of_question={"dosing_protocol", "pharmacology"},
        expected_specialty="neurology",
        gold_pmids=[
            "33480639", "22528278", "27842615", "33470361", "33682040",
            "18239182", "11322432", "34478028", "32217799",
        ],
        must_have_anchors=["milrinone", "subarachnoid hemorrhage"],
        negative_concepts=[
            "CT angiography", "CTA", "computed tomographic angiography",
            "bone-subtraction CTA", "bone subtraction",
            "digital subtraction angiography", "DSA",
        ],
        negative_waiver_concepts=["milrinone"],
        negative_title_patterns=[
            r"\bCTA\b", r"\bbone[- ]subtraction\b",
            r"\bCT angiography\b", r"\bDSA\b",
            r"\b(transcranial )?Doppler\b", r"\bMR(A)? angiography\b",
        ],
        minimum_gold_in_top_10=3,
        minimum_gold_in_top_20=6,
        max_negative_in_top_10=2,
        required_intervention_anchor="milrinone",
        expected_query_features={
            "mesh_descriptors": {
                "pubmed": ["Milrinone", "Subarachnoid Hemorrhage"],
                "europe_pmc": ["Milrinone", "Subarachnoid Hemorrhage"],
            },
            "clinical_queries_category": "therapy",
            "must_exclude_terms": [
                "difference between", "what is the dose of",
                "what is the role of",
            ],
        },
    ),
    BenchmarkCase(
        id="ddimer_pe_lowrisk",
        question=(
            "What is the diagnostic accuracy of D-dimer for pulmonary "
            "embolism in low-risk adults?"
        ),
        expected_type_of_question={"diagnosis"},
        must_have_anchors=["D-dimer", "pulmonary embolism"],
        gold_pmids=["91001", "91002"],
        placeholder_gold=True,
        status="needs_gold_set",
        minimum_gold_in_top_10=1,
        minimum_gold_in_top_20=2,
        required_intervention_anchor="D-dimer",
        expected_query_features={
            "mesh_descriptors": {
                "pubmed": ["Pulmonary Embolism"],
                "europe_pmc": ["Pulmonary Embolism"],
            },
            "clinical_queries_category": "diagnosis",
        },
    ),
    BenchmarkCase(
        id="apixaban_warfarin_afib",
        question=(
            "Is apixaban better than warfarin for stroke prevention in "
            "nonvalvular atrial fibrillation?"
        ),
        expected_type_of_question={"therapy", "prevention"},
        must_have_anchors=["apixaban", "atrial fibrillation"],
        gold_pmids=["55001", "55002"],
        placeholder_gold=True,
        status="needs_gold_set",
        minimum_gold_in_top_10=1,
        minimum_gold_in_top_20=2,
        required_intervention_anchor="apixaban",
    ),
    BenchmarkCase(
        id="nsaid_aki_ckd_older",
        question=(
            "Do NSAIDs cause acute kidney injury in older adults with "
            "chronic kidney disease?"
        ),
        expected_type_of_question={"etiology_harm"},
        must_have_anchors=["NSAID", "acute kidney injury"],
        gold_pmids=["61001", "61002"],
        placeholder_gold=True,
        status="needs_gold_set",
        minimum_gold_in_top_10=1,
        minimum_gold_in_top_20=2,
        required_intervention_anchor="NSAID",
    ),
    BenchmarkCase(
        id="septic_shock_background",
        question="What is septic shock?",
        description="Pure background question — should route to overview/guideline sources.",
        expected_type_of_question={"background"},
        must_have_anchors=["septic shock"],
        gold_pmids=["71001", "71002"],
        placeholder_gold=True,
        status="needs_gold_set",
        minimum_gold_in_top_10=1,
        minimum_gold_in_top_20=2,
        required_intervention_anchor="septic shock",
    ),
    BenchmarkCase(
        id="colorectal_screening_guideline",
        question=(
            "What are the guidelines for colorectal cancer screening in "
            "average-risk adults?"
        ),
        expected_type_of_question={"guideline", "prevention"},
        must_have_anchors=["colorectal cancer"],
        gold_pmids=["81001", "81002"],
        placeholder_gold=True,
        status="needs_gold_set",
        minimum_gold_in_top_10=1,
        minimum_gold_in_top_20=2,
        required_intervention_anchor="colorectal cancer",
    ),
    # Phase 5B — AMI hemoglobin / transfusion threshold case.
    BenchmarkCase(
        id="ami_hemoglobin_transfusion",
        question=(
            "What is the difference between high hemoglobin target and low "
            "hemoglobin target in patients that are having acute myocardial "
            "infarction?"
        ),
        description=(
            "Phase 5B — transfusion threshold comparison in AMI/anemia. "
            "MINT and REALITY trials should rank top. Cardiac surgery / "
            "ICU / trauma / GI-bleeding transfusion-threshold trials and "
            "generic hemoglobin/AMI papers must NOT dominate top-10."
        ),
        expected_type_of_question={"therapy"},
        expected_specialty="cardiology",
        gold_pmids=[
            "37952133", "33560322", "39714935", "39475958",  # primary
            "15824203", "38408702", "39221566",              # secondary
        ],
        must_have_anchors=[
            "liberal transfusion strategy",
            "restrictive transfusion strategy",
            "acute myocardial infarction",
            "anemia",
        ],
        negative_concepts=[
            "cardiac surgery transfusion threshold",
            "ICU transfusion threshold",
            "trauma transfusion threshold",
            "GI bleeding transfusion threshold",
            "hemoglobin A1c",
        ],
        # Decoys with these patterns in their titles are flagged when the
        # article ALSO lacks the AMI/ACS context — i.e., no waiver concept.
        negative_waiver_concepts=[
            "acute myocardial infarction", "AMI", "ACS",
            "acute coronary syndrome",
        ],
        negative_title_patterns=[
            r"\bcardiac surgery\b",
            r"\bICU transfusion\b",
            r"\btrauma transfusion\b",
            r"\b(?:upper )?gastrointestinal bleeding\b",
            r"\bGI bleeding\b",
            r"\bhemoglobin A1c\b|\bHbA1c\b",
            r"\bchronic kidney disease anemia\b",
        ],
        minimum_gold_in_top_10=2,
        minimum_gold_in_top_20=4,
        max_negative_in_top_10=2,
        required_intervention_anchor="liberal transfusion strategy",
        expected_query_features={
            "mesh_descriptors": {
                "pubmed": ["Myocardial Infarction", "Anemia"],
                "europe_pmc": ["Myocardial Infarction", "Anemia"],
            },
            "trial_acronyms": ["MINT", "REALITY"],
            "clinical_queries_category": "therapy",
            "must_exclude_terms": [
                "difference between", "in patients that are",
                "what is the role of",
            ],
        },
    ),
]


# ---------------------------------------------------------------------------
# Benchmark harness — runs parsing + concept-block query generation +
# (optional) ranker post-process against caller-supplied retrieval data.
# ---------------------------------------------------------------------------


async def _run_parse_only(case: BenchmarkCase) -> dict[str, Any]:
    parsed = await clinical_question_parser.parse(case.question)
    picott = parsed.get("picott_json") or {}
    return {
        "case_id": case.id,
        "question": case.question,
        "parser_output": {
            "clinical_question_type": parsed.get("clinical_question_type"),
            "specialty": parsed.get("specialty"),
        },
        "picott": picott,
        "spelling_corrections": parsed.get("spelling_corrections") or [],
    }


def _emit_concept_block_queries(picott: dict[str, Any]) -> list[dict[str, Any]]:
    if not picott or not picott.get("must_have_concepts"):
        return []
    out: list[dict[str, Any]] = []
    for q in concept_block_query_builder.build(picott, sources=["europe_pmc", "pubmed"]):
        out.append({
            "source": q.source,
            "query_family": q.query_family,
            "query_string": q.query_string,
            "mandatory_anchors": q.mandatory_anchors,
            "rationale": q.rationale,
            "max_results": q.max_results,
        })
    return out


def _emit_phase5d_queries(picott: dict[str, Any]) -> list[dict[str, Any]]:
    """Phase 5D — return compiled queries from BooleanQueryCompilerService.

    Returns ``CompiledQuery`` records in a plain-dict shape so the benchmark
    response is JSON-serializable.
    """
    if not picott or not picott.get("must_have_concepts"):
        return []
    out: list[dict[str, Any]] = []
    try:
        for q in boolean_query_compiler.compile(
            picott, sources=["pubmed", "europe_pmc"],
        ):
            out.append({
                "source": q.source,
                "query_family": q.query_family,
                "query_string": q.query_string,
                "mandatory_anchors": q.mandatory_anchors,
                "rationale": q.rationale,
                "max_results": q.max_results,
                "field_tags_used": list(q.field_tags_used),
                "mesh_terms_used": list(q.mesh_terms_used),
                "cq_category": q.cq_category,
                "cq_scope": q.cq_scope,
                "syntax_valid": q.syntax_validation.valid,
                "syntax_errors": list(q.syntax_validation.errors),
            })
    except Exception:  # noqa: BLE001
        # Fail open — benchmark response will surface absent features
        # via the ``query_features_pass`` flag below.
        return []
    return out


def _evaluate_query_features(
    case: BenchmarkCase,
    compiled_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Phase 5D — evaluate ``expected_query_features`` against compiled queries.

    Supported feature keys (see ``BenchmarkCase.expected_query_features``):
      mesh_descriptors:           {source: [...]}  — must appear in the
                                  mesh_keyword family for the source.
      trial_acronyms:             [...] — must appear in the
                                  trial_acronym family on at least one source.
      clinical_queries_category:  expected cq_category on the
                                  clinical_queries family.
      must_include_terms:         substring(s) that must appear somewhere.
      must_exclude_terms:         substring(s) that must NOT appear anywhere
                                  (STOP_PHRASES leak detector).
    Returns a diagnostics dict with a closed-set of strings — no PHI.
    """
    features = case.expected_query_features or {}
    if not features:
        return {"checked": False}

    failures: list[str] = []
    passes: list[str] = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for q in compiled_queries:
        by_family.setdefault(q.get("query_family") or "", []).append(q)

    # mesh_descriptors per source
    mesh_expected = features.get("mesh_descriptors") or {}
    if mesh_expected:
        mk_queries = by_family.get("mesh_keyword") or []
        for src, descriptors in mesh_expected.items():
            src_qs = [q for q in mk_queries if q.get("source") == src]
            joined = " ".join((q.get("query_string") or "") for q in src_qs)
            for d in descriptors or []:
                if d in joined:
                    passes.append(f"mesh:{src}:{d}")
                else:
                    failures.append(f"missing_mesh:{src}:{d}")

    # trial_acronyms
    trial_acronyms = features.get("trial_acronyms") or []
    if trial_acronyms:
        ta_queries = by_family.get("trial_acronym") or []
        joined_ta = " ".join((q.get("query_string") or "") for q in ta_queries)
        for a in trial_acronyms:
            if a in joined_ta:
                passes.append(f"trial_acronym:{a}")
            else:
                failures.append(f"missing_trial_acronym:{a}")

    # clinical_queries_category
    cq_cat_expected = features.get("clinical_queries_category")
    if cq_cat_expected is not None:
        cq_queries = by_family.get("clinical_queries") or []
        if cq_cat_expected:
            cats_seen = {q.get("cq_category") for q in cq_queries}
            if cq_cat_expected in cats_seen:
                passes.append(f"clinical_queries:{cq_cat_expected}")
            else:
                failures.append(f"missing_clinical_queries:{cq_cat_expected}")
        else:
            # Expected absence: if any cq_query exists, fail.
            if cq_queries:
                failures.append("unexpected_clinical_queries_present")
            else:
                passes.append("clinical_queries:absent_as_expected")

    # must_include_terms
    all_strings = " ".join((q.get("query_string") or "") for q in compiled_queries)
    for term in (features.get("must_include_terms") or []):
        if term and term in all_strings:
            passes.append(f"included:{term}")
        elif term:
            failures.append(f"missing_term:{term}")

    # must_exclude_terms
    lower_strings = all_strings.lower()
    for term in (features.get("must_exclude_terms") or []):
        if term and term.lower() in lower_strings:
            failures.append(f"leaked_stop_phrase:{term}")
        elif term:
            passes.append(f"excluded:{term}")

    return {
        "checked": True,
        "passes": passes,
        "failures": failures,
        "pass": not failures,
    }


def _evaluate_ranked(
    case: BenchmarkCase,
    ranked: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute top-10 / top-20 retrieval/precision/anchor metrics for a
    ranked list of articles. Each entry is in ranker-output shape:
    ``{_article: {...}, overall_rank_score, matched_must_have, ...}``.

    Emits the full Phase 5A retrieval-side response shape:
      top_10_pmids, top_20_pmids,
      gold_retrieved_count, gold_missed_count, missed_gold_pmids,
      negatives_in_top_10_count, retrieval_pass,
      diagnostic_notes (closed-set safe strings, no PHI),
      + legacy fields (gold_in_top10, top10_pmids) for BC.
    """
    top10 = ranked[:10]
    top20 = ranked[:20]
    gold_set = {str(p).strip() for p in (case.gold_pmids or []) if p}

    def _ids(arr: list[dict[str, Any]]) -> list[str]:
        return [str((r.get("_article") or {}).get("pmid") or "").strip() for r in arr]

    pmids_top10 = [x for x in _ids(top10) if x]
    pmids_top20 = [x for x in _ids(top20) if x]
    gold_in_top10 = sorted(gold_set & set(pmids_top10))
    gold_in_top20 = sorted(gold_set & set(pmids_top20))
    missed_gold = sorted(gold_set - set(pmids_top20))

    # Negative title-pattern check against top-10 — only counts as a decoy
    # when the article also lacks the required intervention anchor.
    title_patterns = [re.compile(p, re.IGNORECASE)
                       for p in (case.negative_title_patterns or [])]
    required_anchor = (case.required_intervention_anchor
                        or (case.must_have_anchors or [""])[0]).lower()
    negatives_ranked: list[str] = []
    for idx, r in enumerate(top10):
        art = r.get("_article") or {}
        title = art.get("title") or ""
        if title_patterns and any(p.search(title) for p in title_patterns):
            matched_mh = {(m or "").lower() for m in (r.get("matched_must_have") or [])}
            if required_anchor and required_anchor not in matched_mh:
                negatives_ranked.append(str(art.get("pmid") or art.get("id") or ""))

    # Pass criteria — all three must hold.
    retrieval_pass = (
        len(gold_in_top10) >= case.minimum_gold_in_top_10
        and len(gold_in_top20) >= case.minimum_gold_in_top_20
        and len(negatives_ranked) <= case.max_negative_in_top_10
    )

    # Closed-set diagnostic notes.
    notes: list[str] = []
    if required_anchor:
        present_top10 = sum(
            1 for r in top10
            if required_anchor in {(m or "").lower()
                                     for m in (r.get("matched_must_have") or [])}
        )
        notes.append(
            f"expected_must_have '{required_anchor}' present in "
            f"{present_top10}/{len(top10)} top results"
        )
    if negatives_ranked:
        for idx, r in enumerate(top10):
            art = r.get("_article") or {}
            pmid = str(art.get("pmid") or "")
            if pmid in negatives_ranked:
                cap_reasons = r.get("cap_reasons") or []
                cap_str = ",".join(cap_reasons) if cap_reasons else "no_cap_recorded"
                notes.append(
                    f"decoy ranked at position {idx + 1} (pmid={pmid}) "
                    f"with cap_reasons=[{cap_str}]"
                )
                break  # one diagnostic note about decoys is enough
    if not gold_set:
        notes.append("no gold PMIDs configured for this case")
    elif gold_in_top10:
        notes.append(f"gold PMIDs in top-10: {len(gold_in_top10)}/{len(gold_set)}")

    return {
        # ---- New Phase 5A retrieval-side response shape (required by spec). ----
        "top_10_pmids": pmids_top10,
        "top_20_pmids": pmids_top20,
        "gold_retrieved_count": len(gold_in_top20),
        "gold_missed_count": len(missed_gold),
        "missed_gold_pmids": missed_gold,
        "negatives_in_top_10_count": len(negatives_ranked),
        "negatives_in_top_10_pmids": negatives_ranked,
        "retrieval_pass": bool(retrieval_pass),
        "diagnostic_notes": notes,
        # ---- Legacy fields kept for BC with existing tests. ----
        "top10_pmids": pmids_top10,
        "top20_pmids": pmids_top20,
        "gold_in_top10": gold_in_top10,
        "gold_in_top20": gold_in_top20,
        "negatives_ranked_in_top10": negatives_ranked,
        "pass": bool(retrieval_pass),
    }


class SearchBenchmarkService:
    """Public service used by the admin endpoint and the pytest harness."""

    cases = BENCHMARK_CASES

    def list_all(self) -> list[BenchmarkCase]:
        """Phase 6C — return the full benchmark case roster.

        Combines the legacy 7-case set + the Phase 5C provisional set so
        callers (tests, recommendation engine, sweep runner) operate on
        the same authoritative list.

        Phase Q1 — the previous lazy import was needed because
        ``search_benchmark_phase5c`` used to import ``BenchmarkCase`` from
        this module. That circular dependency was broken by moving
        ``BenchmarkCase`` into ``services.search_benchmark_common``;
        the top-of-file import of ``PHASE5C_PROVISIONAL_CASES`` is now
        sufficient.
        """
        return list(BENCHMARK_CASES) + list(PHASE5C_PROVISIONAL_CASES)

    async def run_case_parse(self, case: BenchmarkCase) -> dict[str, Any]:
        """Parse-only run (no retrieval). Returns parser + picott + concept-
        block queries that *would* be generated for the case."""
        out = await _run_parse_only(case)
        picott = out["picott"]
        out["concept_block_queries"] = _emit_concept_block_queries(picott)
        # Phase 5D — emit the formal Boolean-compiler queries and evaluate
        # ``expected_query_features``.
        phase5d_queries = _emit_phase5d_queries(picott)
        out["phase5d_queries"] = phase5d_queries
        # All compiled queries (legacy + phase5d) feed the features check.
        all_compiled = list(out["concept_block_queries"]) + list(phase5d_queries)
        out["query_features"] = _evaluate_query_features(case, all_compiled)
        # Parser-level pass/fail criteria.
        type_ok = (
            not case.expected_type_of_question
            or picott.get("type_of_question") in case.expected_type_of_question
        )
        anchors_ok = True
        if case.must_have_anchors:
            mh_concepts = {
                (m.get("concept") or "").lower()
                for m in (picott.get("must_have_concepts") or [])
            }
            def _matches_any_anchor(needed: str) -> bool:
                nl = needed.lower()
                if nl in mh_concepts:
                    return True
                # Lenient match: accept singular/plural variants (e.g., NSAID
                # vs NSAIDs, statin vs statins).
                if (nl + "s") in mh_concepts or (nl.rstrip("s") in mh_concepts):
                    return True
                # Lenient match: accept any must-have concept that starts
                # with the anchor text or vice-versa.
                return any(c.startswith(nl) or nl.startswith(c) for c in mh_concepts)
            anchors_ok = all(_matches_any_anchor(a) for a in case.must_have_anchors)
        out["parser_pass"] = bool(type_ok and anchors_ok)
        out["parser_diagnostics"] = {
            "type_ok": type_ok, "anchors_ok": anchors_ok,
        }
        return out

    async def run_case_full(
        self, case: BenchmarkCase, *, mocked_articles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Full run with a mocked corpus. When ``mocked_articles`` is not
        supplied, the per-case fixture corpus from
        ``services.search_benchmark_fixtures`` is used (so the harness is
        always deterministic for CI).

        Items must be in pre-rank shape:
            {"_article": {"id", "pmid", "title", "abstract",
                           "article_type_classification", ...},
             "overall_rank_score": int, ...}
        """
        parse_out = await self.run_case_parse(case)
        # Default to the per-case fixture corpus when no explicit mock is
        # supplied. Cases without a fixture skip retrieval metrics.
        if mocked_articles is None:
            mocked_articles = FIXTURES_BY_CASE.get(case.id)
        if not mocked_articles:
            return parse_out | {
                "ranked_metrics": None,
                "skipped": "no_fixture_or_mocked_articles",
            }
        # Deep-ish copy so multiple calls don't mutate the fixture.
        ranked = [dict(r, _article=dict(r.get("_article") or {})) for r in mocked_articles]
        ranked = picott_aware_ranker.postprocess(ranked, parse_out["picott"])
        metrics = _evaluate_ranked(case, ranked)
        return parse_out | {"ranked_metrics": metrics}

    async def run_all_parse_only(self) -> dict[str, Any]:
        results = []
        for c in self.cases:
            results.append(await self.run_case_parse(c))
        passed = sum(1 for r in results if r.get("parser_pass"))
        return {
            "summary": {"total": len(results), "parser_passed": passed},
            "cases": results,
        }

    async def run_all_with_fixtures(self) -> dict[str, Any]:
        """Phase 5A — runs all 6 cases through parser + ranker against the
        per-case fixture corpus. Deterministic; safe for CI."""
        results = []
        for c in self.cases:
            results.append(await self.run_case_full(c))
        passed = sum(1 for r in results if r.get("parser_pass"))
        ret_passed = sum(
            1 for r in results
            if (r.get("ranked_metrics") or {}).get("retrieval_pass") is True
        )
        # Phase 5D — aggregate query_features results.
        feat_checked = sum(
            1 for r in results
            if (r.get("query_features") or {}).get("checked")
        )
        feat_passed = sum(
            1 for r in results
            if (r.get("query_features") or {}).get("pass")
        )
        return {
            "summary": {
                "total": len(results),
                "parser_passed": passed,
                "retrieval_passed": ret_passed,
                # Phase 5D
                "query_features_checked": feat_checked,
                "query_features_passed": feat_passed,
            },
            "cases": results,
        }


search_benchmark_service = SearchBenchmarkService()
