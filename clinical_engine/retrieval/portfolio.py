"""PRESS-informed query portfolio (W2.1), built only from the approved search terms.

Families per source, grouped into the three evidence lanes:

  foundational_primary  main (MeSH + title/abstract words, every "must appear"
                        concept AND-ed), original wording (the clinician's own
                        words, minimally transformed), narrow and broad
                        question-type filters
  aggregated            guidelines, systematic reviews / meta-analyses,
                        narrative reviews, Cochrane reviews (PubMed-indexed)
  recent_primary        the main search, last 5 years
  alternative           ONE bounded, justified relaxation — runs only when the
                        main search finds almost nothing (never turns the
                        condition AND treatment into OR)

The update search (newer than a verified guideline/review cutoff) is added
later by the lineage step, from the same blocks.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from clinical_engine.retrieval.run import QuerySpec
from services import engine_concept_plan as ecp

# PubMed Clinical Queries filters by question type (NLM, PubMed Help —
# "Clinical Queries filters"); see CLINICAL_QUERIES below.
CQ_CATEGORY = {
    "treatment": "therapy", "prevention": "therapy", "clinical_management": "therapy",
    "health_services": "therapy", "screening": "therapy",
    "diagnosis": "diagnosis", "diagnostic_workup": "diagnosis", "monitoring": "diagnosis",
    "harm": "etiology", "etiology_risk": "etiology", "complications": "etiology",
    "prognosis": "prognosis", "prognostic_factor": "prognosis",
    "prediction_model": "clinical_prediction_guides",
}

# NLM PubMed Clinical Queries filters (Haynes et al., revised December 2011),
# verbatim from PubMed Help — checked 2026-10-03.
CLINICAL_QUERIES: dict[str, dict[str, str]] = {
    "therapy": {
        "broad": ("((clinical[Title/Abstract] AND trial[Title/Abstract]) OR clinical trials as topic[MeSH Terms] "
                  "OR clinical trial[Publication Type] OR random*[Title/Abstract] OR random allocation[MeSH Terms] "
                  "OR therapeutic use[MeSH Subheading])"),
        "narrow": ("(randomized controlled trial[Publication Type] OR (randomized[Title/Abstract] AND "
                   "controlled[Title/Abstract] AND trial[Title/Abstract]))"),
    },
    "diagnosis": {
        "broad": ("(sensitiv*[Title/Abstract] OR sensitivity and specificity[MeSH Terms] OR diagnose[Title/Abstract] "
                  "OR diagnosed[Title/Abstract] OR diagnoses[Title/Abstract] OR diagnosing[Title/Abstract] OR "
                  "diagnosis[Title/Abstract] OR diagnostic[Title/Abstract] OR diagnosis[MeSH:noexp] OR (diagnostic "
                  "equipment[MeSH:noexp] OR diagnostic errors[MeSH:noexp] OR diagnostic imaging[MeSH:noexp] OR "
                  "diagnostic services[MeSH:noexp]) OR diagnosis, differential[MeSH:noexp] OR "
                  "diagnosis[Subheading:noexp])"),
        "narrow": "(specificity[Title/Abstract])",
    },
    "etiology": {
        "broad": ("(risk*[Title/Abstract] OR risk*[MeSH:noexp] OR (risk adjustment[MeSH:noexp] OR risk "
                  "assessment[MeSH:noexp] OR risk factors[MeSH:noexp] OR risk management[MeSH:noexp] OR risk "
                  "taking[MeSH:noexp]) OR cohort studies[MeSH Terms] OR group[Text Word] OR groups[Text Word] OR "
                  "grouped [Text Word])"),
        "narrow": ("((relative[Title/Abstract] AND risk*[Title/Abstract]) OR (relative risk[Text Word]) OR "
                   "risks[Text Word] OR cohort studies[MeSH:noexp] OR (cohort[Title/Abstract] AND "
                   "study[Title/Abstract]) OR (cohort[Title/Abstract] AND studies[Title/Abstract]))"),
    },
    "prognosis": {
        "broad": ("(incidence[MeSH:noexp] OR mortality[MeSH Terms] OR follow up studies[MeSH:noexp] OR "
                  "prognos*[Text Word] OR predict*[Text Word] OR course*[Text Word])"),
        "narrow": ("(prognos*[Title/Abstract] OR (first[Title/Abstract] AND episode[Title/Abstract]) OR "
                   "cohort[Title/Abstract])"),
    },
    "clinical_prediction_guides": {
        "broad": ("(predict*[Title/Abstract] OR predictive value of tests[MeSH Terms] OR score[Title/Abstract] OR "
                  "scores[Title/Abstract] OR scoring system[Title/Abstract] OR scoring systems[Title/Abstract] OR "
                  "observ*[Title/Abstract] OR observer variation[MeSH Terms])"),
        "narrow": "(validation[Title/Abstract] OR validate[Title/Abstract])",
    },
}

# ("Consensus Development Conference"[pt] matches nothing in PubMed today — checked
# 2026-10-03 — so consensus statements are found by title.)
_GUIDELINES_PUBMED = ('("Practice Guideline"[pt] OR "Guideline"[pt] OR guideline*[ti] OR recommendation*[ti] '
                      'OR consensus[ti])')
_GUIDELINES_EPMC = ('(PUB_TYPE:"practice guideline" OR PUB_TYPE:"guideline" OR TITLE:guideline* '
                    'OR TITLE:recommendation* OR TITLE:consensus)')
# Condition-level guideline search: guideline publication types OR "guideline" in
# the title — PubMed indexes some major guidelines (e.g. Surviving Sepsis Campaign
# 2021, PMID 34605781) only as "Journal Article".
_GUIDELINE_TYPES_PUBMED = '("Practice Guideline"[pt] OR "Guideline"[pt] OR guideline*[ti])'
_GUIDELINE_TYPES_EPMC = '(PUB_TYPE:"practice guideline" OR PUB_TYPE:"guideline" OR TITLE:guideline*)'
_SR_PUBMED = '(systematic[sb] OR "Meta-Analysis"[pt])'
_SR_EPMC = ('(PUB_TYPE:"systematic review" OR PUB_TYPE:"meta-analysis" OR TITLE:"systematic review" '
            'OR TITLE:"meta-analysis")')
_REVIEW_PUBMED = '"Review"[pt]'
_REVIEW_EPMC = 'PUB_TYPE:"review"'
_COCHRANE_PUBMED = '"Cochrane Database Syst Rev"[ta]'

# Budgets per family (records read per query).
MAX_RECORDS = {
    ("main", "pubmed"): 200, ("main", "europepmc"): 100,
    ("original_wording", "pubmed"): 50, ("original_wording", "europepmc"): 40,
    ("narrow", "pubmed"): 100, ("narrow", "europepmc"): 60,
    ("broad", "pubmed"): 100,
    ("guidelines", "pubmed"): 60, ("guidelines", "europepmc"): 40,
    ("condition_guidelines", "pubmed"): 100, ("condition_guidelines", "europepmc"): 60,
    ("systematic_reviews", "pubmed"): 60, ("systematic_reviews", "europepmc"): 40,
    ("reviews", "pubmed"): 30, ("reviews", "europepmc"): 20,
    ("cochrane", "pubmed"): 20,
    ("recent", "pubmed"): 100, ("recent", "europepmc"): 60,
    ("long_term", "pubmed"): 100, ("long_term", "europepmc"): 60,
    ("landmark", "pubmed"): 100, ("landmark", "europepmc"): 100,
    ("alternative", "pubmed"): 100, ("alternative", "europepmc"): 60,
    ("update", "pubmed"): 60, ("update", "europepmc"): 40,
}

FAMILY_LABELS = {
    "main": "Main search",
    "original_wording": "Your own wording",
    "narrow": "Narrow filter for this question type",
    "broad": "Broad filter for this question type",
    "guidelines": "Guidelines",
    "condition_guidelines": "Guidelines on the condition",
    "systematic_reviews": "Systematic reviews & meta-analyses",
    "reviews": "Reviews",
    "cochrane": "Cochrane reviews",
    "recent": "Last 5 years",
    "long_term": "Long-term outcomes (prognosis)",
    "landmark": "Most-cited papers",
    "alternative": "Wider alternative (only if the main search finds almost nothing)",
    "update": "Newer than the latest guideline or review",
}
LANE_LABELS = {
    "foundational_primary": "Studies, all years",
    "aggregated": "Guidelines & reviews",
    "recent_primary": "Recent studies",
    "alternative": "Wider alternative",
    "update": "Newer evidence",
}
_SOURCE_ORDER = {"pubmed": 0, "europepmc": 1}


def in_planned_order(ledger: list[dict[str, Any]], planned: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Executed queries in the portfolio's own order (main search first; PubMed, then
    Europe PMC), not the order they happened to finish in. Searches added during the
    run (the update search) follow, PubMed first."""
    index = {s.get("id"): i for i, s in enumerate(planned or [])}
    return sorted(ledger, key=lambda q: (index.get(q.get("spec_id"), len(index)),
                                         _SOURCE_ORDER.get(q.get("source"), len(_SOURCE_ORDER))))


_STOP = frozenset("""
a an the of in on for to with among and or by at from as into is are was were be been being does do did can could
should would will may might must what which who whom whose how when why where there than that this these those
patients patient people persons individuals subjects improve improves improved reduce reduces reduced increase
increases increased decrease decreases decreased lower lowers higher affect affects effect effects effective
effectiveness efficacy efficacious compared compare comparison versus vs use using used benefit beneficial
help helps any some it its their associated association
""".split())
_NEGATION = re.compile(r"\b(?:no|not|non|without|never|absence of|excluding)\b", re.I)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def original_wording(question: str) -> tuple[list[str], Optional[str]]:
    """Content words of the clinician's question, order kept. Second value is a
    reason when plain wording cannot express the question safely (negation)."""
    if _NEGATION.search(question or ""):
        return [], "The question contains a negation (e.g. 'without'), which plain wording cannot express safely."
    words = []
    for w in _WORD.findall(question or ""):
        lw = w.lower().strip("'-")
        if lw and lw not in _STOP and len(lw) > 1 and lw not in words:
            words.append(lw)
    return words[:12], None


def _core(plan: dict[str, Any], source: str) -> str:
    return ecp.core_query(plan, source)


def _condition_core(plan: dict[str, Any], source: str) -> str:
    """The required condition/population concepts only (for guideline searches)."""
    cond = [c for c in plan["concepts"] if c.get("required")
            and c.get("role") in ("condition", "presentation", "population")]
    if not cond:
        return ""
    return ecp.core_query({**plan, "concepts": cond}, source)


def _with_filters(query: str, plan: dict[str, Any], source: str) -> str:
    return f"({query}){ecp.source_filters(plan, source)}"[: ecp.MAX_QUERY_LEN]


_MODIFIERS = re.compile(
    r"\b(?:early|late|delayed|prompt|initial|timely|low[- ]dose|high[- ]dose|prolonged|short[- ]course|"
    r"long[- ]term|adjunctive|adjuvant|prophylactic|continuous|intermittent|routine|empiric(?:al)?|standard|"
    r"intravenous|iv|oral|inhaled|systemic|stress[- ]dose)\b", re.I)
_HEADS = re.compile(r"\b(?:treatment|treatments|therapy|therapies|administration|use|regimen|infusion|"
                    r"supplementation|management)\b(?:\s+(?:of|with))?", re.I)
_ACTIVE_ROLES = ("intervention", "exposure", "index_test", "prognostic_factor")


def core_head(term: str) -> Optional[str]:
    """'early corticosteroid treatment' → 'corticosteroid'; 'early administration of
    corticosteroids' → 'corticosteroids'. None when nothing is left or nothing changed."""
    out = " ".join(_HEADS.sub(" ", _MODIFIERS.sub(" ", term or "")).split()).strip(" ,-")
    return out if out and len(out) >= 4 and out.lower() != (term or "").strip().lower() else None


def _narrow_phrase_concept(c: dict[str, Any], plan: dict[str, Any]) -> Optional[str]:
    """A 'must appear' treatment/exposure/test concept with no MeSH heading whose
    every wording is a modified phrase — papers rarely use the exact phrase."""
    if not c.get("required") or c.get("role") not in _ACTIVE_ROLES or c.get("mesh"):
        return None
    words = ecp.concept_words(c, plan)
    heads = {core_head(w) for w in words}
    if None in heads or not heads:
        return None
    head = core_head(c["term"])
    if head and head.lower() not in {w.lower() for w in words}:
        return head
    return None


def _relaxation(plan: dict[str, Any]) -> Optional[tuple[dict[str, Any], str, list[dict[str, Any]]]]:
    """The single bounded alternative: (relaxed plan, reason, the term edits that
    express it). Order: complete the vocabulary of "must appear" concepts that have
    no MeSH heading (the broader heading and the same-meaning wordings the clinician
    was offered but left off) → drop an outcome/comparator "must appear" → add the
    plain term to an over-specific treatment phrase → drop the year limit.
    Condition and treatment always stay AND-ed."""
    concepts = plan["concepts"]
    required = [c for c in concepts if c.get("required")]
    adds: dict[str, dict[str, Any]] = {}
    for c in required:
        if c.get("mesh") or not c.get("id"):
            continue
        off = [t for t in c.get("off_equivalents") or [] if t.get("id") and t.get("text")]
        if c.get("broader_mesh") or off:
            adds[c["id"]] = {"mesh": c.get("broader_mesh"), "off": off}
    if adds:
        relaxed = {**plan, "concepts": [
            {**x, "mesh": adds[x["id"]]["mesh"] or x.get("mesh"),
             "synonyms": list(x.get("synonyms") or []) + [t["text"] for t in adds[x["id"]]["off"]]}
            if x.get("id") in adds else x for x in concepts]}
        parts, edits = [], []
        for c in required:
            a = adds.get(c.get("id"))
            if not a:
                continue
            bits = ([f"the broader MeSH heading '{a['mesh']}'"] if a["mesh"] else []) + (
                [f"{len(a['off'])} same-meaning wording(s) (" + ", ".join(f"'{t['text']}'" for t in a["off"][:3])
                 + ("…" if len(a["off"]) > 3 else "") + ")"] if a["off"] else [])
            parts.append(f"'{c['term']}': " + " and ".join(bits))
            edits += ([{"op": "set_mesh_enabled", "concept_id": c["id"], "enabled": True}] if a["mesh"] else []) + [
                {"op": "set_term_enabled", "concept_id": c["id"], "term_id": t["id"], "enabled": True} for t in a["off"]]
        return (relaxed,
                "Adds what was offered but left off — " + "; ".join(parts) + ". Every concept must still appear.",
                edits)
    core_roles = {"condition", "presentation", "population", "intervention", "exposure", "index_test",
                  "prognostic_factor", "phenomenon", "concept"}
    if len(required) > 2:
        for role in ("outcome", "comparator"):
            target = next((c for c in required if c.get("role") == role), None)
            if target and len([c for c in required if c.get("role") in core_roles]) >= 2:
                relaxed = {**plan, "concepts": [{**c, "required": False} if c is target else c for c in concepts]}
                return (relaxed,
                        f"Same search without requiring the {role} '{target['term']}' (it is often not named in "
                        f"titles or abstracts); every other concept must still appear.",
                        [{"op": "set_required", "concept_id": target.get("id"), "enabled": False}])
    for c in required:
        head = _narrow_phrase_concept(c, plan)
        if head:
            # Quoted [tiab] phrases do not match plurals: search both forms.
            forms = [head, head[:-1] if head.endswith("s") else f"{head}s"]
            relaxed = {**plan, "concepts": [{**x, "synonyms": list(x.get("synonyms") or []) + forms} if x is c else x
                                            for x in concepts], "use_synonyms": True}
            return (relaxed,
                    f"Adds the plain term '{head}' to '{c['term']}' — the exact phrase is rare in titles and "
                    "abstracts; the modifier is then judged when each paper is reviewed.",
                    [{"op": "add_term", "concept_id": c.get("id"), "text": head}]
                    + ([] if plan.get("use_synonyms", True) else [{"op": "set_use_synonyms", "enabled": True}]))
    years = (plan.get("filters") or {}).get("publication_years")
    if years:
        relaxed = {**plan, "filters": {**plan["filters"], "publication_years": None}}
        return relaxed, f"Searches all years instead of the last {years}.", [{"op": "set_publication_years", "years": 0}]
    return None


# Overall prognosis: the Clinical Queries prognosis filters were tuned to inception cohorts and miss
# long-term trial reports and cohorts that never say "prognosis" (Wilczynski & Haynes 2004; Boulos
# 2021); a lane for survival / long-term follow-up wording finds them (PROGRESS; Iorio 2015: trial
# arms are valid overall-prognosis evidence).
_PROGNOSIS_TYPES = frozenset({"prognosis", "prognostic_factor"})
_LONG_TERM_WORDS = ("long-term", "long term", "10-year", "15-year", "20-year", "10-yr", "15-yr", "final results",
                    "final report", "final analysis", "median follow-up", "survival")
_LONG_TERM_PUBMED = ("(" + " OR ".join(f'"{w}"[tiab]' for w in _LONG_TERM_WORDS)
                     + ' OR "survival analysis"[MeSH Terms] OR mortality[Subheading])')
_LONG_TERM_EPMC = "(" + " OR ".join(f'TITLE_ABS:"{w}"' for w in _LONG_TERM_WORDS) + ")"


def build(plan: dict[str, Any], question: str, *, sources: tuple[str, ...] = ("pubmed", "europepmc")) -> dict[str, Any]:
    """The full portfolio for an approved plan: specs + what was skipped and why."""
    qtype = plan.get("question_type") or "other"
    design_key = ecp.DESIGN_ALIASES.get(qtype, qtype)
    cq = CLINICAL_QUERIES.get(CQ_CATEGORY.get(qtype, ""), {})
    years = (plan.get("filters") or {}).get("publication_years")
    words, wording_skip = original_wording(question)
    relax = _relaxation(plan)
    specs: list[QuerySpec] = []
    skipped: list[dict[str, str]] = []

    def add(lane: str, family: str, source: str, query: str, purpose: str, *, condition: Optional[str] = None,
            sort: Optional[str] = None) -> None:
        if not query:
            return
        specs.append(QuerySpec(
            id=f"{lane}:{family}:{source}", lane=lane, family=family, source=source,
            query=query[: ecp.MAX_QUERY_LEN], purpose=purpose,
            max_records=MAX_RECORDS.get((family, source), 50), condition=condition, sort=sort,
        ))

    for source in sources:
        core = _core(plan, source)
        if not core:
            continue
        add("foundational_primary", "main", source, _with_filters(core, plan, source),
            "Every 'must appear' concept, AND-ed; each concept = its MeSH heading OR its words in the title/abstract.")
        # Relevance ranking favours recent, popular records and buries landmark papers in large
        # result sets; the same search read most-cited first finds them.
        add("foundational_primary", "landmark", source, _with_filters(core, plan, source),
            "The main search read most-cited first ("
            + ("NIH iCite citation counts" if source == "pubmed" else "Europe PMC's citation sort")
            + ") — landmark papers that relevance ranking buries.", sort="cited")
        if words:
            plain = " ".join(words) if source == "pubmed" else " AND ".join(words)
            add("foundational_primary", "original_wording", source, _with_filters(plain, plan, source),
                "Your question's own words with filler words removed"
                + (" — PubMed maps them to its vocabulary itself." if source == "pubmed" else "."))
        if source == "pubmed" and cq:
            add("foundational_primary", "narrow", source, _with_filters(f"({core}) AND {cq['narrow']}", plan, source),
                f"The main search with PubMed's narrow (specific) {CQ_CATEGORY[qtype].replace('_', ' ')} filter.")
            add("foundational_primary", "broad", source, _with_filters(f"({core}) AND {cq['broad']}", plan, source),
                f"The main search with PubMed's broad (sensitive) {CQ_CATEGORY[qtype].replace('_', ' ')} filter.")
        else:
            designs = (ecp.DESIGNS_PUBMED if source == "pubmed" else ecp.DESIGNS_EPMC).get(design_key)
            if designs:
                add("foundational_primary", "narrow", source,
                    _with_filters(f"({core}) AND ({' OR '.join(designs)})", plan, source),
                    "The main search narrowed to the study designs that answer this kind of question.")
        if qtype in _PROGNOSIS_TYPES:
            lt = _LONG_TERM_PUBMED if source == "pubmed" else _LONG_TERM_EPMC
            add("foundational_primary", "long_term", source, _with_filters(f"({core}) AND {lt}", plan, source),
                "The main search narrowed to survival and long-term follow-up reports — cohorts and trial arms "
                "that report outcomes over time (prognosis filters miss many of them).")
        guide = _GUIDELINES_PUBMED if source == "pubmed" else _GUIDELINES_EPMC
        add("aggregated", "guidelines", source, _with_filters(f"({core}) AND {guide}", plan, source),
            "Guidelines and consensus statements on the approved concepts (indexed in "
            f"{'PubMed' if source == 'pubmed' else 'Europe PMC'}).")
        cond = _condition_core(plan, source)
        if cond and cond != core:
            gtypes = _GUIDELINE_TYPES_PUBMED if source == "pubmed" else _GUIDELINE_TYPES_EPMC
            add("aggregated", "condition_guidelines", source, _with_filters(f"({cond}) AND {gtypes}", plan, source),
                "Guidelines on the condition itself — broad guidelines rarely name every treatment in their "
                "abstract.")
        srev = _SR_PUBMED if source == "pubmed" else _SR_EPMC
        add("aggregated", "systematic_reviews", source, _with_filters(f"({core}) AND {srev}", plan, source),
            "Systematic reviews and meta-analyses on the approved concepts.")
        rev = _REVIEW_PUBMED if source == "pubmed" else _REVIEW_EPMC
        add("aggregated", "reviews", source, _with_filters(f"({core}) AND {rev}", plan, source),
            "Reviews and summaries on the approved concepts.")
        if source == "pubmed":
            add("aggregated", "cochrane", source, _with_filters(f"{_COCHRANE_PUBMED} AND ({core})", plan, source),
                "Cochrane systematic reviews (PubMed-indexed; link-out only — not used by AI).")
        if not years or int(years) > 5:
            recent = ' AND ("last 5 years"[PDat])' if source == "pubmed" else f" AND ({ecp.epmc_since(5)})"
            add("recent_primary", "recent", source, _with_filters(f"({core}){recent}", plan, source),
                "The main search, publications from the last 5 years first.")
        if relax:
            relaxed_core = _core(relax[0], source)
            if relaxed_core and relaxed_core != core:
                add("alternative", "alternative", source, _with_filters(relaxed_core, relax[0], source),
                    relax[1], condition="if_sparse")
            elif not relaxed_core:
                skipped.append({"family": "alternative", "source": source,
                                "reason": "The relaxation leaves nothing to search."})
            elif (relax[0].get("filters") or {}) != (plan.get("filters") or {}):
                add("alternative", "alternative", source, _with_filters(core, relax[0], source),
                    relax[1], condition="if_sparse")
    if wording_skip:
        skipped.append({"family": "original_wording", "source": "all", "reason": wording_skip})
    if not relax:
        skipped.append({"family": "alternative", "source": "all",
                        "reason": "No justified relaxation: only the core concepts must appear and no limit is set."})
    return {
        "specs": specs,
        "skipped": skipped,
        "alternative": ({"reason": relax[1], "edits": relax[2]} if relax else None),
        "original_wording": words,
        "clinical_queries_category": CQ_CATEGORY.get(qtype) if cq else None,
    }


def update_specs(plan: dict[str, Any], cutoff: str, *, sources: tuple[str, ...] = ("pubmed", "europepmc"),
                 anchor_label: str = "the newest review") -> list[QuerySpec]:
    """Newer evidence: the main search, narrowed to designs that answer this
    question type, for records ADDED after the verified search cutoff (PubMed
    create date [crdt]; Europe PMC first index date) — the cutoff, not a
    publication date, anchors the update."""
    from datetime import date

    qtype = plan.get("question_type") or "other"
    design_key = ecp.DESIGN_ALIASES.get(qtype, qtype)
    cq = CLINICAL_QUERIES.get(CQ_CATEGORY.get(qtype, ""), {})
    y, m, d = cutoff[:10].split("-")
    specs = []
    for source in sources:
        core = _core(plan, source)
        if not core:
            continue
        if source == "pubmed":
            design = cq.get("narrow") or ("(" + " OR ".join(ecp.DESIGNS_PUBMED.get(design_key, [])) + ")"
                                          if ecp.DESIGNS_PUBMED.get(design_key) else "")
            since = f'("{y}/{m}/{d}"[crdt] : "3000"[crdt])'
        else:
            designs = ecp.DESIGNS_EPMC.get(design_key)
            design = "(" + " OR ".join(designs) + ")" if designs else ""
            since = f"FIRST_IDATE:[{y}-{m}-{d} TO {date.today().isoformat()}]"
        query = f"({core}) AND {design} AND {since}" if design else f"({core}) AND {since}"
        specs.append(QuerySpec(
            id=f"update:update:{source}", lane="update", family="update", source=source,
            query=_with_filters(query, plan, source),
            purpose=f"Studies added after {anchor_label}'s search cutoff ({cutoff[:10]}).",
            max_records=MAX_RECORDS.get(("update", source), 50),
        ))
    return specs
