"""Search plan approved by the clinician in the question-review flow.

The clinical engine (``clinical_engine/``) lets the clinician approve the search
terms — each concept, its MeSH heading and the wordings to include. It stores
that plan on the search session as ``engine_concept_plan``; this module is how
the pipeline honours it. With a plan:

* the must-have / ranking concepts and primary anchors ARE the approved concepts
  (the parser's own concepts, ontology lifts and AI expansions are not used);
* queries are built from the approved terms only: one block per required
  concept (MeSH OR title/abstract words), AND-ed together;
* the legacy planner's fallback / thin-pool broadening is not run, because it
  would re-introduce terms the clinician did not approve.

Deliberately self-contained (plain dicts, no ``clinical_engine`` import): the
worker images ship ``services/`` but not ``clinical_engine/``.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

PLAN_VERSION = "ecp.v1"
PLANNER_SOURCE = "engine_approved_terms"
_SOURCES = ("pubmed", "europe_pmc")
_MAX_QUERY_LEN = 3800

# engine role -> (legacy PICOTT role, anchor entity type, missing-concept relevance cap)
_ROLE_MAP: dict[str, tuple[str, str, int]] = {
    "condition": ("condition", "disease_or_problem", 50),
    "presentation": ("condition", "disease_or_problem", 50),
    "population": ("population", "disease_or_problem", 50),
    "intervention": ("intervention", "management_intervention", 30),
    "exposure": ("intervention", "exposure", 30),
    "index_test": ("intervention", "diagnostic_test", 30),
    "prognostic_factor": ("intervention", "exposure", 40),
    "phenomenon": ("intervention", "management_intervention", 40),
    "concept": ("intervention", "management_intervention", 40),
    "comparator": ("comparator", "management_intervention", 40),
    "outcome": ("outcome", "disease_or_problem", 50),
}
_PICO_BLOCK = {
    "condition": "population", "population": "population",
    "intervention": "intervention_or_exposure_or_index_test",
    "comparator": "comparator_or_reference_standard", "outcome": "outcomes_or_target_diagnosis",
}

# Design filters per engine question type (PubMed publication types / MeSH;
# Europe PMC publication types).
_PT = '"{}"[Publication Type]'
_MH = '"{}"[MeSH Terms]'
_DESIGNS_PUBMED: dict[str, list[str]] = {
    "treatment": [_PT.format("Randomized Controlled Trial"), _PT.format("Systematic Review"), _PT.format("Meta-Analysis")],
    "harm": [_MH.format("Cohort Studies"), _MH.format("Case-Control Studies"), _PT.format("Systematic Review"),
             _PT.format("Meta-Analysis")],
    "diagnosis": [_MH.format("Sensitivity and Specificity"), _PT.format("Systematic Review"),
                  _PT.format("Meta-Analysis")],
    "prognosis": [_MH.format("Cohort Studies"), _MH.format("Prognosis"), _PT.format("Systematic Review")],
    "prediction_model": [_PT.format("Validation Study"), _MH.format("Risk Assessment"), _PT.format("Systematic Review")],
    "qualitative": [_MH.format("Qualitative Research"), _MH.format("Interviews as Topic")],
    "epidemiology": [_MH.format("Prevalence"), _MH.format("Incidence"), _MH.format("Cross-Sectional Studies"),
                     _MH.format("Cohort Studies")],
}
_DESIGN_ALIASES = {
    "prevention": "treatment", "screening": "treatment", "clinical_management": "treatment",
    "etiology_risk": "harm", "complications": "harm", "diagnostic_workup": "diagnosis",
    "monitoring": "diagnosis", "prognostic_factor": "prognosis",
}
_DESIGNS_EPMC: dict[str, list[str]] = {
    "treatment": ['PUB_TYPE:"randomized controlled trial"', 'PUB_TYPE:"systematic review"', 'PUB_TYPE:"meta analysis"'],
    "harm": ['PUB_TYPE:"systematic review"', 'PUB_TYPE:"meta analysis"', 'TITLE_ABS:"cohort"'],
    "diagnosis": ['TITLE_ABS:"sensitivity"', 'TITLE_ABS:"diagnostic accuracy"', 'PUB_TYPE:"systematic review"'],
    "prognosis": ['TITLE_ABS:"cohort"', 'TITLE_ABS:"prognosis"', 'PUB_TYPE:"systematic review"'],
    "prediction_model": ['PUB_TYPE:"validation study"', 'TITLE_ABS:"validation"', 'TITLE_ABS:"prediction model"'],
    "qualitative": ['TITLE_ABS:"qualitative"', 'TITLE_ABS:"interviews"'],
    "epidemiology": ['TITLE_ABS:"prevalence"', 'TITLE_ABS:"incidence"'],
}
_REVIEWS_PUBMED = ("(" + " OR ".join(_PT.format(t) for t in (
    "Practice Guideline", "Guideline", "Systematic Review", "Meta-Analysis")) + ")")
_REVIEWS_EPMC = ('(PUB_TYPE:"practice guideline" OR PUB_TYPE:"guideline" OR PUB_TYPE:"systematic review" '
                 'OR PUB_TYPE:"meta analysis")')
_QUOTAS = {"mesh_keyword": 100, "high_precision": 50, "study_type_filtered": 50, "guideline_review": 40,
           "recent_title_abstract": 50, "cochrane_review": 20}

# Europe PMC has no humans-only filter: exclude animal-model studies by title.
_EPMC_NOT_ANIMAL = " NOT (" + " OR ".join(f'TITLE:"{w}"' for w in (
    "porcine", "swine", "pigs", "piglets", "rat", "rats", "mice", "mouse", "murine", "canine", "dogs",
    "sheep", "ovine", "rabbit", "rabbits", "bovine", "equine", "feline", "zebrafish", "in vitro",
)) + ")"
_UNSAFE = re.compile(r'["\[\]{}()*?\\]')


def _phrase(term: str) -> str:
    return " ".join(_UNSAFE.sub(" ", term or "").split())


def validate_plan(raw: Any) -> Optional[dict[str, Any]]:
    """The plan if it is one this code understands, else None (legacy behaviour)."""
    if not isinstance(raw, dict) or raw.get("plan_version") != PLAN_VERSION:
        return None
    concepts = [c for c in raw.get("concepts") or [] if isinstance(c, dict) and _phrase(c.get("term") or "")]
    if not any(c.get("required") for c in concepts):
        return None
    return {**raw, "concepts": concepts}


def _words(concept: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    words = [concept["term"]] + (list(concept.get("synonyms") or []) if plan.get("use_synonyms", True) else [])
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        p = _phrase(w)
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def _block(concept: dict[str, Any], plan: dict[str, Any], source: str) -> str:
    words = _words(concept, plan)
    if source == "pubmed":
        parts = [f'"{concept["mesh"]}"[MeSH Terms]'] if concept.get("mesh") else []
        parts += [f'"{w}"[tiab]' for w in words]
    else:  # Europe PMC's MESH field misbehaves inside OR groups: title/abstract words only
        parts = [f'TITLE_ABS:"{w}"' for w in words]
    return "(" + " OR ".join(parts) + ")" if parts else ""


def _epmc_since(years: int) -> str:
    today = datetime.now(timezone.utc).date()
    start = today.replace(year=today.year - years) if not (today.month == 2 and today.day == 29) \
        else today.replace(year=today.year - years, day=28)
    return f"FIRST_PDATE:[{start.isoformat()} TO {today.isoformat()}]"


def _filters(plan: dict[str, Any], source: str) -> str:
    f = plan.get("filters") or {}
    years = f.get("publication_years")
    out = ""
    if source == "pubmed":
        if years:
            out += f' AND ("last {int(years)} years"[PDat])'
        if f.get("exclude_animal_only", True):
            out += ' NOT ("animals"[MeSH Terms] NOT "humans"[MeSH Terms])'
        out += ' NOT "preprint"[Publication Type]'
    else:
        if years:
            out += f" AND ({_epmc_since(int(years))})"
        if f.get("exclude_animal_only", True):
            out += _EPMC_NOT_ANIMAL
        out += " NOT SRC:PPR"
    return out


def core_query(plan: dict[str, Any], source: str) -> str:
    blocks = [b for b in (_block(c, plan, source) for c in plan["concepts"] if c.get("required")) if b]
    return " AND ".join(blocks)


def full_query(plan: dict[str, Any], source: str) -> str:
    core = core_query(plan, source)
    return (f"({core}){_filters(plan, source)}" if core else "")[:_MAX_QUERY_LEN]


def _record(source: str, family: str, query: str, plan: dict[str, Any], rationale: str) -> dict[str, Any]:
    required = [c for c in plan["concepts"] if c.get("required")]
    return {
        "source": source, "query_type": family, "query_family": family,
        "query_string": query[:_MAX_QUERY_LEN],
        "concept_blocks_json": {c["term"]: _words(c, plan) for c in plan["concepts"]},
        "mandatory_anchors": [c["term"] for c in required],
        "optional_terms": [c["term"] for c in plan["concepts"] if not c.get("required")],
        "rationale": rationale, "max_results": _QUOTAS.get(family, 50), "planner_source": PLANNER_SOURCE,
        "field_tags_used": ["MeSH Terms", "tiab"] if source == "pubmed" else ["TITLE_ABS"],
        "mesh_terms_used": [{"concept": c["term"], "mesh_descriptor": c["mesh"], "source": "ncbi_mesh"}
                            for c in required if c.get("mesh") and source == "pubmed"],
    }


def build_queries(plan: dict[str, Any], sources: list[str]) -> list[dict[str, Any]]:
    """Query records (same shape as the concept-block builder's) from the approved plan."""
    from services.query_compiler import validate_query_syntax

    qtype = plan.get("question_type") or "other"
    design_key = _DESIGN_ALIASES.get(qtype, qtype)
    years = (plan.get("filters") or {}).get("publication_years")
    outcomes = [c for c in plan["concepts"] if c.get("role") == "outcome" and not c.get("required")]
    out: list[dict[str, Any]] = []
    for source in (s for s in _SOURCES if s in sources):
        core = core_query(plan, source)
        if not core:
            continue
        filt = _filters(plan, source)
        specs: list[tuple[str, str, str]] = [
            ("mesh_keyword", f"({core}){filt}", "All approved concepts (MeSH + title/abstract words), AND-ed."),
        ]
        if outcomes:
            ob = " OR ".join(b[1:-1] for b in (_block(c, plan, source) for c in outcomes) if b)
            specs.append(("high_precision", f"({core}) AND ({ob}){filt}",
                          "Approved concepts AND the outcomes of interest."))
        designs = (_DESIGNS_PUBMED if source == "pubmed" else _DESIGNS_EPMC).get(design_key)
        if designs:
            specs.append(("study_type_filtered", f"({core}) AND ({' OR '.join(designs)}){filt}",
                          "Approved concepts narrowed to study designs that answer this kind of question."))
        reviews = _REVIEWS_PUBMED if source == "pubmed" else _REVIEWS_EPMC
        specs.append(("guideline_review", f"({core}) AND {reviews}{filt}",
                      "Approved concepts narrowed to guidelines, systematic reviews and meta-analyses."))
        if not years or int(years) > 5:
            recent = ' AND ("last 5 years"[PDat])' if source == "pubmed" else f" AND ({_epmc_since(5)})"
            specs.append(("recent_title_abstract", f"({core}){recent}{filt}",
                          "Approved concepts, publications from the last 5 years."))
        if source == "pubmed":
            specs.append(("cochrane_review", f'"Cochrane Database Syst Rev"[ta] AND ({core}){filt}',
                          "Approved concepts in Cochrane systematic reviews (PubMed-indexed)."))
        for family, query, why in specs:
            if validate_query_syntax(query[:_MAX_QUERY_LEN], source=source).valid:
                out.append(_record(source, family, query, plan, why))
    return out


def apply_to_picott(picott: dict[str, Any], anchors_out: dict[str, Any], plan: dict[str, Any]) -> None:
    """Make the approved concepts the pipeline's concepts (in place)."""
    must: list[dict[str, Any]] = []
    should: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    pico: dict[str, list[str]] = {}
    for c in plan["concepts"]:
        legacy_role, entity_type, cap = _ROLE_MAP.get(c.get("role") or "", ("intervention", "other", 40))
        words = _words(c, plan)
        synonyms = words[1:]
        pico.setdefault(_PICO_BLOCK.get(legacy_role, "population"), []).append(c["term"])
        if c.get("required"):
            must.append({"concept": c["term"], "role": legacy_role, "synonyms": synonyms,
                         "required_for_high_relevance": True, "missing_relevance_cap": cap,
                         "source": PLANNER_SOURCE})
            anchors.append({"text": c["term"], "canonical_concept": c["term"], "entity_type": entity_type,
                            "role": legacy_role, "synonyms": synonyms, "source": PLANNER_SOURCE,
                            "confidence": 1.0, "required_for_search": True, "missing_relevance_cap": cap})
        elif legacy_role == "comparator":
            should.append({"concept": c["term"], "role": "comparator", "synonyms": synonyms, "boost_weight": 4})
        else:
            ranking.append({"concept": c["term"], "role": legacy_role, "synonyms": synonyms, "boost_weight": 4})
    picott["must_have_concepts"] = must
    picott["should_have_concepts"] = should
    picott["ranking_concepts"] = ranking
    kept = {k: v for k, v in (picott.get("pico") or {}).items() if k in ("time_horizon", "setting")}
    picott["pico"] = {**kept, **pico}
    if plan.get("legacy_type_of_question"):
        picott["type_of_question"] = plan["legacy_type_of_question"]
    if plan.get("legacy_subtype") is not None:
        picott["question_subtype"] = plan["legacy_subtype"]
    picott["engine_concept_plan_applied"] = True
    anchors_out["primary_anchors"] = anchors
    anchors_out["anchor_failure_reason"] = None
    anchors_out["anchor_repair_applied"] = False
    anchors_out["picott_overrides"] = {}
    anchors_out["interpretation_source"] = PLANNER_SOURCE


# Public names for the engine's query portfolio (clinical_engine/retrieval/portfolio.py),
# which builds its search families from the same approved-terms blocks and filters.
concept_words = _words
concept_block = _block
source_filters = _filters
epmc_since = _epmc_since
DESIGNS_PUBMED = _DESIGNS_PUBMED
DESIGNS_EPMC = _DESIGNS_EPMC
DESIGN_ALIASES = _DESIGN_ALIASES
REVIEWS_PUBMED = _REVIEWS_PUBMED
REVIEWS_EPMC = _REVIEWS_EPMC
MAX_QUERY_LEN = _MAX_QUERY_LEN
