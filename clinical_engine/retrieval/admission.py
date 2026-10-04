"""Clinical relevance and admission (W2.4).

Every publication gets ONE state with a reason, judged on what the paper is
about — population, condition, intervention/exposure, comparison, outcome,
setting and publication role — not on word overlap:

  direct           answers the question as asked
  broad_answering  answers it for a broader population/intervention class that
                   includes the asked one (or a guideline/review covering it)
  supportive       useful for the answer but not itself answering it
                   (secondary/subgroup analyses, follow-up reports, …)
  indirect         related, but a different population/intervention/comparison
  uncertain        the record does not say enough to judge
  excluded         wrong condition/population/intervention, not a study,
                   no results (protocol), animal/lab only, retracted, preprint

Deterministic checks run first and can only make a state stricter: safety
exclusions (retractions, errata, preprints), animal-only, publication role
(protocols, commentaries, corrections; subgroup/secondary reports of a trial
whose main report is present), and negation ("patients without X"). The model
judges the rest in batches through the model gateway; without a model, records
are "uncertain" (never silently admitted). Open access, citations and source
counts play no part.

Sections: about 15 qualifying items (direct, then broad-answering; at most 3
guidelines among them), up to 5 optional extras, everything else listed with
its reason. No filler: a short list stays short.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from clinical_engine.config import settings
from clinical_engine.gateway import ModelCallResult, ModelGateway
from services import engine_concept_plan as ecp

STATES = ("direct", "broad_answering", "supportive", "indirect", "uncertain", "excluded")
QUALIFYING = ("direct", "broad_answering")
STATE_LABELS = {
    "direct": "Directly answers", "broad_answering": "Answers more broadly", "supportive": "Supportive",
    "indirect": "Indirect", "uncertain": "Uncertain", "excluded": "Excluded",
}
EXCLUSIONS = {
    "wrong_condition": "Different condition",
    "wrong_population": "Different population",
    "wrong_intervention": "Different intervention or exposure",
    "wrong_comparison": "Different comparison",
    "not_a_study": "Not a study (editorial, letter, comment or erratum)",
    "no_results": "No results (protocol or design paper)",
    "animal_or_lab": "Animal or laboratory study",
    "retracted": "Retracted or a retraction notice",
    "preprint": "Preprint",
    "other": "Does not answer the question",
}
ROLES = ("primary_result", "secondary_analysis", "subgroup_analysis", "follow_up", "protocol", "guideline",
         "systematic_review", "meta_analysis", "narrative_review", "commentary", "correction", "case_report", "other")
ROLE_LABELS = {
    "primary_result": "Main report", "secondary_analysis": "Secondary analysis", "subgroup_analysis": "Subgroup analysis",
    "follow_up": "Follow-up report", "protocol": "Protocol", "guideline": "Guideline",
    "systematic_review": "Systematic review", "meta_analysis": "Meta-analysis", "narrative_review": "Review",
    "commentary": "Commentary", "correction": "Correction", "case_report": "Case report", "other": "Study",
}
MAIN_LIMIT = 15
MAIN_GUIDELINES = 3
OTHER_LIMIT = 5
MATCH = ("match", "broader", "narrower", "related", "different", "unclear", "na")

_NEG_CUE = re.compile(r"(?:\bno\b|\bnot\b|\bnon[- ]?|\bwithout\b|\babsence of\b|\bfree of\b|\bexcluding\b|"
                      r"\bexcept\b|\bnegative for\b)\W*(?:\w+\W+){0,2}$", re.I)

SYSTEM_PROMPT = """You screen search results for ONE clinical question that a clinician has approved.
For EACH candidate decide how well it answers THAT question. Judge what the paper actually studied —
its population, condition, intervention or exposure, comparison, outcomes, setting and publication role —
never word overlap, the journal, open access or citation counts.

States (choose exactly one):
- "direct": same condition and population, same intervention/exposure (and comparison when the question
  names one), and it reports an outcome the question cares about. Guidelines and systematic reviews focused
  on this question are direct.
- "broad_answering": answers it for a broader population or a broader intervention class that INCLUDES the
  asked one (e.g. all corticosteroids when hydrocortisone was asked; sepsis when septic shock was asked), or
  a guideline/review whose scope covers the question among others.
- "supportive": useful for the answer but not itself answering it: secondary, post hoc or subgroup analyses
  and follow-up reports of a trial (unless the question is about that subgroup), cost-effectiveness, dosing
  or mechanism work in the same clinical setting.
- "indirect": related but different population, intervention, comparison or setting (e.g. children when
  adults were asked; a different drug class; a different comparator).
- "uncertain": the title/abstract does not say enough to judge.
- "excluded": does not answer it. Give "exclusion": wrong_condition (a different disease or clinical
  problem, even if the drug matches), wrong_population (e.g. patients WITHOUT the condition), wrong_intervention,
  wrong_comparison, not_a_study (editorial, letter, comment, erratum — a guideline or consensus statement IS
  a study here), no_results (protocol or design paper), animal_or_lab, other.

Rules:
- Judge relevance only — do not downgrade for study design (randomized or observational) or age of the paper.
- Outcomes: the same outcome at another time point (e.g. 90-day instead of 28-day mortality) or the same
  outcome measured another way (e.g. mortality when survival was asked) is outcome "match" — say so in the
  reason. Use outcome "related" only for a surrogate or neighbouring outcome (e.g. shock reversal, physiology or
  transfusion volume when survival was asked). Concepts listed as "used to rank" are not required for "direct".
- A multi-component intervention (a bundle or protocol) that merely contains the asked intervention is
  "indirect", not "direct".
- Drug classes: when the question names a class (e.g. "corticosteroids"), a specific member or a combination
  of members of that class (e.g. hydrocortisone, hydrocortisone plus fludrocortisone) is a match — not broader.
  When the question names one drug and the paper studies the whole class, that is "broad_answering".
- A guideline or review whose recommendations or analyses specifically address the asked intervention for
  the asked condition is "direct", even if it also covers other conditions or populations.
- Guidelines: a comprehensive guideline on the condition's overall management covers the asked treatment —
  "broad_answering" (or "direct" if it is about this treatment) even if the abstract does not name it. A
  guideline on a different specific topic (e.g. fluids, ultrasound) is excluded as wrong_intervention.
- A narrative (non-systematic) review — including practical "how to" articles, overviews and expert
  opinion — is at most "supportive" (except for background questions — see "How to read the dimensions");
  give it role "narrative_review".
- Read negation carefully ("patients without diabetes" is a different population from diabetes).
- A secondary, post hoc or subgroup report is at most "supportive" unless the question is about that subgroup.
- Use the deterministic signals given with each candidate; do not contradict a signal marked "certain".

For each candidate return:
{"id": "<the candidate id>",
 "population": "match|broader|narrower|related|different|unclear|na",
 "condition": "match|broader|narrower|related|different|unclear|na",
 "intervention": "match|broader|narrower|related|different|unclear|na",
 "comparison": "match|broader|narrower|related|different|unclear|na",
 "outcome": "match|broader|narrower|related|different|unclear|na",
 "setting": "match|broader|narrower|related|different|unclear|na",
 "role": "primary_result|secondary_analysis|subgroup_analysis|follow_up|protocol|guideline|systematic_review|meta_analysis|narrative_review|commentary|correction|case_report|other",
 "state": "direct|broad_answering|supportive|indirect|uncertain|excluded",
 "exclusion": null or one of the exclusion codes above,
 "reason": "<one sentence, at most 30 words, naming what matches or differs>",
 "confidence": "high|medium|low"}
Reply with JSON only: {"items": [ ... one object per candidate, same ids ... ]}"""


# ── Question context ────────────────────────────────────────────────────────

TYPE_FAMILY = {
    "treatment": "treatment", "prevention": "treatment", "clinical_management": "treatment",
    "health_services": "treatment", "diagnosis": "diagnosis", "diagnostic_workup": "diagnosis",
    "screening": "diagnosis", "monitoring": "diagnosis", "harm": "harm", "etiology_risk": "harm",
    "complications": "harm", "prognosis": "prognosis", "prognostic_factor": "prognosis",
    "prediction_model": "prognosis", "qualitative": "qualitative", "epidemiology": "epidemiology",
    "background": "background", "evidence_map": "background", "other": "background",
}
# What each judged dimension means for the question's type (the model labels the
# same six dimensions for every question; their meaning follows the framework).
DIMENSION_GUIDE = {
    "treatment": {
        "intervention": "the treatment or preventive intervention",
        "comparison": "its comparator — when none is stated: no treatment, placebo or usual care "
                      "(a head-to-head comparison of two regimens or doses is a different comparison)",
        "outcome": "the clinical outcomes asked about",
    },
    "diagnosis": {
        "condition": "the target condition the test should detect",
        "intervention": "the index test",
        "comparison": "the reference standard (or a comparator test)",
        "outcome": "diagnostic accuracy for the target condition (sensitivity, specificity, AUC, predictive values)",
        "extra": "A study that uses the test to GUIDE treatment (e.g. test-guided antibiotic stopping or "
                 "stewardship), or that only describes test levels without accuracy for the target condition, "
                 "answers a different question: comparison or outcome 'different'. A different target condition "
                 "(e.g. viral pneumonia, one infection site, a specific pathogen) is condition 'different' or "
                 "'related'.",
    },
    "harm": {
        "intervention": "the exposure", "comparison": "no exposure or another exposure level",
        "outcome": "the harm or adverse outcome asked about",
    },
    "prognosis": {
        "intervention": "the prognostic factor or prediction model ('na' for overall prognosis)",
        "comparison": "the reference level of the factor ('na' when none)",
        "outcome": "the future outcome over the time horizon",
        "extra": "When no factor is named (overall prognosis), 'na' means there is no index factor — it does NOT "
                 "make a treatment study off-topic: the treatment received is context (setting). A cohort, registry "
                 "or single trial arm reporting outcomes over time (survival, disease-specific death, metastasis, "
                 "recurrence) in the asked population or stage is 'direct'; a wider population in which the asked "
                 "group is the majority (e.g. 'locally advanced' including the asked stage) is 'broad_answering'; "
                 "prognostic factors or models within the group, a pathological-stage cohort for a clinical-stage "
                 "question, or a mix where the asked group is a minority is 'supportive'; a comparison of treatments "
                 "that reports only relative effects (hazard or odds ratios, efficacy) without outcomes over time in "
                 "the group is 'indirect'; another disease state "
                 "(e.g. metastatic disease) is 'excluded' as wrong_population; a paper with no outcomes over time "
                 "(toxicity, quality of life or dose planning only) is 'excluded'. If the abstract does not give the "
                 "population or stage mix, use 'uncertain', not 'excluded'.",
    },
    "qualitative": {
        "intervention": "the phenomenon of interest", "comparison": "'na'",
        "outcome": "experiences, views or meanings",
    },
    "epidemiology": {
        "intervention": "'na'", "comparison": "'na'",
        "outcome": "the frequency measure (prevalence, incidence) in the stated population and period",
    },
    "background": {
        "condition": "the condition or topic the question is about",
        "intervention": "'na' unless the question names a specific test, treatment or exposure",
        "comparison": "'na'",
        "outcome": "the aspect asked about (definition, criteria, mechanism, features, course)",
        "extra": "For this background question a definition or consensus statement, a guideline, or a review "
                 "(narrative or systematic) that explains the asked aspect is 'direct' — the narrative-review limit "
                 "does not apply. A study of one treatment or test for the condition is 'indirect'.",
    },
}


def question_context(cqm: Any, plan: dict[str, Any]) -> dict[str, Any]:
    roles: dict[str, list[str]] = {}
    for e in cqm.elements:
        if e.active and not e.negated:
            roles.setdefault(e.role, []).append(e.value)
    return {
        "question": cqm.question_text,
        "summary": cqm.summary,
        "question_type": cqm.question_type,
        "framework": cqm.framework,
        "elements": roles,
        "excluded": [e.value for e in cqm.elements if e.negated],
        "age_group": cqm.age_group(),
        "must_appear": [c["term"] for c in plan["concepts"] if c.get("required")],
        "ranking_only": [c["term"] for c in plan["concepts"] if not c.get("required")],
    }


# ── Deterministic signals ───────────────────────────────────────────────────


def _role(pub: dict[str, Any]) -> tuple[str, str]:
    """(role, how it was decided: pubtype | title | report | none)."""
    from clinical_engine.retrieval.lineage import guideline_role

    art = pub.get("article") or {}
    types = {t.lower() for t in art.get("publication_types_json") or []}
    title = (art.get("title") or "").lower()
    g = guideline_role(art)
    if g:
        return "guideline", g
    return _report_role(pub, types, title), "report"


_FROM_TRIAL = re.compile(r"\b(?:from|of|in) the [\w-]{3,}(?: \w+){0,3} (?:randomi[sz]ed )?(?:controlled )?trials?\b")


_CASE = re.compile(r"\bcase reports?\b|\ba case of\b|\bcase series\b|\bcase presentation\b")
# PubMed publication types that mark a study's own results (not "comparative" or
# "evaluation" study, which PubMed also gives to models and lab work).
_PRIMARY_TYPES = frozenset({
    "randomized controlled trial", "controlled clinical trial", "clinical trial", "clinical trial, phase i",
    "clinical trial, phase ii", "clinical trial, phase iii", "clinical trial, phase iv", "pragmatic clinical trial",
    "equivalence trial", "observational study", "multicenter study",
})


def _report_role(pub: dict[str, Any], types: set[str], title: str) -> str:
    base = pub.get("report_role") or "unknown"
    if "case reports" in types or _CASE.search(title):
        return "case_report"
    if base in ("unknown", "primary_result") and _FROM_TRIAL.search(title):
        base = "secondary_analysis"  # e.g. "… From the VANISH Randomized Trial"
    if base in ("protocol", "design_rationale"):
        return "protocol"
    if base in ("correction",):
        return "correction"
    if base == "commentary" or types & {"editorial", "letter", "comment"}:
        return "commentary"
    if base == "subgroup_analysis":
        return "subgroup_analysis"
    if base == "secondary_analysis":
        return "secondary_analysis"
    if base == "long_term_follow_up":
        return "follow_up"
    if "meta-analysis" in types:
        return "meta_analysis"
    if "systematic review" in types or "systematic review" in title:
        return "systematic_review"
    if "review" in types or "historical article" in types:
        return "narrative_review"
    if base == "primary_result" or (base == "unknown" and types & _PRIMARY_TYPES):
        return "primary_result"
    return "other"


def _occurrences_all_negated(text: str, words: list[str]) -> Optional[bool]:
    """True when every mention of the concept is negated; None when not mentioned."""
    low = text.lower()
    seen = negated = 0
    for w in words:
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", low):
            seen += 1
            before = low[max(0, m.start() - 40): m.start()]
            if _NEG_CUE.search(before):
                negated += 1
    if not seen:
        return None
    return negated == seen


def signals(pub: dict[str, Any], plan: dict[str, Any], *, primary_trials: set[str]) -> dict[str, Any]:
    from services.safety_filter import SafetyFilterService

    art = pub.get("article") or {}
    text = f"{art.get('title') or ''}. {art.get('abstract') or ''}"
    mesh = {m.lower() for m in ((pub.get("pubmed") or {}).get("mesh") or []) + ((pub.get("epmc") or {}).get("mesh") or [])}
    role, role_source = _role(pub)
    out: dict[str, Any] = {"role": role, "role_source": role_source, "concepts": [], "certain_exclusion": None}
    safety = SafetyFilterService._exclusion_reason(art)  # noqa: SLF001 — shared closed-set rule
    if safety:
        out["certain_exclusion"] = {"preprint_excluded_by_policy": "preprint", "retracted_publication": "retracted",
                                    "retraction_notice": "retracted"}.get(safety, "not_a_study")
    elif "animals" in mesh and "humans" not in mesh:
        out["certain_exclusion"] = "animal_or_lab"
    elif out["role"] in ("protocol",):
        out["certain_exclusion"] = "no_results"
    elif out["role"] in ("commentary", "correction"):
        out["certain_exclusion"] = "not_a_study"
    for c in plan["concepts"]:
        if not c.get("required"):
            continue
        words = ecp.concept_words(c, plan)
        by_mesh = bool(c.get("mesh") and c["mesh"].lower() in mesh)
        negated = _occurrences_all_negated(text, words)
        out["concepts"].append({"term": c["term"], "role": c.get("role"), "in_text": negated is not None,
                                "by_mesh": by_mesh, "only_negated": bool(negated)})
    title_low = (art.get("title") or "").lower()
    out["names_intervention"] = any(
        re.search(r"(?<![a-z0-9])" + re.escape(re.sub(r"s$", "", w.lower())), title_low)
        for c in plan["concepts"] if c.get("required") and c.get("role") in (
            "intervention", "exposure", "index_test", "prognostic_factor", "phenomenon", "concept")
        for w in ecp.concept_words(c, plan) + ([c["mesh"]] if c.get("mesh") else []))
    neg = [c for c in out["concepts"] if c["only_negated"] and not c["by_mesh"]
           and c["role"] in ("condition", "population", "presentation")]
    if neg and not out["certain_exclusion"]:
        out["negated_condition"] = neg[0]["term"]
    if out["role"] in ("subgroup_analysis", "secondary_analysis", "follow_up"):
        linked = [t for t in pub.get("trial_ids") or [] if t in primary_trials]
        if linked:
            out["secondary_of_trial"] = linked[0]
    return out


# ── Priority (which records the model reviews first) ────────────────────────

_FAMILY_WEIGHT = {"main": 10, "narrow": 7, "broad": 5, "recent": 6, "original_wording": 5, "guidelines": 8,
                  "condition_guidelines": 7, "systematic_reviews": 8, "cochrane": 6, "reviews": 3,
                  "alternative": 7, "update": 7, "long_term": 7, "landmark": 8}
AGGREGATED_FAMILIES = frozenset({"guidelines", "condition_guidelines", "systematic_reviews", "cochrane", "reviews"})


def _family_ranks(pub: dict[str, Any]) -> dict[str, int]:
    best: dict[str, int] = {}
    for h in pub.get("hits") or []:
        f = h.get("family")
        if f:
            best[f] = min(best.get(f, 10**6), int(h.get("rank") or 10**6))
    return best


def priority(pub: dict[str, Any]) -> float:
    """How strongly the approved searches point at a record (no access, citation
    or source-count bonus): for each search family that found it, the family's
    weight scaled by the record's rank in that family (rank 1 → full, 21 → half)."""
    return sum(_FAMILY_WEIGHT.get(f, 2) / (1.0 + (r - 1) / 20.0) for f, r in _family_ranks(pub).items())


# Searches that AND every "must appear" concept (PubMed checked the record against all of them,
# MeSH explosion included); the relaxed alternative and the free-wording search do not.
_ALL_CONCEPT_FAMILIES = frozenset({"main", "narrow", "broad", "guidelines", "systematic_reviews", "reviews",
                                   "cochrane", "recent", "update", "long_term", "landmark"})
_DESIGN_TYPES = frozenset({
    "randomized controlled trial", "controlled clinical trial", "clinical trial", "clinical trial, phase ii",
    "clinical trial, phase iii", "clinical trial, phase iv", "pragmatic clinical trial", "equivalence trial",
    "multicenter study", "observational study", "meta-analysis", "systematic review", "practice guideline",
    "guideline",
})


def design_lane(pub: dict[str, Any]) -> bool:
    """A trial, systematic review or guideline that every approved concept matched."""
    types = {t.lower() for t in (pub.get("article") or {}).get("publication_types_json") or []}
    return bool(types & _DESIGN_TYPES) and bool(set(_family_ranks(pub)) & _ALL_CONCEPT_FAMILIES)


def aggregated_priority(pub: dict[str, Any]) -> float:
    return sum(_FAMILY_WEIGHT.get(f, 2) / (1.0 + (r - 1) / 20.0)
               for f, r in _family_ranks(pub).items() if f in AGGREGATED_FAMILIES)


# ── Model judgment ──────────────────────────────────────────────────────────


def _candidate_block(cid: str, pub: dict[str, Any], sig: dict[str, Any]) -> str:
    art = pub.get("article") or {}
    abstract = (art.get("abstract") or "").strip()
    mesh = ((pub.get("pubmed") or {}).get("mesh") or [])[:14]
    lines = [
        f"[{cid}] {art.get('title') or '(no title)'}",
        f"Journal: {art.get('journal') or '?'} ({art.get('publication_year') or '?'}). "
        f"Types: {', '.join((art.get('publication_types_json') or [])[:6]) or 'n/a'}",
        f"MeSH: {', '.join(mesh) or 'n/a'}",
        f"Abstract: {abstract[:1800] + ('…' if len(abstract) > 1800 else '') if abstract else '(no abstract)'}",
    ]
    notes = [f"role (from title/types): {sig['role']}"]
    if sig.get("certain_exclusion"):
        notes.append(f"certain: {sig['certain_exclusion']}")
    if sig.get("negated_condition"):
        notes.append(f"every mention of '{sig['negated_condition']}' is negated")
    if sig.get("secondary_of_trial"):
        notes.append(f"a separate report of trial {sig['secondary_of_trial']}, whose main report is also listed")
    lines.append("Signals: " + "; ".join(notes))
    return "\n".join(lines)


def _question_block(ctx: dict[str, Any]) -> str:
    parts = [f"Question: {ctx['question']}", f"Approved reading: {ctx['summary']}",
             f"Question type: {ctx['question_type']} ({ctx['framework']})"]
    for role, values in ctx["elements"].items():
        parts.append(f"- {role}: {', '.join(values)}")
    if ctx["excluded"]:
        parts.append(f"- excluded by the clinician: {', '.join(ctx['excluded'])}")
    parts.append(f"Age group: {ctx['age_group'] or 'not stated (do not assume adults)'}")
    guide = DIMENSION_GUIDE.get(TYPE_FAMILY.get(ctx["question_type"], "treatment"), DIMENSION_GUIDE["treatment"])
    parts.append("How to read the dimensions for THIS question: "
                 + "; ".join(f"{k} = {v}" for k, v in guide.items() if k != "extra")
                 + (". " + guide["extra"] if guide.get("extra") else "."))
    parts.append(f"Must appear (approved): {', '.join(ctx['must_appear']) or 'n/a'}")
    if ctx["ranking_only"]:
        parts.append(f"Used to rank, not required: {', '.join(ctx['ranking_only'])}")
    return "\n".join(parts)


_ROLE_EXCLUSIONS = frozenset({"not_a_study", "no_results", "animal_or_lab", "retracted", "preprint"})


def policy_state(dims: dict[str, str], model_state: str, model_excl: Optional[str], *,
                 comparator_stated: bool = True, na_intervention_ok: bool = False,
                 stated: frozenset[str] = frozenset(), guideline: bool = False) -> tuple[str, Optional[str]]:
    """The admission state from the model's per-dimension labels — a fixed,
    explainable rule (more stable than one holistic call):

      condition different                    → excluded (wrong condition)
      intervention different                 → excluded (wrong intervention)
      population different / any core related → indirect
      comparison or outcome different        → supportive (when the question names no
                                               comparator, any non-matching comparison is
                                               a head-to-head study → supportive)
      outcome related (a surrogate or nearby
        outcome, not the one asked)          → supportive (not for guidelines: they
                                               recommend, they do not report outcomes)
      population or condition narrower (a subgroup) → supportive
      any core dimension broader             → broad-answering
      population and condition match, intervention match or narrower
        (a specific member of the asked class)  → direct

    The model's own call is kept when it excluded for a role reason (not a study,
    no results, animal/lab) or when two or more core dimensions are unclear. When
    the outcome (or a stated comparator) could not be judged from what was read,
    the rule never makes a paper stronger than the model's own call; when the
    labels say "answers" but the model's own call is indirect or excluded, the
    paper is shown as uncertain, not as an answer.

    ``stated`` names the dimensions the question states: "na" on one of them means
    the paper does not address it (e.g. a training study has no patients), which
    is treated as unclear — never as compatible with an answer."""
    dims = {k: ("unclear" if v == "na" and k in stated else v) for k, v in dims.items()}
    state, excl = _policy_state(dims, model_state, model_excl, comparator_stated=comparator_stated,
                                na_intervention_ok=na_intervention_ok, guideline=guideline)
    unclear = dims["outcome"] == "unclear" or (comparator_stated and dims["comparison"] == "unclear")
    if unclear and state in _STRENGTH and model_state in _STRENGTH and _STRENGTH[model_state] > _STRENGTH[state]:
        return model_state, None
    if state in QUALIFYING and model_state in ("indirect", "excluded"):
        return "uncertain", None
    return state, excl


# Weaker → larger. "uncertain" sits between a partial answer and an indirect one.
_STRENGTH = {"direct": 0, "broad_answering": 1, "supportive": 2, "uncertain": 3, "indirect": 4}


def _policy_state(dims: dict[str, str], model_state: str, model_excl: Optional[str], *,
                  comparator_stated: bool, na_intervention_ok: bool, guideline: bool) -> tuple[str, Optional[str]]:
    if model_state == "excluded" and model_excl in _ROLE_EXCLUSIONS:
        return "excluded", model_excl
    core = [dims["population"], dims["condition"], dims["intervention"]]
    if dims["condition"] == "different":
        return "excluded", "wrong_condition"
    if dims["intervention"] == "different":
        return "excluded", "wrong_intervention"
    if sum(v == "unclear" for v in core) >= 2:
        return model_state, model_excl if model_state == "excluded" else None
    if dims["population"] == "different" or "related" in core:
        return "indirect", None
    if dims["intervention"] == "na" and not na_intervention_ok:
        return "indirect", None  # the paper does not study the asked treatment / exposure / test
    head_to_head = dims["comparison"] in ("different", "related") or (
        not comparator_stated and dims["comparison"] in ("narrower", "broader"))
    if head_to_head or dims["outcome"] == "different" or (dims["outcome"] == "related" and not guideline):
        return "supportive", None
    if "narrower" in (dims["population"], dims["condition"]):
        return "supportive", None
    if "broader" in core:
        return "broad_answering", None
    if dims["population"] in ("match", "na") and dims["condition"] in ("match", "na") \
            and dims["intervention"] in ("match", "narrower", "na"):
        return "direct", None
    return model_state, model_excl if model_state == "excluded" else None


def validate_items(data: Any, ids: list[str], *, comparator_stated: bool = True,
                   na_intervention_ok: bool = False, stated: frozenset[str] = frozenset(),
                   roles: Optional[dict[str, str]] = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    items = data.get("items") if isinstance(data, dict) else None
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or it.get("id") not in ids or it["id"] in out:
            continue
        state = it.get("state")
        if state not in STATES:
            continue
        excl = it.get("exclusion")
        if state == "excluded" and excl not in EXCLUSIONS:
            excl = "other"
        if state != "excluded":
            excl = None
        reason = " ".join(str(it.get("reason") or "").split())[:240]
        dims = {k: (it.get(k) if it.get(k) in MATCH else "unclear")
                for k in ("population", "condition", "intervention", "comparison", "outcome", "setting")}
        model_state = state
        guideline = it.get("role") == "guideline" or (roles or {}).get(it["id"]) == "guideline"
        state, excl = policy_state(dims, state, excl, comparator_stated=comparator_stated,
                                   na_intervention_ok=na_intervention_ok, stated=stated, guideline=guideline)
        out[it["id"]] = {
            "state": state, "model_state": model_state, "exclusion": excl, "reason": reason or STATE_LABELS[state],
            "role": it.get("role") if it.get("role") in ROLES else None,
            "dimensions": dims,
            "confidence": it.get("confidence") if it.get("confidence") in ("high", "medium", "low") else "low",
        }
    return out


# A second look at an excluded paper (asked for by the clinician) leans towards keeping it:
# missing a relevant paper costs more than listing one more.
SECOND_LOOK = ("This paper was excluded earlier and the clinician asked for a second look. Re-read it against the "
               "question; exclude it only if it clearly does not study the asked condition, population or "
               "treatment/test/exposure, or is not a study. If it plausibly informs the answer, keep it "
               "(supportive or uncertain at least) and say why.")


async def judge_batch(gateway: ModelGateway, ctx: dict[str, Any], batch: list[tuple[str, dict, dict]],
                      note: Optional[str] = None) -> tuple[dict[str, dict[str, Any]], ModelCallResult]:
    user = _question_block(ctx) + ("\n\n" + note if note else "") + "\n\nCandidates:\n\n" + "\n\n".join(
        _candidate_block(cid, pub, sig) for cid, pub, sig in batch)
    res = await gateway.complete_json(SYSTEM_PROMPT, user, max_tokens=settings.ADMISSION_MAX_TOKENS,
                                      timeout_s=settings.ADMISSION_TIMEOUT_S)
    if res.data is None:
        return {}, res
    treatment_like = TYPE_FAMILY.get(ctx["question_type"], "treatment") == "treatment"
    family = TYPE_FAMILY.get(ctx["question_type"], "treatment")
    return validate_items(res.data, [cid for cid, _, _ in batch],
                          comparator_stated=bool(ctx["elements"].get("comparator")) or not treatment_like,
                          na_intervention_ok=family in ("prognosis", "epidemiology", "qualitative", "background"),
                          stated=stated_dimensions(ctx["elements"]),
                          roles={cid: sig.get("role") for cid, _, sig in batch}), res


_STATED_BY = {"population": ("population", "age_group", "sex", "pregnancy"), "condition": ("condition",),
              "outcome": ("outcome",)}


def stated_dimensions(elements: dict[str, list[str]]) -> frozenset[str]:
    """The judgment dimensions the question itself states (from its active elements)."""
    return frozenset(d for d, roles in _STATED_BY.items() if any(elements.get(r) for r in roles))


# ── Final state ─────────────────────────────────────────────────────────────


def finalize(sig: dict[str, Any], judged: Optional[dict[str, Any]], *, method: str,
             narrative_ok: bool = False, has_abstract: bool = True) -> dict[str, Any]:
    """Combine deterministic signals and the model's judgment. Signals can only
    make the state stricter."""
    if sig.get("certain_exclusion"):
        code = sig["certain_exclusion"]
        return {"state": "excluded", "exclusion": code, "reason": EXCLUSIONS[code], "role": sig["role"],
                "method": "rules", "dimensions": (judged or {}).get("dimensions"), "confidence": "high"}
    if judged is None:
        return {"state": "uncertain", "exclusion": None, "role": sig["role"], "method": method,
                "reason": "Not reviewed by the model (unavailable or over budget) — check the abstract.",
                "dimensions": None, "confidence": "low"}
    out = dict(judged)
    # A specific role from the record's own types wins; a title-only guess or a
    # generic role defers to the model, which read the abstract.
    keep = sig["role"] not in ("other", "primary_result") and sig.get("role_source") != "title"
    generic = judged.get("role") in (None, "", "other")
    out["role"] = sig["role"] if keep or generic else judged["role"]
    out["method"] = method
    if sig.get("negated_condition") and out["state"] in QUALIFYING + ("supportive",):
        out.update(state="excluded", exclusion="wrong_population",
                   reason=f"Studies patients without {sig['negated_condition']}.")
    if out.get("role") == "case_report" and out["state"] in QUALIFYING:
        out.update(state="supportive", reason="Case report — it cannot show the effect on its own. "
                                              + (judged.get("reason") or ""))
    if out.get("role") == "narrative_review" and out["state"] in QUALIFYING and not narrative_ok:
        out.update(state="supportive", reason="Narrative review (not systematic). " + (judged.get("reason") or ""))
    if not has_abstract and out["state"] in QUALIFYING and out.get("role") != "guideline":
        # A title alone is word presence, not a judgment of population, comparison and outcome.
        out.update(state="uncertain", reason="No abstract — judged on the title only; check the paper. "
                                             + (judged.get("reason") or ""))
    if sig.get("secondary_of_trial") and out["state"] in QUALIFYING:
        out.update(state="supportive",
                   reason=f"A separate report of trial {sig['secondary_of_trial']}; its main report is listed. "
                          + (judged.get("reason") or ""))
    return out


# ── Sections ────────────────────────────────────────────────────────────────

_TYPE_ORDER = {"guideline": 0, "systematic_review": 1, "meta_analysis": 1, "primary_result": 2, "other": 3,
               "follow_up": 4, "secondary_analysis": 4, "subgroup_analysis": 4, "narrative_review": 5,
               "case_report": 6}


def order_key(pub: dict[str, Any]) -> tuple:
    adm = pub.get("admission") or {}
    year = (pub.get("article") or {}).get("publication_year") or 0
    state_rank = STATES.index(adm["state"]) if adm.get("state") in STATES else len(STATES)
    return (state_rank, _TYPE_ORDER.get(adm.get("role") or "other", 3),
            -priority(pub), -year)


MAIN_REVIEWS = 5
_CURRENCY_ORDER = {"current": 0, None: 1, "older_than_window": 2, "superseded": 3}


def _is_listed(p: dict[str, Any]) -> bool:
    """Co-publications and executive summaries are listed under their edition's lead record."""
    lin = p.get("lineage") or {}
    return not lin.get("edition_id") or bool(lin.get("lead"))


def assign_sections(pubs: list[dict[str, Any]], *, newest_first: bool = False) -> dict[str, list[dict[str, Any]]]:
    """main (≈15: ≤3 current-first guidelines, ≤5 reviews, then primary studies) /
    other (≤5) / more / excluded / held_back / not_assessed. One entry per guideline edition.
    ``newest_first`` (background questions: definitions and explanations are replaced
    over time) orders the remaining studies by year before search strength."""
    sections: dict[str, list[dict[str, Any]]] = {k: [] for k in
                                                 ("main", "other", "more", "excluded", "held_back", "not_assessed")}

    def key(p: dict[str, Any]) -> tuple:
        adm = p.get("admission") or {}
        lin = p.get("lineage") or {}
        state = STATES.index(adm["state"]) if adm.get("state") in STATES else len(STATES)
        role = adm.get("role") or "other"
        year = (p.get("article") or {}).get("publication_year") or 0
        if role == "guideline":
            # Current editions first; guidelines on the asked treatment before general
            # ones; newest first; then the answer state.
            specific = 0 if (p.get("admission") or {}).get("signals", {}).get("names_intervention") else 1
            return (0, _CURRENCY_ORDER.get(lin.get("currency"), 1), specific, -(lin.get("edition_year") or year), state)
        if role in ("systematic_review", "meta_analysis"):
            # Newest cumulative evidence first (no age window for reviews): a newer review that
            # answers more broadly usually contains the older direct ones' trials and more.
            return (1, -year, state)
        if newest_first:
            return (2 + _TYPE_ORDER.get(role, 3), state, -year, -priority(p))
        return (2 + _TYPE_ORDER.get(role, 3), state, -priority(p), -year)

    ranked = sorted((p for p in pubs if _is_listed(p)), key=key)
    counts = {"guideline": 0, "review": 0}
    for p in ranked:
        adm = p.get("admission") or {}
        role = adm.get("role")
        kind = "guideline" if role == "guideline" else "review" if role in ("systematic_review", "meta_analysis") \
            else "other"
        cap_ok = (kind == "other" or (kind == "guideline" and counts["guideline"] < MAIN_GUIDELINES
                                      and (p.get("lineage") or {}).get("currency") != "superseded")
                  or (kind == "review" and counts["review"] < MAIN_REVIEWS))
        if p.get("identity_status") == "quarantined":
            sections["held_back"].append(p)
        elif adm.get("state") == "not_assessed":
            sections["not_assessed"].append(p)
        elif adm.get("state") == "excluded":
            sections["excluded"].append(p)
        elif adm.get("state") in QUALIFYING and len(sections["main"]) < MAIN_LIMIT and cap_ok:
            sections["main"].append(p)
            if kind != "other":
                counts[kind] += 1
        else:
            sections["more"].append(p)
    # Up to 5 optional extras, chosen in a stable, explainable order: supportive,
    # then uncertain, then indirect — never a superseded or out-of-window guideline.
    pref = {"supportive": 0, "uncertain": 1, "indirect": 2}
    pool = [p for p in sections["more"] if (p.get("admission") or {}).get("state") in pref
            and (p.get("lineage") or {}).get("currency") not in ("superseded", "older_than_window")]
    pool.sort(key=lambda p: (pref[p["admission"]["state"]], -priority(p)))
    sections["other"] = pool[:OTHER_LIMIT]
    chosen = {p["id"] for p in sections["other"]}
    sections["more"] = [p for p in sections["more"] if p["id"] not in chosen]
    return sections


def admission_gateway() -> ModelGateway:
    """Relevance judgments need the stronger model (as term classification does):
    CLINICAL_ENGINE_ADMISSION_MODEL, else the terms model, else GPT-4.1 on the provider."""
    base = ModelGateway()
    chosen = settings.ADMISSION_MODEL or settings.TERMS_MODEL or {
        "openai": "gpt-4.1", "openrouter": "openai/gpt-4.1"}.get(base.provider, "")
    return ModelGateway(model=chosen) if chosen else base


async def admit(pubs: list[dict[str, Any]], cqm: Any, plan: dict[str, Any], *,
                gateway: Optional[ModelGateway] = None, max_judged: Optional[int] = None,
                on_tokens: Any = None, call: Any = None) -> dict[str, Any]:
    """Attach ``admission`` to each publication (in place). Returns usage/method info."""
    ctx = question_context(cqm, plan)
    gateway = gateway or admission_gateway()
    primary_trials = {t for p in pubs if (p.get("report_role") == "primary_result") for t in p.get("trial_ids") or []}
    live = [p for p in pubs if p.get("identity_status") != "quarantined"]
    sigs = {p["id"]: signals(p, plan, primary_trials=primary_trials) for p in live}
    limit = settings.ADMISSION_MAX_JUDGED if max_judged is None else max_judged
    to_judge = sorted((p for p in live if not sigs[p["id"]].get("certain_exclusion")),
                      key=priority, reverse=True)
    judged_ids = {p["id"] for p in to_judge[:limit]}
    # Guidelines and reviews have their own share of the budget: their searches
    # rank them, not the main search.
    aggregated = sorted((p for p in to_judge if p["id"] not in judged_ids and aggregated_priority(p) > 0),
                        key=aggregated_priority, reverse=True)
    judged_ids |= {p["id"] for p in aggregated[: settings.ADMISSION_AGGREGATED_EXTRA]}
    designs = [p for p in to_judge if p["id"] not in judged_ids and design_lane(p)]
    judged_ids |= {p["id"] for p in designs[: settings.ADMISSION_DESIGN_EXTRA]}
    landmarks = sorted((p for p in to_judge if p["id"] not in judged_ids and "landmark" in _family_ranks(p)),
                       key=lambda p: _family_ranks(p)["landmark"])
    judged_ids |= {p["id"] for p in landmarks[: settings.ADMISSION_LANDMARK_EXTRA]}
    to_judge = [p for p in to_judge if p["id"] in judged_ids]
    batches: list[list[tuple[str, dict, dict]]] = []
    current: list[tuple[str, dict, dict]] = []
    for i, p in enumerate(to_judge):
        current.append((f"c{i + 1}", p, sigs[p["id"]]))
        if len(current) >= settings.ADMISSION_BATCH_SIZE:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    usage = {"model": gateway.model, "calls": 0, "tokens": 0, "failed_batches": 0, "judged": 0}
    results: dict[str, dict[str, Any]] = {}
    method = "model" if gateway.is_configured() else "rules_only"
    if method == "model":
        sem = asyncio.Semaphore(max(1, settings.ADMISSION_CONCURRENCY))

        async def run(batch: list[tuple[str, dict, dict]]) -> None:
            async with sem:
                if call is not None:  # through the run's tool gateway (allowlist + call budget)
                    got, res = await call(judge_batch, gateway, ctx, batch)
                else:
                    got, res = await judge_batch(gateway, ctx, batch)
                usage["calls"] += 1
                usage["tokens"] += res.tokens
                if on_tokens and res.tokens:
                    await on_tokens(res.tokens)
                missing = [b for b in batch if b[0] not in got]
                if res.data is None or len(missing) == len(batch):
                    usage["failed_batches"] += 1
                for cid, pub, _ in batch:
                    if cid in got:
                        results[pub["id"]] = got[cid]

        await asyncio.gather(*(run(b) for b in batches))
    usage["judged"] = len(results)
    for p in pubs:
        if p.get("identity_status") == "quarantined":
            p["admission"] = {"state": "held_back", "reason": "Identifiers from different sources disagree — held back.",
                              "method": "rules", "role": None, "exclusion": None}
            continue
        sig = sigs[p["id"]]
        if not sig.get("certain_exclusion") and p["id"] not in judged_ids:
            p["admission"] = {"state": "not_assessed", "method": "budget", "role": sig["role"], "exclusion": None,
                              "reason": f"Beyond the review budget ({len(judged_ids)} records reviewed, "
                                        "in search-strength order)."}
        else:
            p["admission"] = finalize(sig, results.get(p["id"]), method=method,
                                      narrative_ok=TYPE_FAMILY.get(ctx["question_type"]) == "background",
                                      has_abstract=bool(((p.get("article") or {}).get("abstract") or "").strip()))
        p["admission"]["signals"] = {k: v for k, v in sig.items() if k != "concepts"}
        p["admission"]["concepts"] = sig["concepts"]
    usage["method"] = method
    usage["context"] = {k: ctx[k] for k in ("question_type", "framework", "age_group", "must_appear")}
    return usage


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
