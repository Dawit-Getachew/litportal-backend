"""Versioned task plan for an approved question (W1.2) and its search plan.

The search plan lays out the three evidence lanes of the target architecture
from the *approved* interpretation only (suggestions the clinician did not
accept are excluded). In Week 1 it is recorded for inspection; retrieval runs
through the adapter named on the run until the W2 lanes are built.
"""
from __future__ import annotations

from typing import Any, Optional

from clinical_engine.config import settings
from clinical_engine.contracts.question import ClinicalQuestionMap
from clinical_engine.contracts.run import TaskDoc
from clinical_engine.interpretation.present import build_summary

PLAN_VERSION = "plan.v1"

_LANE_DESIGNS: dict[str, list[str]] = {
    "treatment": ["Randomized Controlled Trial", "Controlled Clinical Trial"],
    "prevention": ["Randomized Controlled Trial", "Controlled Clinical Trial", "Cohort Study"],
    "harm": ["Randomized Controlled Trial", "Cohort Study", "Case-Control Study"],
    "etiology_risk": ["Cohort Study", "Case-Control Study"],
    "diagnosis": ["Diagnostic accuracy study", "Cross-Sectional Study"],
    "screening": ["Randomized Controlled Trial", "Cohort Study"],
    "prognosis": ["Cohort Study"],
    "epidemiology": ["Cross-Sectional Study", "Cohort Study", "Registry"],
    "clinical_management": ["Randomized Controlled Trial", "Cohort Study"],
    "diagnostic_workup": ["Diagnostic accuracy study", "Cohort Study"],
    "prognostic_factor": ["Cohort Study"],
    "prediction_model": ["Validation Study", "Cohort Study"],
    "qualitative": ["Qualitative Research"],
    "evidence_map": [],
    "monitoring": ["Cohort Study", "Randomized Controlled Trial"],
    "complications": ["Cohort Study", "Case-Control Study"],
    "health_services": ["Randomized Controlled Trial", "Controlled Before-After Study"],
    "background": [],
    "other": [],
}
_AGGREGATED_TYPES = ["Practice Guideline", "Guideline", "Systematic Review", "Meta-Analysis"]
_BLOCK_ROLES = ("condition", "presentation", "population", "intervention", "exposure", "index_test",
                "prognostic_factor", "phenomenon", "concept", "comparator", "outcome")


def _term(value: str) -> str:
    return f'"{value}"' if " " in value else value


def concept_blocks(cqm: ClinicalQuestionMap) -> list[dict[str, Any]]:
    blocks = []
    for role in _BLOCK_ROLES:
        terms = [e.value for e in cqm.active(role)]
        if terms:
            blocks.append({"role": role, "terms": terms, "mesh": "pending (W2.1)"})
    return blocks


def _sketch(blocks: list[dict[str, Any]], excluded: list[str], filter_types: list[str],
            recent: bool) -> str:
    # Comparators and outcomes rank rather than restrict (AND-ing them loses
    # relevant trials); the sketch ANDs only the core question concepts.
    core = [b for b in blocks if b["role"] not in ("comparator", "outcome")] or blocks
    parts = ["(" + " OR ".join(_term(t) for t in b["terms"]) + ")" for b in core]
    query = " AND ".join(parts) if parts else "(original wording)"
    if excluded:
        query += " NOT (" + " OR ".join(_term(t) for t in excluded) + ")"
    if filter_types:
        query += " AND (" + " OR ".join(f"{t}[pt]" for t in filter_types) + ")"
    if recent:
        query += " AND (last 3 years[dp])"
    return query


def search_plan(cqm: ClinicalQuestionMap, version: int,
                concept_plan: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    blocks = concept_blocks(cqm)
    excluded = [e.value for e in cqm.elements if e.negated and e.role != "age_group"]
    excluded += [{"child": "child", "pediatric": "pediatric", "adult": "adult",
                  "older_adult": "aged"}.get(e.value, e.value)
                 for e in cqm.elements if e.negated and e.role == "age_group"]
    designs = _LANE_DESIGNS.get(cqm.question_type, [])
    lanes = [
        {"lane": "aggregated", "purpose": "Guidelines, systematic reviews and meta-analyses (all years).",
         "publication_types": _AGGREGATED_TYPES, "date_window": "all years",
         "query_sketch": _sketch(blocks, excluded, _AGGREGATED_TYPES, recent=False), "executed": False},
        {"lane": "foundational_primary", "purpose": "Direct primary studies with a design suited to the question.",
         "publication_types": designs, "date_window": "all years",
         "query_sketch": _sketch(blocks, excluded, designs, recent=False), "executed": False},
        {"lane": "recent_primary", "purpose": "Recent primary studies first; targeted update search after cutoff verification (W2.3).",
         "publication_types": designs, "date_window": "last 3 years",
         "query_sketch": _sketch(blocks, excluded, designs, recent=True), "executed": False},
    ]
    return {
        "plan_version": PLAN_VERSION,
        "question_version": version,
        "question_type": cqm.question_type,
        "framework": cqm.framework,
        "original_wording": cqm.question_text,
        "concept_blocks": blocks,
        "excluded_terms": excluded,
        "constraints": {
            "age_group": cqm.age_group(),
            "setting": [e.value for e in cqm.active("setting")],
            "time_frame": [e.value for e in cqm.active("time_frame")],
            "publication_window": [e.value for e in cqm.active("publication_window")],
        },
        "not_applied_suggestions": [{"role": e.role, "value": e.value} for e in cqm.suggestions()],
        "lanes": lanes,
        "limits": {"per_lane_candidates": 50, "max_total_candidates": 150},
        "approved_search": _approved_search(concept_plan),
        "notes": [
            "Engine retrieval (Week 2) runs the approved query portfolio in these lanes; questions without "
            "reviewed search terms still use the existing pipeline.",
            "With approved search terms, the retrieval adapter searches exactly those terms "
            "(see approved_search); otherwise the existing pipeline reads the question itself.",
        ],
    }


def _approved_search(concept_plan: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not concept_plan:
        return None
    from services.engine_concept_plan import full_query, validate_plan

    plan = validate_plan(concept_plan)
    return {
        "terms_version": concept_plan.get("terms_version"),
        "terms_hash": concept_plan.get("terms_hash"),
        "use_synonyms": concept_plan.get("use_synonyms"),
        "filters": concept_plan.get("filters"),
        "concepts": concept_plan.get("concepts"),
        "queries": {s: full_query(plan, s) for s in ("pubmed", "europe_pmc")} if plan else {},
    }


_POPULATION_FOCUS = {
    "adult": "adults", "older_adult": "older_adults", "child": "pediatrics", "pediatric": "pediatrics",
    "adolescent": "pediatrics", "infant": "pediatrics", "neonate": "pediatrics",
}


def legacy_search_inputs(cqm: ClinicalQuestionMap, change_kind: str) -> dict[str, str]:
    """What the Week-1 legacy adapter sends to the existing pipeline.

    Unedited interpretations keep the clinician's own words; after chip edits
    the approved one-line interpretation is sent, so edits reach the search.
    """
    text = cqm.question_text if change_kind != "chip_edit" else build_summary(cqm).rstrip(".")
    focus = "pregnancy" if cqm.active("pregnancy") else _POPULATION_FOCUS.get(cqm.age_group(), "any")
    return {"question": text[:5000], "population_focus": focus,
            "source": "original_wording" if change_kind != "chip_edit" else "approved_interpretation"}


_ENGINE_TASKS: list[tuple[str, str, list[str], list[str]]] = [
    ("plan_search", "plan_search", [], []),
    ("retrieve", "engine_retrieve", ["plan_search"], ["pubmed.search", "europepmc.search"]),
    ("union", "engine_union", ["retrieve"], []),
    ("admit", "engine_admit", ["union"], ["model.json"]),
    ("lineage", "engine_lineage", ["admit"], ["pubmed.search", "europepmc.search", "model.json"]),
    ("project", "engine_project", ["lineage"], []),
    ("access", "engine_access", ["project"], ["access.resolve"]),
]


def build_tasks(run: dict, *, mode: str, adapter: str) -> list[dict]:
    if adapter == "engine":
        specs = list(_ENGINE_TASKS)
        if mode == "answer_for_me":
            specs.append(("delegated_selection", "delegated_selection", ["access"], []))
        return [TaskDoc(
            run_id=run["id"], question_version_id=run["question_version_id"], key=key, type=ttype,
            seq=seq, depends_on=deps, tools=tools, max_attempts=settings.TASK_MAX_ATTEMPTS,
            idempotency_key=f"{run['id']}:{key}",
        ).model_dump() for seq, (key, ttype, deps, tools) in enumerate(specs, start=1)]
    specs: list[tuple[str, str, list[str], list[str]]] = [
        ("plan_search", "plan_search", [], []),
        ("retrieve", "retrieve",
         ["plan_search"], ["legacy.search_session" if adapter == "legacy_pipeline" else "mock.evidence_search"]),
    ]
    last = "retrieve"
    if adapter == "legacy_pipeline":
        specs.append(("await_retrieval", "await_retrieval", ["retrieve"], []))
        last = "await_retrieval"
    specs.append(("integrity_check", "integrity_check", [last], []))
    if mode == "answer_for_me":
        specs.append(("delegated_selection", "delegated_selection", ["integrity_check"], []))
    tasks = []
    for seq, (key, ttype, deps, tools) in enumerate(specs, start=1):
        tasks.append(TaskDoc(
            run_id=run["id"], question_version_id=run["question_version_id"], key=key, type=ttype,
            seq=seq, depends_on=deps, tools=tools, max_attempts=settings.TASK_MAX_ATTEMPTS,
            idempotency_key=f"{run['id']}:{key}",
        ).model_dump())
    return tasks
