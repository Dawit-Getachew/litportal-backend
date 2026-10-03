"""Approved search terms → the plan the search pipeline runs, plus previews."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

from clinical_engine.contracts.question import ClinicalQuestionMap
from clinical_engine.frameworks import FRAMEWORK_LABELS, INTENT_FOR_TYPE
from clinical_engine.search_terms.contracts import SearchTerms, terms_hash
from services.engine_concept_plan import PLAN_VERSION, full_query, validate_plan

logger = logging.getLogger("litportal.clinical_engine.search_plan")

# Engine question type → what the legacy ranker / study-type router expect.
LEGACY_TYPE: dict[str, str] = {
    "treatment": "therapy", "prevention": "prevention", "harm": "etiology_harm",
    "etiology_risk": "etiology_harm", "complications": "etiology_harm", "diagnosis": "diagnosis",
    "diagnostic_workup": "diagnosis", "screening": "prevention", "prognosis": "prognosis",
    "prognostic_factor": "prognosis", "prediction_model": "prognosis", "epidemiology": "epidemiology",
    "clinical_management": "guideline", "monitoring": "guideline", "qualitative": "background",
    "evidence_map": "background", "health_services": "background", "background": "background",
}
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def to_concept_plan(terms: SearchTerms, cqm: ClinicalQuestionMap) -> dict[str, Any]:
    """Plain-dict plan stored on the search session (read by services/engine_concept_plan)."""
    return {
        "plan_version": PLAN_VERSION,
        "question_type": cqm.question_type,
        "framework": cqm.framework,
        "framework_label": FRAMEWORK_LABELS.get(cqm.framework, cqm.framework),
        "intent_label": INTENT_FOR_TYPE[cqm.question_type].label if cqm.question_type in INTENT_FOR_TYPE else None,
        "legacy_type_of_question": LEGACY_TYPE.get(cqm.question_type),
        "legacy_subtype": "prediction_model" if cqm.question_type == "prediction_model" else None,
        "question_version": terms.question_version,
        "terms_version": terms.version,
        "terms_hash": terms_hash(terms),
        "use_synonyms": terms.use_synonyms,
        "filters": {"publication_years": terms.filters.publication_years,
                    "exclude_animal_only": terms.filters.exclude_animal_only},
        "age_group": cqm.age_group(),
        "concepts": [
            {
                "id": c.id, "role": c.role, "term": c.term, "required": c.required,
                "mesh": c.mesh.descriptor if c.mesh and c.mesh.enabled else None,
                # A broader heading the clinician was offered but left off — only the
                # bounded alternative search may use it, and says so.
                "broader_mesh": (c.mesh.descriptor if c.mesh and not c.mesh.enabled and c.mesh.match == "broader"
                                 else None),
                "synonyms": [t.text for t in c.terms if t.enabled] if terms.use_synonyms else [],
            }
            for c in terms.concepts
        ],
    }


def query_preview(terms: SearchTerms, cqm: ClinicalQuestionMap) -> dict[str, Optional[str]]:
    plan = validate_plan(to_concept_plan(terms, cqm))
    if plan is None:
        return {"pubmed": None, "europe_pmc": None}
    return {"pubmed": full_query(plan, "pubmed") or None, "europe_pmc": full_query(plan, "europe_pmc") or None}


# Counts go through the engine's source client: one rate limit per source host
# (shared with the search-plan preview and retrieval, which run at the same
# moment in step 2) and 429/5xx retries — a burst no longer shows "?".
async def _pubmed_count(client: httpx.AsyncClient, query: str) -> Optional[int]:
    from clinical_engine.retrieval import pubmed as pubmed_src

    return (await pubmed_src.esearch(client, query, retmax=0)).count


async def _epmc_count(client: httpx.AsyncClient, query: str) -> Optional[int]:
    from clinical_engine.retrieval.http import request

    resp = await request(client, "GET", EPMC_SEARCH, host="europepmc",
                         params={"query": query, "format": "json", "pageSize": 1, "resultType": "idlist"})
    return int(resp.json().get("hitCount"))


async def hit_counts(preview: dict[str, Optional[str]], *,
                     transport: Optional[httpx.AsyncBaseTransport] = None) -> dict[str, Optional[int]]:
    """How many records the approved search matches in each source — counts only,
    nothing is retrieved. None when a source could not be reached."""
    from clinical_engine.retrieval.http import SourceError

    async def safe(fn, query: Optional[str]) -> Optional[int]:
        if not query:
            return None
        try:
            return await fn(client, query)
        except (SourceError, httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning("hit count failed: %s", getattr(exc, "code", type(exc).__name__))
            return None

    async with httpx.AsyncClient(timeout=8.0, transport=transport) as client:
        pubmed, epmc = await asyncio.gather(safe(_pubmed_count, preview.get("pubmed")),
                                            safe(_epmc_count, preview.get("europe_pmc")))
    return {"pubmed": pubmed, "europe_pmc": epmc}
