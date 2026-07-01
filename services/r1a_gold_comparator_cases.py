"""Phase R1A — Part 5 — gold-comparator benchmark cases seed.

DETERMINISTIC. NO PHI. NO LLM. Closed-set schema.

Seeds the 4 physician-curated cases from the M3.2 soak closeout
into ``r1a_gold_comparator_cases``. Idempotent — uses ``case_id``
as the unique key.

Schema (closed-set, append-only):
    case_id, question, shape, gold_article_dois (list),
    gold_article_pmids (list), gold_article_titles (list),
    expected_expansion_concepts (list),
    expected_downrank_lanes (list), created_at, notes (optional).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from core.db import Collections, get_db

logger = logging.getLogger("clinevidence.r1a_gold_cases")


# Closed-set seed payload. Each entry is what the M3.2 closeout
# captured verbatim. DO NOT add fabricated PMIDs or DOIs.
R1A_GOLD_COMPARATOR_CASES_SEED: list[dict[str, Any]] = [
    {
        "case_id": "r1a-case-A-cholestatic-pruritus",
        "question": "What is the mechanism of pruritus in cholestatic liver disease?",
        "shape": "pathophysiology_mechanism",
        "gold_article_dois": ["10.1038/s41575-022-00687-7"],
        "gold_article_pmids": [],
        "gold_article_titles": [
            "Mechanisms of pruritus in cholestasis: understanding and treating the itch.",
        ],
        "expected_expansion_concepts": [
            "cholestasis", "cholestatic liver disease",
            "cholestatic pruritus", "cholestasis-associated itch",
            "pruritus", "itch", "mechanisms", "pathophysiology",
            "molecular pathogenesis", "pruritogens",
            "lysophosphatidic acid", "autotaxin", "bile acids",
            "sulfated progesterone metabolites", "MRGPRX4", "TGR5",
            "C-fibres", "itch pathway",
        ],
        "expected_downrank_lanes": [
            "treatment_modality_indirect_for_mechanism",
            "IBAT inhibitor treatment",
        ],
        "notes": (
            "Beuers U, Wolters F, Oude Elferink RPJ. "
            "Nat Rev Gastroenterol Hepatol 2023;20:26–36."
        ),
    },
    {
        "case_id": "r1a-case-B-pcos-insulin-resistance",
        "question": "What is the pathophysiology of insulin resistance in polycystic ovary syndrome?",
        "shape": "pathophysiology_mechanism",
        "gold_article_dois": [],
        "gold_article_pmids": [],
        "gold_article_titles": [],
        "expected_expansion_concepts": [
            "polycystic ovary syndrome", "PCOS", "insulin resistance",
            "hyperinsulinemia", "insulin signaling",
            "androgen excess", "hyperandrogenism", "ovarian theca cell",
            "adipose tissue insulin resistance", "mitochondrial",
            "inflammation", "pathophysiology", "mechanism",
        ],
        "expected_downrank_lanes": [
            "treatment_modality_indirect_for_mechanism",
            "management_guideline_indirect_for_mechanism",
        ],
        "notes": (
            "Gold comparator NOT yet specified by operator — "
            "physician to curate. Run shadow simulation without a "
            "gold target; report label distribution + rank delta only."
        ),
    },
    {
        "case_id": "r1a-case-C-amyloid-diastolic-dysfunction",
        "question": "What is the mechanism of diastolic dysfunction in amyloid cardiomyopathy?",
        "shape": "pathophysiology_mechanism",
        "gold_article_dois": ["10.1016/j.jcmg.2018.12.034"],
        "gold_article_pmids": [],
        "gold_article_titles": [
            (
                "Diastolic Dysfunction and Heart Failure With Preserved Ejection "
                "Fraction: Understanding Mechanisms by Using Noninvasive Methods."
            ),
        ],
        "expected_expansion_concepts": [
            "cardiac amyloidosis", "amyloid cardiomyopathy",
            "transthyretin amyloid cardiomyopathy", "ATTR-CM",
            "AL amyloidosis", "restrictive cardiomyopathy",
            "diastolic dysfunction", "HFpEF", "myocardial stiffness",
            "impaired relaxation", "extracellular volume", "strain",
            "echocardiography", "noninvasive imaging",
            "myocardial infiltration",
        ],
        "expected_downrank_lanes": [
            "treatment_modality_indirect_for_mechanism",
            "diagnostic_modality_indirect_for_mechanism",
            "management_guideline_indirect_for_mechanism",
        ],
        "notes": (
            "Obokata M, Reddy YNV, Borlaug BA. JACC Cardiovasc Imaging. "
            "Comparator is not amyloid-specific; mechanism-rescue lane."
        ),
    },
    {
        "case_id": "r1a-case-D-cvc-pneumothorax",
        "question": "What is the risk of pneumothorax after central line placement?",
        "shape": "procedure_complication_risk",
        "gold_article_dois": [],
        "gold_article_pmids": [],
        "gold_article_titles": [
            (
                "Pneumothorax as a complication of central venous catheter "
                "insertion."
            ),
            (
                "Pneumothorax is a rare complication of thoracic central venous "
                "catheterization in community EDs."
            ),
            (
                "Incidence and trends of central line associated pneumothorax "
                "using radiograph report text search versus administrative "
                "database codes."
            ),
            "Central line complications.",
            (
                "Pneumothorax in the intensive care unit: incidence, risk "
                "factors, and outcome."
            ),
        ],
        "expected_expansion_concepts": [
            "central venous catheterization", "CVC",
            "central venous access", "central line placement",
            "subclavian catheterization",
            "internal jugular catheterization", "pneumothorax",
            "iatrogenic pneumothorax", "catheter-related pneumothorax",
            "procedure-related pneumothorax", "incidence", "risk",
            "risk factors", "complication rate", "mechanical complications",
        ],
        "expected_downrank_lanes": [
            "routine_imaging_utility_indirect",
            "training_or_detection_method_indirect",
            "diagnostic_followup_indirect_for_complication_risk",
        ],
        "notes": (
            "Tsotsolis 2015, Vinson 2015, Reeson 2018, Kornbau 2015, "
            "de Lassence 2006. Multi-comparator."
        ),
    },
]


async def seed_r1a_gold_comparator_cases() -> dict[str, Any]:
    """Idempotent upsert of the 4 closed-set R1A benchmark cases.

    Returns ``{seeded, inserted, updated}`` counters. Never raises
    on individual case failure.
    """
    db = get_db()
    inserted = 0
    updated = 0
    now = datetime.now(timezone.utc)
    for case in R1A_GOLD_COMPARATOR_CASES_SEED:
        case_with_meta = dict(case)
        case_with_meta["created_at"] = case_with_meta.get(
            "created_at", now,
        )
        case_with_meta["updated_at"] = now
        existing = await db[Collections.R1A_GOLD_COMPARATOR_CASES].find_one(
            {"case_id": case["case_id"]}, {"_id": 0, "case_id": 1},
        )
        result = await db[Collections.R1A_GOLD_COMPARATOR_CASES].update_one(
            {"case_id": case["case_id"]},
            {"$set": case_with_meta},
            upsert=True,
        )
        if existing:
            updated += 1 if result.modified_count else 0
        else:
            inserted += 1
    logger.info(
        "r1a_gold_comparator_cases_seeded inserted=%d updated=%d",
        inserted, updated,
    )
    return {
        "seeded": True,
        "inserted": inserted,
        "updated": updated,
        "total_cases": len(R1A_GOLD_COMPARATOR_CASES_SEED),
    }


async def list_gold_comparator_cases() -> list[dict[str, Any]]:
    db = get_db()
    out: list[dict[str, Any]] = []
    cursor = db[Collections.R1A_GOLD_COMPARATOR_CASES].find(
        {}, {"_id": 0},
    ).sort([("case_id", 1)])
    async for d in cursor:
        ca = d.get("created_at")
        if isinstance(ca, datetime):
            d["created_at"] = ca.isoformat()
        ua = d.get("updated_at")
        if isinstance(ua, datetime):
            d["updated_at"] = ua.isoformat()
        out.append(d)
    return out


async def get_gold_comparator_case(case_id: str) -> dict[str, Any] | None:
    db = get_db()
    d = await db[Collections.R1A_GOLD_COMPARATOR_CASES].find_one(
        {"case_id": case_id}, {"_id": 0},
    )
    if not d:
        return None
    ca = d.get("created_at")
    if isinstance(ca, datetime):
        d["created_at"] = ca.isoformat()
    ua = d.get("updated_at")
    if isinstance(ua, datetime):
        d["updated_at"] = ua.isoformat()
    return d


__all__ = [
    "R1A_GOLD_COMPARATOR_CASES_SEED",
    "seed_r1a_gold_comparator_cases",
    "list_gold_comparator_cases",
    "get_gold_comparator_case",
]
