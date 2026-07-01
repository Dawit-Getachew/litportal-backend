"""Phase K5.1 UI-verification seed.

Creates two synthetic clinician-owned sessions (idempotent) so the frontend
agent can VISUALLY verify the guideline-lane badge without depending on live
PubMed retrieval (which 429s with no API key):

  * guideline session  → primary_intent=prevention_screening + a Practice
    Guideline article → packet branch has guideline_lane_active=true.
  * mechanism session  → primary_intent=pathophysiology_mechanism + a
    guideline-pub-type article → packet must NOT surface the lane.

Run:  python -m scripts.seed_k51_ui_demo
"""
from __future__ import annotations

import asyncio

from core.db import Collections, get_db
from models.common import utcnow
from services.k4_evidence_packet_builder import build_evidence_packet

GUIDELINE_SID = "k51-ui-guideline-demo"
MECHANISM_SID = "k51-ui-mechanism-demo"
CLINICIAN_EMAIL = "clinician@test.com"


def _article(aid, title, year, *, is_guideline=False, pub_types=None):
    return {
        "id": aid,
        "pmid": aid.replace("art-", ""),
        "doi": f"10.5555/{aid}",
        "title": title,
        "abstract": "MOCK ABSTRACT — synthetic demo article (no real PHI).",
        "journal": "Demo Journal",
        "publication_year": year,
        "source": "pubmed",
        "full_text_status": "abstract_only",
        "is_guideline": is_guideline,
        "publication_types_json": pub_types or ["Journal Article"],
        "canonical_key": aid,
    }


def _sa(i, sid, aid):
    return {
        "id": f"sa-{i}-{sid}",
        "search_session_id": sid,
        "article_id": aid,
        "overall_rank": 100 - i,
        "relevance_score": 50 - i,
        "evidence_design_score": 10 - i,
        "included_by_default": True,
        "user_included": False,
        "exclusion_reason": None,
        "query_types_matched": ["high_recall"],
        "matched_branch_ids": ["b_main"],
        "sources_present": ["pubmed"],
        "created_at": utcnow(),
        "updated_at": utcnow(),
    }


def _session(sid, uid, intent, question):
    bqp = {
        "schema_version": "1.0",
        "mode": "branch_aware",
        "branches": [{
            "branch_id": "b_main",
            "intent": intent,
            "framework": intent,
            "query_families": [{"family_name": "high_recall"}],
            "required_concepts_used": [],
        }],
    }
    return {
        "id": sid,
        "user_id": uid,
        "original_question": question,
        "sanitized_question": question,
        "status": "awaiting_selection",
        "clinical_question_type": intent,
        "clinical_question_map_json": {
            "primary_intent": intent,
            "framework": intent,
            "branch_query_plan": bqp,
            "anchors": [],
            "required_anchors": [],
        },
        "created_at": utcnow(),
        "updated_at": utcnow(),
        "expires_at": utcnow().replace(year=utcnow().year + 1),
        "deleted_at": None,
    }


async def main():
    db = get_db()
    user = await db[Collections.USERS].find_one(
        {"email": CLINICIAN_EMAIL}, {"_id": 0, "id": 1},
    )
    if not user:
        raise SystemExit(f"clinician user {CLINICIAN_EMAIL} not found")
    uid = user["id"]

    plan = [
        (GUIDELINE_SID, "prevention_screening",
         "What are the guidelines for colorectal cancer screening in average-risk adults?",
         [
             _article("art-g1", "USPSTF-style practice guideline for colorectal cancer screening", 2024,
                      is_guideline=True, pub_types=["Practice Guideline"]),
             _article("art-g2", "Colorectal cancer screening adherence cohort study", 2023),
         ]),
        (MECHANISM_SID, "pathophysiology_mechanism",
         "What is the mechanism of pruritus in cholestatic liver disease?",
         [
             _article("art-m1", "Practice guideline on cholestatic pruritus management", 2024,
                      is_guideline=True, pub_types=["Practice Guideline"]),
             _article("art-m2", "Autotaxin and bile acids in cholestatic itch: mechanism review", 2023,
                      pub_types=["Review"]),
         ]),
    ]

    for sid, intent, question, arts in plan:
        # idempotent reset
        await db[Collections.SEARCH_SESSIONS].delete_one({"id": sid})
        await db[Collections.SEARCH_SESSION_ARTICLES].delete_many({"search_session_id": sid})
        await db[Collections.EVIDENCE_PACKETS].delete_one({"search_session_id": sid})
        for a in arts:
            await db[Collections.ARTICLES].update_one(
                {"id": a["id"]}, {"$set": a}, upsert=True,
            )
        await db[Collections.SEARCH_SESSIONS].insert_one(_session(sid, uid, intent, question))
        for i, a in enumerate(arts, start=1):
            await db[Collections.SEARCH_SESSION_ARTICLES].insert_one(_sa(i, sid, a["id"]))
        packet = await build_evidence_packet(search_session_id=sid)
        lane = [
            (b["branch_id"], b.get("guideline_lane_active"),
             [x.get("guideline_lane") for x in b["articles"]])
            for b in packet["branches"]
        ]
        print(f"SEED {sid} intent={intent} version={packet['evidence_packet_version']}")
        print(f"   lane_by_branch={lane}")
        print(f"   deep-link: /search/{sid}")

    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
