"""Phase K4 — branch-aware evidence packet builder.

DETERMINISTIC. NO LLM. NO retrieval. NO new connector calls. Builds
a clinician-visible packet by:

  1. Reading existing ``search_session_articles`` flat list.
  2. Reading ``clinical_question_map_json.branch_query_plan.branches``.
  3. Assigning each article to a branch using existing
     ``search_queries`` provenance (``query_family``) intersected
     with each branch's ``query_families[*].family_name``.
  4. Computing R1A + R1C directness labels from each article's
     metadata via existing label detectors (display-only — does NOT
     mutate active ranking, selection, or summary).
  5. Mapping internal labels to clinician-readable strings via
     ``k4_evidence_packet_display_map`` (raises ``KeyError`` if any
     label is unmapped — fail loud).
  6. Leak-scanning the persisted doc.

Branch-priority rule for multi-branch articles (closed-set,
operator-confirmed):
  * (1) If any of the article's ``matched_branch_ids`` matches the
    branch whose ``intent`` equals the session's ``primary_intent``,
    assign to THAT branch.
  * (2) Else assign to the branch with the LOWEST index in
    ``branch_query_plan.branches`` whose family set intersects the
    article's query-family provenance.
  * (3) Articles with NO branch provenance → ``branch_id = "unbranched"``.

NO clinician-visible ranking change. NO default-selection change.
NO summary or citation-verifier change. R1A label weights, R1B
configuration, R1C anchor labels — UNCHANGED.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from core.db import Collections, get_db
from models.evidence_packets import (
    EVIDENCE_PACKET_VERSION,
    EvidencePacketArticle,
    EvidencePacketBranch,
    EvidencePacketDoc,
    EvidencePacketProvenance,
)
from models.common import utcnow
from services.k4_evidence_packet_display_map import (
    R_SERIES_DISPLAY_MAP,
    map_labels_to_displays,
)
from services.r1a_directness_labels import (
    detect_labels_for_shape,
)
from services.r1c_anchor_labels import detect_r1c_labels
from services.guideline_lane import (
    GUIDELINE_LANE_LABEL,
    GUIDELINE_LANE_TOOLTIP,
    article_guideline_provenance,
    classify_guideline_intent,
)
from services.evidence_lens import preference_packet_metadata

logger = logging.getLogger("clinevidence.k4_packet")

UNBRANCHED_ID = "unbranched"

# Closed-set R-series version sentinels. Bumped when R1A/B/C/C.1
# label catalogs or weights are changed.
R_SERIES_VERSIONS: dict[str, str] = {
    "r1a": "1.0",
    "r1b": "1.0",
    "r1c": "1.0",
    "r1c_1": "1.1",
}


# ---------------------------------------------------------------------
# Leak-scan helpers — fail loud at write time.
# ---------------------------------------------------------------------
_FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "abstract", "abstractText", "fullText", "raw_full_text",
    "prompt", "completion", "system_prompt", "user_prompt",
    "api_key", "password", "secret", "x_api_key",
    "evidence_text", "uploaded_text",
})

_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "bearer ", "sk-", "system:", "assistant:",
    "authorization:", "x-api-key", "password=", "secret=",
)


def _leak_scan(doc: dict) -> None:
    """Raise ``ValueError`` if any forbidden key or substring is present.

    Closed-set guard — never persist abstracts, prompts, completions,
    credentials, or PHI-like strings to the evidence_packets collection.
    """
    def walk(obj: Any, path: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                if kl in _FORBIDDEN_KEYS:
                    raise ValueError(
                        f"k4 packet leak-scan: forbidden key {k!r} at {path}"
                    )
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, x in enumerate(obj):
                walk(x, f"{path}[{i}]")
        elif isinstance(obj, str):
            sl = obj.lower()
            for sub in _FORBIDDEN_SUBSTRINGS:
                if sub in sl:
                    raise ValueError(
                        f"k4 packet leak-scan: forbidden substring "
                        f"{sub!r} at {path}"
                    )
    walk(doc)


# ---------------------------------------------------------------------
# Branch-priority assignment.
# ---------------------------------------------------------------------
def _normalize_family(name: str) -> str:
    """Strip the ``cb_`` prefix the pipeline prepends to query_type
    entries (e.g., ``cb_high_recall`` → ``high_recall``)."""
    s = str(name or "").strip().lower()
    if s.startswith("cb_"):
        return s[3:]
    return s


def _branch_families(branch: dict) -> set[str]:
    """Return the set of normalized family_name strings for a branch."""
    out: set[str] = set()
    for qf in branch.get("query_families") or []:
        if isinstance(qf, dict):
            fam = qf.get("family_name")
            if fam:
                out.add(_normalize_family(fam))
    return out


def _article_family_provenance(sa_row: dict) -> set[str]:
    """Return the set of normalized family_name strings that this
    article's ``query_types_matched`` / ``query_family_matches`` carry.
    """
    fams: set[str] = set()
    for k in ("query_family_matches", "query_types_matched"):
        for fam in sa_row.get(k) or []:
            n = _normalize_family(fam)
            if n:
                fams.add(n)
    return fams


def _matched_branch_ids_for_article(
    sa_row: dict,
    article_doc: dict,
) -> set[str]:
    """Collect closed-set branch_id strings already attached to the
    article record (from K3A lane classification — additive metadata,
    NOT clinician-visible) so we honor them in priority rule (1).
    """
    out: set[str] = set()
    for k in ("matched_branch_ids",):
        for b in (sa_row.get(k) or []):
            if isinstance(b, str) and b:
                out.add(b)
        for b in (article_doc.get(k) or []):
            if isinstance(b, str) and b:
                out.add(b)
    return out


def assign_branch_for_article(
    *,
    sa_row: dict,
    article_doc: dict,
    branches: list[dict],
    primary_intent: str,
) -> str:
    """Closed-set deterministic branch assignment per K4 rule (c).

    K4 fix-up: when the session has exactly ONE planner branch, every
    article belongs to that branch — there is nowhere else for an
    article to go (legacy retrieval families don't carry a
    ``branch_id``, and the planner declares one branch's worth of
    work). Falling back to "unbranched" for a single-branch session
    would empty the only branch and defeat the K4 promise.

    Returns the chosen ``branch_id`` (or ``"unbranched"`` if no
    branch's family set intersects the article's provenance AND
    the article carries no ``matched_branch_ids``).
    """
    # Closed-set single-branch fallback (K4 fix-up).
    if len(branches) == 1:
        only_bid = branches[0].get("branch_id") or ""
        if only_bid:
            return only_bid

    matched_ids = _matched_branch_ids_for_article(sa_row, article_doc)
    article_fams = _article_family_provenance(sa_row)
    pi = (primary_intent or "").strip().lower()

    # Rule (1) — primary_intent's branch wins if it's in matched_ids
    # OR if its family set intersects the article's provenance.
    if pi:
        for br in branches:
            br_id = br.get("branch_id") or ""
            if not br_id:
                continue
            br_intent = str(br.get("intent") or "").strip().lower()
            if br_intent != pi:
                continue
            if br_id in matched_ids:
                return br_id
            if article_fams & _branch_families(br):
                return br_id

    # Rule (2) — plan order: first branch whose family set or
    # matched_branch_ids intersects.
    for br in branches:
        br_id = br.get("branch_id") or ""
        if not br_id:
            continue
        if br_id in matched_ids:
            return br_id
        if article_fams & _branch_families(br):
            return br_id

    # Rule (3) — fallback.
    return UNBRANCHED_ID


# ---------------------------------------------------------------------
# R-series label computation.
# ---------------------------------------------------------------------
def _r_series_labels_for_article(
    *,
    article_doc: dict,
    shape: str,
    case_id: Optional[str] = None,
) -> tuple[list[str], bool]:
    """Compute the closed-set union of R1A + R1C labels for ONE
    article. Returns ``(labels, is_mixed_signal)``.

    ``is_mixed_signal`` fires when the article carries BOTH a
    positive AND a negative R-series label — mirrors the existing
    R1A mixed-signal posture.
    """
    labels = detect_labels_for_shape(shape, article_doc)
    if case_id:
        labels = labels + detect_r1c_labels(
            case_id=case_id, article=article_doc, shape=shape,
        )
    # dedupe preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for lab in labels:
        if lab in seen:
            continue
        seen.add(lab)
        deduped.append(lab)
    # mixed-signal: has both polarities.
    polarities = {
        R_SERIES_DISPLAY_MAP[lab]["polarity"]
        for lab in deduped
        if lab in R_SERIES_DISPLAY_MAP
    }
    is_mixed = "positive" in polarities and "negative" in polarities
    return deduped, is_mixed


# ---------------------------------------------------------------------
# Per-article eligibility helpers.
# ---------------------------------------------------------------------
def _access_eligibility(article_doc: dict) -> str:
    val = article_doc.get("full_text_status") or "unknown"
    if val not in (
        "full_text_available", "free_full_text_link",
        "abstract_only", "unknown",
    ):
        return "unknown"
    return val


def _ai_ingestion_eligibility(article_doc: dict) -> str:
    """Closed-set: eligible if OA full text available; requires_upload
    if abstract_only; not_eligible if explicit retraction; else unknown.
    """
    if article_doc.get("is_retracted"):
        return "not_eligible"
    fts = article_doc.get("full_text_status")
    if fts == "full_text_available":
        return "eligible"
    if fts == "free_full_text_link":
        return "eligible"
    if fts == "abstract_only":
        return "requires_upload"
    return "unknown"


def _selected_state(sa_row: dict) -> str:
    if sa_row.get("user_included"):
        return "selected_by_user"
    if sa_row.get("included_by_default"):
        return "selected_by_default"
    return "unselected"


def _summary_eligibility(*, sa_row: dict, article_doc: dict) -> bool:
    state = _selected_state(sa_row)
    if state == "unselected":
        return False
    return _ai_ingestion_eligibility(article_doc) in ("eligible",)


def _citation_verification_readiness(article_doc: dict) -> str:
    doi = (article_doc.get("doi") or "").strip()
    pmid = (article_doc.get("pmid") or "").strip()
    if doi and pmid:
        return "ready"
    if not doi:
        return "missing_doi"
    return "missing_pmid"


# ---------------------------------------------------------------------
# Closed-set CQM readers.
# ---------------------------------------------------------------------
def _cqm_primary_framework(cqm: dict) -> Optional[str]:
    """Return the CQM's primary framework, or None.

    The persisted ``ClinicalQuestionMap`` carries ``frameworks`` (a
    LIST, ``clinical_question_map.py`` ``frameworks: List[Framework]``)
    under ``ConfigDict(extra="forbid")`` — there is NO scalar
    ``framework`` key at the top level. Reading ``cqm["framework"]``
    therefore always yielded None, silently nulling every framework
    signal in this builder (guideline-lane gate, R-series shape
    dispatch, and the persisted snapshot). Read the list.
    """
    fws = cqm.get("frameworks")
    if isinstance(fws, list):
        for fw in fws:
            if isinstance(fw, str) and fw.strip():
                return fw.strip()
    return None


def _cqm_anchor_concepts(cqm: dict, *, required_only: bool) -> list[str]:
    """Canonical concepts from ``framework_neutral_slots``.

    Same class of defect as ``framework``: the CQM has no ``anchors`` /
    ``required_anchors`` keys. The anchors live on the slots, where
    ``required_for_search`` is the closed-set retrieval flag (kept in
    lock-step with ``slot_search_policy`` by the Slot validator).
    """
    out: list[str] = []
    for slot in (cqm.get("framework_neutral_slots") or []):
        if not isinstance(slot, dict):
            continue
        if required_only and not slot.get("required_for_search"):
            continue
        concept = str(slot.get("canonical_concept") or "").strip()
        if concept and concept not in out:
            out.append(concept)
    return out


# ---------------------------------------------------------------------
# Public builder.
# ---------------------------------------------------------------------
async def build_evidence_packet(
    *,
    search_session_id: str,
) -> dict:
    """Build (or rebuild) the evidence packet for ``search_session_id``.

    Idempotent. Reads from the existing collections; writes ONE
    document into ``evidence_packets`` (upsert on
    ``search_session_id``). Returns the persisted packet dict.

    Raises ``ValueError`` if the leak scan fails.
    Raises ``KeyError`` (via display map) if any R-series label is
    missing a clinician-readable string.
    """
    db = get_db()
    session_doc = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": search_session_id}, {"_id": 0},
    )
    if not session_doc:
        raise LookupError(f"search_session not found: {search_session_id}")

    cqm = session_doc.get("clinical_question_map_json") or {}
    bqp = cqm.get("branch_query_plan") or {}
    branches_raw = list(bqp.get("branches") or [])
    primary_intent = str(cqm.get("primary_intent") or "").strip().lower()
    primary_framework = _cqm_primary_framework(cqm)

    # Phase K5.1 — guideline-lane gate. Surfaced only for guideline /
    # management / protocol / stabilization / monitoring / prevention /
    # screening / diagnostic-workup questions. Deterministic; no network.
    guideline_intent = classify_guideline_intent(
        primary_intent=cqm.get("primary_intent"),
        secondary_intents=(
            cqm.get("secondary_intents")
            or cqm.get("secondary_intents_json")
        ),
        framework=primary_framework,
        question=(
            session_doc.get("sanitized_question")
            or session_doc.get("original_question")
        ),
    )
    guideline_applicable = bool(guideline_intent.get("applicable"))

    # Shape used to dispatch R-series detectors. R1A only fires for
    # the closed set ``{pathophysiology_mechanism,
    # procedure_complication_risk}`` — every other shape returns []
    # from ``detect_labels_for_shape``.
    #
    # K4 fix-up: derive shape primarily from
    # ``clinical_question_map_json.primary_intent`` (operator-curated)
    # rather than the legacy ``clinical_question_type`` field, which
    # often carries coarser labels (``other``, ``harm``, etc.) that
    # don't map to R1A's supported shapes. Apply a closed-set alias
    # so equivalent intents (e.g., ``risk_etiology_harm``) route to
    # the right detector.
    shape_source = (
        cqm.get("primary_intent")
        or primary_framework
        or session_doc.get("clinical_question_type")
        or ""
    )
    shape_raw = str(shape_source).strip().lower()
    _SHAPE_ALIASES: dict[str, str] = {
        # primary_intent aliases that route to R1A procedure detector
        "risk_etiology_harm": "procedure_complication_risk",
        "procedure_complication": "procedure_complication_risk",
        "procedure_complication_risk": "procedure_complication_risk",
        # primary_intent aliases that route to R1A mechanism detector
        "pathophysiology_mechanism": "pathophysiology_mechanism",
        "concept_mechanism": "pathophysiology_mechanism",
    }
    shape = _SHAPE_ALIASES.get(shape_raw, shape_raw)

    # Read the flat article list.
    sa_rows = []
    async for r in db[Collections.SEARCH_SESSION_ARTICLES].find(
        {"search_session_id": search_session_id}, {"_id": 0},
    ):
        sa_rows.append(r)
    # deterministic ordering by overall_rank desc then created_at
    sa_rows.sort(
        key=lambda r: (-int(r.get("overall_rank") or 0), r.get("created_at") or 0),
    )

    article_ids = [r["article_id"] for r in sa_rows]
    articles_by_id: dict[str, dict] = {}
    if article_ids:
        async for art in db[Collections.ARTICLES].find(
            {"id": {"$in": article_ids}}, {"_id": 0},
        ):
            articles_by_id[art["id"]] = art

    # Snapshot CQM (closed-set canonical fields only — no raw user text)
    cqm_snapshot = {
        "primary_intent": cqm.get("primary_intent"),
        "framework": primary_framework,
        "frameworks": list(cqm.get("frameworks") or []),
        "anchors": _cqm_anchor_concepts(cqm, required_only=False),
        "required_anchors": _cqm_anchor_concepts(cqm, required_only=True),
        "branch_query_plan_summary": {
            "schema_version": bqp.get("schema_version"),
            "mode": bqp.get("mode"),
            "branch_count": len(branches_raw),
        },
    }

    # Bucket articles per branch_id.
    bucket: dict[str, list[tuple[dict, dict, int]]] = {}
    for global_rank, r in enumerate(sa_rows, start=1):
        article_doc = articles_by_id.get(r["article_id"])
        if not article_doc:
            continue
        if r.get("exclusion_reason"):
            # Excluded articles do NOT appear in evidence packet branches;
            # they remain in the legacy ``excluded_articles`` payload.
            continue
        branch_id = assign_branch_for_article(
            sa_row=r,
            article_doc=article_doc,
            branches=branches_raw,
            primary_intent=primary_intent,
        )
        bucket.setdefault(branch_id, []).append((r, article_doc, global_rank))

    # Build per-branch records in plan order; unbranched last.
    branches_out: list[EvidencePacketBranch] = []
    seen_ids: set[str] = set()
    for br in branches_raw:
        br_id = br.get("branch_id") or ""
        if not br_id:
            continue
        seen_ids.add(br_id)
        items = bucket.get(br_id, [])
        articles_out = _build_article_records(
            items=items, shape=shape, case_id=None,
            guideline_applicable=guideline_applicable,
        )
        branches_out.append(_apply_guideline_lane_to_branch(
            EvidencePacketBranch(
                branch_id=br_id,
                framework=str(br.get("framework") or ""),
                intent=str(br.get("intent") or ""),
                branch_summary=_branch_summary_text(br),
                query_terms_used=_branch_query_terms_used(br),
                articles=articles_out,
            ),
            guideline_applicable=guideline_applicable,
        ))
    # Buckets the planner never declared (e.g., legacy or unbranched).
    extra_ids = [bid for bid in bucket.keys() if bid not in seen_ids and bid != UNBRANCHED_ID]
    for bid in extra_ids:
        items = bucket.get(bid, [])
        branches_out.append(_apply_guideline_lane_to_branch(
            EvidencePacketBranch(
                branch_id=bid,
                framework="",
                intent="",
                branch_summary=f"Unplanned branch: {bid}",
                query_terms_used=[],
                articles=_build_article_records(
                    items=items, shape=shape, case_id=None,
                    guideline_applicable=guideline_applicable,
                ),
            ),
            guideline_applicable=guideline_applicable,
        ))
    if UNBRANCHED_ID in bucket and bucket[UNBRANCHED_ID]:
        items = bucket[UNBRANCHED_ID]
        branches_out.append(_apply_guideline_lane_to_branch(
            EvidencePacketBranch(
                branch_id=UNBRANCHED_ID,
                framework="",
                intent="",
                branch_summary="Unbranched articles",
                query_terms_used=[],
                articles=_build_article_records(
                    items=items, shape=shape, case_id=None,
                    guideline_applicable=guideline_applicable,
                ),
            ),
            guideline_applicable=guideline_applicable,
        ))

    packet = EvidencePacketDoc(
        search_session_id=search_session_id,
        evidence_packet_version=EVIDENCE_PACKET_VERSION,
        provenance=EvidencePacketProvenance(
            r_series_labels_source="computed_at_packet_build_time",
            r_series_labels_versions=dict(R_SERIES_VERSIONS),
        ),
        question=session_doc.get("sanitized_question") or session_doc.get(
            "original_question"
        ),
        clinical_question_map_snapshot=cqm_snapshot,
        branches=branches_out,
        preference_metadata=preference_packet_metadata(
            session_doc.get("l1_preference_diagnostics_json")
        ),
    )
    packet_dict = packet.model_dump(mode="json")
    _leak_scan(packet_dict)

    now = utcnow()
    # Strip created_at from the $set payload — it goes in $setOnInsert
    # so we don't conflict with the upsert path.
    packet_dict.pop("created_at", None)
    packet_dict.pop("updated_at", None)
    await db[Collections.EVIDENCE_PACKETS].update_one(
        {"search_session_id": search_session_id},
        {
            "$set": {**packet_dict, "updated_at": now},
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    persisted = await db[Collections.EVIDENCE_PACKETS].find_one(
        {"search_session_id": search_session_id}, {"_id": 0},
    )
    return persisted


def _build_article_records(
    *,
    items: list,
    shape: str,
    case_id: Optional[str] = None,
    guideline_applicable: bool = False,
) -> list[EvidencePacketArticle]:
    """Compose the per-article packet record for a branch."""
    out: list[EvidencePacketArticle] = []
    for br_rank, (sa_row, article_doc, global_rank) in enumerate(items, start=1):
        labels, is_mixed = _r_series_labels_for_article(
            article_doc=article_doc, shape=shape, case_id=case_id,
        )
        humans = map_labels_to_displays(labels)
        guideline_lane = (
            article_guideline_provenance(article_doc)
            if guideline_applicable else None
        )
        out.append(EvidencePacketArticle(
            article_id=article_doc.get("id") or sa_row.get("article_id") or "",
            pmid=article_doc.get("pmid"),
            doi=article_doc.get("doi"),
            title=article_doc.get("title"),
            journal=article_doc.get("journal"),
            year=_safe_year(article_doc.get("publication_year")),
            source=(article_doc.get("source") or None),
            branch_rank=br_rank,
            global_rank=global_rank,
            directness_labels=labels,
            directness_label_human_strings=humans,
            is_mixed_signal=is_mixed,
            source_provenance=list(sa_row.get("sources_present") or []),
            access_eligibility=_access_eligibility(article_doc),
            ai_ingestion_eligibility=_ai_ingestion_eligibility(article_doc),
            selected_state=_selected_state(sa_row),
            summary_eligibility=_summary_eligibility(
                sa_row=sa_row, article_doc=article_doc,
            ),
            citation_verification_readiness=(
                _citation_verification_readiness(article_doc)
            ),
            guideline_lane=guideline_lane,
        ))
    return out


def _apply_guideline_lane_to_branch(
    branch: EvidencePacketBranch, *, guideline_applicable: bool,
) -> EvidencePacketBranch:
    """Set the closed-set guideline-lane surfacing fields on a branch.

    Active only when the question is a guideline-type question AND the
    branch contains at least one PubMed-indexed guideline article.
    Display-only; no ranking / selection / summary effect."""
    if guideline_applicable and any(a.guideline_lane for a in branch.articles):
        branch.guideline_lane_active = True
        branch.guideline_lane_label = GUIDELINE_LANE_LABEL
        branch.guideline_lane_tooltip = GUIDELINE_LANE_TOOLTIP
    return branch


def _safe_year(val: Any) -> Optional[int]:
    try:
        if val is None:
            return None
        return int(val)
    except (TypeError, ValueError):
        return None


def _branch_summary_text(br: dict) -> str:
    """Closed-set human-readable branch label. Pulls from the planner
    fields the FE already renders; never echoes free-text user input."""
    fw = str(br.get("framework") or "").strip()
    intent = str(br.get("intent") or "").strip()
    if fw and intent:
        return f"{fw.replace('_', ' ').title()} — {intent.replace('_', ' ')}"
    if fw:
        return fw.replace("_", " ").title()
    if intent:
        return intent.replace("_", " ").title()
    return br.get("branch_id") or "Branch"


def _branch_query_terms_used(br: dict) -> list[str]:
    """Display-only: a small, closed-set list of search terms /
    concepts surfaced from the planner output. Never includes raw
    user text — only canonical concepts and validated MeSH terms.
    """
    out: list[str] = []
    for cs in (br.get("required_concepts_used") or [])[:10]:
        if isinstance(cs, dict):
            v = cs.get("canonical_concept")
            if isinstance(v, str) and v:
                out.append(v)
    for mt in (br.get("validated_MeSH_terms") or [])[:10]:
        if isinstance(mt, str) and mt and mt not in out:
            out.append(mt)
    return out


__all__ = [
    "EVIDENCE_PACKET_VERSION",
    "UNBRANCHED_ID",
    "R_SERIES_VERSIONS",
    "build_evidence_packet",
    "assign_branch_for_article",
]
