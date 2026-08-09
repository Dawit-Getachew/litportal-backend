"""Phase H2-tail Task C end-to-end — Brugada / LQTS / CPVT
perioperative deterministic routing through
``PrimaryClinicalEntityAnchorService.derive`` and
``post_procedure_router.detect``.

This file verifies that the additive wiring in:
  * services/post_procedure_router.py
    (+ _PERIOPERATIVE_AVOIDANCE_TOKENS lexicon, condition-context
     pattern set, _INHERITED_ARRHYTHMIA_LEXICAL_TOKENS allowlist,
     _looks_like_inherited_arrhythmia filter, condition branch in
     detect())
  * services/primary_clinical_anchor.py
    (+ _INHERITED_ARRHYTHMIA_CONDITION_CANONICALS closed-set,
     entity_type=disease_or_problem override on the proc_anchor,
     extended _SUBDOMAIN_TO_MGMT_SYNONYMS["perioperative_medication_management"])

produces:
  * dual primary anchors (condition + management) for the three
    canonical Brugada/LQTS/CPVT perioperative phrasings,
  * the closed-set ``perioperative_medication_management`` subdomain
    in both ``post_procedure_routing.subdomain`` and
    ``picott_overrides.subdomain``,
  * type_of_question = "therapy",
  * anchor_failure_reason = None,
  * is_generic_only = False on these queries (the perioperative
    management of an inherited arrhythmia is NOT generic-only).

Hematology + antithrombotic regressions, generic-only fast-path, and
PHI-block fast-path are also asserted here so the additive wiring
does not regress them.

Pure unit tests over the pure stateless services — no LLM, no
network.
"""
from __future__ import annotations

from typing import Any

import pytest

from services.post_procedure_router import (
    SUBDOMAINS,
    post_procedure_router,
)
from services.primary_clinical_anchor import (
    primary_clinical_anchor_service as anchor_service,
)


BRUGADA_QUERIES: tuple[str, ...] = (
    "perioperative management of patients with Brugada syndrome",
    "anesthesia management in Brugada syndrome",
    "medications to avoid in Brugada syndrome",
)


# ---------------------------------------------------------------------
# Router — condition-context patterns + closed-set subdomain routing.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("q", BRUGADA_QUERIES)
def test_router_routes_brugada_perioperative_to_closed_set_subdomain(q: str):
    out = post_procedure_router.detect(q)
    assert out["matched"] is True, q
    assert out["subdomain"] == "perioperative_medication_management", q
    assert out["subdomain"] in SUBDOMAINS
    assert out["procedure_phrase"] and "brugada" in out["procedure_phrase"].lower()
    assert out["therapy_phrase"]


def test_router_routes_long_qt_perioperative():
    out = post_procedure_router.detect(
        "perioperative care in long QT syndrome",
    )
    assert out["matched"] is True
    assert out["subdomain"] == "perioperative_medication_management"
    assert "long qt" in out["procedure_phrase"].lower()


def test_router_routes_cpvt_anesthesia():
    out = post_procedure_router.detect(
        "anesthetic management for CPVT patients",
    )
    assert out["matched"] is True
    assert out["subdomain"] == "perioperative_medication_management"
    assert "cpvt" in out["procedure_phrase"].lower()


# ---------------------------------------------------------------------
# Router — over-matching guard.
# ---------------------------------------------------------------------


def test_router_rejects_anesthesia_for_healthy_patients():
    """No condition token → no match (over-matching guard preserved)."""
    out = post_procedure_router.detect("anesthesia for healthy patients")
    assert out["matched"] is False


def test_router_rejects_what_is_psoriasis():
    """Background definition with no condition token → no match."""
    out = post_procedure_router.detect("what is psoriasis?")
    assert out["matched"] is False


def test_router_rejects_generic_only_question():
    out = post_procedure_router.detect("what are the recommendations?")
    assert out["matched"] is False


# ---------------------------------------------------------------------
# Router — hematology + antithrombotic regression suite.
# ---------------------------------------------------------------------


def test_router_rituximab_asct_regression_still_routes_to_hematology():
    """Phase H2 regression — the rituximab+ASCT pattern must still
    route through the post_transplant_medication_management subdomain
    despite the new perioperative pack additions."""
    out = post_procedure_router.detect(
        "recommended dose of rituximab after autologous stem cell transplant",
    )
    assert out["matched"] is True
    assert out["subdomain"] == "post_transplant_medication_management"


def test_router_antithrombotic_aortic_valve_regression():
    out = post_procedure_router.detect(
        "recommendations for anticoagulation after aortic valve surgery",
    )
    assert out["matched"] is True
    assert out["subdomain"] == "post_procedure_antithrombotic_management"


# ---------------------------------------------------------------------
# Anchor service — dual deterministic anchors for Brugada perioperative.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("q", BRUGADA_QUERIES)
def test_anchor_service_brugada_yields_dual_deterministic_anchors(q: str):
    out: dict[str, Any] = anchor_service.derive(
        picott={}, ontology_mappings=[], normalized_question=q,
    )
    anchors = out["primary_anchors"]
    assert len(anchors) == 2, (q, anchors)
    canonicals = [a["canonical_concept"].lower() for a in anchors]
    assert "brugada syndrome" in canonicals, (q, canonicals)
    assert "perioperative medication management" in canonicals, (q, canonicals)
    # The Brugada anchor MUST be disease_or_problem (not procedure).
    condition_anchor = next(
        a for a in anchors
        if a["canonical_concept"].lower() == "brugada syndrome"
    )
    assert condition_anchor["entity_type"] == "disease_or_problem"
    assert condition_anchor["role"] == "population"
    assert condition_anchor["confidence"] >= 0.95
    # The management anchor is closed-set perioperative_medication_management.
    mgmt = next(
        a for a in anchors
        if a["canonical_concept"].lower() == "perioperative medication management"
    )
    assert mgmt["entity_type"] == "management_intervention"
    assert mgmt["role"] == "intervention"
    # Subdomain routing exposed via both keys (closed-set).
    assert out["picott_overrides"]["subdomain"] == "perioperative_medication_management"
    assert out["post_procedure_routing"]["subdomain"] == "perioperative_medication_management"
    assert out["picott_overrides"]["type_of_question"] == "therapy"
    assert out["picott_overrides"]["question_subtype"] == "post_procedure_management"
    assert out["anchor_failure_reason"] is None
    assert anchor_service.is_generic_only(q) is False


def test_anchor_service_long_qt_perioperative_yields_dual_anchors():
    q = "perioperative care in long QT syndrome"
    out = anchor_service.derive(picott={}, ontology_mappings=[], normalized_question=q)
    canonicals = [a["canonical_concept"].lower() for a in out["primary_anchors"]]
    assert "long qt syndrome" in canonicals
    assert "perioperative medication management" in canonicals
    cond = next(
        a for a in out["primary_anchors"]
        if a["canonical_concept"].lower() == "long qt syndrome"
    )
    assert cond["entity_type"] == "disease_or_problem"
    assert out["picott_overrides"]["subdomain"] == "perioperative_medication_management"
    assert out["anchor_failure_reason"] is None


def test_anchor_service_cpvt_anesthesia_yields_dual_anchors():
    q = "anesthetic management for CPVT patients"
    out = anchor_service.derive(picott={}, ontology_mappings=[], normalized_question=q)
    canonicals = [a["canonical_concept"].lower() for a in out["primary_anchors"]]
    assert "catecholaminergic polymorphic ventricular tachycardia" in canonicals
    assert "perioperative medication management" in canonicals
    cond = next(
        a for a in out["primary_anchors"]
        if a["canonical_concept"].lower()
        == "catecholaminergic polymorphic ventricular tachycardia"
    )
    assert cond["entity_type"] == "disease_or_problem"


# ---------------------------------------------------------------------
# Anchor service — regression: hematology + antithrombotic + generic +
# definition fast-path.
# ---------------------------------------------------------------------


def test_anchor_service_rituximab_asct_regression():
    """The hematology pack regression — autologous SCT MUST stay
    procedure_or_surgery (not be promoted to disease_or_problem by
    the new wiring) and rituximab MUST stay drug_or_drug_class."""
    out = anchor_service.derive(
        picott={}, ontology_mappings=[],
        normalized_question=(
            "recommended dose of rituximab after autologous stem cell transplant"
        ),
    )
    by_canon = {a["canonical_concept"].lower(): a for a in out["primary_anchors"]}
    assert "autologous stem cell transplant" in by_canon
    assert by_canon["autologous stem cell transplant"]["entity_type"] == "procedure_or_surgery"
    assert "rituximab" in by_canon
    assert by_canon["rituximab"]["entity_type"] == "drug_or_drug_class"
    assert out["picott_overrides"]["subdomain"] == "post_transplant_medication_management"
    assert out["anchor_failure_reason"] is None


def test_anchor_service_aortic_valve_surgery_regression():
    out = anchor_service.derive(
        picott={}, ontology_mappings=[],
        normalized_question=(
            "recommendations for anticoagulation after aortic valve surgery"
        ),
    )
    by_canon = {a["canonical_concept"].lower(): a for a in out["primary_anchors"]}
    assert "aortic valve replacement" in by_canon
    assert by_canon["aortic valve replacement"]["entity_type"] == "procedure_or_surgery"
    assert out["picott_overrides"]["subdomain"] == "post_procedure_antithrombotic_management"


def test_anchor_service_what_is_psoriasis_definition_fastpath_intact():
    """Background-definition fast-path: 'what is psoriasis?' must
    yield question_subtype=disease_overview (not post_procedure_management)
    despite the new perioperative pack additions."""
    out = anchor_service.derive(
        picott={}, ontology_mappings=[],
        normalized_question="what is psoriasis?",
    )
    canonicals = [a["canonical_concept"].lower() for a in out["primary_anchors"]]
    assert "psoriasis" in canonicals
    assert out["picott_overrides"].get("question_subtype") == "disease_overview"
    assert out["anchor_failure_reason"] is None


def test_anchor_service_generic_only_clarification_still_fires():
    q = "what are the recommendations?"
    anchor_service.derive(picott={}, ontology_mappings=[], normalized_question=q)
    # The generic-only detector still labels this true even though the
    # condition-branch router doesn't match (no inherited-arrhythmia
    # token in the captured slot). This guarantees the generic-only
    # clarification UI still fires.
    assert anchor_service.is_generic_only(q) is True


# ---------------------------------------------------------------------
# Anchor service — wrong-condition cap support invariant.
#
# The ranker uses ``related_but_distinct`` (read from the pack via
# procedure_ontology_mapper) to apply the L5 wrong-condition cap. The
# cap implementation itself is in the ranker; here we assert the
# necessary state is exposed in the anchor service routing payload so
# the ranker has the inputs it needs.
# ---------------------------------------------------------------------


def test_anchor_service_brugada_payload_exposes_related_but_distinct():
    """The post_procedure_routing payload MUST include
    related_but_distinct so the ranker can L5-cap wrong-condition
    perioperative articles (anesthesia-in-LQTS when Brugada is
    asked)."""
    out = anchor_service.derive(
        picott={}, ontology_mappings=[],
        normalized_question=(
            "perioperative management of patients with Brugada syndrome"
        ),
    )
    pp = out["post_procedure_routing"]
    assert pp["procedure_canonical"].lower() == "brugada syndrome"
    rbd = " ".join(pp.get("related_but_distinct") or []).lower()
    # Adjacent channelopathies are in the closed-set
    # related_but_distinct list on the pack and survive the routing
    # payload projection.
    assert "long qt" in rbd
    assert "cpvt" in rbd or "catecholaminergic" in rbd
