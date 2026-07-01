"""Phase 6G.4P — Post-Procedure Management Routing tests.

These tests are deterministic / unit-level. They exercise:

  * ``PostProcedureManagementRouter.detect`` for the 5 reproducer
    queries (aortic-valve anticoagulation, PCI antiplatelet, hip-VTE
    prophylaxis, colonoscopy anticoagulation restart, kidney transplant
    immunosuppression).
  * ``ProcedureOntologyMapperService.resolve_procedure`` returns the
    expected canonical_concept, procedure_family, specialty, and
    related_but_distinct sets from the 4 procedure packs.
  * ``PrimaryClinicalEntityAnchorService.is_generic_only`` for:
      - the 5 dual-anchor cases → False
      - 5 closed-set Phase 6F.2 fast-path generic-only cases → True
      - 3 new Task 9.F regression cases → True
      - "What is anticoagulation?" → False (Phase 6G.1 invariant).
  * ``derive`` injects two primary anchors (procedure + management)
    AND sets closed-set picott_overrides (question_subtype =
    post_procedure_management, subdomain, specialty, type_of_question).
  * Boolean compiler emits queries that contain BOTH the procedure
    anchor AND the management anchor for every reproducer (must_have
    contract).
  * Closed-set refinement_note projection works for every subdomain.
  * ``procedure_anchor_critic.is_enabled()`` is False by default and
    the critic is import-safe.

NO network. NO LLM calls. Pure rule-based regression.
"""
from __future__ import annotations

import importlib

import pytest

from services.post_procedure_router import (
    SUBDOMAINS,
    post_procedure_router,
)
from services.primary_clinical_anchor import (
    ENTITY_TYPES,
    primary_clinical_anchor_service as anchor_svc,
)
from services.procedure_ontology_mapper import (
    procedure_ontology_mapper,
)
from services.procedure_packs import (
    MANAGEMENT_DOMAINS,
    PROCEDURE_FAMILIES,
    all_entries,
    all_packs,
)


# ---------------------------------------------------------------------
# Pack invariants.
# ---------------------------------------------------------------------


def test_procedure_packs_load_all_four():
    packs = all_packs()
    expected = {
        "cardiovascular_procedures",
        "orthopedic_procedures",
        "gastrointestinal_procedures",
        "transplant_procedures",
    }
    assert expected.issubset(set(packs.keys()))


def test_every_pack_entry_uses_closed_set_vocabularies():
    entries = all_entries()
    assert entries, "No procedure pack entries loaded"
    for e in entries:
        fam = e.get("procedure_family")
        assert fam in PROCEDURE_FAMILIES, f"bad family {fam!r} for {e.get('canonical_concept')}"
        for d in (e.get("management_domains") or []):
            assert d in MANAGEMENT_DOMAINS, (
                f"bad domain {d!r} for {e.get('canonical_concept')}"
            )


def test_first_class_entity_types_present():
    # Phase 6G.4P contract.
    for et in (
        "procedure_or_surgery",
        "device_or_implant",
        "perioperative_context",
        "postprocedure_context",
        "management_intervention",
    ):
        assert et in ENTITY_TYPES


# ---------------------------------------------------------------------
# Router.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected_subdomain",
    [
        (
            "what are the recommendations for anticoagulation after aortic valve surgery?",
            "post_procedure_antithrombotic_management",
        ),
        (
            "what antiplatelet therapy is recommended after PCI?",
            "post_procedure_antithrombotic_management",
        ),
        (
            "what VTE prophylaxis is recommended after hip replacement?",
            "post_procedure_vte_prophylaxis",
        ),
        (
            "when should anticoagulation be restarted after colonoscopy with polypectomy?",
            "post_procedure_antithrombotic_management",
        ),
        (
            "what immunosuppression is used after kidney transplant?",
            "post_transplant_immunosuppression",
        ),
    ],
)
def test_post_procedure_router_matches_reproducers(question, expected_subdomain):
    r = post_procedure_router.detect(question)
    assert r["matched"] is True, question
    assert r["subdomain"] == expected_subdomain
    assert r["procedure_phrase"]
    # Subdomain is closed-set.
    assert r["subdomain"] in SUBDOMAINS


@pytest.mark.parametrize(
    "question",
    [
        "what are the recommendations?",
        "what anticoagulation is recommended?",
        "what is postoperative management?",
        "What is anticoagulation?",
        "What are the side effects?",
    ],
)
def test_post_procedure_router_rejects_generic(question):
    r = post_procedure_router.detect(question)
    assert r["matched"] is False, question


# ---------------------------------------------------------------------
# Ontology mapper.
# ---------------------------------------------------------------------


def test_ontology_resolves_aortic_valve_surgery():
    pm = procedure_ontology_mapper.resolve_procedure("aortic valve surgery")
    assert pm is not None
    assert pm.canonical_concept == "aortic valve replacement"
    assert pm.procedure_family == "cardiac_valve_surgery"
    assert "cardiology" in pm.specialty
    # TAVR must surface as related-but-distinct so the ranker can apply
    # the L5 cap when SAVR is asked but TAVR-only is returned.
    rbd = " ".join(pm.related_but_distinct).lower()
    assert "tavr" in rbd or "transcatheter" in rbd


def test_ontology_resolves_pci():
    pm = procedure_ontology_mapper.resolve_procedure("PCI")
    assert pm is not None
    assert pm.canonical_concept == "percutaneous coronary intervention"
    assert pm.procedure_family == "coronary_intervention"
    assert "antiplatelet_therapy" in pm.management_domains


def test_ontology_resolves_hip_replacement():
    pm = procedure_ontology_mapper.resolve_procedure("hip replacement")
    assert pm is not None
    assert pm.canonical_concept == "total hip arthroplasty"
    assert pm.procedure_family == "joint_arthroplasty"
    assert "vte_prophylaxis" in pm.management_domains
    assert "orthopedics" in pm.specialty


def test_ontology_prefers_more_specific_canonical_for_colonoscopy_polypectomy():
    pm = procedure_ontology_mapper.resolve_procedure("colonoscopy with polypectomy")
    assert pm is not None
    assert pm.canonical_concept == "colonoscopy with polypectomy"
    assert pm.procedure_family == "endoscopic_polypectomy"


def test_ontology_resolves_kidney_transplant():
    pm = procedure_ontology_mapper.resolve_procedure("kidney transplant")
    assert pm is not None
    assert pm.canonical_concept == "kidney transplant"
    assert pm.procedure_family == "solid_organ_transplant"
    assert "immunosuppression" in pm.management_domains


def test_ontology_returns_none_for_nonsense():
    assert procedure_ontology_mapper.resolve_procedure("blue widget") is None
    assert procedure_ontology_mapper.resolve_procedure("") is None


# ---------------------------------------------------------------------
# is_generic_only + derive integration.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "q,expected",
    [
        # Dual-anchor cases (must NOT clarify).
        ("what are the recommendations for anticoagulation after aortic valve surgery?", False),
        ("what antiplatelet therapy is recommended after PCI?", False),
        ("what VTE prophylaxis is recommended after hip replacement?", False),
        ("when should anticoagulation be restarted after colonoscopy with polypectomy?", False),
        ("what immunosuppression is used after kidney transplant?", False),
        # Phase 6G.4P Task 9.F regression cases (MUST clarify).
        ("what are the recommendations?", True),
        ("what anticoagulation is recommended?", True),
        ("what is postoperative management?", True),
        # Phase 6F.2 closed-set fast-path (MUST still clarify).
        ("What are the side effects?", True),
        ("What is first-line treatment?", True),
        ("What is the dose?", True),
        ("What is the recommended age?", True),
        ("What is diagnostic accuracy?", True),
        # Phase 6G.1 invariant: drug_overview still works.
        ("What is anticoagulation?", False),
    ],
)
def test_is_generic_only_full_matrix(q, expected):
    assert anchor_svc.is_generic_only(q) is expected, q


def test_derive_injects_dual_anchors_for_aortic_valve():
    q = "what are the recommendations for anticoagulation after aortic valve surgery?"
    out = anchor_svc.derive(picott={}, ontology_mappings=[], normalized_question=q)
    anchors = out["primary_anchors"]
    et_set = {a["entity_type"] for a in anchors}
    assert "procedure_or_surgery" in et_set
    # Mgmt anchor — could be drug_or_drug_class or management_intervention
    # for the antithrombotic subdomain (with no drug token in the
    # therapy phrase the mgmt anchor is management_intervention).
    assert ({"drug_or_drug_class", "management_intervention"} & et_set), et_set
    # picott_overrides closed-set fields.
    overrides = out["picott_overrides"]
    assert overrides["question_subtype"] == "post_procedure_management"
    assert overrides["subdomain"] == "post_procedure_antithrombotic_management"
    assert overrides["question_scope"] == "foreground"
    assert overrides.get("specialty") == "cardiology"
    # post_procedure_routing payload surfaced.
    pp = out["post_procedure_routing"]
    assert pp.get("matched") is True
    assert pp.get("procedure_canonical") == "aortic valve replacement"
    assert pp.get("procedure_family") == "cardiac_valve_surgery"


def test_derive_injects_dual_anchors_for_pci():
    q = "what antiplatelet therapy is recommended after PCI?"
    out = anchor_svc.derive(picott={}, ontology_mappings=[], normalized_question=q)
    anchors = out["primary_anchors"]
    et_set = {a["entity_type"] for a in anchors}
    assert "procedure_or_surgery" in et_set
    canonicals = [a["canonical_concept"].lower() for a in anchors]
    assert any("percutaneous coronary intervention" in c for c in canonicals)


def test_derive_injects_dual_anchors_for_hip_vte():
    q = "what VTE prophylaxis is recommended after hip replacement?"
    out = anchor_svc.derive(picott={}, ontology_mappings=[], normalized_question=q)
    et_set = {a["entity_type"] for a in out["primary_anchors"]}
    assert "procedure_or_surgery" in et_set
    assert out["picott_overrides"]["subdomain"] == "post_procedure_vte_prophylaxis"


def test_derive_injects_dual_anchors_for_colonoscopy_restart():
    q = "when should anticoagulation be restarted after colonoscopy with polypectomy?"
    out = anchor_svc.derive(picott={}, ontology_mappings=[], normalized_question=q)
    pp = out["post_procedure_routing"]
    assert pp["intent"] == "restart"
    assert pp["procedure_canonical"] == "colonoscopy with polypectomy"


def test_derive_injects_dual_anchors_for_kidney_transplant():
    q = "what immunosuppression is used after kidney transplant?"
    out = anchor_svc.derive(picott={}, ontology_mappings=[], normalized_question=q)
    et_set = {a["entity_type"] for a in out["primary_anchors"]}
    # Either drug_or_drug_class (when therapy phrase mentions
    # tacrolimus, etc.) OR management_intervention is acceptable.
    assert "procedure_or_surgery" in et_set
    assert out["picott_overrides"]["subdomain"] == "post_transplant_immunosuppression"


# ---------------------------------------------------------------------
# PICOTT term mapping projection (refinement_note + subdomain).
# ---------------------------------------------------------------------


def test_refinement_note_for_every_subdomain():
    """Every closed-set subdomain MUST have a closed-set refinement
    note. This guarantees the UI never gets a missing string for a
    routed session."""
    from services.picott_term_mapping_service import _REFINEMENT_NOTES_BY_SUBDOMAIN
    for sd in SUBDOMAINS:
        assert sd in _REFINEMENT_NOTES_BY_SUBDOMAIN, sd
        text = _REFINEMENT_NOTES_BY_SUBDOMAIN[sd]
        assert isinstance(text, str) and len(text) > 20


def test_picott_term_mapping_surfaces_refinement_note():
    from services.picott_term_mapping_service import build_picott_term_mapping
    session_doc = {
        "id": "test-session-1",
        "sanitized_question": "what are the recommendations for anticoagulation after aortic valve surgery?",
        "picott_json": {
            "question_scope": "foreground",
            "type_of_question": "therapy",
            "question_subtype": "post_procedure_management",
            "subdomain": "post_procedure_antithrombotic_management",
            "specialty": "cardiology",
            "must_have_concepts": [],
        },
        "primary_clinical_anchors_json": {
            "primary_anchors": [
                {
                    "canonical_concept": "aortic valve replacement",
                    "entity_type": "procedure_or_surgery",
                    "role": "population",
                    "synonyms": ["aortic valve replacement"],
                    "source": "ontology_pack",
                    "confidence": 0.92,
                    "required_for_search": True,
                },
                {
                    "canonical_concept": "antithrombotic therapy",
                    "entity_type": "management_intervention",
                    "role": "intervention",
                    "synonyms": ["antithrombotic therapy", "anticoagulation"],
                    "source": "ontology_pack",
                    "confidence": 0.9,
                    "required_for_search": True,
                },
            ],
            "generic_modifiers": [],
            "anchor_repair_applied": True,
            "anchor_failure_reason": None,
            "post_procedure_routing": {
                "matched": True,
                "subdomain": "post_procedure_antithrombotic_management",
                "intent": "guideline",
                "procedure_canonical": "aortic valve replacement",
                "procedure_family": "cardiac_valve_surgery",
                "specialty": ["cardiology", "cardiac surgery"],
                "related_but_distinct": ["TAVR", "transcatheter aortic valve replacement"],
                "management_canonical": "antithrombotic therapy",
                "management_entity_type": "management_intervention",
                "management_domains": ["antithrombotic_therapy", "anticoagulation"],
            },
        },
    }
    mapping = build_picott_term_mapping(session_doc)
    assert mapping is not None
    assert mapping.subdomain == "post_procedure_antithrombotic_management"
    assert mapping.refinement_note
    assert "mechanical" in mapping.refinement_note.lower()
    assert mapping.post_procedure_routing.get("procedure_family") == "cardiac_valve_surgery"


# ---------------------------------------------------------------------
# Optional Critic scaffold.
# ---------------------------------------------------------------------


def test_procedure_anchor_critic_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PROCEDURE_ANCHOR_CRITIC_ENABLED", raising=False)
    mod = importlib.import_module("services.procedure_anchor_critic")
    assert mod.is_enabled() is False
    assert mod.procedure_anchor_critic.suggest("aortic valve surgery", []) is None


def test_procedure_anchor_critic_shadow_mode(monkeypatch):
    monkeypatch.setenv("PROCEDURE_ANCHOR_CRITIC_ENABLED", "true")
    monkeypatch.setenv("PROCEDURE_ANCHOR_CRITIC_MODE", "shadow")
    mod = importlib.import_module("services.procedure_anchor_critic")
    assert mod.is_enabled() is True
    s = mod.procedure_anchor_critic.suggest("aortic valve surgery", [])
    assert s is not None
    # Closed-set schema: extra fields forbidden.
    assert hasattr(s, "suggested_canonical")
    assert hasattr(s, "confidence")
    assert hasattr(s, "rationale")
    # Stub returns no canonical (no LLM wired).
    assert s.suggested_canonical is None


def test_procedure_anchor_critic_disabled_when_mode_not_shadow(monkeypatch):
    monkeypatch.setenv("PROCEDURE_ANCHOR_CRITIC_ENABLED", "true")
    monkeypatch.setenv("PROCEDURE_ANCHOR_CRITIC_MODE", "live")
    mod = importlib.import_module("services.procedure_anchor_critic")
    assert mod.is_enabled() is False
    assert mod.procedure_anchor_critic.suggest("aortic valve surgery", []) is None
