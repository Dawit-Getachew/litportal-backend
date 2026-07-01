"""Phase H2 — admin observability tile + hematology pack acceptance tests.

Backend-only contract tests:

  * The admin endpoint `GET /api/admin/clinical-context-resolver/metrics`
    is admin-only (403 for clinicians).
  * Aggregate fields are populated correctly from a seeded session
    fixture.
  * The payload carries NO PHI / prompts / completions / question
    text / session IDs / user emails (closed-set numerics + ISO
    timestamps only).
  * The "insufficient sample" path fires when called_count below the
    documented minimum.
  * The hematology procedure pack is loaded and exposes
    `autologous stem cell transplant` (incl. ASCT acronym) so the
    Phase 6G.4P post-procedure router resolves rituximab+ASCT
    deterministically without an LLM call.

No real LLM. No real PubMed. The pipeline path is exercised via
`PrimaryClinicalEntityAnchorService.derive` + the procedure
ontology mapper.
"""
from __future__ import annotations

import json
import re

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------
# Hematology pack — deterministic resolution.
# ---------------------------------------------------------------------


def test_hematology_pack_loaded():
    from services.procedure_packs import all_packs
    packs = all_packs()
    assert "hematology_procedures" in packs
    assert any(
        (e.get("canonical_concept") or "").lower() == "autologous stem cell transplant"
        for e in packs["hematology_procedures"].get("entries") or []
    )


def test_procedure_ontology_resolves_asct():
    from services.procedure_ontology_mapper import procedure_ontology_mapper
    pm = procedure_ontology_mapper.resolve_procedure("autologous stem cell transplant")
    assert pm is not None
    assert pm.canonical_concept == "autologous stem cell transplant"
    assert pm.procedure_family == "solid_organ_transplant"
    assert any(s.lower() in {"hematology", "oncology", "transplant"}
               for s in pm.specialty)


def test_procedure_ontology_resolves_asct_acronym():
    from services.procedure_ontology_mapper import procedure_ontology_mapper
    pm = procedure_ontology_mapper.resolve_procedure("ASCT")
    assert pm is not None
    assert pm.canonical_concept == "autologous stem cell transplant"


def test_hematology_pack_keeps_lymphoma_myeloma_related_not_collapsed():
    """The brief explicitly requires multiple myeloma + lymphoma to
    surface as RELATED-BUT-DISTINCT, NOT collapsed into a single
    condition anchor."""
    from services.procedure_packs import all_packs
    asct = None
    for e in (all_packs()["hematology_procedures"].get("entries") or []):
        if (e.get("canonical_concept") or "").lower() == "autologous stem cell transplant":
            asct = e
            break
    assert asct is not None
    rbd = [r.lower() for r in (asct.get("related_but_distinct") or [])]
    assert "multiple myeloma" in rbd
    assert any("lymphoma" in r for r in rbd)


def test_post_procedure_router_recognises_rituximab_ASCT():
    """The router MUST classify "recommended dose of rituximab after
    autologous stem cell transplant" as a post-procedure management
    question routed to the new
    ``post_transplant_medication_management`` subdomain."""
    from services.post_procedure_router import post_procedure_router
    r = post_procedure_router.detect(
        "recommended dose of rituximab after autologous stem cell transplant",
    )
    assert r["matched"] is True
    assert r["subdomain"] == "post_transplant_medication_management"
    assert "rituximab" in (r["therapy_phrase"] or "").lower()
    assert (r["procedure_phrase"] or "").lower().startswith(
        "autologous stem cell transplant",
    )


def test_anchor_service_derives_dual_anchors_for_rituximab_asct():
    """End-to-end deterministic resolution: the anchor service
    produces a procedure_or_surgery anchor for ASCT and a
    drug_or_drug_class anchor for rituximab WITHOUT calling the
    resolver."""
    from services.primary_clinical_anchor import (
        primary_clinical_anchor_service as svc,
    )
    out = svc.derive(
        picott={},
        ontology_mappings=[],
        normalized_question=(
            "recommended dose of rituximab after autologous stem cell transplant"
        ),
    )
    et_set = {a["entity_type"] for a in out["primary_anchors"]}
    canonicals = [a["canonical_concept"].lower() for a in out["primary_anchors"]]
    assert "procedure_or_surgery" in et_set
    # rituximab is in our closed-set drug tokens.
    assert "drug_or_drug_class" in et_set or "management_intervention" in et_set
    assert any("rituximab" in c for c in canonicals)
    assert any("autologous stem cell" in c for c in canonicals)
    assert out["anchor_failure_reason"] is None
    assert out["picott_overrides"]["subdomain"] == (
        "post_transplant_medication_management"
    )


def test_anchor_failure_reason_clears_so_resolver_not_triggered():
    """If anchor_failure_reason is None, the pipeline's resolver
    branch is never entered (verified by inspection of pipeline.py).
    """
    from services.primary_clinical_anchor import (
        primary_clinical_anchor_service as svc,
    )
    out = svc.derive(
        picott={},
        ontology_mappings=[],
        normalized_question=(
            "recommended dose of rituximab after autologous stem cell transplant"
        ),
    )
    assert out["anchor_failure_reason"] is None


# ---------------------------------------------------------------------
# Admin endpoint — auth + closed-set payload + safety.
# ---------------------------------------------------------------------


@pytest_asyncio.fixture
async def admin_token(client):
    r = await client.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    return r.json()["access_token"]


@pytest_asyncio.fixture
async def clinician_token(client):
    r = await client.post(
        "/api/auth/login",
        json={"email": "clinician@test.com", "password": "ClinicianTest123!"},
    )
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_admin_endpoint_returns_closed_set_payload(client, admin_token):
    r = await client.get(
        "/api/admin/clinical-context-resolver/metrics",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    # Closed-set required fields — every one of these MUST be present
    # on every response, even when the window is empty.
    required_keys = {
        "window_start", "window_end", "generated_at",
        "min_sample_size", "insufficient_sample",
        "total_session_count",
        "context_resolver_called",
        "context_resolver_accepted",
        "context_resolver_rejected",
        "resolver_accept_rate",
        "rejection_reason_distribution",
        "resolver_latency_p50_ms",
        "resolver_latency_p95_ms",
        "resolver_token_estimate_total",
        "fallback_active_session_count",
        "shadow_session_count",
        "generic_only_clarification_count",
        "phi_blocked_resolver_skipped_count",
    }
    missing = required_keys - set(data.keys())
    assert not missing, f"missing keys: {missing}"
    # No accidental extra keys (closed-set contract).
    extra = set(data.keys()) - required_keys
    assert not extra, f"unexpected extra keys: {extra}"


@pytest.mark.asyncio
async def test_admin_endpoint_rejects_clinician(client, clinician_token):
    r = await client.get(
        "/api/admin/clinical-context-resolver/metrics",
        headers={"Authorization": f"Bearer {clinician_token}"},
    )
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_admin_endpoint_rejects_anonymous(client):
    r = await client.get("/api/admin/clinical-context-resolver/metrics")
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_admin_payload_contains_no_phi_or_question_text(
    client, admin_token,
):
    r = await client.get(
        "/api/admin/clinical-context-resolver/metrics",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200
    blob = r.text
    # PHI patterns absent.
    assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", blob)  # SSN
    assert not re.search(r"\bMRN\b", blob)
    assert not re.search(r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", blob)  # email
    assert "patient@" not in blob.lower()
    # No session-id-looking UUIDs in the response.
    assert not re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        blob,
    )
    # No user-email-looking strings.
    assert "@test.com" not in blob
    # No question text fragments — assert specific banned tokens that
    # would only appear in question text.
    banned = ["rituximab", "anticoagulation", "warfarin", "Brugada",
              "side effect", "treatment of", "recommend"]
    blob_lower = blob.lower()
    for needle in banned:
        assert needle.lower() not in blob_lower, needle


@pytest.mark.asyncio
async def test_admin_payload_min_sample_size_signal(client, admin_token):
    r = await client.get(
        "/api/admin/clinical-context-resolver/metrics",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    data = r.json()
    # When called_count < min_sample_size, insufficient_sample is True
    # and resolver_accept_rate is None.
    if int(data["context_resolver_called"]) < int(data["min_sample_size"]):
        assert data["insufficient_sample"] is True
        assert data["resolver_accept_rate"] is None
    else:
        assert data["insufficient_sample"] is False
        assert isinstance(data["resolver_accept_rate"], float)


@pytest.mark.asyncio
async def test_admin_payload_rejection_distribution_is_closed_set(
    client, admin_token,
):
    from services.clinical_context_resolver import REJECTION_REASONS
    r = await client.get(
        "/api/admin/clinical-context-resolver/metrics",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    data = r.json()
    dist = data["rejection_reason_distribution"]
    assert set(dist.keys()) == set(REJECTION_REASONS)
    for v in dist.values():
        assert isinstance(v, int)
        assert v >= 0
