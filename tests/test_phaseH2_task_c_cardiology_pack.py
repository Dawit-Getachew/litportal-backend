"""Phase H2-tail Task C — inherited cardiac arrhythmia syndrome pack
acceptance tests.

Scope:
  * The cardiology_inherited_arrhythmias.json pack loads.
  * Every entry uses ONLY closed-set procedure_family + management_domains
    tokens (the additive ``perioperative_medication_management``
    token is registered).
  * `ProcedureOntologyMapperService.resolve_procedure` returns the
    Brugada / LQTS / CPVT canonical mappings for canonical text +
    common acronyms (BrS / LQTS / CPVT).
  * Brugada keeps long QT / CPVT / SQTS / ARVC in
    `related_but_distinct` so the ranker can L5-cap wrong-condition
    perioperative articles. Long QT keeps Brugada / CPVT in
    `related_but_distinct` for the symmetric guarantee.
  * The pack does NOT collapse separate conditions into a shared
    anchor: long QT and Brugada each resolve to themselves.
  * The pack is pure additive — no service-code change required to
    register it (just the closed-set token addition in
    procedure_packs/__init__.py).
  * `perioperative_medication_management` is in MANAGEMENT_DOMAINS,
    SUBDOMAINS, and the refinement-note table — the three closed-set
    surfaces stay consistent.

No network. No LLM. Pure deterministic ontology testing.
"""
from __future__ import annotations


from services.procedure_ontology_mapper import procedure_ontology_mapper
from services.procedure_packs import (
    MANAGEMENT_DOMAINS,
    PROCEDURE_FAMILIES,
    all_entries,
    get_pack,
)


# ---------------------------------------------------------------------
# Pack registration + closed-set vocab compliance.
# ---------------------------------------------------------------------


def test_cardiology_inherited_arrhythmias_pack_loaded():
    pack = get_pack("cardiology_inherited_arrhythmias")
    assert pack is not None, "cardiology_inherited_arrhythmias pack not loaded"
    entries = pack.get("entries") or []
    canonical = [e.get("canonical_concept") for e in entries]
    assert "Brugada syndrome" in canonical
    assert "long QT syndrome" in canonical
    assert "catecholaminergic polymorphic ventricular tachycardia" in canonical


def test_pack_uses_only_closed_set_tokens():
    """Hard regression for the original bug: the leftover file used
    ``perioperative_medication_management`` and
    ``post_procedure_followup_surveillance`` as management_domain
    tokens — neither was in the closed-set enum. The rewrite must
    use only closed-set tokens."""
    pack = get_pack("cardiology_inherited_arrhythmias")
    for e in (pack.get("entries") or []):
        fam = e.get("procedure_family")
        assert fam in PROCEDURE_FAMILIES, (
            f"bad procedure_family {fam!r} for "
            f"{e.get('canonical_concept')}"
        )
        for d in (e.get("management_domains") or []):
            assert d in MANAGEMENT_DOMAINS, (
                f"bad management_domain {d!r} for "
                f"{e.get('canonical_concept')}"
            )


def test_perioperative_medication_management_in_closed_set_management_domains():
    """The additive closed-set token must be present in
    MANAGEMENT_DOMAINS (single source of truth for pack
    management_domain validation)."""
    assert "perioperative_medication_management" in MANAGEMENT_DOMAINS


def test_perioperative_medication_management_in_closed_set_subdomains():
    """And the SUBDOMAIN surface (closed-set router subdomains) was
    already present — we must NOT regress it."""
    from services.post_procedure_router import SUBDOMAINS
    assert "perioperative_medication_management" in SUBDOMAINS


def test_perioperative_medication_management_has_refinement_note():
    """And the refinement-note table — required to keep the closed
    set complete across all three surfaces."""
    from services.picott_term_mapping_service import (
        _REFINEMENT_NOTES_BY_SUBDOMAIN,
    )
    note = _REFINEMENT_NOTES_BY_SUBDOMAIN.get(
        "perioperative_medication_management",
    )
    assert isinstance(note, str) and len(note) > 20


def test_every_pack_entry_uses_closed_set_vocabularies_global():
    """Cross-pack guarantee: with the additive
    `perioperative_medication_management` token now in
    MANAGEMENT_DOMAINS, the global closed-set vocab check across
    ALL packs (cardiology_inherited_arrhythmias + the 5 procedure
    packs) must pass."""
    entries = all_entries()
    assert entries
    for e in entries:
        fam = e.get("procedure_family")
        assert fam in PROCEDURE_FAMILIES, fam
        for d in (e.get("management_domains") or []):
            assert d in MANAGEMENT_DOMAINS, d


# ---------------------------------------------------------------------
# Brugada canonical resolution.
# ---------------------------------------------------------------------


def test_ontology_resolves_brugada_canonical():
    pm = procedure_ontology_mapper.resolve_procedure("Brugada syndrome")
    assert pm is not None
    assert pm.canonical_concept == "Brugada syndrome"
    assert pm.procedure_family == "cardiac_arrhythmia_intervention"
    # Target deterministic confidence ≥ 0.95.
    assert pm.confidence >= 0.95
    # Closed-set specialty tags.
    assert any(s in {"cardiology", "electrophysiology", "anesthesia"}
               for s in pm.specialty)


def test_ontology_resolves_brugada_acronym_brs():
    pm = procedure_ontology_mapper.resolve_procedure("BrS")
    assert pm is not None
    assert pm.canonical_concept == "Brugada syndrome"
    # Acronym match scores 0.9 per the scorer.
    assert pm.confidence >= 0.9


def test_ontology_resolves_brugada_in_full_phrase():
    """The brief's canonical query phrase: the mapper must still
    extract the Brugada anchor even when wrapped in 'perioperative
    management of patients with ...'."""
    pm = procedure_ontology_mapper.resolve_procedure(
        "perioperative management of patients with Brugada syndrome",
    )
    assert pm is not None
    assert pm.canonical_concept == "Brugada syndrome"
    assert pm.procedure_family == "cardiac_arrhythmia_intervention"
    assert pm.confidence >= 0.95


# ---------------------------------------------------------------------
# Adjacent channelopathies — LQTS / CPVT do NOT collapse into Brugada.
# ---------------------------------------------------------------------


def test_long_qt_syndrome_does_not_collapse_into_brugada():
    pm = procedure_ontology_mapper.resolve_procedure("long QT syndrome")
    assert pm is not None
    assert pm.canonical_concept == "long QT syndrome"
    assert pm.canonical_concept != "Brugada syndrome"


def test_lqts_acronym_resolves():
    pm = procedure_ontology_mapper.resolve_procedure("LQTS")
    assert pm is not None
    assert pm.canonical_concept == "long QT syndrome"


def test_cpvt_canonical_and_acronym():
    pm1 = procedure_ontology_mapper.resolve_procedure(
        "catecholaminergic polymorphic ventricular tachycardia",
    )
    pm2 = procedure_ontology_mapper.resolve_procedure("CPVT")
    assert pm1 is not None
    assert pm2 is not None
    assert pm1.canonical_concept == (
        "catecholaminergic polymorphic ventricular tachycardia"
    )
    assert pm2.canonical_concept == (
        "catecholaminergic polymorphic ventricular tachycardia"
    )


# ---------------------------------------------------------------------
# related_but_distinct invariants — wrong-condition cap support.
# ---------------------------------------------------------------------


def test_brugada_related_but_distinct_includes_lqts_cpvt_sqts_arvc():
    """The L5 wrong-condition cap on the ranker keys off this list;
    if these are missing, an anesthesia-in-LQTS article ranking for
    a Brugada query won't be capped."""
    pm = procedure_ontology_mapper.resolve_procedure("Brugada syndrome")
    assert pm is not None
    rbd = " ".join((pm.related_but_distinct or [])).lower()
    assert "long qt" in rbd
    assert "cpvt" in rbd or "catecholaminergic" in rbd
    assert "short qt" in rbd or "sqts" in rbd
    assert "arvc" in rbd or "arrhythmogenic right ventricular" in rbd


def test_long_qt_related_but_distinct_includes_brugada_cpvt():
    pm = procedure_ontology_mapper.resolve_procedure("long QT syndrome")
    assert pm is not None
    rbd = " ".join((pm.related_but_distinct or [])).lower()
    assert "brugada" in rbd
    assert "cpvt" in rbd or "catecholaminergic" in rbd


def test_cpvt_related_but_distinct_includes_brugada_long_qt():
    pm = procedure_ontology_mapper.resolve_procedure("CPVT")
    assert pm is not None
    rbd = " ".join((pm.related_but_distinct or [])).lower()
    assert "brugada" in rbd
    assert "long qt" in rbd or "lqts" in rbd


# ---------------------------------------------------------------------
# Pack is pure-additive (no negative side effect on existing procedure
# resolutions). Pick one regression: aortic valve surgery still
# resolves to its existing canonical_concept.
# ---------------------------------------------------------------------


def test_existing_procedure_packs_still_resolve_after_pack_addition():
    pm = procedure_ontology_mapper.resolve_procedure("aortic valve surgery")
    assert pm is not None
    assert pm.canonical_concept == "aortic valve replacement"
    assert pm.procedure_family == "cardiac_valve_surgery"


def test_pack_contains_no_pmid_or_doi_or_article_id_forcing():
    """Hard safety invariant: the pack must NEVER force a specific
    PMID / DOI / article reference. Pure ontology only."""
    import json
    blob = json.dumps(get_pack("cardiology_inherited_arrhythmias"))
    assert "pmid" not in blob.lower()
    assert "doi" not in blob.lower()
    assert "10." not in blob  # no DOI prefix
    # Allow the word "guideline" since it's a search term;
    # but no exact title forcing.
    assert "article_id" not in blob.lower()
