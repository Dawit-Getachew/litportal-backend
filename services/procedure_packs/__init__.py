"""Phase 6G.4P — procedure pack loaders.

Each pack is a JSON file in this directory. Packs are loaded lazily on
first access and exposed through ``get_pack(name)`` and ``all_packs()``.

Procedure packs are DISTINCT from the broader ``specialty_packs`` —
they capture procedure ontology entries (canonical concept, family,
synonyms, related_but_distinct, management_domains, search terms,
optional curated MeSH descriptors, and negative_patterns) used by
``ProcedureOntologyMapperService`` and the ``PostProcedureManagementRouter``.

Closed-set vocabularies (defined at module top):

* ``PROCEDURE_FAMILIES`` — the only acceptable values for
  ``procedure_family`` on any pack entry.
* ``MANAGEMENT_DOMAINS`` — the only acceptable values for the
  ``management_domains`` list on any pack entry.

Bad packs MUST NOT crash startup; they are silently skipped (with a
log line at debug-level).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("clinevidence.procedure_packs")

_HERE = Path(__file__).resolve().parent
_PACK_CACHE: dict[str, dict[str, Any]] = {}


# Closed-set procedure families. Additions require a regression test.
PROCEDURE_FAMILIES: frozenset[str] = frozenset({
    "cardiac_valve_surgery",
    "coronary_intervention",
    "coronary_artery_bypass",
    "cardiac_arrhythmia_intervention",
    "joint_arthroplasty",
    "fracture_repair",
    "spinal_surgery",
    "endoscopic_polypectomy",
    "endoscopy_diagnostic",
    "colorectal_resection",
    "bariatric_surgery",
    "solid_organ_transplant",
    "general_abdominal_surgery",
})


# Closed-set management domains. Drawn from the Phase 6G.4P brief
# (Task 3) — any new domain MUST be added here AND in the
# subdomain→refinement_note table in
# ``services.picott_term_mapping_service``.
#
# Phase H2-tail Task C — ``perioperative_medication_management`` is
# the single additive token introduced for inherited cardiac
# arrhythmia syndromes (Brugada / LQTS / CPVT). It also exists as a
# closed-set SUBDOMAIN string in ``post_procedure_router.SUBDOMAINS``
# and as a refinement-note key in
# ``picott_term_mapping_service._REFINEMENT_NOTES_BY_SUBDOMAIN``, so
# the additive change is consistent across the three closed-set
# surfaces.
MANAGEMENT_DOMAINS: frozenset[str] = frozenset({
    "anticoagulation",
    "antithrombotic_therapy",
    "antiplatelet_therapy",
    "dual_antiplatelet_therapy",
    "vte_prophylaxis",
    "antibiotic_prophylaxis",
    "immunosuppression",
    "infection_prophylaxis",
    "rejection_surveillance",
    "bridging_anticoagulation",
    "restarting_anticoagulation",
    "pain_control",
    "rehabilitation",
    "follow_up_imaging",
    "surveillance",
    "endocarditis_prophylaxis",
    "postoperative_atrial_fibrillation",
    "diet",
    "perioperative_medication_management",
})


def _load() -> None:
    if _PACK_CACHE:
        return
    for path in sorted(_HERE.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            name = data.get("name") or path.stem
            _PACK_CACHE[name] = data
        except Exception:  # noqa: BLE001 — bad pack must not crash startup
            logger.debug("procedure_pack_load_failed", extra={"path": str(path)})
            continue


def get_pack(name: str) -> dict[str, Any] | None:
    _load()
    return _PACK_CACHE.get(name)


def all_packs() -> dict[str, dict[str, Any]]:
    _load()
    return dict(_PACK_CACHE)


def all_entries() -> list[dict[str, Any]]:
    """Flatten every entry across loaded packs into a single closed-set
    list. Each entry is a dict with the fields documented in the pack
    schema (canonical_concept, procedure_family, specialty, synonyms,
    acronyms, related_but_distinct, management_domains, search_terms,
    mesh_terms_curated [optional], negative_patterns). Pack name is
    propagated under ``_pack``.
    """
    out: list[dict[str, Any]] = []
    for pack_name, pack in all_packs().items():
        for entry in (pack.get("entries") or []):
            if not isinstance(entry, dict):
                continue
            row = dict(entry)
            row["_pack"] = pack_name
            out.append(row)
    return out


__all__ = [
    "PROCEDURE_FAMILIES",
    "MANAGEMENT_DOMAINS",
    "get_pack",
    "all_packs",
    "all_entries",
]
