"""ClinEvidence Search — Phase 4A OA full-text fetcher package.

NEVER fetches publisher HTML. NEVER follows institutional / subscription
links to ingest text. NEVER calls an LLM. NEVER scrapes. Sources are
strictly:
  - Europe PMC `fullTextXML` REST API (returns JATS XML)
  - NCBI PMC OA Web Service `oa.fcgi` (discovery)
  - NCBI PMC BioC REST API (`BioC_xml/{PMCID}/unicode`)

License gate is conservative: CC0/BY/BY-SA → ingest by default; everything
else (NC, ND, unknown, restricted) → skip unless an explicit `OA_ALLOW_*`
env flag is flipped on. User-uploaded full text is NOT governed by this
gate (it carries the user's attestation; see Phase 2D).
"""
from .service import OAFetcherService, oa_fetcher

__all__ = ["OAFetcherService", "oa_fetcher"]
