"""Phase 6E — Closed-set trusted-domain allowlist for ``official_source_url``.

Validates that an ``official_source_url`` candidate (added by an admin
during benchmark curation) belongs to a known trustworthy clinical-
guideline / professional-society / federal-health / open-data domain.

Implementation:
- HTTPS-only.
- Hostname must equal a trusted host OR end with ``.<trusted_host>``
  (subdomain match).
- Closed-set list — adding a new domain requires a code change + audit
  review.

Public API:
- ``TRUSTED_OFFICIAL_DOMAINS`` — frozen tuple of trusted hosts.
- ``is_trusted_official_url(url) -> bool`` — boolean check.
- ``classify_official_url(url) -> dict`` — returns ``{"is_trusted": bool,
  "host": str, "scheme": str, "reason": str | None}`` for audit-friendly
  diagnostics. Never returns the full URL outside ``host`` + ``scheme``.

NEVER log the path / query / fragment of an arbitrary user-supplied URL
— only the host. Audit payloads use the domain part only.
"""
from __future__ import annotations

from urllib.parse import urlparse


TRUSTED_OFFICIAL_DOMAINS: tuple[str, ...] = (
    "cdc.gov",
    "uspreventiveservicestaskforce.org",
    "who.int",
    "nice.org.uk",
    "acpjournals.org",
    "americanheart.org",
    "professional.heart.org",
    "acc.org",
    "aap.org",
    "acg.gi.org",
    "gi.org",
    "cancer.org",
    "ahrq.gov",
    "nih.gov",
    "nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "europepmc.org",
    "doi.org",
    "sccm.org",
)


def _host_matches_trusted(host: str) -> bool:
    if not host:
        return False
    h = host.lower().strip()
    # Strip a possible leading ``www.``.
    if h.startswith("www."):
        h = h[4:]
    for d in TRUSTED_OFFICIAL_DOMAINS:
        if h == d or h.endswith(f".{d}"):
            return True
    return False


def is_trusted_official_url(url: str | None) -> bool:
    return classify_official_url(url)["is_trusted"]


def classify_official_url(url: str | None) -> dict:
    """Returns ``{is_trusted, host, scheme, reason}``.

    ``reason`` is a closed-set snake_case code when ``is_trusted=False``.
    Possible reasons: ``invalid_url``, ``not_https``, ``empty_host``,
    ``untrusted_domain``.
    """
    if not url or not isinstance(url, str):
        return {"is_trusted": False, "host": "", "scheme": "",
                "reason": "invalid_url"}
    try:
        parsed = urlparse(url.strip())
    except Exception:  # noqa: BLE001
        return {"is_trusted": False, "host": "", "scheme": "",
                "reason": "invalid_url"}
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    if scheme != "https":
        return {"is_trusted": False, "host": host, "scheme": scheme,
                "reason": "not_https"}
    if not host:
        return {"is_trusted": False, "host": "", "scheme": scheme,
                "reason": "empty_host"}
    if _host_matches_trusted(host):
        return {"is_trusted": True, "host": host, "scheme": scheme,
                "reason": None}
    return {"is_trusted": False, "host": host, "scheme": scheme,
            "reason": "untrusted_domain"}
