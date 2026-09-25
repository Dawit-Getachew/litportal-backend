"""Publication-date parsing that keeps precision ("2019" is not "2019-01-01")."""
from __future__ import annotations

import re
from typing import Optional

from clinical_engine.contracts.publication import FieldProvenance, PublicationDate

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_SEASONS = {"spring", "summer", "fall", "autumn", "winter"}

_ISO = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2})(?:[T ][\d:.+\-Z]*)?)?)?$")
_YEAR = re.compile(r"\b(1[89]\d{2}|2[01]\d{2})\b")


def parse_publication_date(
    raw: Optional[str], provenance: Optional[FieldProvenance] = None,
) -> PublicationDate:
    if raw is None or not str(raw).strip():
        return PublicationDate(raw=raw, precision="unknown", provenance=provenance)
    text = str(raw).strip()
    if text.lower().startswith("epub "):
        text = text[5:].strip()

    iso = _ISO.match(text)
    if iso:
        year = int(iso.group(1))
        month = int(iso.group(2)) if iso.group(2) else None
        day = int(iso.group(3)) if iso.group(3) else None
        if month is not None and not 1 <= month <= 12:
            month, day = None, None
        if day is not None and not 1 <= day <= 31:
            day = None
        precision = "day" if day else "month" if month else "year"
        return PublicationDate(
            raw=raw, year=year, month=month, day=day, precision=precision, provenance=provenance,
        )

    year_match = _YEAR.search(text)
    if not year_match:
        return PublicationDate(raw=raw, precision="unknown", provenance=provenance)
    year = int(year_match.group(1))
    rest = text[year_match.end():].strip().lower()
    tokens = re.split(r"[\s/,\-]+", rest)
    month = None
    day = None
    for i, tok in enumerate(tokens):
        key = tok[:4] if tok[:4] == "sept" else tok[:3]
        if key in _MONTHS:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if nxt[:3] in _MONTHS:
                # "2019 Mar-Apr": a multi-month issue — coarser than a month.
                return PublicationDate(raw=raw, year=year, precision="season", provenance=provenance)
            month = _MONTHS[key]
            if nxt.isdigit():
                d = int(nxt)
                day = d if 1 <= d <= 31 else None
            break
        if tok in _SEASONS:
            return PublicationDate(raw=raw, year=year, precision="season", provenance=provenance)
    precision = "day" if day else "month" if month else "year"
    return PublicationDate(
        raw=raw, year=year, month=month, day=day, precision=precision, provenance=provenance,
    )
