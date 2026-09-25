"""Identity-safe union of candidate records (W1.3; used by retrieval from W2.2).

Replaces the legacy behaviour for the new path:
  * legacy dedupe merged on ANY single shared identifier and kept the longer
    abstract/author list → one row with paper X's title and paper Y's abstract;
  * it also missed transitive matches (A~B~C → two clusters).

Here records are clustered transitively on identifiers of the same kind, and a
cluster is only merged when it is internally consistent: at most one value per
identifier kind, compatible titles and years. Anything else is quarantined —
every member stays a separate, flagged Publication and nothing is merged.
Fields come from the most authoritative source, never "the longest".
Same-trial reports (shared NCT, different PMIDs) are linked as a Study, not merged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional

from clinical_engine.contracts.publication import (
    FieldProvenance,
    Identifier,
    IdentifierConflict,
    Publication,
    Study,
    StudyReport,
)
from clinical_engine.identity.dates import parse_publication_date
from services.canonical_identity import normalize_doi, normalize_trial_id

# Higher = more authoritative for bibliographic fields.
SOURCE_AUTHORITY = {
    "pubmed": 50, "europepmc": 40, "crossref": 30, "semantic_scholar": 20,
    "openalex": 15, "legacy_session": 10, "fixture": 5,
}
_MERGE_KINDS = ("pmid", "doi", "pmcid")
_PMID_RE = re.compile(r"^\d{1,9}$")
_PMCID_RE = re.compile(r"^PMC\d{1,10}$")
_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_pmid(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower().startswith("pmid:"):
        s = s[5:].strip()
    return s if _PMID_RE.match(s) and s.strip("0") else ""


def normalize_pmcid(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().upper()
    if s.isdigit():
        s = f"PMC{s}"
    return s if _PMCID_RE.match(s) else ""


def normalize_doi_strict(value: Any) -> str:
    s = normalize_doi(str(value) if value is not None else None)
    return s.rstrip(".,;:)]}") if s else ""


def _title_tokens(title: Optional[str]) -> list[str]:
    return _WORD_RE.findall((title or "").lower())


def titles_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """Same paper despite punctuation/case/subtitle differences? Unknown → compatible."""
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return True
    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / len(sa | sb)
    ratio = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    # One title may be a prefix of the other (subtitle dropped by a source).
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    prefix = len(shorter) >= 4 and longer[: len(shorter)] == shorter
    return jaccard >= 0.6 or ratio >= 0.85 or prefix


@dataclass
class CandidateRecord:
    record_id: str
    source: str
    pmid: str = ""
    doi: str = ""
    pmcid: str = ""
    nct: str = ""
    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    date_raw: Optional[str] = None
    abstract: Optional[str] = None
    publication_types: list[str] = field(default_factory=list)
    dropped_identifiers: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "CandidateRecord":
        rec = cls(
            record_id=str(raw.get("record_id") or raw.get("id") or ""),
            source=str(raw.get("source") or "unknown").lower(),
            title=raw.get("title"),
            journal=raw.get("journal"),
            abstract=raw.get("abstract"),
            date_raw=raw.get("publication_date") or raw.get("date"),
            publication_types=list(raw.get("publication_types") or []),
        )
        year = raw.get("year") or raw.get("publication_year")
        try:
            rec.year = int(year) if year not in (None, "") else None
        except (TypeError, ValueError):
            rec.year = None
        for kind, norm in (
            ("pmid", normalize_pmid), ("doi", normalize_doi_strict),
            ("pmcid", normalize_pmcid), ("nct", normalize_trial_id),
        ):
            raw_value = raw.get(kind)
            if raw_value in (None, ""):
                continue
            value = norm(raw_value)
            if value:
                setattr(rec, kind, value)
            else:
                rec.dropped_identifiers.append(f"{kind}:{str(raw_value)[:40]}")
        return rec


@dataclass
class ResolutionReport:
    publications: list[Publication]
    studies: list[Study]
    quarantined_record_ids: list[str]
    conflicts: list[dict[str, Any]]
    merged_groups: int
    input_records: int


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _cluster_conflicts(members: list[CandidateRecord]) -> list[IdentifierConflict]:
    conflicts: list[IdentifierConflict] = []
    for kind in (*_MERGE_KINDS, "nct"):
        values = sorted({getattr(m, kind) for m in members if getattr(m, kind)})
        if len(values) > 1:
            conflicts.append(IdentifierConflict(
                kind=kind, values=values,
                sources=sorted({m.source for m in members if getattr(m, kind)}),
                reason=f"records sharing an identifier disagree on {kind}",
            ))
    titled = [m for m in members if m.title]
    for i, a in enumerate(titled):
        for b in titled[i + 1:]:
            if not titles_compatible(a.title, b.title):
                conflicts.append(IdentifierConflict(
                    kind="other", values=[a.record_id, b.record_id],
                    sources=sorted({a.source, b.source}),
                    reason="shared identifier but different titles (possible grafted identifier)",
                ))
    years = sorted({m.year for m in members if m.year})
    if years and years[-1] - years[0] > 1:
        conflicts.append(IdentifierConflict(
            kind="other", values=[str(y) for y in years],
            sources=sorted({m.source for m in members if m.year}),
            reason="shared identifier but publication years differ by more than one year",
        ))
    return conflicts


def _prov(rec: CandidateRecord) -> FieldProvenance:
    return FieldProvenance(source=rec.source, record_id=rec.record_id)


def _best(members: list[CandidateRecord], attr: str) -> Optional[CandidateRecord]:
    """Most authoritative member that has *attr* — never the longest value."""
    have = [m for m in members if getattr(m, attr)]
    if not have:
        return None
    return max(have, key=lambda m: (SOURCE_AUTHORITY.get(m.source, 0), -members.index(m)))


def _publication(members: list[CandidateRecord], status: str,
                 conflicts: list[IdentifierConflict]) -> Publication:
    identifiers: list[Identifier] = []
    for kind in (*_MERGE_KINDS, "nct"):
        best = _best(members, kind)
        if best is not None:
            identifiers.append(Identifier(kind=kind, value=getattr(best, kind), provenance=_prov(best)))
    title_src = _best(members, "title")
    abstract_src = _best(members, "abstract")
    date_src = _best(members, "date_raw") or _best(members, "year")
    if date_src is not None and date_src.date_raw:
        pub_date = parse_publication_date(date_src.date_raw, _prov(date_src))
    elif date_src is not None:
        pub_date = parse_publication_date(str(date_src.year), _prov(date_src))
    else:
        pub_date = parse_publication_date(None)
    journal_src = _best(members, "journal")
    ptypes = sorted({t for m in members for t in m.publication_types})
    return Publication(
        identifiers=identifiers,
        title=title_src.title if title_src else None,
        title_provenance=_prov(title_src) if title_src else None,
        journal=journal_src.journal if journal_src else None,
        publication_date=pub_date,
        publication_types=ptypes,
        abstract=abstract_src.abstract if abstract_src else None,
        abstract_provenance=_prov(abstract_src) if abstract_src else None,
        identity_status=status,  # type: ignore[arg-type]
        identity_conflicts=conflicts,
        merged_from=[m.record_id for m in members],
    )


_ROLE_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("protocol", ("study protocol", "trial protocol", "protocol for", "statistical analysis plan")),
    ("design_rationale", ("design and rationale", "rationale and design")),
    ("correction", ("erratum", "correction to", "corrigendum")),
    ("commentary", ("comment on", "editorial", "letter to the editor")),
    ("subgroup_analysis", ("subgroup", "sub-group", "in patients with and without")),
    ("secondary_analysis", ("secondary analysis", "post hoc", "post-hoc", "ancillary", "exploratory analysis")),
    ("long_term_follow_up", ("long-term follow-up", "long term follow up", "extended follow-up", "year follow-up")),
]


def report_role(pub: Publication) -> str:
    title = (pub.title or "").lower()
    for role, cues in _ROLE_CUES:
        if any(c in title for c in cues):
            return role
    types = {t.lower() for t in pub.publication_types}
    if types & {"comment", "editorial", "letter"}:
        return "commentary"
    if types & {"published erratum"}:
        return "correction"
    if types & {"randomized controlled trial", "clinical trial"}:
        return "primary_result"
    return "unknown"


def link_studies(publications: Iterable[Publication]) -> list[Study]:
    """Group reports by trial registration. Reports stay separate Publications."""
    by_nct: dict[str, list[Publication]] = {}
    for pub in publications:
        nct = pub.identifier("nct")
        if nct and pub.identity_status != "quarantined":
            by_nct.setdefault(nct, []).append(pub)
    studies = []
    for nct, pubs in sorted(by_nct.items()):
        studies.append(Study(
            registration_ids=[nct],
            reports=[StudyReport(publication_id=p.id, report_role=report_role(p)) for p in pubs],
        ))
    return studies


def resolve(raw_records: Iterable[dict[str, Any]]) -> ResolutionReport:
    records = [CandidateRecord.from_raw(r) for r in raw_records]
    uf = _UnionFind(len(records))
    first_seen: dict[tuple[str, str], int] = {}
    for i, rec in enumerate(records):
        for kind in _MERGE_KINDS:
            value = getattr(rec, kind)
            if not value:
                continue
            key = (kind, value)
            if key in first_seen:
                uf.union(first_seen[key], i)
            else:
                first_seen[key] = i

    clusters: dict[int, list[CandidateRecord]] = {}
    for i, rec in enumerate(records):
        clusters.setdefault(uf.find(i), []).append(rec)

    publications: list[Publication] = []
    quarantined: list[str] = []
    conflict_log: list[dict[str, Any]] = []
    merged_groups = 0
    for members in clusters.values():
        conflicts = _cluster_conflicts(members) if len(members) > 1 else []
        if conflicts:
            for m in members:
                publications.append(_publication([m], "quarantined", conflicts))
                quarantined.append(m.record_id)
            conflict_log.append({
                "record_ids": [m.record_id for m in members],
                "conflicts": [c.model_dump() for c in conflicts],
            })
            continue
        if len(members) > 1:
            merged_groups += 1
        authoritative = {m.source for m in members} & {"pubmed", "europepmc"}
        has_id = any(getattr(m, k) for m in members for k in _MERGE_KINDS)
        status = "verified" if (authoritative and has_id) else "unverified"
        publications.append(_publication(members, status, []))

    return ResolutionReport(
        publications=publications,
        studies=link_studies(publications),
        quarantined_record_ids=quarantined,
        conflicts=conflict_log,
        merged_groups=merged_groups,
        input_records=len(records),
    )
