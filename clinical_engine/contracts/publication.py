"""Publication / Study / Result / Assessment / AuthorizedEvidenceSet contracts (W1.3).

Rules these types encode:
  * Every identifier and bibliographic field records which source supplied it.
  * A publication whose identifiers disagree is ``conflict``/``quarantined`` —
    never silently merged (the legacy dedupe merged on any one identifier and
    kept the longer abstract, producing mixed records).
  * A Study links its reports; separate reports of one trial stay separate
    Publications with a ``report_role`` (main result, secondary analysis, …).
  * Dates carry their precision; "2019" is not "2019-01-01".
  * Assessments keep source-reported, AI-provisional and reviewed judgments
    distinct; missing information is not a grade.
  * An AuthorizedEvidenceSet lists every consumed source, how it was selected
    and what text was actually permitted/used.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from models.common import new_id, utcnow

IdentifierKind = Literal["pmid", "doi", "pmcid", "nct", "other"]
DatePrecision = Literal["day", "month", "season", "year", "unknown"]
IdentityStatus = Literal["verified", "unverified", "conflict", "quarantined"]


class FieldProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str  # e.g. "pubmed", "europepmc", "openalex", "fixture", "legacy_session"
    retrieved_at: Optional[datetime] = None
    record_id: Optional[str] = None  # the source's own id for the record


class Identifier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: IdentifierKind
    value: str  # normalized (see clinical_engine.identity)
    provenance: FieldProvenance


class PublicationDate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1800, le=2200)
    month: Optional[int] = Field(default=None, ge=1, le=12)
    day: Optional[int] = Field(default=None, ge=1, le=31)
    precision: DatePrecision = "unknown"
    provenance: Optional[FieldProvenance] = None


class IdentifierConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: IdentifierKind
    values: list[str]
    sources: list[str]
    reason: str


class Publication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    identifiers: list[Identifier] = Field(default_factory=list)
    title: Optional[str] = None
    title_provenance: Optional[FieldProvenance] = None
    journal: Optional[str] = None
    publication_date: PublicationDate = Field(default_factory=PublicationDate)
    publication_types: list[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    abstract_provenance: Optional[FieldProvenance] = None
    identity_status: IdentityStatus = "unverified"
    identity_conflicts: list[IdentifierConflict] = Field(default_factory=list)
    # Which input records were combined into this one (only same-publication merges).
    merged_from: list[str] = Field(default_factory=list)

    def identifier(self, kind: str) -> Optional[str]:
        for ident in self.identifiers:
            if ident.kind == kind:
                return ident.value
        return None


ReportRole = Literal[
    "primary_result", "secondary_analysis", "subgroup_analysis", "long_term_follow_up",
    "protocol", "design_rationale", "correction", "commentary", "unknown",
]


class StudyReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    publication_id: str
    report_role: ReportRole = "unknown"


class Study(BaseModel):
    """One study (e.g. a trial) and the separate publications that report it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    registration_ids: list[str] = Field(default_factory=list)  # e.g. NCT numbers
    acronym: Optional[str] = None
    design: Optional[str] = None
    reports: list[StudyReport] = Field(default_factory=list)


class Result(BaseModel):
    """One reported result — always tied to the report it came from."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    study_id: Optional[str] = None
    publication_id: str
    outcome: str
    comparison: Optional[str] = None
    timepoint: Optional[str] = None
    effect_measure: Optional[str] = None  # e.g. "RR", "HR", "MD"
    value: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    subgroup: Optional[str] = None
    passage: Optional[str] = Field(default=None, max_length=2000)  # supporting text
    input_scope: Literal["abstract", "full_text", "metadata_only"] = "abstract"


AssessmentFramework = Literal["GRADE", "RoB2", "ROBINS-I", "AMSTAR2", "QUADAS-2", "other"]
AssessmentSource = Literal["source_reported", "ai_assisted_provisional", "reviewed"]


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    framework: AssessmentFramework
    # GRADE → outcome/comparison/body of evidence; RoB 2 → a trial result;
    # AMSTAR 2 → an eligible review. The target must match the framework.
    target_kind: Literal["body_of_evidence", "result", "publication", "study"]
    target_id: str
    outcome: Optional[str] = None
    comparison: Optional[str] = None
    judgment: Optional[str] = None  # None = not assessed; never a default grade
    missing_information: list[str] = Field(default_factory=list)
    source: AssessmentSource
    provenance: Optional[FieldProvenance] = None
    input_scope: Literal["abstract", "full_text", "metadata_only"] = "abstract"


SelectionMode = Literal["manual", "delegated"]


class EvidenceSetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    publication_ref: str  # engine publication id, or the legacy session row id
    article_id: Optional[str] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    selected_by: Literal["user", "policy"]
    selection_reason: str
    rank: Optional[int] = None
    input_scope: Literal["abstract", "full_text", "metadata_only"] = "abstract"
    ai_processing_allowed: bool
    permission_basis: str
    used: bool = False  # set true only when a synthesis actually consumed it


class AuthorizedEvidenceSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    user_id: str
    question_id: str
    question_version_id: str
    run_id: str
    selection_mode: SelectionMode
    delegation_id: Optional[str] = None  # required when selection_mode == delegated
    policy: Optional[str] = None  # documented policy id for delegated selection
    items: list[EvidenceSetItem] = Field(default_factory=list)
    # Selected but not usable (e.g. AI processing not permitted) — disclosed, not dropped.
    ineligible: list[EvidenceSetItem] = Field(default_factory=list)
    omitted_notice: Optional[str] = None
    manifest_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)
