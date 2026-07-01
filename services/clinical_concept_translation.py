"""Phase 5B Task 1 + 2 — ClinicalConceptTranslationService.

Translates user-facing phrases (e.g. "high hemoglobin target") into
canonical clinical concepts (e.g. "liberal transfusion strategy") +
synonym lists, and detects implicit comparison patterns
("high vs low", "liberal vs restrictive", etc.).

Output shape (per spec):
{
  "translated_question": str,
  "concept_translations": [
    {
      "original_phrase": str,
      "canonical_concept": str,
      "role": "intervention"|"comparator"|"condition"|"context"|"outcome"|...,
      "expanded_terms": [str, ...],
      "confidence": float,
      "rationale": str
    }, ...
  ],
  "latent_concepts": [str, ...],
  "must_have_concepts_added": [<MustHaveConcept-like dict>, ...],
  "ranking_concepts_added": [<RankingConcept-like dict>, ...],
  "negative_concepts_added": [<NegativeConcept-like dict>, ...],
  "warnings": [str, ...]
}

Hard rules (DO NOT relax):
- Pure-deterministic. No LLM, no external network calls.
- Refuses to run when ``status == "blocked_phi"`` (caller's responsibility
  to gate, but we never store raw question text in audit either way).
- Context-aware dispatch: every translation pair is gated by a presence
  predicate so "high LDL target" does NOT trigger a transfusion strategy.
- Translations + comparison-pattern inferences are recorded with a
  short ``rationale`` string (closed-set tokens) that the UI surfaces in
  the search-transparency panel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from services.specialty_packs import get_pack


# ---------------------------------------------------------------------------
# Dispatch table — each rule maps an "original phrase" to a canonical concept.
# Each rule carries a ``context_predicate`` that returns True/False on the
# normalized question. Multiple rules can fire; later rules see earlier
# additions in ``warnings`` only.
# ---------------------------------------------------------------------------

@dataclass
class TranslationRule:
    pattern: re.Pattern
    canonical_concept: str
    role: str
    expanded_terms: list[str]
    rationale: str
    confidence: float = 0.85
    # Predicate: must return True for the rule to fire. Receives the
    # normalized question text (lowercased).
    requires_any: list[str] = field(default_factory=list)
    requires_none: list[str] = field(default_factory=list)


def _has_any(haystack: str, needles: list[str]) -> bool:
    if not needles:
        return True
    s = haystack.lower()
    return any(re.search(rf"\b{re.escape(n.lower())}\b", s) for n in needles)


def _has_none(haystack: str, needles: list[str]) -> bool:
    if not needles:
        return True
    s = haystack.lower()
    return not any(re.search(rf"\b{re.escape(n.lower())}\b", s) for n in needles)


# Canonical concept → expanded synonym list, used everywhere downstream.
_CONCEPT_EXPANSION: dict[str, list[str]] = {
    "liberal transfusion strategy": [
        "liberal transfusion",
        "liberal red blood cell transfusion",
        "liberal RBC transfusion",
        "higher hemoglobin threshold",
        "higher haemoglobin threshold",
        "high hemoglobin target",
        "higher hemoglobin target",
        "transfusion threshold 10",
    ],
    "restrictive transfusion strategy": [
        "restrictive transfusion",
        "restrictive red blood cell transfusion",
        "restrictive RBC transfusion",
        "lower hemoglobin threshold",
        "lower haemoglobin threshold",
        "low hemoglobin target",
        "lower hemoglobin target",
        "transfusion threshold 7",
        "transfusion threshold 8",
    ],
    "acute myocardial infarction": [
        "AMI", "acute MI", "myocardial infarction", "MI", "ACS",
        "acute coronary syndrome", "STEMI", "NSTEMI",
        "ST-elevation myocardial infarction",
        "non-ST-elevation myocardial infarction",
        "myocardial ischemia",
    ],
    "anemia": [
        "anaemia", "hemoglobin", "haemoglobin", "Hb", "Hgb",
        "low hemoglobin", "low haemoglobin",
    ],
    "red blood cell transfusion": [
        "RBC transfusion", "blood transfusion", "transfusion",
        "transfusion strategy", "transfusion threshold",
    ],
    "intensive lipid lowering": [
        "high-intensity statin", "intensive statin therapy",
        "high-intensity lipid lowering", "high-dose statin",
    ],
    "standard lipid lowering": [
        "moderate-intensity statin", "standard statin therapy",
        "moderate lipid lowering",
    ],
    "blood pressure target": [
        "BP target", "intensive BP target", "tight BP control",
        "blood pressure goal",
    ],
}


# Cardiology / transfusion: high X vs low X where X is hemoglobin/Hb target,
# but only when AMI or anemia or transfusion context is present.
_AMI_TOKENS = [
    "acute myocardial infarction", "AMI", "acute MI", "myocardial infarction",
    "MI", "ACS", "acute coronary syndrome", "STEMI", "NSTEMI",
    "myocardial ischemia", "ischemic heart disease",
]
_ANEMIA_TOKENS = [
    "anemia", "anaemia", "hemoglobin", "haemoglobin", "Hb", "Hgb",
]
_TRANSFUSION_TOKENS = [
    "transfusion", "RBC", "red blood cell",
]

_LIPID_TOKENS = [
    "LDL", "statin", "lipid", "cholesterol",
]
_BP_TOKENS = [
    "blood pressure", "BP", "hypertension", "systolic", "diastolic",
]


_RULES: list[TranslationRule] = [
    # ---- Transfusion threshold strategy ----
    TranslationRule(
        pattern=re.compile(
            r"\b(?:high(?:er)?|elevated)\s+(?:hemoglobin|haemoglobin|Hb|Hgb)\s+(?:target|threshold|goal|level)\b",
            re.IGNORECASE,
        ),
        canonical_concept="liberal transfusion strategy",
        role="intervention",
        expanded_terms=_CONCEPT_EXPANSION["liberal transfusion strategy"],
        rationale="High hemoglobin target → liberal transfusion strategy (cardiology/transfusion pack)",
        confidence=0.92,
        requires_any=_AMI_TOKENS + _ANEMIA_TOKENS + _TRANSFUSION_TOKENS,
    ),
    TranslationRule(
        pattern=re.compile(
            r"\b(?:low(?:er)?|reduced)\s+(?:hemoglobin|haemoglobin|Hb|Hgb)\s+(?:target|threshold|goal|level)\b",
            re.IGNORECASE,
        ),
        canonical_concept="restrictive transfusion strategy",
        role="comparator",
        expanded_terms=_CONCEPT_EXPANSION["restrictive transfusion strategy"],
        rationale="Low hemoglobin target → restrictive transfusion strategy (cardiology/transfusion pack)",
        confidence=0.92,
        requires_any=_AMI_TOKENS + _ANEMIA_TOKENS + _TRANSFUSION_TOKENS,
    ),
    # ---- Comparison-pattern interpreter, transfusion variants ----
    TranslationRule(
        pattern=re.compile(
            r"\bliberal\s+(?:vs|versus|or)\s+restrictive\b|"
            r"\brestrictive\s+(?:vs|versus|or)\s+liberal\b",
            re.IGNORECASE,
        ),
        canonical_concept="liberal vs restrictive transfusion strategy",
        role="intervention",
        expanded_terms=_CONCEPT_EXPANSION["liberal transfusion strategy"]
            + _CONCEPT_EXPANSION["restrictive transfusion strategy"],
        rationale="Explicit liberal-vs-restrictive comparison detected",
        confidence=0.95,
        requires_any=_TRANSFUSION_TOKENS + _ANEMIA_TOKENS + ["transfusion strategy", "transfusion threshold"],
    ),
    # ---- Lipid/statin domain ----
    TranslationRule(
        pattern=re.compile(
            r"\b(?:intensive|high[- ]intensity|high[- ]dose)\s+(?:statin|lipid[- ]lowering|LDL[- ]lowering)\b",
            re.IGNORECASE,
        ),
        canonical_concept="intensive lipid lowering",
        role="intervention",
        expanded_terms=_CONCEPT_EXPANSION["intensive lipid lowering"],
        rationale="Intensive lipid lowering pattern detected",
        confidence=0.9,
        requires_any=_LIPID_TOKENS,
    ),
    TranslationRule(
        pattern=re.compile(
            r"\b(?:moderate|standard)\s+(?:statin|lipid[- ]lowering)\b",
            re.IGNORECASE,
        ),
        canonical_concept="standard lipid lowering",
        role="comparator",
        expanded_terms=_CONCEPT_EXPANSION["standard lipid lowering"],
        rationale="Standard lipid lowering pattern detected",
        confidence=0.85,
        requires_any=_LIPID_TOKENS,
    ),
]


# Implicit "X vs Y" / "difference between X and Y" comparison interpreter.
# Each pattern returns (intervention_text, comparator_text) on match.
@dataclass
class ComparisonPattern:
    pattern: re.Pattern
    domain_tokens: list[str]   # Must have AT LEAST ONE to fire — context guard.
    semantic_label: str         # e.g. "transfusion threshold comparison"
    rationale: str

    def match(self, q: str) -> tuple[str, str] | None:
        m = self.pattern.search(q)
        if not m:
            return None
        if not _has_any(q, self.domain_tokens):
            return None
        groups = m.groupdict() if m.groupdict() else None
        if groups and "intervention" in groups and "comparator" in groups:
            return groups["intervention"].strip(), groups["comparator"].strip()
        # Otherwise the pattern has 2 positional groups (low, high).
        if m.lastindex and m.lastindex >= 2:
            return m.group(1).strip(), m.group(2).strip()
        return None


_COMPARISON_PATTERNS: list[ComparisonPattern] = [
    # "difference between X and Y" / "X vs Y" / "X versus Y" for hi/low targets.
    ComparisonPattern(
        pattern=re.compile(
            r"\b(?:difference between|comparison of|comparing)\s+"
            r"(?P<intervention>high(?:er)?\s+(?:hemoglobin|haemoglobin|Hb|Hgb)\s+(?:target|threshold|level))"
            r"\s+(?:and|vs|versus|or)\s+"
            r"(?P<comparator>low(?:er)?\s+(?:hemoglobin|haemoglobin|Hb|Hgb)\s+(?:target|threshold|level))",
            re.IGNORECASE,
        ),
        domain_tokens=_AMI_TOKENS + _ANEMIA_TOKENS + _TRANSFUSION_TOKENS,
        semantic_label="transfusion_threshold_comparison",
        rationale="Comparison pattern: difference between high vs low hemoglobin target",
    ),
    ComparisonPattern(
        pattern=re.compile(
            r"\bhigh(?:er)?\s+(?:hemoglobin|haemoglobin|Hb|Hgb)\s+(?:target|threshold|level)"
            r"\s+(?:vs|versus|or)\s+"
            r"low(?:er)?\s+(?:hemoglobin|haemoglobin|Hb|Hgb)\s+(?:target|threshold|level)\b",
            re.IGNORECASE,
        ),
        domain_tokens=_AMI_TOKENS + _ANEMIA_TOKENS + _TRANSFUSION_TOKENS,
        semantic_label="transfusion_threshold_comparison",
        rationale="Comparison pattern: high vs low hemoglobin target",
    ),
    ComparisonPattern(
        pattern=re.compile(
            r"\bliberal(?:\s+transfusion)?\s+(?:vs|versus|or)\s+restrictive(?:\s+transfusion)?\b",
            re.IGNORECASE,
        ),
        domain_tokens=_TRANSFUSION_TOKENS + _ANEMIA_TOKENS,
        semantic_label="transfusion_threshold_comparison",
        rationale="Comparison pattern: liberal vs restrictive transfusion",
    ),
]


@dataclass
class TranslationResult:
    translated_question: str
    concept_translations: list[dict[str, Any]]
    latent_concepts: list[str]
    must_have_concepts_added: list[dict[str, Any]]
    ranking_concepts_added: list[dict[str, Any]]
    negative_concepts_added: list[dict[str, Any]]
    warnings: list[str]
    question_subtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "translated_question": self.translated_question,
            "concept_translations": self.concept_translations,
            "latent_concepts": self.latent_concepts,
            "must_have_concepts_added": self.must_have_concepts_added,
            "ranking_concepts_added": self.ranking_concepts_added,
            "negative_concepts_added": self.negative_concepts_added,
            "warnings": self.warnings,
            "question_subtype": self.question_subtype,
        }


class ClinicalConceptTranslationService:
    """Phase 5B — concept translation + comparison-pattern interpreter.

    Pure-deterministic. Refuses PHI-blocked sessions (caller passes
    ``is_phi_blocked=True`` — we then return an empty no-op result).
    """

    def translate(
        self,
        normalized_question: str,
        picott: dict[str, Any] | None = None,
        *,
        is_phi_blocked: bool = False,
    ) -> dict[str, Any]:
        if is_phi_blocked:
            return TranslationResult(
                translated_question=normalized_question or "",
                concept_translations=[],
                latent_concepts=[],
                must_have_concepts_added=[],
                ranking_concepts_added=[],
                negative_concepts_added=[],
                warnings=["phi_blocked_no_translation"],
            ).to_dict()

        q = (normalized_question or "").strip()
        translations: list[dict[str, Any]] = []
        latent: list[str] = []
        must_have_added: list[dict[str, Any]] = []
        ranking_added: list[dict[str, Any]] = []
        negative_added: list[dict[str, Any]] = []
        warnings: list[str] = []
        question_subtype: str | None = None

        # ---- 1) Apply translation rules ----
        rule_canonicals_added: set[str] = set()
        for rule in _RULES:
            if not rule.pattern.search(q):
                continue
            if rule.requires_any and not _has_any(q, rule.requires_any):
                warnings.append(
                    f"rule_skipped_missing_context:{rule.canonical_concept}"
                )
                continue
            if rule.requires_none and not _has_none(q, rule.requires_none):
                warnings.append(
                    f"rule_skipped_blocked_context:{rule.canonical_concept}"
                )
                continue
            original_phrase = rule.pattern.search(q).group(0)
            translations.append({
                "original_phrase": original_phrase,
                "canonical_concept": rule.canonical_concept,
                "role": rule.role,
                "expanded_terms": list(rule.expanded_terms),
                "confidence": rule.confidence,
                "rationale": rule.rationale,
            })
            if rule.canonical_concept not in rule_canonicals_added:
                rule_canonicals_added.add(rule.canonical_concept)
                must_have_added.append({
                    "concept": rule.canonical_concept,
                    "role": rule.role if rule.role in {
                        "intervention", "comparator", "condition",
                        "outcome", "population", "context", "setting",
                        "study_design", "time_horizon",
                    } else "context",
                    "synonyms": list(rule.expanded_terms),
                    "required_for_high_relevance": True,
                    # Strategy-level concepts carry a moderate cap (65)
                    # because the article may discuss the topic via a
                    # synonym we missed.
                    "missing_relevance_cap": 65,
                })

        # ---- 2) Apply comparison-pattern interpreter ----
        intervention_text: str | None = None
        comparator_text: str | None = None
        for cp in _COMPARISON_PATTERNS:
            res = cp.match(q)
            if res is None:
                continue
            intervention_text = res[0]
            comparator_text = res[1]
            question_subtype = cp.semantic_label
            warnings.append(f"comparison_pattern_matched:{cp.semantic_label}")
            translations.append({
                "original_phrase": f"{intervention_text} vs {comparator_text}",
                "canonical_concept": cp.semantic_label,
                "role": "context",
                "expanded_terms": [],
                "confidence": 0.9,
                "rationale": cp.rationale,
            })
            break

        # ---- 3) Cardiology/transfusion pack — latent concepts when relevant ----
        if _has_any(q, _AMI_TOKENS) and _has_any(q, _ANEMIA_TOKENS):
            pack = get_pack("cardiology_transfusion") or {}
            # Add anemia + AMI canonical must-haves (cap 30 for missing
            # intervention/AMI/condition — drives the transfusion-strategy
            # ranking rules in Task 8).
            if "acute myocardial infarction" not in rule_canonicals_added:
                rule_canonicals_added.add("acute myocardial infarction")
                must_have_added.append({
                    "concept": "acute myocardial infarction",
                    "role": "condition",
                    "synonyms": list(pack.get("ami_variants") or _CONCEPT_EXPANSION["acute myocardial infarction"]),
                    "required_for_high_relevance": True,
                    "missing_relevance_cap": 40,
                })
            if "anemia" not in rule_canonicals_added:
                rule_canonicals_added.add("anemia")
                must_have_added.append({
                    "concept": "anemia",
                    "role": "context",
                    "synonyms": list(pack.get("hemoglobin_anemia_variants") or _CONCEPT_EXPANSION["anemia"]),
                    "required_for_high_relevance": True,
                    "missing_relevance_cap": 30,
                })
            if "red blood cell transfusion" not in rule_canonicals_added:
                rule_canonicals_added.add("red blood cell transfusion")
                must_have_added.append({
                    "concept": "red blood cell transfusion",
                    "role": "context",
                    "synonyms": list(pack.get("transfusion_strategy_terms") or _CONCEPT_EXPANSION["red blood cell transfusion"]),
                    "required_for_high_relevance": True,
                    "missing_relevance_cap": 65,
                })
            # Ranking concepts — trial acronyms (boost-only).
            for trial in (pack.get("trial_acronyms") or []):
                ranking_added.append({
                    "concept": trial,
                    "role": "context",
                    "synonyms": [],
                    "boost_weight": 6,
                })
            # Ranking concepts — outcomes (boost-only).
            for outcome in (pack.get("outcomes") or [])[:8]:
                ranking_added.append({
                    "concept": outcome,
                    "role": "outcome",
                    "synonyms": [],
                    "boost_weight": 3,
                })
            # Latent concepts to surface in the transparency panel.
            latent.extend([
                "anemia", "transfusion threshold",
                "red blood cell transfusion", "AMI", "ACS",
            ])
            # Negative penalty concepts — wrong-population transfusion threshold
            # questions get capped (e.g. cardiac surgery without AMI).
            for neg, reason in (
                ("cardiac surgery transfusion threshold",
                 "wrong population (cardiac surgery transfusion threshold without AMI)"),
                ("ICU transfusion threshold",
                 "wrong population (ICU transfusion threshold without MI/ACS)"),
                ("trauma transfusion threshold",
                 "wrong population (trauma transfusion threshold without AMI)"),
                ("GI bleeding transfusion threshold",
                 "wrong population (GI bleeding transfusion threshold without AMI)"),
            ):
                negative_added.append({
                    "concept": neg,
                    "penalty_reason": reason,
                    "unless_concepts_present": list(_AMI_TOKENS),
                    "penalty_cap": 50,
                })

            # Sub-type label for the UI.
            if question_subtype is None:
                question_subtype = "transfusion_threshold_comparison"

        # ---- 4) Build the translated question (deterministic rewrite) ----
        translated = q
        if intervention_text and comparator_text and (
            "liberal" in [t["canonical_concept"] for t in translations]
            or any(t["canonical_concept"] == "liberal transfusion strategy"
                    for t in translations)
        ):
            # Already canonical — leave as-is.
            pass
        if question_subtype == "transfusion_threshold_comparison" \
                and _has_any(q, _AMI_TOKENS) and _has_any(q, _ANEMIA_TOKENS):
            translated = (
                "In patients with acute myocardial infarction and anemia, "
                "how does a liberal red blood cell transfusion strategy or "
                "higher hemoglobin threshold compare with a restrictive "
                "transfusion strategy or lower hemoglobin threshold for "
                "death, recurrent MI, and major cardiovascular events?"
            )

        return TranslationResult(
            translated_question=translated,
            concept_translations=translations,
            latent_concepts=latent,
            must_have_concepts_added=must_have_added,
            ranking_concepts_added=ranking_added,
            negative_concepts_added=negative_added,
            warnings=warnings,
            question_subtype=question_subtype,
        ).to_dict()


clinical_concept_translation = ClinicalConceptTranslationService()
