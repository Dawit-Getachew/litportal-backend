"""Phase K1.F — Deterministic Clinical Signal Router.

Closed-set, deterministic, no LLM. Reads the (already PHI-screened +
spelling + shorthand normalized) question text and emits closed-set
lexical routing hints used to decide whether the bounded resolver
needs to be invoked.

This is NOT the main semantic interpreter. It exists ONLY to:
  * Surface ``possible_intents`` from CLOSED-SET lexical patterns.
  * Detect multi-intent language (closed-set conjunctions / mixed
    question-stems).
  * Provide ``needs_resolver: bool`` for the pipeline to gate the
    fallback path.

The pipeline (K1.I) MAY override ``needs_resolver`` to False when the
deterministic anchor service already produced a confident primary
anchor — i.e. when a deterministic parser / concept pack succeeded.

Safety invariants (must remain true after every change):
  * NO PHI handling here (the PHI guard owns that — runs before).
  * NO external network / DB / LLM calls.
  * Pure / sync / stateless.
  * Output is a ``DeterministicSignalSummary`` (closed-set schema).
"""
from __future__ import annotations

import re
from typing import Iterable

from services.clinical_question_map import (
    ACTIVE_INTENTS,
    ClinicalIntent,
    DeterministicSignalSummary,
)


# ---------------------------------------------------------------------
# Lexical patterns — closed-set. Each entry is
# ``(intent_label, signal_label, compiled_regex)``.
# ---------------------------------------------------------------------

# (Intent labels MUST come from ``ACTIVE_INTENTS``; the validator
# below verifies this at import time.)
_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    # ---- Therapy / intervention ---------------------------------
    ("therapy", "therapy_token", re.compile(
        r"\b(?:treatment|treat|therapy|manage|management|"
        r"first[-\s]line|second[-\s]line|"
        r"recommended\s+(?:treatment|therapy)|"
        r"treatment\s+(?:options?|algorithm|sequence))\b",
        re.IGNORECASE,
    )),
    # ---- Diagnosis accuracy / workup / differential -------------
    ("diagnosis_accuracy", "diagnostic_accuracy_token", re.compile(
        r"\b(?:diagnostic\s+accuracy|sensitivity\s+and\s+specificity|"
        r"sensitivity|specificity|likelihood\s+ratio|"
        r"test\s+characteristics|positive\s+predictive\s+value|"
        r"negative\s+predictive\s+value|auc|roc\s+curve)\b",
        re.IGNORECASE,
    )),
    ("diagnostic_workup", "diagnostic_workup_token", re.compile(
        r"\b(?:work[-\s]?up|workup|how\s+to\s+evaluate|"
        r"how\s+do\s+I\s+evaluate|"
        r"initial\s+evaluation|approach\s+to)\b",
        re.IGNORECASE,
    )),
    ("differential_diagnosis", "differential_diagnosis_token", re.compile(
        r"(?:"
        r"\bdifferential\s+diagnos[ei]s\b"
        r"|\bddx\b"
        r"|\bdon[-'’]?t\s+miss\b"
        r"|\bdo\s+not\s+miss\b"
        r"|\bdiagnoses?\s+(?:should\s+I\s+)?(?:not|never)\s+miss\b"
        r"|\bcannot\s+miss\b"
        r"|\bcan[\'’]?t\s+miss\b"
        r"|\bmust\s+rule\s+out\b"
        r"|\bred\s+flags?\b"
        r"|\bcauses?\s+of\b"
        r"|\betiolog(?:y|ies)\s+of\b"
        r")",
        re.IGNORECASE,
    )),
    # ---- Prognosis / prediction ---------------------------------
    ("prognosis_natural_history", "prognosis_token", re.compile(
        r"\b(?:prognosis|natural\s+history|outcome|mortality|"
        r"survival|recurrence|progression)\b",
        re.IGNORECASE,
    )),
    ("prognostic_factor", "prognostic_factor_token", re.compile(
        r"\b(?:prognostic\s+(?:factor|marker)|predict(?:s|or)?\s+"
        r"(?:outcome|mortality|recurrence)|"
        r"associated\s+with\s+(?:worse|better)\s+outcome)\b",
        re.IGNORECASE,
    )),
    ("prediction_model", "prediction_model_token", re.compile(
        r"\b(?:prediction\s+model|risk\s+score|"
        r"clinical\s+score|validation\s+(?:cohort|study)|"
        r"calibration|discrimination)\b",
        re.IGNORECASE,
    )),
    # ---- Risk / etiology / harm + complication risk -------------
    ("risk_etiology_harm", "risk_etiology_token", re.compile(
        r"\b(?:risk\s+factor|caused?\s+by|associated\s+with|"
        r"exposure\s+to|etiology|harm|harmful|adverse\s+effect|"
        r"\w+\s+risk\b|risk\s+of\s+\w+)\b",
        re.IGNORECASE,
    )),
    ("complication_risk", "complication_token", re.compile(
        r"\b(?:risk\s+of\s+(?:complication|infection|bleed(?:ing)?|"
        r"migration|dislodg(?:e?ment|ed)|displacement|injury|stroke|"
        r"mortality|death|adverse)|"
        r"complications?\s+of|complications?\s+after|"
        r"complications?\s+following)\b",
        re.IGNORECASE,
    )),
    ("device_complication", "device_complication_token", re.compile(
        r"\b(?:pacemaker|icd|defibrillator|stent|"
        r"prosthesis|prosthetic\s+(?:valve|joint)|"
        r"catheter|central\s+(?:line|venous\s+catheter)|"
        r"tracheostomy|endotracheal\s+tube|implant)\b.*?"
        r"\b(?:infection|dislodg|displace|migration|fail|"
        r"thrombosis|complication|adverse|malfunction)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    ("procedure_complication", "procedure_complication_token", re.compile(
        r"\b(?:after|following|post[-\s]?(?:op|operative|procedure))\b"
        r".{0,40}\b(?:complication|infection|bleed|stroke|mortality|"
        r"thrombosis|dehiscence|leak)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    ("drug_complication", "drug_safety_token", re.compile(
        r"\b(?:side\s+effects?|adverse\s+(?:effects?|reactions?|events?)|"
        r"toxicity|tolerability|drug\s+safety|"
        r"safety\s+(?:profile|of)|"
        r"treatment[-\s]emergent\s+adverse\s+events?)\b",
        re.IGNORECASE,
    )),
    # ---- Epidemiology -------------------------------------------
    ("epidemiology_prevalence_incidence", "epidemiology_token", re.compile(
        r"\b(?:prevalence|incidence|epidemiology|"
        r"how\s+common|how\s+frequent|"
        r"population[-\s]based\s+(?:study|estimate))\b",
        re.IGNORECASE,
    )),
    # ---- Pathophysiology / mechanism ----------------------------
    ("pathophysiology_mechanism", "mechanism_token", re.compile(
        r"\b(?:pathophysiology|mechanism|mechanistic|"
        r"how\s+does\s+.*?\s+(?:cause|lead\s+to|result\s+in)|"
        r"why\s+does\s+.*?\s+(?:occur|happen)|"
        r"molecular\s+basis|biological\s+pathway)\b",
        re.IGNORECASE | re.DOTALL,
    )),
    # ---- Monitoring / follow-up / surveillance ------------------
    ("monitoring_followup_surveillance", "monitoring_token", re.compile(
        r"\b(?:monitor(?:ing)?|follow[-\s]?up|surveillance|"
        r"how\s+often\s+(?:to\s+)?(?:check|monitor|screen)|"
        r"surveillance\s+(?:interval|strategy))\b",
        re.IGNORECASE,
    )),
    # ---- Stabilization / emergency management -------------------
    ("stabilization_emergency_management", "stabilization_token", re.compile(
        r"\b(?:emergency\s+management|acute\s+management|"
        r"initial\s+(?:management|stabilization|approach)|"
        r"ED\s+management|code\s+(?:blue|stemi|stroke))\b",
        re.IGNORECASE,
    )),
    # ---- Guideline-first ----------------------------------------
    ("guideline_recommendation", "guideline_token", re.compile(
        r"\b(?:guideline|guidelines|recommend(?:s|ed|ation)?|"
        r"society\s+(?:recommend|guidance)|aha|esc|acc|ash|nice|"
        r"practice\s+(?:guideline|recommendation)|"
        r"consensus\s+statement)\b",
        re.IGNORECASE,
    )),
    # ---- Background / overview ----------------------------------
    ("background_overview", "background_token", re.compile(
        r"^\s*(?:what\s+(?:is|are)|define|definition\s+of|"
        r"tell\s+me\s+about|overview\s+of|explain|describe)\b",
        re.IGNORECASE,
    )),
    # ---- Prevention / screening ---------------------------------
    ("prevention_screening", "prevention_token", re.compile(
        r"\b(?:prevention|prevent|prophylaxis|screening|"
        r"primary\s+prevention|secondary\s+prevention|"
        r"vaccin(?:e|ation))\b",
        re.IGNORECASE,
    )),
    # ---- Therapy safety -----------------------------------------
    ("therapy_safety", "therapy_safety_token", re.compile(
        r"\b(?:benefits?\s+and\s+risks?|"
        r"safety\s+and\s+efficacy|"
        r"adverse\s+effects?\s+of\s+(?:therapy|treatment))\b",
        re.IGNORECASE,
    )),
    # ---- Drug safety (legacy spec line) -------------------------
    ("drug_safety", "drug_safety_phrase", re.compile(
        r"\b(?:adverse\s+drug\s+(?:reaction|event)|"
        r"black[-\s]box\s+warning|FDA\s+warning)\b",
        re.IGNORECASE,
    )),
]


# Closed-set multi-intent conjunctions / mixed-stem signals. When ANY
# of these patterns matches AND the lexical scan emitted ≥2 distinct
# intent labels, we set ``possible_multi_intent=True``.
_MULTI_INTENT_RE = re.compile(
    r"(?:"
    r"\?\s*(?:and|;|,)\s+"                     # "X? and Y?"
    r"|\s+and\s+(?:what|how|when|why|which)"   # "X and what diagnoses..."
    r"|\s+,\s*and\s+(?:what|how|when|why|which)"  # ", and what is Y"
    r"|\s+or\s+(?:what|how|when|why|which)"    # "X or what..."
    r"|\s+as\s+well\s+as\s+"                   # "X as well as Y"
    r"|\s+plus\s+(?:what|how)"                 # "X plus what is Y"
    r"|\s+also\s+(?:what|how|consider)"
    r"|\s+and\s+(?:also|additionally)\s+"
    r"|\s+and\s+(?:diagnos|differential|cause|"
    r"prognos|risk|complication|mechanism)"    # "X and diagnoses to miss"
    r")",
    re.IGNORECASE,
)


# Validate that every pattern's intent_label is in ACTIVE_INTENTS.
# Importing module fails fast on drift.
def _validate_patterns() -> None:
    bad = [
        (intent, sig) for (intent, sig, _) in _PATTERNS
        if intent not in ACTIVE_INTENTS
    ]
    if bad:
        raise RuntimeError(
            f"clinical_signal_router: patterns reference intents outside "
            f"ACTIVE_INTENTS: {bad}",
        )


_validate_patterns()


# ---------------------------------------------------------------------
# Service.
# ---------------------------------------------------------------------


class ClinicalSignalRouterService:
    """Deterministic, closed-set lexical router.

    Public surface: ``analyze(question) -> DeterministicSignalSummary``.
    NEVER raises. NEVER makes any I/O call.
    """

    def analyze(self, question: str | None) -> DeterministicSignalSummary:
        q = (question or "").strip()
        if not q:
            return DeterministicSignalSummary(
                detected_signals=[],
                possible_intents=[],
                possible_multi_intent=False,
                deterministic_confidence=0.0,
                needs_resolver=True,
                reason="empty_question",
            )

        detected: list[str] = []
        intents: list[ClinicalIntent] = []  # ordered, closed-set
        seen_intents: set[str] = set()
        for intent_label, signal_label, patt in _PATTERNS:
            if patt.search(q):
                if signal_label not in detected:
                    detected.append(signal_label)
                if intent_label not in seen_intents:
                    seen_intents.add(intent_label)
                    intents.append(intent_label)  # type: ignore[arg-type]

        possible_multi_intent = (
            len(intents) >= 2 and bool(_MULTI_INTENT_RE.search(q))
        )
        # Phase K1.1 — broader multi-intent gate. If the lexical scan
        # produced ≥2 distinct intent labels AND the question contains
        # any explicit conjunction (``and`` / ``or`` / ``,`` / ``;`` /
        # ``plus``) anywhere, treat it as multi-intent. This handles
        # phrasings the focused regex misses (e.g. "and red flags to
        # consider", "and complications to monitor").
        if not possible_multi_intent and len(intents) >= 2:
            if re.search(
                r"(?:\band\b|\bor\b|,|;|\bplus\b|\bas\s+well\s+as\b|\balso\b)",
                q,
                re.IGNORECASE,
            ):
                possible_multi_intent = True

        # Deterministic confidence — coarse but closed-set:
        #   0.0  : no signals
        #   0.4  : 1 signal
        #   0.6  : 2-3 signals
        #   0.75 : 4+ signals (almost certainly a concrete clinical
        #          question)
        if not detected:
            det_conf = 0.0
        elif len(detected) == 1:
            det_conf = 0.4
        elif len(detected) <= 3:
            det_conf = 0.6
        else:
            det_conf = 0.75

        # ``needs_resolver`` defaults to True unless the deterministic
        # scan produced ≥2 signals AND no multi-intent ambiguity. The
        # pipeline's existing deterministic anchor service may further
        # override this to False when it produces a confident anchor.
        if not detected:
            needs_resolver = True
            reason = "no_lexical_signal_detected"
        elif possible_multi_intent:
            needs_resolver = True
            reason = "multi_intent_signal_detected"
        elif len(detected) >= 2:
            needs_resolver = False
            reason = "two_or_more_concrete_lexical_signals"
        else:
            needs_resolver = True
            reason = "single_signal_insufficient_for_structuring"

        return DeterministicSignalSummary(
            detected_signals=detected,
            possible_intents=intents,
            possible_multi_intent=possible_multi_intent,
            deterministic_confidence=det_conf,
            needs_resolver=needs_resolver,
            reason=reason,
        )


clinical_signal_router = ClinicalSignalRouterService()


__all__ = [
    "ClinicalSignalRouterService",
    "clinical_signal_router",
]
