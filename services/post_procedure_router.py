"""Phase 6G.4P — PostProcedureManagementRouter.

Deterministic, closed-set regex/pattern detector that recognises
post-procedure management questions BEFORE the generic-only fast-path
fires. Runs in front of the ``PrimaryClinicalEntityAnchorService``
generic-only short-circuit so questions like
``"what are the recommendations for anticoagulation after aortic
valve surgery?"`` are NOT misclassified as generic-only.

Output (closed-set):
    {
      "matched": bool,
      "therapy_phrase": str | None,
      "procedure_phrase": str | None,
      "intent": "guideline" | "therapy" | "surveillance" | "restart",
      "subdomain": str   # closed-set, see _SUBDOMAINS
    }

NO LLM. NO network. Pure regex over closed-set lexicons.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("clinevidence.post_procedure_router")


# Closed-set subdomains (mirror Phase 6G.4P brief Task 4).
SUBDOMAINS: frozenset[str] = frozenset({
    "post_procedure_antithrombotic_management",
    "post_procedure_antibiotic_management",
    "post_procedure_vte_prophylaxis",
    "post_procedure_followup_surveillance",
    "perioperative_medication_management",
    "post_transplant_immunosuppression",
    "post_procedure_pain_management",
    # Phase H2 — additive subdomain for post-transplant medication
    # management (e.g. anti-CD20 / rituximab maintenance after
    # autologous stem cell transplant). Closed-set; routed via the
    # hematology_procedures.json pack + the new
    # _POST_TRANSPLANT_MED_TOKENS lexicon below.
    "post_transplant_medication_management",
})


# Therapy / management lexicons, indexed by the subdomain they route to.
_ANTITHROMBOTIC_TOKENS: tuple[str, ...] = (
    "anticoagulation", "anticoagulant therapy", "anticoagulants",
    "antithrombotic therapy", "antithrombotic treatment",
    "antiplatelet therapy", "antiplatelets", "dual antiplatelet",
    "DAPT",
    "warfarin", "vitamin K antagonist", "VKA",
    "DOAC", "NOAC", "direct oral anticoagulant",
    "apixaban", "rivaroxaban", "dabigatran", "edoxaban",
    "aspirin", "clopidogrel", "ticagrelor", "prasugrel",
    "heparin", "LMWH", "enoxaparin",
)

_VTE_TOKENS: tuple[str, ...] = (
    "VTE prophylaxis", "venous thromboembolism prophylaxis",
    "thromboprophylaxis", "DVT prophylaxis",
    "thromboembolism prophylaxis",
)

_ANTIBIOTIC_TOKENS: tuple[str, ...] = (
    "antibiotic prophylaxis", "antibiotics",
    "perioperative antibiotic", "surgical site infection prophylaxis",
    "endocarditis prophylaxis",
)

_IMMUNOSUPPRESSION_TOKENS: tuple[str, ...] = (
    "immunosuppression", "immunosuppressive therapy",
    "immunosuppressants",
    "tacrolimus", "cyclosporine", "ciclosporin",
    "mycophenolate", "mycophenolate mofetil", "MMF",
    "azathioprine", "sirolimus", "everolimus",
    "induction therapy", "maintenance immunosuppression",
)

_PAIN_TOKENS: tuple[str, ...] = (
    "pain control", "pain management",
    "postoperative analgesia", "post-operative analgesia",
    "multimodal analgesia",
)

_SURVEILLANCE_TOKENS: tuple[str, ...] = (
    "follow-up", "follow up", "surveillance",
    "surveillance colonoscopy", "surveillance interval",
    "post-treatment surveillance",
)


# Phase H2 — closed-set post-transplant medication tokens (rituximab
# maintenance / consolidation after ASCT, anti-CD20 maintenance, etc.).
# Routed to the new ``post_transplant_medication_management`` subdomain
# via the order in ``_THERAPY_TO_SUBDOMAIN`` below.
_POST_TRANSPLANT_MED_TOKENS: tuple[str, ...] = (
    "rituximab", "rituximab maintenance", "rituximab consolidation",
    "anti-CD20", "anti-CD20 therapy",
    "post-transplant maintenance", "post transplant maintenance",
    "post-transplant consolidation", "post transplant consolidation",
    "maintenance therapy",
)


# Phase H2-tail Task C — closed-set perioperative / anesthesia /
# medication-avoidance tokens for inherited cardiac arrhythmia
# syndromes (Brugada / LQTS / CPVT). Mirrors the
# ``_POST_TRANSPLANT_MED_TOKENS`` pattern: routes to the existing
# closed-set ``perioperative_medication_management`` subdomain so the
# anchor service produces dual (condition + perioperative) primary
# anchors for queries like
# ``"perioperative management of patients with Brugada syndrome"``,
# ``"anesthesia management in Brugada syndrome"``,
# ``"medications to avoid in Brugada syndrome"``.
_PERIOPERATIVE_AVOIDANCE_TOKENS: tuple[str, ...] = (
    "perioperative management", "perioperative care",
    "perioperative drug avoidance",
    "anesthesia", "anaesthesia",
    "anesthesia management", "anaesthesia management",
    "anesthetic management", "anaesthetic management",
    "drug avoidance", "drugs to avoid",
    "medication avoidance", "medications to avoid",
    "sodium channel blocker avoidance",
    "qt-prolonging drug avoidance", "qt prolonging drug avoidance",
    "fever management",
)


# Order matters — more specific tokens first.
_THERAPY_TO_SUBDOMAIN: tuple[tuple[tuple[str, ...], str], ...] = (
    (_VTE_TOKENS, "post_procedure_vte_prophylaxis"),
    (_ANTIBIOTIC_TOKENS, "post_procedure_antibiotic_management"),
    # Phase H2 — anti-CD20 / rituximab maintenance MUST route to the
    # new post_transplant_medication_management subdomain BEFORE the
    # immunosuppression bucket so rituximab+ASCT is captured
    # specifically rather than collapsed into the generic
    # immunosuppression bucket.
    (_POST_TRANSPLANT_MED_TOKENS, "post_transplant_medication_management"),
    (_IMMUNOSUPPRESSION_TOKENS, "post_transplant_immunosuppression"),
    (_PAIN_TOKENS, "post_procedure_pain_management"),
    (_ANTITHROMBOTIC_TOKENS, "post_procedure_antithrombotic_management"),
    (_SURVEILLANCE_TOKENS, "post_procedure_followup_surveillance"),
    # Phase H2-tail Task C — perioperative / anesthesia / medication-
    # avoidance lexicon for inherited cardiac arrhythmia syndromes
    # routes to the existing perioperative_medication_management
    # subdomain. Placed LAST so it does not pre-empt the more
    # specific procedure-domain buckets above.
    (_PERIOPERATIVE_AVOIDANCE_TOKENS, "perioperative_medication_management"),
)


# Pattern templates. Two-slot patterns capture both the therapy and
# procedure phrases; one-slot patterns capture only the procedure
# (intent derives from the surrounding lexical signal).
_PATTERNS_THERAPY_THEN_PROCEDURE: tuple[re.Pattern[str], ...] = (
    # "recommendations for [therapy] after [procedure]"
    re.compile(
        r"\brecommendations?\s+(?:for|on|about)\s+(?P<therapy>.+?)\s+"
        r"(?:after|following|post)\s+(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
    # "guidelines for [therapy] after [procedure]"
    re.compile(
        r"\bguidelines?\s+(?:for|on|about)\s+(?P<therapy>.+?)\s+"
        r"(?:after|following|post)\s+(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
    # "[therapy] after [procedure]" — capturing version
    re.compile(
        r"^\s*(?:what\s+(?:is|are)\s+(?:the\s+)?)?(?P<therapy>.+?)\s+"
        r"(?:after|following|post)\s+(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
    # "postoperative [therapy] after [procedure]"
    re.compile(
        r"\bpost[-\s]?operative\s+(?P<therapy>.+?)\s+"
        r"(?:after|following|post)\s+(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
    # "perioperative [therapy] for [procedure]"
    re.compile(
        r"\bperi[-\s]?operative\s+(?P<therapy>.+?)\s+(?:for|in|during)\s+"
        r"(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
    # "when to restart [therapy] after [procedure]"
    re.compile(
        r"\bwhen\s+(?:to\s+|should\s+\w+\s+be\s+)?restart(?:ed)?\s+"
        r"(?P<therapy>.+?)\s+(?:after|following|post)\s+(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
    # "prophylaxis after [procedure]" — therapy implicit
    re.compile(
        r"\b(?P<therapy>(?:antibiotic|VTE|venous\s+thromboembolism|"
        r"thromboembolism|infection|DVT|endocarditis)\s+prophylaxis)\s+"
        r"(?:after|following|post)\s+(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
    # "what [therapy] is (recommended|used) after [procedure]"
    re.compile(
        r"\bwhat\s+(?P<therapy>.+?)\s+is\s+(?:recommended|used|appropriate|"
        r"indicated|preferred|advised|given)\s+"
        r"(?:after|following|post|for)\s+(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
)


_PATTERNS_PROCEDURE_ONLY: tuple[re.Pattern[str], ...] = (
    # "follow-up after [procedure]"
    re.compile(
        r"\bfollow[-\s]?up\s+(?:after|following|post)\s+(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
    # "surveillance after [procedure]"
    re.compile(
        r"\bsurveillance\s+(?:after|following|post)\s+(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
    # "post [procedure] [therapy]" — handled below by therapy-after-noun
)


# Phase H2-tail Task C — condition-context patterns. Mirror the
# "therapy then procedure" two-slot family but with the linking
# preposition being "in / of / for / with" instead of
# "after / following / post" — that is the structural difference
# between perioperative-management-of-a-condition queries (Brugada)
# and post-procedure-management queries (CABG / ASCT).
#
# These patterns are evaluated AFTER the procedure family so a
# question that fits both is preferentially routed via the procedure
# family.
_PATTERNS_THERAPY_IN_CONDITION: tuple[re.Pattern[str], ...] = (
    # "[therapy] (of|in|for) (?:patients with )?[condition]"
    re.compile(
        r"^\s*(?:what\s+(?:is|are)\s+(?:the\s+)?)?(?P<therapy>.+?)\s+"
        r"(?:of|in|for)\s+(?:patients?\s+with\s+)?(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
    # "medications? to avoid (?:in|with|during|for) [condition]"
    re.compile(
        r"\b(?P<therapy>medications?\s+to\s+avoid|drugs?\s+to\s+avoid|"
        r"drug\s+avoidance|medication\s+avoidance|"
        r"perioperative\s+(?:management|care|drug\s+avoidance)|"
        r"anaesthetic\s+management|anesthetic\s+management|"
        r"anaesthesia\s+management|anesthesia\s+management)\s+"
        r"(?:in|with|during|for)\s+"
        r"(?:patients?\s+with\s+)?(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
)


# Phase J — passive-voice medication-avoidance patterns. Routed
# directly to ``perioperative_medication_management`` (the closed-set
# subdomain), with the captured "condition" slot still filtered by
# ``_looks_like_inherited_arrhythmia``. The therapy slot here is a
# closed-set verb cluster (medications/drugs/medicines/agents +
# should/must/can/may + be + avoided/used/...) so we can route
# without going through the lexical _therapy_to_subdomain lookup —
# the pattern IS the lexical signal.
_PATTERNS_PASSIVE_MEDICATION_AVOIDANCE: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(?:what\s+)?(?P<therapy>medications?|drugs?|medicines?|"
        r"agents?|anesthetics?|anaesthetics?)\s+(?:should|must|can|may|"
        r"are)\s+be\s+"
        r"(?:avoided|used|considered|administered|given|prescribed|"
        r"selected|chosen)\s+"
        r"(?:in|with|during|for)\s+"
        r"(?:patients?\s+with\s+)?(?P<procedure>.+?)\??\s*$",
        re.IGNORECASE,
    ),
)


# Phase H2-tail Task C — closed-set lexical signal that the captured
# "procedure" slot is actually an inherited arrhythmia condition.
# Mirrors the structure of ``_PROCEDURE_LEXICAL_TOKENS`` but used by
# the condition-context branch only.
_INHERITED_ARRHYTHMIA_LEXICAL_TOKENS: tuple[str, ...] = (
    "brugada", "brugada syndrome", "brugada pattern",
    "brs",
    "long qt", "long qt syndrome", "lqts",
    "lqt1", "lqt2", "lqt3",
    "catecholaminergic polymorphic ventricular tachycardia",
    "cpvt",
    "channelopathy", "channelopathies",
    "sodium channelopathy", "sodium channelopathies",
    "inherited arrhythmia", "inherited arrhythmia syndrome",
    "inherited arrhythmias",
    "short qt syndrome", "sqts",
    "arrhythmogenic right ventricular cardiomyopathy", "arvc",
    "andersen-tawil syndrome", "andersen tawil syndrome",
)


# Procedure-word allowlist (closed-set tokens that signal a procedure
# is present in the captured phrase). The router consults the procedure
# ontology mapper to confirm — this set is only a coarse pre-filter.
_PROCEDURE_LEXICAL_TOKENS: tuple[str, ...] = (
    "surgery", "surgical", "operation",
    "replacement", "repair", "arthroplasty",
    "transplant", "transplantation",
    "bypass", "stent", "stenting",
    "intervention", "implant", "implantation",
    "colonoscopy", "endoscopy", "polypectomy",
    "resection", "colectomy",
    "fracture", "fixation",
    "PCI", "TAVR", "TAVI", "CABG",
    "SAVR", "AVR", "MVR", "THA", "TKA", "THR", "TKR", "ORIF",
    "valve", "valvular",
    "EMR", "ESD",
    "gastric bypass", "sleeve gastrectomy",
    "kidney transplant", "liver transplant", "heart transplant",
)


def _therapy_to_subdomain(therapy_phrase: str) -> Optional[str]:
    if not therapy_phrase:
        return None
    t = therapy_phrase.strip().lower()
    for tokens, subdomain in _THERAPY_TO_SUBDOMAIN:
        for tok in tokens:
            if tok.lower() in t:
                return subdomain
    return None


def _intent_for(therapy: str, question: str, subdomain: str) -> str:
    q = (question or "").lower()
    t = (therapy or "").lower()
    if "restart" in q or "restart" in t:
        return "restart"
    if subdomain == "post_procedure_followup_surveillance":
        return "surveillance"
    if "guideline" in q or "recommendation" in q or "recommend" in q:
        return "guideline"
    return "therapy"


def _looks_like_procedure(phrase: str) -> bool:
    if not phrase:
        return False
    p = phrase.strip().lower()
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(tok.lower())}(?![a-z0-9])", p)
        for tok in _PROCEDURE_LEXICAL_TOKENS
    )


def _looks_like_inherited_arrhythmia(phrase: str) -> bool:
    """Phase H2-tail Task C — sibling allowlist filter for the
    condition-context branch. Returns True when the captured slot
    contains a closed-set inherited-arrhythmia condition token.
    Pure regex word-boundary match, no LLM.
    """
    if not phrase:
        return False
    p = phrase.strip().lower()
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(tok.lower())}(?![a-z0-9])", p)
        for tok in _INHERITED_ARRHYTHMIA_LEXICAL_TOKENS
    )


def _clean_phrase(s: str) -> str:
    if not s:
        return ""
    s = s.strip().strip("?.,!")
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.IGNORECASE).strip()
    return s


class PostProcedureManagementRouter:
    """Closed-set post-procedure question router. Stateless / pure.

    Use ``detect(question)`` from the pipeline BEFORE the generic-only
    fast-path. When ``matched=True``, the pipeline emits a closed-set
    routing record and proceeds with deterministic procedure +
    management anchor extraction (see
    ``PrimaryClinicalEntityAnchorService.detect_post_procedure``).
    """

    def detect(self, question: str) -> dict:
        empty = {
            "matched": False,
            "therapy_phrase": None,
            "procedure_phrase": None,
            "intent": "therapy",
            "subdomain": "post_procedure_antithrombotic_management",
        }
        if not question:
            return empty
        q = question.strip().strip(".!").strip()
        if not q:
            return empty

        # Try two-slot patterns first.
        for pat in _PATTERNS_THERAPY_THEN_PROCEDURE:
            m = pat.search(q)
            if not m:
                continue
            therapy = _clean_phrase(m.group("therapy"))
            procedure = _clean_phrase(m.group("procedure"))
            if not therapy or not procedure:
                continue
            if not _looks_like_procedure(procedure):
                continue
            subdomain = _therapy_to_subdomain(therapy)
            if not subdomain:
                # Therapy phrase doesn't carry a closed-set management
                # signal — refuse to over-match.
                continue
            return {
                "matched": True,
                "therapy_phrase": therapy,
                "procedure_phrase": procedure,
                "intent": _intent_for(therapy, q, subdomain),
                "subdomain": subdomain,
            }

        # Procedure-only patterns (follow-up / surveillance).
        for pat in _PATTERNS_PROCEDURE_ONLY:
            m = pat.search(q)
            if not m:
                continue
            procedure = _clean_phrase(m.group("procedure"))
            if not procedure or not _looks_like_procedure(procedure):
                continue
            return {
                "matched": True,
                "therapy_phrase": None,
                "procedure_phrase": procedure,
                "intent": "surveillance",
                "subdomain": "post_procedure_followup_surveillance",
            }

        # Phase H2-tail Task C — condition-context patterns
        # ("[therapy] in/of/for [inherited arrhythmia]"). Evaluated
        # AFTER the procedure family so questions that fit both are
        # preferentially routed via the procedure family. Captured
        # therapy MUST map to a closed-set subdomain via
        # ``_therapy_to_subdomain`` AND the captured "procedure" slot
        # MUST contain a closed-set inherited-arrhythmia token. If
        # either filter fails, we refuse to match (preserves the
        # over-matching guard that ``_looks_like_procedure``
        # establishes for the procedure family).
        for pat in _PATTERNS_THERAPY_IN_CONDITION:
            m = pat.search(q)
            if not m:
                continue
            therapy = _clean_phrase(m.group("therapy"))
            procedure = _clean_phrase(m.group("procedure"))
            if not therapy or not procedure:
                continue
            if not _looks_like_inherited_arrhythmia(procedure):
                continue
            subdomain = _therapy_to_subdomain(therapy)
            if not subdomain:
                continue
            return {
                "matched": True,
                "therapy_phrase": therapy,
                "procedure_phrase": procedure,
                "intent": _intent_for(therapy, q, subdomain),
                "subdomain": subdomain,
            }

        # Phase J — passive-voice medication-avoidance patterns. The
        # pattern itself encodes the closed-set verb cluster
        # (medications/drugs/medicines/agents + should/must/can/may +
        # be + avoided/used/...) so we route directly to the
        # ``perioperative_medication_management`` subdomain without
        # the lexical token table lookup. Condition slot is still
        # filtered by the closed-set inherited-arrhythmia allowlist.
        for pat in _PATTERNS_PASSIVE_MEDICATION_AVOIDANCE:
            m = pat.search(q)
            if not m:
                continue
            therapy = _clean_phrase(m.group("therapy"))
            procedure = _clean_phrase(m.group("procedure"))
            if not therapy or not procedure:
                continue
            if not _looks_like_inherited_arrhythmia(procedure):
                continue
            subdomain = "perioperative_medication_management"
            return {
                "matched": True,
                "therapy_phrase": therapy,
                "procedure_phrase": procedure,
                "intent": _intent_for(therapy, q, subdomain),
                "subdomain": subdomain,
            }
        return empty


post_procedure_router = PostProcedureManagementRouter()


__all__ = [
    "SUBDOMAINS",
    "PostProcedureManagementRouter",
    "post_procedure_router",
]
