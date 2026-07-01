"""Phase M4.6 — deterministic prognosis/stage cancer retrieval rescue.

Cancer-GENERIC (not site-specific). Detects obvious natural-history /
prognosis questions about a staged cancer — e.g. *"What is the prognosis of
stage 3 prostate cancer?"*, *"survival rate for stage II colon cancer"*,
*"recurrence risk after stage I colon cancer"*, *"prognosis for HER2 positive
stage 2 breast cancer"* — and rewrites the PICOTT so that:

  * the CONDITION anchor is the cancer itself (``prostate cancer``), NOT the
    glued full phrase ``prognosis of stage 3 prostate cancer`` (which matches
    no article and caps the entire pool as ``missing_condition_anchor``);
  * ``type_of_question = prognosis`` (was mis-set to ``background`` /
    ``disease_overview``);
  * stage is preserved as a HIGH-PRECISION / ranking concept (NOT a hard
    must-have, so recall is not destroyed by AND-ing ``"stage III"``);
  * prognosis / survival / recurrence / mortality outcomes ride as
    should-have + legacy pico outcomes for the broad-recall + ranking lanes.

Deterministic. NO LLM. NO network. NO scraping. PHI guard already ran first
in the pipeline. This is a "protected fast path" — it only fires on a
high-confidence closed-set question shape.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# ---- prognosis / natural-history signal ----
_PROGNOSIS_SIGNAL = re.compile(
    r"\b(?:prognosis|prognostic|natural\s+history|life\s+expectancy|"
    r"survival|overall\s+survival|disease[-\s]free\s+survival|"
    r"progression[-\s]free\s+survival|metastasis[-\s]free\s+survival|"
    r"cancer[-\s]specific\s+survival|recurrence|mortality|death\s+rate)\b",
    re.IGNORECASE,
)

# ---- cancer head + (≤4) preceding site/qualifier words ----
_CANCER_HEAD = (
    r"(?:cancers?|carcinomas?|adenocarcinomas?|tumou?rs?|neoplasms?|"
    r"malignanc(?:y|ies)|sarcomas?|melanomas?|lymphomas?|leuk[ae]mias?|"
    r"myelomas?|glioblastomas?|gliomas?|mesotheliomas?|blastomas?)"
)
_CANCER_RE = re.compile(
    r"((?:[a-z][a-z-]+\s+){0,4}" + _CANCER_HEAD + r")\b", re.IGNORECASE
)
# Heads that are standalone conditions (need no site word).
_STANDALONE_HEADS = (
    "melanoma", "lymphoma", "leukemia", "leukaemia", "myeloma",
    "glioblastoma", "glioma", "mesothelioma",
)

# Front-strip tokens (stage / grade / biomarker / qualifier / stopwords).
_STRIP_FRONT = {
    "the", "of", "for", "in", "with", "a", "an", "and", "to", "on", "is",
    "patients", "patient", "stage", "grade", "risk", "early", "late",
    "advanced", "localized", "localised", "metastatic", "recurrent",
    "newly", "diagnosed", "positive", "negative", "her2", "her-2", "her",
    "er", "pr", "egfr", "braf", "brca", "brca1", "brca2", "kras", "alk",
    "pdl1", "pd-l1", "triple", "high", "low", "intermediate", "primary",
    "secondary", "node", "lymph", "stage-", "b", "c",
    "i", "ii", "iii", "iv", "v",
}

_ARABIC_TO_ROMAN = {"0": "0", "1": "I", "2": "II", "3": "III", "4": "IV"}
_ROMAN_TO_ARABIC = {"0": "0", "i": "1", "ii": "2", "iii": "3", "iv": "4"}
_STAGE_RE = re.compile(
    r"\bstage\s+(0|1|2|3|4|iv|iii|ii|i)\s*([abc])?\b", re.IGNORECASE
)

# Curated, conservative site → synonym lifts (safe, non-hallucinated).
_SITE_SYNONYMS = {
    "colon": ["colorectal", "colonic"],
    "colorectal": ["colon", "rectal"],
    "rectal": ["rectum", "colorectal"],
    "prostate": ["prostatic"],
    "renal": ["kidney"],
    "kidney": ["renal"],
    "hepatocellular": ["liver", "hepatic"],
    "gastric": ["stomach"],
    "stomach": ["gastric"],
}

# Closed-set prognosis outcome concepts (ranking / should-have).
_OUTCOMES = (
    "prognosis", "overall survival", "cancer-specific survival",
    "disease-free survival", "progression-free survival",
    "metastasis-free survival", "recurrence", "biochemical recurrence",
    "mortality", "survival",
)

# question_subtypes already owned by an earlier K4A deterministic phase —
# never override those.
_PRIOR_PHASE_SUBTYPES = {
    "severity_marker_overview", "rule_out_test_utility",
    "epidemiology_incidence_complication_risk", "drug_safety",
}


def prognosis_outcome_concepts() -> list[str]:
    return list(_OUTCOMES)


def _extract_cancer(question: str) -> Optional[str]:
    q = question or ""
    best: Optional[str] = None
    for m in _CANCER_RE.finditer(q):
        phrase = m.group(1)
        toks = [t for t in re.split(r"[\s]+", phrase.strip()) if t]
        # strip leading stage / biomarker / stopword tokens + pure numbers
        while toks and (
            toks[0].lower() in _STRIP_FRONT or toks[0].isdigit()
        ):
            toks.pop(0)
        if not toks:
            continue
        head = toks[-1].lower().rstrip("s")
        # standalone head (melanoma/lymphoma/…) is a valid condition alone
        is_standalone = any(head.startswith(h[:6]) for h in _STANDALONE_HEADS)
        if len(toks) == 1 and not is_standalone:
            # bare "cancer" with no site word — too generic, skip
            continue
        best = " ".join(toks)  # last match wins (questions end w/ the cancer)
    return best


def _condition_synonyms(cancer: str) -> list[str]:
    syns = [cancer]
    low = cancer.lower()
    toks = low.split()
    head = toks[-1] if toks else ""
    site = " ".join(toks[:-1]) if len(toks) > 1 else ""
    # head variants when head == cancer
    if head == "cancer" and site:
        syns.append(f"{site} carcinoma")
        syns.append(f"{site} neoplasm")
    # curated site synonyms
    if site:
        for s_tok, lifts in _SITE_SYNONYMS.items():
            if s_tok in toks:
                for lift in lifts:
                    syns.append(f"{lift} {head}")
    # dedup, keep order
    seen, out = set(), []
    for s in syns:
        sl = s.strip().lower()
        if sl and sl not in seen:
            seen.add(sl)
            out.append(s.strip())
    return out


def _extract_stage(question: str) -> Optional[dict[str, Any]]:
    m = _STAGE_RE.search(question or "")
    if not m:
        return None
    raw = m.group(1).lower()
    sub = (m.group(2) or "").upper()
    if raw.isdigit():
        arabic, roman = raw, _ARABIC_TO_ROMAN.get(raw, raw)
    else:
        arabic, roman = _ROMAN_TO_ARABIC.get(raw, raw), raw.upper()
    label = f"stage {arabic}{sub}".strip()
    synonyms = [f"stage {arabic}{sub}", f"stage {roman}{sub}"]
    # dedup
    synonyms = list(dict.fromkeys(s for s in synonyms if s))
    return {"label": label, "arabic": arabic, "roman": roman,
            "substage": sub, "synonyms": synonyms}


def extract_components(question: str) -> Optional[dict[str, Any]]:
    """Return closed-set components if this is an obvious prognosis-cancer
    question, else ``None``. Requires BOTH a prognosis signal AND a cancer
    condition; stage is optional (added as ranking context when present)."""
    q = question or ""
    if not _PROGNOSIS_SIGNAL.search(q):
        return None
    cancer = _extract_cancer(q)
    if not cancer:
        return None
    stage = _extract_stage(q)
    return {
        "cancer": cancer,
        "condition_synonyms": _condition_synonyms(cancer),
        "stage": stage,
        "outcomes": list(_OUTCOMES),
    }


def is_prognosis_stage_cancer_question(question: str) -> bool:
    return extract_components(question) is not None


def _is_question_lead_concept(concept_low: str) -> bool:
    """A glued/mis-extracted concept that begins with (or is dominated by) a
    prognosis lead phrase — e.g. ``prognosis of stage 3 prostate cancer``."""
    if not concept_low:
        return True
    return bool(_PROGNOSIS_SIGNAL.match(concept_low)) or bool(
        re.match(r"^(?:what|stage)\b", concept_low)
    )


def apply_rescue(picott_payload: dict, question: str) -> dict[str, Any]:
    """Mutate ``picott_payload`` in place when the question is an obvious
    prognosis/stage cancer question. Returns a closed-set diagnostics dict.

    Does NOT fire if an earlier deterministic K4A phase already routed the
    question (``question_subtype`` in the prior-phase set).
    """
    existing_subtype = (picott_payload.get("question_subtype") or "").strip().lower()
    if existing_subtype in _PRIOR_PHASE_SUBTYPES:
        return {"applied": False, "reason": "prior_phase_owns_route"}

    comp = extract_components(question)
    if not comp:
        return {"applied": False, "reason": "no_match"}

    cancer = comp["cancer"]
    stage = comp["stage"]
    syns = comp["condition_synonyms"]

    # ---- type / subtype / scope ----
    picott_payload["type_of_question"] = "prognosis"
    picott_payload["question_subtype"] = "stage_specific_prognosis"
    picott_payload["question_scope"] = "foreground"
    if not (picott_payload.get("specialty") or "").strip():
        picott_payload["specialty"] = "oncology"
    picott_payload["prognosis_stage_cancer_condition_anchor"] = cancer
    if stage:
        picott_payload["prognosis_stage_cancer_stage"] = stage["label"]

    # ---- must_have: drop the glued question-lead concept, inject clean
    #      cancer CONDITION anchor (required; stage is NOT a hard must-have). ----
    cleaned: list[dict] = []
    for mh in picott_payload.get("must_have_concepts") or []:
        if not isinstance(mh, dict):
            continue
        c = (mh.get("concept") or "").strip()
        if not c:
            continue
        if _is_question_lead_concept(c.lower()):
            continue
        cleaned.append(mh)
    have = {(m.get("concept") or "").strip().lower() for m in cleaned}
    if cancer.lower() not in have:
        cleaned.insert(0, {
            "concept": cancer,
            "role": "condition",
            "synonyms": syns,
            "required_for_high_relevance": True,
        })
    picott_payload["must_have_concepts"] = cleaned

    # ---- should_have: prognosis/survival/recurrence outcomes (soft). ----
    should = list(picott_payload.get("should_have_concepts") or [])
    have_should = {
        (s.get("concept") or "").strip().lower()
        for s in should if isinstance(s, dict)
    }
    for o in comp["outcomes"]:
        if o.lower() not in have_should:
            should.append({"concept": o, "role": "outcome",
                           "synonyms": [], "boost_weight": 2})
            have_should.add(o.lower())
    picott_payload["should_have_concepts"] = should

    # ---- ranking: stage as high-precision context (NOT a cap). ----
    if stage:
        ranking = list(picott_payload.get("ranking_concepts") or [])
        have_rank = {
            (r.get("concept") or "").strip().lower()
            for r in ranking if isinstance(r, dict)
        }
        if stage["label"].lower() not in have_rank:
            ranking.append({"concept": stage["label"], "role": "context",
                            "synonyms": stage["synonyms"], "boost_weight": 5})
        picott_payload["ranking_concepts"] = ranking

    # ---- legacy pico (read by the concept_block_query_builder). ----
    pico = dict(picott_payload.get("pico") or {})
    pop_label = (
        f"patients with {stage['label']} {cancer}".strip() if stage
        else f"patients with {cancer}".strip()
    )
    pico["population"] = list(dict.fromkeys([pop_label, cancer] + syns))
    existing_out = [o for o in (pico.get("outcomes_or_target_diagnosis") or []) if o]
    seen_out = {o.lower() for o in existing_out}
    for o in comp["outcomes"]:
        if o.lower() not in seen_out:
            existing_out.append(o)
            seen_out.add(o.lower())
    pico["outcomes_or_target_diagnosis"] = existing_out
    picott_payload["pico"] = pico

    # ---- bump confidence + clear misleading flags. ----
    try:
        picott_payload["confidence"] = max(
            float(picott_payload.get("confidence") or 0.0), 0.85)
    except (TypeError, ValueError):
        picott_payload["confidence"] = 0.85
    me = dict(picott_payload.get("missing_elements") or {})
    me.pop("outcomes_or_target_diagnosis", None)
    me.pop("population", None)
    picott_payload["missing_elements"] = me

    return {
        "applied": True,
        "reason": "prognosis_stage_cancer_rescue",
        "cancer": cancer,
        "stage": (stage or {}).get("label"),
        "type_of_question": "prognosis",
        "question_subtype": "stage_specific_prognosis",
    }


__all__ = [
    "apply_rescue",
    "extract_components",
    "is_prognosis_stage_cancer_question",
    "prognosis_outcome_concepts",
]
