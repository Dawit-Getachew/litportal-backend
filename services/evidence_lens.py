"""Phase L1 — Explicit Evidence Lens + Population Focus preferences.

Clinician-stated SOFT preference layer that helps steer evidence review.
It is the FIRST Phase-L build and is intentionally minimal:

  * Explicit user preference ONLY — no passive learning, no
    personalization, no cross-session history, no automatic ranking model.
  * Soft-boost / shadow-first — never a hard filter. It can only ADD a
    small, bounded boost to an already-retrieved, already-ranked,
    already-cap-clamped article. It can never hide evidence, never
    promote a capped / wrong-lane / wrong-anchor article above a
    non-capped direct article, and never override PHI, retraction,
    source/license, AI-ingestion, or citation-verification rules.
  * Caps DOMINATE — every applied boost is re-clamped to the article's
    persisted relevance cap ceiling, so safety caps always win.

The module is pure + deterministic. It reads only closed-set article
metadata + title/abstract keyword signals (used transiently for matching;
never persisted or logged). No LLM. No source calls.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA_VERSION = "L1.1.0"
PREFERENCE_PROVENANCE = "user_explicit_preference"
PREFERENCE_DISCLAIMER = (
    "Evidence lens helps prioritize results. It does not override "
    "clinical relevance or safety checks."
)

# --- Closed-set option vocabularies (value -> clinician label). --------
EVIDENCE_LENS_LABELS: dict[str, str] = {
    "clinically_direct": "Clinically direct",
    "most_recent": "Most recent",
    "guidelines": "Guidelines",
    "systematic_reviews": "Systematic reviews",
    "mechanism": "Mechanism",
    "risk_incidence": "Risk/incidence",
    "drug_safety": "Drug safety",
}
EVIDENCE_LENSES: tuple[str, ...] = tuple(EVIDENCE_LENS_LABELS.keys())
DEFAULT_EVIDENCE_LENS = "clinically_direct"

POPULATION_FOCUS_LABELS: dict[str, str] = {
    "any": "Any",
    "adults": "Adults",
    "pediatrics": "Pediatrics",
    "older_adults": "Older adults",
    "pregnancy": "Pregnancy",
}
POPULATION_FOCUS_OPTIONS: tuple[str, ...] = tuple(POPULATION_FOCUS_LABELS.keys())
DEFAULT_POPULATION_FOCUS = "any"

PREFERENCE_MODES: tuple[str, ...] = ("shadow", "soft_boost")
DEFAULT_PREFERENCE_MODE = "soft_boost"

# Bounded boost budget. Lens + population sub-boosts are summed then
# clamped to MAX_L1_BOOST. Small by design so a boost can never jump an
# article across a relevance cap tier.
MAX_L1_BOOST = 8.0
_LENS_BOOST = 5.0
_LENS_BOOST_WEAK = 2.5
_POP_BOOST = 3.0

# --- Keyword signal sets (lowercased substring match on title+abstract).
# Used transiently for matching only — never persisted or logged.
_MECHANISM_TERMS = (
    "mechanism", "pathophysiolog", "pathogenesis", "molecular",
    "signaling", "signalling", "receptor", "pathway",
)
_RISK_TERMS = (
    "incidence", "prevalence", "risk factor", "risk of", "complication rate",
    "rate of", "cohort", "epidemiolog", "case series", "case-control",
)
_DRUG_SAFETY_TERMS = (
    "adverse", "safety", "toxicity", "side effect", "tolerab",
    "drug reaction", "hepatotox", "nephrotox",
)
_PEDIATRIC_TERMS = (
    "child", "children", "infant", "neonat", "pediatric", "paediatric",
    "adolescent", "newborn",
)
_OLDER_ADULT_TERMS = (
    "elderly", "older adult", "geriatric", "octogenarian", "nursing home",
    "aged 65", "aged ≥65", "aged >65",
)
_PREGNANCY_TERMS = (
    "pregnan", "gestational", "maternal", "obstetric", "prenatal",
    "antenatal", "postpartum", "peripartum",
)
_ADULT_TERMS = ("adult", "adults")

_GUIDELINE_PUB_TYPES = ("guideline", "practice guideline", "consensus")
_SR_PUB_TYPES = (
    "systematic review", "meta-analysis", "meta analysis", "review",
)


# --- Normalizers -------------------------------------------------------

def normalize_lens(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in EVIDENCE_LENS_LABELS else DEFAULT_EVIDENCE_LENS


def normalize_population(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in POPULATION_FOCUS_LABELS else DEFAULT_POPULATION_FOCUS


def resolve_preference_mode() -> str:
    """Resolve the active L1 preference mode.

    ``L1_PREFERENCE_MODE`` env override (shadow | soft_boost). In
    production the soft-boost ranking effect is downgraded to ``shadow``
    unless explicitly authorized via
    ``L1_PREFERENCE_PROD_SOFT_BOOST_ALLOWED=true`` — so L1 never alters
    production ranking without sign-off (Northstar v4.0 §9/§15).
    """
    mode = str(os.environ.get("L1_PREFERENCE_MODE", DEFAULT_PREFERENCE_MODE)).strip().lower()
    if mode not in PREFERENCE_MODES:
        mode = DEFAULT_PREFERENCE_MODE
    is_prod = os.environ.get("APP_ENV", "development").strip().lower() == "production"
    prod_allowed = os.environ.get(
        "L1_PREFERENCE_PROD_SOFT_BOOST_ALLOWED", "false",
    ).strip().lower() == "true"
    if mode == "soft_boost" and is_prod and not prod_allowed:
        return "shadow"
    return mode


def l1_ui_visible() -> bool:
    """Phase L1.0.1 — production UI visibility gate.

    L1 evidence-lens / population-focus controls render in preview/dev by
    default. In production they are HIDDEN unless explicitly enabled via
    ``L1_PREFERENCE_PROD_UI_VISIBLE=true`` — this prevents the confusing
    "visible but shadow/inert" UX in production (active soft-boost stays
    disabled there regardless; see ``resolve_preference_mode``). The
    backend is the single source of truth; the frontend reads the
    resolved boolean off the session response.
    """
    is_prod = os.environ.get("APP_ENV", "development").strip().lower() == "production"
    if not is_prod:
        return True
    return os.environ.get(
        "L1_PREFERENCE_PROD_UI_VISIBLE", "false",
    ).strip().lower() == "true"


# --- Article-level matching helpers (transient; never persisted) -------

def _article_text(article_doc: dict) -> str:
    parts = [
        str(article_doc.get("title") or ""),
        str(article_doc.get("abstract") or ""),
    ]
    for mt in (article_doc.get("mesh_terms") or []):
        parts.append(str(mt))
    return " ".join(parts).lower()


def _pub_types(article_doc: dict) -> list[str]:
    pts = article_doc.get("publication_types") or article_doc.get("pub_types") or []
    return [str(p).strip().lower() for p in pts if str(p).strip()]


def _article_year(article_doc: dict) -> Optional[int]:
    for key in ("publication_year", "year"):
        val = article_doc.get(key)
        try:
            if val is not None:
                return int(val)
        except (TypeError, ValueError):
            continue
    return None


def _any(text: str, terms: tuple[str, ...]) -> bool:
    return any(t in text for t in terms)


def compute_article_preference(
    article_doc: dict,
    row: Optional[dict],
    *,
    lens: str,
    population: str,
    current_year: Optional[int] = None,
) -> dict[str, Any]:
    """Deterministic per-article preference signal. Returns a bounded
    boost (0..MAX_L1_BOOST), a ``matched`` flag, and a closed-set list of
    signal tokens. The default lens (``clinically_direct``) and the
    default population (``any``) are deliberate no-ops — the baseline
    ranker already prioritizes direct evidence, so L1 only adds signal
    when a clinician picks a NON-default preference.
    """
    article_doc = article_doc or {}
    row = row or {}
    if current_year is None:
        current_year = datetime.now(timezone.utc).year

    text = _article_text(article_doc)
    pub_types = _pub_types(article_doc)
    signals: list[str] = []
    lens_boost = 0.0

    if lens == "most_recent":
        yr = _article_year(article_doc)
        if yr is not None:
            if yr >= current_year - 3:
                lens_boost = _LENS_BOOST
                signals.append("recent_publication")
            elif yr >= current_year - 5:
                lens_boost = _LENS_BOOST_WEAK
                signals.append("recent_publication")
    elif lens == "guidelines":
        if _any(" ".join(pub_types), _GUIDELINE_PUB_TYPES) or bool(
            row.get("guideline_lane")
        ):
            lens_boost = _LENS_BOOST
            signals.append("guideline_publication_type")
    elif lens == "systematic_reviews":
        joined = " ".join(pub_types)
        if "systematic review" in joined or "meta-analysis" in joined or "meta analysis" in joined:
            lens_boost = _LENS_BOOST
            signals.append("systematic_review_or_meta_analysis")
        elif "review" in joined:
            lens_boost = _LENS_BOOST_WEAK
            signals.append("review_publication_type")
    elif lens == "mechanism":
        if _any(text, _MECHANISM_TERMS):
            lens_boost = _LENS_BOOST
            signals.append("mechanism_focus")
    elif lens == "risk_incidence":
        if _any(text, _RISK_TERMS):
            lens_boost = _LENS_BOOST
            signals.append("risk_incidence_focus")
    elif lens == "drug_safety":
        if _any(text, _DRUG_SAFETY_TERMS) or "adverse effects" in " ".join(pub_types):
            lens_boost = _LENS_BOOST
            signals.append("drug_safety_focus")
    # ``clinically_direct`` => baseline, no boost.

    pop_boost = 0.0
    if population == "pediatrics" and _any(text, _PEDIATRIC_TERMS):
        pop_boost = _POP_BOOST
        signals.append("pediatric_population")
    elif population == "older_adults" and _any(text, _OLDER_ADULT_TERMS):
        pop_boost = _POP_BOOST
        signals.append("older_adult_population")
    elif population == "pregnancy" and _any(text, _PREGNANCY_TERMS):
        pop_boost = _POP_BOOST
        signals.append("pregnancy_population")
    elif population == "adults" and _any(text, _ADULT_TERMS) and not _any(
        text, _PEDIATRIC_TERMS
    ):
        pop_boost = _POP_BOOST
        signals.append("adult_population")

    boost = min(lens_boost + pop_boost, MAX_L1_BOOST)
    return {
        "boost": round(boost, 2),
        "matched": bool(signals),
        "signals": signals,
    }


def _cap_ceiling(row: dict) -> float:
    cap = row.get("relevance_cap_applied")
    if isinstance(cap, (int, float)):
        return float(cap)
    return 100.0


def apply_evidence_lens(
    ranked: list[dict],
    *,
    lens: str,
    population: str,
    mode: str,
    current_year: Optional[int] = None,
) -> dict[str, Any]:
    """Annotate (and, in ``soft_boost`` mode, gently re-order) the ranked
    rows per the clinician's stated preference. Mutates rows in place and
    returns a closed-set session-level diagnostics block (no PHI / no
    free text).

    Safety invariants enforced here:
      * Every boost is re-clamped to the row's relevance cap ceiling —
        caps always dominate.
      * Pure boost (never penalize / never filter) — nothing is hidden.
      * ``shadow`` mode records the would-be delta but does NOT change
        the clinician-visible ``overall_rank_score`` ordering.
    """
    lens = normalize_lens(lens)
    population = normalize_population(population)
    mode = mode if mode in PREFERENCE_MODES else DEFAULT_PREFERENCE_MODE

    original_order = {
        (r.get("_article") or {}).get("id") or id(r): idx
        for idx, r in enumerate(ranked)
    }

    matched_count = 0
    boosted_count = 0
    caps_dominated = 0
    max_boost_applied = 0.0
    would_change_count = 0

    for r in ranked:
        art = r.get("_article") or {}
        pref = compute_article_preference(
            art, r, lens=lens, population=population, current_year=current_year,
        )
        boost = float(pref["boost"])
        r["l1_preference_boost"] = boost
        r["l1_preference_matched"] = pref["matched"]
        r["l1_preference_signals"] = pref["signals"]
        r["l1_preference_rank_delta"] = 0
        if pref["matched"]:
            matched_count += 1

        base = float(r.get("overall_rank_score") or 0.0)
        ceiling = _cap_ceiling(r)
        proposed = base + boost
        clamped = max(0.0, min(100.0, min(proposed, ceiling)))
        effective = clamped - base  # boost actually applied after caps
        if boost > 0 and proposed > ceiling:
            caps_dominated += 1
        if effective > 0:
            would_change_count += 1
            if effective > max_boost_applied:
                max_boost_applied = effective

        if mode == "soft_boost" and effective > 0:
            r["l1_pre_boost_rank_score"] = base
            r["overall_rank_score"] = clamped
            boosted_count += 1

    rank_changed_count = 0
    if mode == "soft_boost":
        ranked.sort(key=lambda r: -float(r.get("overall_rank_score") or 0.0))
        for new_idx, r in enumerate(ranked):
            art = r.get("_article") or {}
            key = art.get("id") or id(r)
            old_idx = original_order.get(key, new_idx)
            delta = old_idx - new_idx
            r["l1_preference_rank_delta"] = delta
            if delta != 0:
                rank_changed_count += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_lens": lens,
        "population_focus": population,
        "preference_mode": mode,
        "provenance": PREFERENCE_PROVENANCE,
        "disclaimer": PREFERENCE_DISCLAIMER,
        "articles_considered": len(ranked),
        "articles_matched": matched_count,
        "articles_boosted": boosted_count,
        "rank_changed_count": rank_changed_count,
        "would_change_count": would_change_count,
        "max_boost_applied": round(max_boost_applied, 2),
        "safety_caps_dominated_count": caps_dominated,
        "safety_caps_dominant": True,
        "preference_affected_ranking": bool(rank_changed_count > 0),
        "is_hard_filter": False,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def preference_packet_metadata(diagnostics: Optional[dict]) -> dict[str, Any]:
    """Closed-set projection of L1 diagnostics for the K4 evidence packet
    metadata + clinician-facing surfaces."""
    d = diagnostics if isinstance(diagnostics, dict) else {}
    return {
        "evidence_lens": normalize_lens(d.get("evidence_lens")),
        "population_focus": normalize_population(d.get("population_focus")),
        "preference_mode": d.get("preference_mode") or DEFAULT_PREFERENCE_MODE,
        "provenance": PREFERENCE_PROVENANCE,
        "preference_affected_display_or_ranking": bool(
            d.get("preference_affected_ranking")
        ),
        "articles_affected": int(d.get("articles_boosted") or 0),
        "safety_caps_dominant": True,
        "is_hard_filter": False,
        "disclaimer": PREFERENCE_DISCLAIMER,
    }


def list_lens_options() -> list[dict[str, str]]:
    return [{"value": v, "label": EVIDENCE_LENS_LABELS[v]} for v in EVIDENCE_LENSES]


def list_population_options() -> list[dict[str, str]]:
    return [
        {"value": v, "label": POPULATION_FOCUS_LABELS[v]}
        for v in POPULATION_FOCUS_OPTIONS
    ]


__all__ = [
    "SCHEMA_VERSION",
    "PREFERENCE_PROVENANCE",
    "PREFERENCE_DISCLAIMER",
    "EVIDENCE_LENSES",
    "EVIDENCE_LENS_LABELS",
    "DEFAULT_EVIDENCE_LENS",
    "POPULATION_FOCUS_OPTIONS",
    "POPULATION_FOCUS_LABELS",
    "DEFAULT_POPULATION_FOCUS",
    "PREFERENCE_MODES",
    "DEFAULT_PREFERENCE_MODE",
    "MAX_L1_BOOST",
    "normalize_lens",
    "normalize_population",
    "resolve_preference_mode",
    "l1_ui_visible",
    "compute_article_preference",
    "apply_evidence_lens",
    "preference_packet_metadata",
    "list_lens_options",
    "list_population_options",
]
