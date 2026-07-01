"""Phase 6G.3 — overview directness classifier.

Deterministic, closed-set classifier that labels an article's
directness w.r.t. a primary disease anchor. Used by:

  * ``services.picott_ranker`` to apply directness boosts + aspect-
    specific / multi-disease caps when the question subtype is
    ``disease_overview`` or ``definition``.
  * The UI to render the "directness badge" on each article card in
    background overview searches.

NO LLM. NO network. Pure string-pattern lookups against closed-set
lexicons declared in this module.
"""
from __future__ import annotations

import re
from typing import Iterable, Literal

Directness = Literal[
    "general_overview",
    "guideline_overview",
    "disease_seminar",
    "broad_clinical_review",
    "aspect_specific_review",
    "treatment_specific_review",
    "pathogenesis_specific_review",
    "environmental_or_lifestyle_specific",
    "multi_disease_review",
    "wrong_disease",
    "unrelated",
]


# ---------------------------------------------------------------------
# Closed-set lexicons
# ---------------------------------------------------------------------


_GENERAL_OVERVIEW_TITLE_TOKENS: tuple[str, ...] = (
    "overview",
    "review",
    "clinical review",
    "seminar",
    "clinical features",
    "burden",
)

_GUIDELINE_OVERVIEW_TOKENS: tuple[str, ...] = (
    "guideline",
    "practice guideline",
    "consensus",
    "recommendation",
    "consensus statement",
)

_BROAD_DOMAINS: tuple[str, ...] = (
    "epidemiology",
    "pathogenesis",
    "clinical manifestations",
    "diagnosis",
    "classification",
    "treatment overview",
    "management",
    "prognosis",
    "complications",
    "etiology",
)

_TREATMENT_SPECIFIC_TOKENS: tuple[str, ...] = (
    "biologic",
    "biologics",
    "phototherapy",
    "isotretinoin",
    "methotrexate",
    "cyclosporine",
    "topical retinoid",
    "topical retinoids",
    "spironolactone",
    "antibiotic",
    "antibiotics",
    "hormonal therapy",
    "laser therapy",
)

_PATHOGENESIS_SPECIFIC_TOKENS: tuple[str, ...] = (
    "pathogenesis only",
    "genetics only",
    "biomarker",
    "biomarkers",
    "il-17",
    "il-23",
    "tnf",
    "cytokine",
    "cytokines",
    "molecular mechanism",
    "molecular mechanisms",
)

_ASPECT_SPECIFIC_TOKENS: tuple[str, ...] = (
    "quality of life",
    "cost",
    "cost-effectiveness",
    "stem cells",
    "stem cell",
    "exosome",
    "exosomes",
    "nutraceutical",
    "nutraceuticals",
    "probiotic",
    "probiotics",
    "single drug",
    "single agent",
    "single arm",
    "case report",
    "case series",
    "case study",
    "pilot study",
)

_ENVIRONMENTAL_OR_LIFESTYLE_TOKENS: tuple[str, ...] = (
    "environmental pollution",
    "air pollution",
    "climate change",
    "mediterranean diet",
    "vegetarian diet",
    "dietary intervention",
    "diet and",
    "lifestyle intervention",
    "microbiome",
    "gut microbiome",
    "skin microbiome",
    "ultraviolet exposure",
    "smoking and",
    "alcohol and",
)

# Multi-disease detector — closed-set list of dermatologic + other
# conditions that often co-appear in "review of skin diseases" or
# "review of inflammatory disorders" articles.
_DERMA_OTHER_DISEASES: tuple[str, ...] = (
    "atopic dermatitis", "eczema",
    "hidradenitis suppurativa", "acne inversa",
    "rosacea", "chloracne",
    "vitiligo", "alopecia", "alopecia areata",
    "lupus", "systemic lupus", "cutaneous lupus",
    "scleroderma", "lichen planus",
    "seborrheic dermatitis", "contact dermatitis",
    "urticaria",
    "psoriatic arthritis",
)

# Broad "inflammatory skin diseases" / "skin diseases" multi-disease
# umbrella patterns.
_MULTI_DISEASE_UMBRELLAS: tuple[str, ...] = (
    "inflammatory skin diseases",
    "skin diseases",
    "dermatologic diseases",
    "chronic skin conditions",
    "common skin disorders",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _norm(text: str | None) -> str:
    return (text or "").lower()


def _any_present(haystack: str, needles: Iterable[str]) -> bool:
    return any(n in haystack for n in needles)


def _count_present(haystack: str, needles: Iterable[str]) -> int:
    return sum(1 for n in needles if n in haystack)


def _anchor_in(text: str, anchor_canonical: str,
               anchor_synonyms: Iterable[str]) -> bool:
    if not text or not anchor_canonical:
        return False
    if anchor_canonical.lower() in text:
        return True
    for s in anchor_synonyms or []:
        if s and s.lower() in text:
            return True
    return False


_TITLE_DISEASE_ONLY_RE = re.compile(
    r"^\s*[\"']?(?P<d>[a-z\s\-]{3,50}?)[\"']?\s*[:.\-—]?\s*"
    r"(?:overview|review|clinical review|seminar)?\s*[\"']?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------


def classify_overview_directness(
    article: dict,
    anchor_canonical: str,
    anchor_synonyms: Iterable[str] | None = None,
) -> Directness:
    """Return the directness label for ``article`` w.r.t. the primary
    disease anchor.

    ``article`` must expose at least ``title`` and ``abstract`` string
    fields. Missing fields are treated as empty.

    Args:
        article: the article metadata dict
        anchor_canonical: canonical disease anchor name (e.g. "acne
            vulgaris", "psoriasis")
        anchor_synonyms: optional list of pre-validated synonyms
    """
    title = _norm(article.get("title"))
    abstract = _norm(article.get("abstract"))
    pub_types = [
        _norm(p) for p in (article.get("publication_types_json") or [])
        if isinstance(p, str)
    ]
    full = f"{title} {abstract}"
    anchor_canonical = anchor_canonical or ""
    syns = list(anchor_synonyms or [])

    # 0. Multi-disease umbrella detection comes BEFORE the anchor-in-
    #    text check. An article that says "inflammatory skin diseases"
    #    or "skin diseases broadly" implicitly INCLUDES the anchor
    #    (psoriasis, acne, etc.) and is correctly labeled
    #    ``multi_disease_review`` even when the bare anchor token is
    #    absent from title / abstract.
    umbrella_present = _any_present(full, _MULTI_DISEASE_UMBRELLAS)
    if umbrella_present:
        return "multi_disease_review"

    # 1. Wrong disease / unrelated
    if not _anchor_in(full, anchor_canonical, syns):
        # Title is empty AND abstract is empty → unrelated.
        if not title and not abstract:
            return "unrelated"
        return "wrong_disease"

    # 2. Multi-disease review check (≥3 distinct dermatologic /
    #    other diseases mentioned).
    other_present = _count_present(full, _DERMA_OTHER_DISEASES)
    if other_present >= 2:
        # ≥2 OTHER diseases (the anchor itself being one) →
        # multi_disease_review.
        return "multi_disease_review"

    # 3. Environmental / lifestyle aspect (highest specificity caps).
    if _any_present(full, _ENVIRONMENTAL_OR_LIFESTYLE_TOKENS):
        return "environmental_or_lifestyle_specific"

    # 4. Pathogenesis-only / molecular-mechanism aspect.
    if _any_present(full, _PATHOGENESIS_SPECIFIC_TOKENS):
        return "pathogenesis_specific_review"

    # 5. Treatment-specific aspect.
    if _any_present(full, _TREATMENT_SPECIFIC_TOKENS):
        return "treatment_specific_review"

    # 6. Aspect-specific (cost / QoL / stem cells / case reports etc.).
    if _any_present(full, _ASPECT_SPECIFIC_TOKENS):
        return "aspect_specific_review"

    # 7. Guideline-style overview.
    if (
        _any_present(title, _GUIDELINE_OVERVIEW_TOKENS)
        or any("guideline" in p or "consensus" in p for p in pub_types)
    ):
        return "guideline_overview"

    # 8. General-overview: title is essentially the disease name,
    #    optionally followed by an overview/review/seminar keyword.
    title_disease_only = (
        anchor_canonical.lower() in title
        and len(title) <= len(anchor_canonical) + 40
    )
    title_has_overview_token = _any_present(
        title, _GENERAL_OVERVIEW_TITLE_TOKENS,
    )
    if title_disease_only and title_has_overview_token:
        return "general_overview"
    if title_disease_only:
        # "Acne vulgaris" / "Psoriasis" — bare title → disease seminar
        # textbook-style article.
        return "disease_seminar"

    # 9. Broad clinical review: abstract covers ≥3 of the broad domains.
    broad_hits = _count_present(abstract, _BROAD_DOMAINS)
    if broad_hits >= 3:
        return "broad_clinical_review"

    # 10. Fall-through: still mentions the anchor; treat as
    #     aspect_specific_review (no boost, no cap unless invoked by
    #     the ranker for the disease_overview subtype).
    return "aspect_specific_review"
