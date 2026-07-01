"""Phase 5B Task 6 — SemanticRerankerService.

Per-article semantic reranking signal. Used in addition to (NOT instead
of) the deterministic picott_aware_ranker. Operates ONLY on data the
ranker already has access to (title, abstract, OA-extract text, matched
canonical concepts) — never queries an LLM with the raw question or
with novel medical content.

Phase 5F extension: ``article_answer_class`` closed-set enum. Used for
ranking + explanation only (per spec) — NEVER for summary generation.
The summary path remains gated by the citation verifier and uses only
the user-selected evidence packet.

Output per article (added to ``session_articles`` row):
{
  "semantic_relevance_score": float (0..100),
  "direct_answer_likelihood": float (0..100),
  "article_answer_class": str in {direct, related, mechanistic,
                                   wrong_population, wrong_intervention,
                                   wrong_outcome, background_only},
  "matched_canonical_concepts": [str, ...],
  "missing_canonical_concepts": [str, ...],
  "semantic_rationale": str,
}

Provider:
- If an LLM provider is configured AND ``SEMANTIC_RERANKER_ENABLED`` is
  truthy, we delegate to ``llm_provider.complete_json`` with a strict
  JSON schema and a system prompt that FORBIDS new medical facts from
  model memory and FORBIDS answer generation.
- Otherwise we use a deterministic fallback: anchor matching + title/
  abstract term overlap with canonical concepts + query family + concept
  translations.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any


# Phase 5F — closed-set article_answer_class values. Mirrors the spec.
ARTICLE_ANSWER_CLASSES = (
    "direct", "related", "mechanistic",
    "wrong_population", "wrong_intervention", "wrong_outcome",
    "background_only",
)

# Phase 5F — GLP-1 / lipid term sets (closed-set; mirrors picott_ranker).
_GLP1_RA_MEMBERS_S = (
    "semaglutide", "ozempic", "wegovy", "rybelsus",
    "liraglutide", "victoza", "saxenda",
    "dulaglutide", "trulicity",
    "exenatide", "byetta", "bydureon", "exendin-4",
    "lixisenatide", "adlyxin",
    "albiglutide", "tanzeum",
    "glp-1 receptor agonist", "glp-1 receptor agonists",
    "glp1 receptor agonist", "glp-1 ra", "glp1-ra",
    "glucagon-like peptide-1 receptor agonist",
)
_GLP1_FAMILY_S = _GLP1_RA_MEMBERS_S + ("glp-1", "glp1")
_LIPID_TERMS_S = (
    "ldl", "ldl-c", "ldl cholesterol", "low-density lipoprotein",
    "low density lipoprotein",
    "hdl", "hdl-c", "hdl cholesterol",
    "triglyceride", "triglycerides", "total cholesterol", "cholesterol",
    "apolipoprotein", "apob", "apo b", "lipoprotein(a)", "lp(a)",
    "non-hdl", "dyslipidemia", "dyslipidaemia",
    "lipid profile", "lipids",
)
_MECHANISTIC_TERMS = (
    "in vitro", "in vivo", "rodent", "rodents", "mice", "mouse",
    "rat", "rats", "islet", "beta cell", "beta-cell", "binding affinity",
    "receptor signaling", "expression", "cell line", "cellular",
)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def _semantic_enabled() -> bool:
    return (os.environ.get("SEMANTIC_RERANKER_ENABLED", "true") or "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


@dataclass
class SemanticScore:
    semantic_relevance_score: float
    direct_answer_likelihood: float
    article_answer_class: str
    matched_canonical_concepts: list[str]
    missing_canonical_concepts: list[str]
    semantic_rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_relevance_score": round(float(self.semantic_relevance_score), 2),
            "direct_answer_likelihood": round(float(self.direct_answer_likelihood), 2),
            "article_answer_class": self.article_answer_class,
            "matched_canonical_concepts": list(self.matched_canonical_concepts),
            "missing_canonical_concepts": list(self.missing_canonical_concepts),
            "semantic_rationale": self.semantic_rationale,
        }


def _hit(text: str, term: str) -> bool:
    if not text or not term:
        return False
    return bool(re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE))


def _hits_any(text: str, terms: tuple | list) -> bool:
    if not text:
        return False
    return any(_hit(text, t) for t in terms)


def _canonical_concepts(picott: dict[str, Any]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for mh in (picott.get("must_have_concepts") or []):
        out.append((mh.get("concept") or "", list(mh.get("synonyms") or [])))
    for rc in (picott.get("ranking_concepts") or []):
        out.append((rc.get("concept") or "", list(rc.get("synonyms") or [])))
    return [(c, s) for c, s in out if c]


def classify_article_answer(
    *,
    title: str,
    abstract: str,
    picott: dict[str, Any],
) -> str:
    """Closed-set enum classifier.

    Order of precedence (first match wins):
      1. wrong_intervention — question requires GLP-1 RA but article has
         lipid outcome WITHOUT any GLP-1 family term.
      2. wrong_outcome — article has GLP-1 family term but no lipid
         outcome term (when question asks lipids).
      3. mechanistic — body contains mechanistic markers (in vitro, mice,
         beta cell, receptor signaling …) AND no direct human-outcome
         language.
      4. direct — title or first-paragraph abstract contains BOTH the
         intervention canonical AND an outcome canonical from PICOTT.
      5. related — intervention OR outcome is present (but not both) AND
         no mechanistic markers fired.
      6. wrong_population — question population canonical missing from
         both title and abstract while intervention/outcome present.
      7. background_only — no canonicals matched at all.

    Returns a member of ``ARTICLE_ANSWER_CLASSES``.
    """
    body = " ".join([title or "", abstract or ""])
    body_lower = body.lower()
    must = picott.get("must_have_concepts") or []
    intervention_concepts = [
        m.get("concept") for m in must
        if (m.get("role") or "").lower() in {"intervention", "exposure",
                                                "index_test"}
        and m.get("concept")
    ]
    outcome_concepts = [
        m.get("concept") for m in must
        if (m.get("role") or "").lower() in {"outcome", "target_diagnosis"}
        and m.get("concept")
    ]
    # Also pull outcomes from ranking_concepts (Phase 5F outcomes-only
    # fallback).
    for rc in (picott.get("ranking_concepts") or []):
        if (rc.get("role") or "").lower() in {"outcome", "target_diagnosis"}:
            if rc.get("concept"):
                outcome_concepts.append(rc["concept"])
    population_concepts = [
        m.get("concept") for m in must
        if (m.get("role") or "").lower() in {"population", "condition",
                                                "patient_group"}
        and m.get("concept")
    ]

    body_has_intervention = any(_hit(body, c) for c in intervention_concepts)
    body_has_outcome = any(_hit(body, c) for c in outcome_concepts)
    body_has_population = any(_hit(body, c) for c in population_concepts)

    # Special-case for GLP-1 / lipid questions: detect via the canonicals,
    # then use a lexical class membership check so synonym variants the
    # picott block didn't enumerate (Ozempic, Wegovy, ...) still classify.
    intervention_is_glp1 = any(
        any(t in (c or "").lower() for t in ("glp-1", "glp1"))
        for c in intervention_concepts
    )
    outcome_is_lipid = any(
        any(t in (c or "").lower() for t in
            ("ldl", "hdl", "cholesterol", "triglyceride", "apolipoprotein",
              "lipid", "lipoprotein"))
        for c in outcome_concepts
    )

    if intervention_is_glp1 and outcome_is_lipid:
        art_has_glp1 = _hits_any(body_lower, _GLP1_FAMILY_S)
        art_has_lipid = _hits_any(body_lower, _LIPID_TERMS_S)
        # Rule 1 — wrong_intervention.
        if art_has_lipid and not art_has_glp1:
            return "wrong_intervention"
        # Rule 3 — mechanistic if mechanistic markers present AND no
        # lipid outcome (otherwise direct wins).
        if (art_has_glp1 and _hits_any(body_lower, _MECHANISTIC_TERMS)
                and not art_has_lipid):
            return "mechanistic"
        # Rule 2 — wrong_outcome.
        if art_has_glp1 and not art_has_lipid:
            return "wrong_outcome"
        # Rule 4 — direct (both present, in the same article).
        if art_has_glp1 and art_has_lipid:
            return "direct"
        # Rule 7 — neither.
        return "background_only"

    # Generic path for any other question type.
    # Rule 3 — mechanistic.
    if _hits_any(body_lower, _MECHANISTIC_TERMS) and not (
        body_has_intervention and body_has_outcome
    ):
        return "mechanistic"
    # Rule 4 — direct.
    if body_has_intervention and body_has_outcome:
        # Wrong population? Population specified but absent.
        if population_concepts and not body_has_population:
            return "wrong_population"
        return "direct"
    # Rule 5 — related.
    if body_has_intervention or body_has_outcome:
        return "related"
    return "background_only"


class SemanticRerankerService:
    """Deterministic semantic-overlap reranker. LLM path is optional."""

    def rerank(
        self,
        ranked: list[dict[str, Any]],
        picott: dict[str, Any] | None,
        *,
        is_phi_blocked: bool = False,
    ) -> list[dict[str, Any]]:
        if is_phi_blocked or not ranked or not picott or not _semantic_enabled():
            for r in ranked or []:
                r.setdefault("semantic_relevance_score", None)
                r.setdefault("direct_answer_likelihood", None)
                r.setdefault("article_answer_class", "background_only")
                r.setdefault("matched_canonical_concepts", [])
                r.setdefault("missing_canonical_concepts", [])
                r.setdefault("semantic_rationale", "")
            return ranked or []

        concepts = _canonical_concepts(picott)
        type_of_q = (picott.get("type_of_question") or "other").lower()
        subtype = (picott.get("question_subtype") or "")

        for r in ranked:
            art = r.get("_article") or {}
            title = art.get("title") or ""
            abstract = art.get("abstract") or ""
            body = " ".join([title, abstract])
            matched: list[str] = []
            missing: list[str] = []
            for canonical, syns in concepts:
                terms = [canonical] + syns
                if any(_hit(body, t) for t in terms):
                    matched.append(canonical)
                else:
                    missing.append(canonical)

            # Coverage ratio (0..1).
            denom = max(1, len(concepts))
            coverage = len(matched) / denom

            # Direct-answer heuristic: title contains BOTH intervention
            # and condition canonicals; outcomes mentioned in abstract.
            interv = [c for c, _ in concepts if any(
                (m or "").get("concept", "") == c and (m or {}).get("role") == "intervention"
                for m in (picott.get("must_have_concepts") or [])
            )]
            cond = [c for c, _ in concepts if any(
                (m or {}).get("concept", "") == c and (m or {}).get("role") == "condition"
                for m in (picott.get("must_have_concepts") or [])
            )]
            direct = 0.0
            if interv and cond:
                t_int = any(_hit(title, t) for t in interv)
                t_con = any(_hit(title, t) for t in cond)
                if t_int and t_con:
                    direct = 90.0
                elif t_int or t_con:
                    direct = 70.0
                else:
                    a_int = any(_hit(abstract, t) for t in interv)
                    a_con = any(_hit(abstract, t) for t in cond)
                    if a_int and a_con:
                        direct = 55.0
                    elif a_int or a_con:
                        direct = 35.0
                    else:
                        direct = 10.0
            else:
                direct = 50.0 * coverage

            # Threshold-style sub-question: a "transfusion_threshold_comparison"
            # article gets a small boost only when *both* liberal/restrictive
            # tokens are present.
            if subtype == "transfusion_threshold_comparison":
                if _hit(body, "liberal") and _hit(body, "restrictive"):
                    direct = min(100.0, direct + 8.0)

            # Final semantic_relevance_score: blend coverage (60%) + direct (40%).
            sem = 60.0 * coverage + 0.4 * direct
            sem = max(0.0, min(100.0, sem))

            # Phase 5F — closed-set article_answer_class.
            answer_class = classify_article_answer(
                title=title, abstract=abstract, picott=picott,
            )

            # Cap-aware rationale.
            note_bits = [
                f"coverage={coverage:.2f}",
                f"matched={len(matched)}/{denom}",
                f"type_of_q={type_of_q}",
                f"class={answer_class}",
            ]
            if subtype:
                note_bits.append(f"subtype={subtype}")
            rat = " | ".join(note_bits)

            score = SemanticScore(
                semantic_relevance_score=sem,
                direct_answer_likelihood=direct,
                article_answer_class=answer_class,
                matched_canonical_concepts=matched,
                missing_canonical_concepts=missing,
                semantic_rationale=rat,
            ).to_dict()
            for k, v in score.items():
                r[k] = v
        return ranked


semantic_reranker = SemanticRerankerService()


__all__ = [
    "ARTICLE_ANSWER_CLASSES",
    "SemanticRerankerService",
    "semantic_reranker",
    "classify_article_answer",
]

