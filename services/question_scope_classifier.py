"""Phase 5A — Question scope classifier (background vs foreground vs hybrid).

A background question is conceptual ("what is septic shock?") and routes to
guidelines/major reviews/background sections of textbooks. A foreground
question is a specific clinical decision ("What is the evidence for X for
Y?") and runs the full PICOTT workflow.

Deterministic-first. Optional LLM enrichment can be wired later but is NOT
needed to satisfy Phase 5A acceptance criteria.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

QuestionScope = Literal["background", "foreground", "hybrid", "unclear"]


# Strong background patterns — interrogative-style "what is X" / "define X".
_BACKGROUND_PATTERNS = [
    re.compile(r"\bwhat is\b\s+\w", re.IGNORECASE),
    re.compile(r"\bwhat are\b\s+\w", re.IGNORECASE),
    re.compile(r"\bdefine\b\s+\w", re.IGNORECASE),
    re.compile(r"\bdefinition of\b", re.IGNORECASE),
    re.compile(r"\bhow does\b\s+\w+\s+(?:work|cause|develop)", re.IGNORECASE),
    re.compile(r"\bmechanism of\b", re.IGNORECASE),
    re.compile(r"\bpathophysiology of\b", re.IGNORECASE),
    re.compile(r"\boverview of\b", re.IGNORECASE),
    re.compile(r"\b(?:tell me about|explain)\b", re.IGNORECASE),
    re.compile(r"\bsignificance of\b", re.IGNORECASE),
    re.compile(r"\bclassification of\b", re.IGNORECASE),
]

# Strong foreground patterns — anything that pairs an intervention with a
# decision frame ("vs", "compared with", "for [condition]", "in [population]").
_FOREGROUND_PATTERNS = [
    re.compile(r"\b(?:vs\.?|versus|compared (?:with|to))\b", re.IGNORECASE),
    re.compile(r"\b(?:treatment|therapy|management) of\b", re.IGNORECASE),
    # Phase 6F — "treatment for X" / "therapy for X" / "management for X"
    # patterns. Required so Q1 (psoriasis treatment-sequencing) flips
    # from background → hybrid → foreground.
    re.compile(r"\b(?:treatment|therapy|management)\s+for\b", re.IGNORECASE),
    re.compile(r"\b(?:first[-\s]line|second[-\s]line|third[-\s]line)\s+"
                r"(?:treatment|therapy|line)\b", re.IGNORECASE),
    re.compile(r"\b(?:treatment|management|therapy)\s+algorithm\b",
                re.IGNORECASE),
    re.compile(r"\btreatment\s+options\b", re.IGNORECASE),
    re.compile(r"\b(?:diagnostic accuracy|sensitivity|specificity)\b", re.IGNORECASE),
    re.compile(r"\b(?:risk of|harm|safety) of\b", re.IGNORECASE),
    # Phase 6F — drug-safety phrases. Required so Q2 (Seroquel side
    # effects) flips from background → hybrid → foreground.
    re.compile(r"\b(?:side|adverse)\s+(?:effects?|reactions?|events?)\s+of\b",
                re.IGNORECASE),
    re.compile(r"\b(?:toxicity|tolerability|drug\s+safety|safety\s+profile)\s+of\b",
                re.IGNORECASE),
    re.compile(r"\b(?:complications?|long[-\s]term\s+effects?)\s+of\b",
                re.IGNORECASE),
    re.compile(r"\b(?:prognosis|mortality|survival|recurrence)\b", re.IGNORECASE),
    re.compile(r"\b(?:screening|screen for)\b", re.IGNORECASE),
    re.compile(r"\b(?:prevent|prophylax)", re.IGNORECASE),
    re.compile(r"\b(?:dose|dosing|protocol|regimen|infusion|bolus)\b", re.IGNORECASE),
    re.compile(r"\bin\s+(?:patients|adults|children|women|men)\b", re.IGNORECASE),
    re.compile(r"\bevidence for\b", re.IGNORECASE),
    re.compile(r"\bguideline(?:s)?\b", re.IGNORECASE),
    re.compile(r"\b(?:efficacy|effectiveness|effective)\b", re.IGNORECASE),
]


@dataclass
class ScopeResult:
    scope: QuestionScope
    rationale: str
    background_signals: list[str]
    foreground_signals: list[str]


class QuestionScopeClassifierService:
    """Deterministic background-vs-foreground classifier.

    Returns ``background`` if and only if background signals fire and NO
    foreground signals fire. Returns ``hybrid`` when both fire. Returns
    ``foreground`` when only foreground patterns fire. ``unclear`` is
    reserved for very short or signal-free questions — the rest of the
    pipeline treats unclear as foreground for now.
    """

    def classify(self, question: str) -> ScopeResult:
        q = (question or "").strip()
        if not q:
            return ScopeResult(scope="unclear", rationale="empty question",
                                background_signals=[], foreground_signals=[])
        bg_hits: list[str] = []
        fg_hits: list[str] = []
        for patt in _BACKGROUND_PATTERNS:
            m = patt.search(q)
            if m:
                bg_hits.append(m.group(0))
        for patt in _FOREGROUND_PATTERNS:
            m = patt.search(q)
            if m:
                fg_hits.append(m.group(0))

        if bg_hits and not fg_hits:
            return ScopeResult(
                scope="background",
                rationale=(
                    "Background question — definitional/conceptual frame "
                    f"({', '.join(bg_hits[:2])}). Routing to guideline + "
                    "major-review sources."
                ),
                background_signals=bg_hits,
                foreground_signals=[],
            )
        if fg_hits and not bg_hits:
            return ScopeResult(
                scope="foreground",
                rationale=(
                    "Foreground question — specific clinical decision "
                    f"({', '.join(fg_hits[:2])}). Running full PICOTT workflow."
                ),
                background_signals=[],
                foreground_signals=fg_hits,
            )
        if fg_hits and bg_hits:
            return ScopeResult(
                scope="hybrid",
                rationale=(
                    "Hybrid question — both background and decision-frame "
                    "signals present. Running PICOTT plus a background query "
                    "family for guideline/major-review context."
                ),
                background_signals=bg_hits,
                foreground_signals=fg_hits,
            )
        # No signals — treat as unclear; downstream treats this as foreground.
        return ScopeResult(
            scope="unclear",
            rationale="No strong scope signals; defaulting to foreground workflow.",
            background_signals=[],
            foreground_signals=[],
        )


question_scope_classifier = QuestionScopeClassifierService()
