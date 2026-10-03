"""Automated PRESS-informed check of a query portfolio (W2.1).

The six domains follow the PRESS 2015 guideline (McGowan et al.). These are
automated checks only — they are labelled as such and are never presented as
a human PRESS peer review.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from clinical_engine.retrieval.portfolio import CQ_CATEGORY, FAMILY_LABELS, _narrow_phrase_concept
from clinical_engine.search_terms.mesh import _UK_US  # noqa: PLC2701 — shared spelling pairs
from services import engine_concept_plan as ecp

LABEL = "Automated PRESS-informed check"
SPARSE = 30
DISCLAIMER = ("Automated checks informed by the PRESS 2015 guideline. This is not a human PRESS peer review "
              "by an information specialist.")
_PUBMED_TAGS = re.compile(
    r"^(tiab|ti|ab|mh|mesh terms|mesh|mesh:noexp|mh:noexp|majr|pt|publication type|sb|ta|journal|"
    r"title/abstract|title|pdat|dp|edat|crdt|uid|sh|mesh subheading|subheading|subheading:noexp|"
    r"all fields|lang|language|tw|text word|filter)$", re.I)
_EPMC_FIELDS = {"TITLE_ABS", "TITLE", "ABSTRACT", "PUB_TYPE", "SRC", "FIRST_PDATE", "FIRST_IDATE", "MESH",
                "JOURNAL", "OPEN_ACCESS", "KW", "AUTH", "PUB_YEAR", "LANG", "HAS_FT", "IN_EPMC"}


def _check(status: str, text: str) -> dict[str, str]:
    return {"status": status, "text": text}


def _domain(key: str, title: str, checks: list[dict[str, str]]) -> dict[str, Any]:
    order = {"fail": 3, "warn": 2, "pass": 1, "info": 0}
    worst = max((c["status"] for c in checks), key=lambda s: order[s], default="info")
    return {"key": key, "title": title, "status": "pass" if worst == "info" else worst, "checks": checks}


def syntax_problems(query: str, source: str) -> list[str]:
    """Structural problems a source would reject or misread (empty list = fine)."""
    problems: list[str] = []
    depth, in_quote = 0, False
    for ch in query:
        if ch == '"':
            in_quote = not in_quote
        elif not in_quote and ch == "(":
            depth += 1
        elif not in_quote and ch == ")":
            depth -= 1
            if depth < 0:
                problems.append("a closing parenthesis has no opening one")
                break
    if in_quote:
        problems.append("unbalanced quotation marks")
    if depth > 0:
        problems.append("unclosed parenthesis")
    bare = re.sub(r'"[^"]*"', '""', query)
    if re.search(r"\(\s*\)", bare):
        problems.append("an empty group '()'")
    if re.search(r"\b(AND|OR|NOT)\s*\)|\(\s*(AND|OR)\b|^\s*(AND|OR)\b|\b(AND|OR|NOT)\s*$", bare):
        problems.append("an operator with nothing on one side")
    if source == "pubmed":
        for tag in re.findall(r"\[([^\]]+)\]", bare):
            if not _PUBMED_TAGS.match(tag.strip()):
                problems.append(f"unknown PubMed field tag [{tag}]")
    else:
        for field_name in re.findall(r"\b([A-Z_]{2,}):", bare):
            if field_name not in _EPMC_FIELDS:
                problems.append(f"unknown Europe PMC field {field_name}:")
    if len(query) > ecp.MAX_QUERY_LEN:
        problems.append(f"longer than {ecp.MAX_QUERY_LEN} characters")
    return problems


def _spelling_variants(word: str) -> set[str]:
    """UK⇄US spelling forms of a wording (lower-case)."""
    w = word.lower()
    out = {w}
    for uk, us in _UK_US:
        if uk in w:
            out.add(w.replace(uk, us))
        if us in w and uk not in w:
            out.add(w.replace(us, uk))
    return out


def check(plan: dict[str, Any], portfolio: dict[str, Any], *,
          pubmed_main: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """PRESS-informed report. *pubmed_main*: PubMed's reply for the main search
    ({count, translation, warnings, errors}) when available."""
    specs = portfolio["specs"]
    concepts = plan["concepts"]
    required = [c for c in concepts if c.get("required")]
    ranking = [c for c in concepts if not c.get("required")]
    families = {s.family for s in specs}
    qtype = plan.get("question_type") or "other"
    use_syn = bool(plan.get("use_synonyms", True))

    # 1. Translation of the research question
    t: list[dict[str, str]] = []
    t.append(_check("pass" if required else "fail",
                    "Must appear in every search: " + (", ".join(f"'{c['term']}'" for c in required) or "nothing")))
    if ranking:
        t.append(_check("info", "Used to rank, not to restrict: " + ", ".join(f"'{c['term']}'" for c in ranking)))
    if len(required) == 1:
        t.append(_check("warn", "Only one concept must appear — the search may be very broad."))
    if len(required) > 4:
        t.append(_check("warn", f"{len(required)} concepts must all appear — relevant papers that omit one will be missed."))
    cat = portfolio.get("clinical_queries_category")
    if cat:
        t.append(_check("pass", f"Question type '{qtype.replace('_', ' ')}' → PubMed's {cat.replace('_', ' ')} "
                                "filters (narrow and broad) run alongside the unfiltered main search."))
    elif "narrow" in families:
        t.append(_check("info", f"No validated PubMed filter for '{qtype.replace('_', ' ')}' questions; a study-design "
                                "list narrows one search, the main search stays unfiltered."))
    else:
        t.append(_check("info", "No study-design filter for this question type; the searches are unfiltered."))
    if "original_wording" in families:
        t.append(_check("pass", "Your own wording also runs, as a check on the structured search: "
                                + " ".join(portfolio.get("original_wording") or [])))
    else:
        reason = next((s["reason"] for s in portfolio["skipped"] if s["family"] == "original_wording"), "")
        t.append(_check("info", f"Your own wording is not searched separately. {reason}".strip()))

    # 2. Boolean and proximity operators
    b = [
        _check("pass", "Concepts are joined with AND; the wordings of one concept with OR."),
        _check("pass", "NOT is used only for approved limits (animal-only studies, preprints)."),
        _check("pass", "No search turns the condition AND treatment into OR — the wider alternative keeps "
                       "the core concepts AND-ed."),
        _check("info", "Proximity operators are not used; multi-word wordings are searched as exact phrases."),
    ]

    # 3. Subject headings
    sh: list[dict[str, str]] = []
    for c in required:
        if c.get("mesh"):
            sh.append(_check("pass", f"'{c['term']}' → MeSH '{c['mesh']}' (with its narrower headings) OR words."))
        elif c.get("broader_mesh"):
            sh.append(_check("info", f"'{c['term']}': only a broader MeSH heading ('{c['broader_mesh']}') exists; it is "
                                     "left off (only the wider alternative may use it). Searched as words."))
        else:
            sh.append(_check("warn", f"'{c['term']}': no exact MeSH heading — searched as title/abstract words only."))
    sh.append(_check("info", "Europe PMC is searched with words only (its MeSH search is unreliable inside OR groups)."))

    # 4. Text words
    tw: list[dict[str, str]] = []
    if not use_syn:
        tw.append(_check("info", "Alternative wordings are off (your choice): each concept is searched by its own words."))
    for c in required:
        words = ecp.concept_words(c, plan)
        if len(words) == 1 and not c.get("mesh"):
            tw.append(_check("warn", f"'{c['term']}' rests on one wording and no MeSH heading — papers using another "
                                     "term for it will be missed."))
        else:
            tw.append(_check("pass", f"'{c['term']}': {len(words)} wording(s)" + (" + MeSH." if c.get("mesh") else ".")))
        off = [t["text"] for t in c.get("off_equivalents") or [] if t.get("text")]
        if off and not c.get("mesh"):
            tw.append(_check("warn", f"'{c['term']}' has no MeSH heading and {len(off)} same-meaning wording(s) are "
                                     "switched off (" + ", ".join(f"'{t}'" for t in off[:4])
                                     + ("…" if len(off) > 4 else "") + ") — papers that use only those words will "
                                     "be missed. Consider switching them on."))
        head = _narrow_phrase_concept(c, plan)
        if head:
            tw.append(_check("warn", f"'{c['term']}' is searched only as exact phrases and has no MeSH heading — papers "
                                     f"usually say '{head}'. Consider adding '{head}' as a wording."))
        if use_syn:
            have = {w.lower() for w in words}
            missing = sorted({v for w in words for v in _spelling_variants(w)} - have)
            if missing:
                tw.append(_check("warn", f"'{c['term']}': consider the spelling variant(s) "
                                         + ", ".join(f"'{m}'" for m in missing[:3]) + "."))

    # 5. Spelling and syntax
    sp: list[dict[str, str]] = []
    bad = [(s, p) for s in specs for p in syntax_problems(s.query, s.source)]
    if bad:
        for s, p in bad[:6]:
            sp.append(_check("fail", f"{FAMILY_LABELS.get(s.family, s.family)} ({s.source}): {p}."))
    else:
        sp.append(_check("pass", f"All {len(specs)} searches are well formed (brackets, quotes, operators, field tags)."))
    if pubmed_main:
        for w in pubmed_main.get("warnings") or []:
            sp.append(_check("warn", f"PubMed: {w} — check the spelling."))
        for e in pubmed_main.get("errors") or []:
            sp.append(_check("fail", f"PubMed: {e}."))
        if pubmed_main.get("count") is not None and pubmed_main["count"] < SPARSE:
            t.append(_check("warn", f"The main search finds only {pubmed_main['count']} PubMed record(s) — it may be "
                                    "too narrow." + (" The wider alternative will run." if portfolio.get("alternative")
                                                     else "")))
        if pubmed_main.get("translation"):
            sp.append(_check("info", "PubMed read the main search as: " + pubmed_main["translation"][:600]))

    # 6. Limits and filters
    lf: list[dict[str, str]] = []
    years = (plan.get("filters") or {}).get("publication_years")
    lf.append(_check("info", f"Limited to the last {years} years (your choice)." if years else "No year limit."))
    if (plan.get("filters") or {}).get("exclude_animal_only", True):
        lf.append(_check("pass", "Animal-only studies are excluded (PubMed: animals NOT humans; Europe PMC: animal "
                                 "words in the title)."))
    else:
        lf.append(_check("info", "Animal-only studies are included (your choice)."))
    lf.append(_check("pass", "Preprints are excluded."))
    lf.append(_check("pass", "No language limit."))
    lf.append(_check("pass", "Study-type filters narrow only some searches; the main search is not design-filtered."))
    if CQ_CATEGORY.get(qtype) and not cat:
        lf.append(_check("info", "PubMed's question-type filters are not configured; the design list is used instead."))

    domains = [
        _domain("translation", "Translation of the question", t),
        _domain("boolean", "Boolean and proximity operators", b),
        _domain("subject_headings", "Subject headings (MeSH)", sh),
        _domain("text_words", "Text words", tw),
        _domain("spelling_syntax", "Spelling and syntax", sp),
        _domain("limits", "Limits and filters", lf),
    ]
    summary = {k: sum(1 for d in domains if d["status"] == k) for k in ("pass", "warn", "fail")}
    return {"label": LABEL, "disclaimer": DISCLAIMER, "domains": domains, "summary": summary}
