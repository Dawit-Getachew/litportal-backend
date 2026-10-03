"""Guideline lineage, review cutoffs and currency (W2.3).

From the admitted guidelines and reviews:

* editions — one guideline co-published in several journals (and its executive
  summary) is ONE work; the extra records are listed under it;
* issuer — matched against the (proposed) issuer registry;
* family — editions of the same issuer on the same topic (adult and paediatric
  guidelines are different families); without a known issuer, only editions of
  one guideline series (near-identical titles) — one body never supersedes another;
* focused updates — linked to the edition they update when the record says so
  ("update to the 2017 guideline…"); a focused update changes some
  recommendations, it does not replace the whole guideline;
* status within a family — the newest full edition is current, older full
  editions are superseded by it; one society never supersedes another;
* currency — soft five-year window measured back from the newest comparable
  applicable guideline (not from today); older guidelines stay when justified
  (parent of a current focused update; the only guideline in the results that
  addresses the asked treatment specifically); the window is not applied to
  reviews;
* search cutoffs — read from the abstract ("searched … to March 2023") and
  verified by quoting the sentence; they — not publication dates — anchor the
  update search for newer evidence.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from clinical_engine.retrieval import registry

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
_MON3 = {k[:3]: v for k, v in _MONTHS.items()}
_META = re.compile(
    r"\b(?:adherence|compliance|implementation|implementing|knowledge|comparison|impact|assessing|assessment of|"
    r"evaluating|evaluation of|what is new|what's new|dissenting|retired|beyond|highlights|implications|survey|"
    r"audit|translating|prescribing|variability|questionnaire|resident|nursing|lessons|critique|challenge|"
    r"controversy|debate|perspective|commentary|editorial|pilot|study|trial|cohort|randomi[sz]ed|"
    r"with and without|how to|review of)\b|^(?:re|point|counterpoint)\s*:", re.I)
_ISSUANCE = re.compile(
    r"\b(?:guidelines?|recommendations|consensus statement|position statement|practice parameters?|"
    r"clinical practice guidelines?)\s+(?:for|on|of|from|in)\b|^\W*(?:\d{4}\s+)?(?:international |clinical practice )?"
    r"guidelines?\b|\binternational guidelines\b|\bfocused update\b|\bguideline update\b|"
    r"\bclinical practice guidelines?\b|\bconsensus statement\b|\bguidance (?:document|on|for)\b", re.I)
_SUMMARY = re.compile(r"^\W*executive summary\b", re.I)
_FOCUSED = re.compile(r"\bfocused update\b", re.I)
_PARENT = re.compile(r"\b(?:update|updates|updated|updating|revision)\s+(?:to|of)\s+(?:the\s+)?(\d{4})\b", re.I)
_PEDIATRIC = re.compile(r"\b(?:child|children|paediatric|pediatric|neonat\w*|infant\w*|newborn\w*)\b", re.I)
_STOP = frozenset("""the a an of for and in on to with from by at international clinical practice guideline guidelines
recommendation recommendations update updated focused executive summary part i ii iii management adult adults
consensus statement statements society""".split())
_WORD = re.compile(r"[a-z][a-z\-]+")
_DATE = (r"(?:(\d{1,2})(?:st|nd|rd|th)?\s+)?(january|february|march|april|may|june|july|august|september|october|"
         r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?,?\s+(?:(\d{1,2}),?\s+)?(\d{4})"
         r"|(\d{4})-(\d{2})-(\d{2})|(?<!\d)(\d{4})(?!\d)")
_CUTOFF = re.compile(
    r"(?:search\w*|database\w*|medline|pubmed|embase|cochrane|literature|records?)[^.;]{0,220}?"
    r"(?:to|until|through|thru|up to|till|and|before)\s+(" + _DATE + r")", re.I)
_INCEPTION = re.compile(r"(?:inception|onset|beginning|earliest)[^.;]{0,40}?(?:to|until|through|up to)\s+(" + _DATE + r")",
                        re.I)
_OTHER_WORK = re.compile(r"\b(?:guidelines?|statement|review)\s+(?:19|20)\d{2}\b|\b(?:19|20)\d{2}\s+(?:guidelines?|edition)\b",
                         re.I)


def guideline_role(article: dict[str, Any]) -> Optional[str]:
    """'pubtype' (indexed as a guideline), 'title' (issuance wording), or None.
    Papers ABOUT guidelines (adherence, what is new, debates) are not guidelines."""
    title = (article.get("title") or "").strip()
    types = {t.lower() for t in article.get("publication_types_json") or []}
    if _META.search(title):
        return None
    if types & {"practice guideline", "guideline"}:
        return "pubtype"
    if _ISSUANCE.search(title):
        return "title"
    return None


def _topic(title: str) -> set[str]:
    low = re.sub(r"\(.*?\)", " ", (title or "").lower())
    low = re.sub(r"\b(?:19|20)\d{2}\b", " ", _SUMMARY.sub("", low))
    return {w for w in _WORD.findall(low) if w not in _STOP and len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def parse_date(text: str) -> Optional[tuple[date, str]]:
    m = re.search(_DATE, text or "", re.I)
    if not m:
        return None
    day, mon, day2, year, iy, im, idd, bare = m.groups()
    try:
        if iy:
            return date(int(iy), int(im), int(idd)), "day"
        if mon:
            month = _MONTHS.get(mon.lower()) or _MON3.get(mon.lower()[:3])
            d = day or day2
            return (date(int(year), month, int(d)), "day") if d else (date(int(year), month, 1), "month")
        if bare and 1950 <= int(bare) <= date.today().year:
            return date(int(bare), 1, 1), "year"
    except ValueError:
        return None
    return None


def search_cutoff(abstract: Optional[str]) -> Optional[dict[str, Any]]:
    """The review's OWN last search date, verified by quoting the sentence it is in.
    A sentence that names other works' searches ("…Guideline 2018 (search until
    2018) with the SSC Guideline 2021…") or holds several candidate dates
    (other than "from inception to …") is ambiguous and skipped."""
    if not abstract:
        return None
    best: Optional[dict[str, Any]] = None
    for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z])", abstract):
        inception = _INCEPTION.search(sentence)
        matches = [inception] if inception else list(_CUTOFF.finditer(sentence))
        if not matches or (not inception and (len(matches) > 1 or _OTHER_WORK.search(sentence))):
            continue
        parsed = parse_date(matches[0].group(1))
        if parsed is None or parsed[0] > date.today():
            continue
        cand = {"date": parsed[0].isoformat(), "precision": parsed[1], "quote": sentence.strip()[:300]}
        if best is None or cand["date"] > best["date"]:
            best = cand
    return best


def _year(p: dict[str, Any]) -> int:
    return int((p.get("article") or {}).get("publication_year") or 0)


def _issuer_text(p: dict[str, Any]) -> str:
    a = p.get("article") or {}
    groups = " ".join(str(x.get("fullName") or "") for x in a.get("authors_json") or []
                      if re.search(r"committee|group|campaign|society|consortium|panel|task force",
                                   str(x.get("fullName") or ""), re.I))
    return f"{a.get('title') or ''} {(a.get('abstract') or '')[:900]} {groups}"


def build(pubs: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach ``lineage`` to admitted guidelines (and ``cutoff`` to reviews), in place."""
    admitted = [p for p in pubs if (p.get("admission") or {}).get("state")
                in ("direct", "broad_answering", "supportive", "indirect", "uncertain")]
    guides = [p for p in admitted if (p.get("admission") or {}).get("role") == "guideline"]

    # 1. Editions (co-publications + executive summaries).
    editions: list[list[dict[str, Any]]] = []
    for p in sorted(guides, key=lambda x: (_year(x), x["id"])):
        t = _topic(p["article"].get("title") or "")
        ped = bool(_PEDIATRIC.search(p["article"].get("title") or ""))
        for ed in editions:
            lead = ed[0]
            if (abs(_year(lead) - _year(p)) <= 1 and ped == bool(_PEDIATRIC.search(lead["article"].get("title") or ""))
                    and _jaccard(t, _topic(lead["article"].get("title") or "")) >= 0.7
                    and bool(_FOCUSED.search(lead["article"]["title"] or "")) == bool(_FOCUSED.search(p["article"]["title"] or ""))):
                ed.append(p)
                break
        else:
            editions.append([p])

    rank = {"direct": 0, "broad_answering": 1, "supportive": 2, "indirect": 3, "uncertain": 4}
    edition_docs = []
    for i, ed in enumerate(editions):
        ed.sort(key=lambda x: (bool(_SUMMARY.search(x["article"].get("title") or "")),
                               rank.get(x["admission"]["state"], 9), x["article"].get("pmid") or ""))
        lead = ed[0]
        text = " ".join(_issuer_text(x) for x in ed)
        issuers = registry.match_issuers(text)
        title = lead["article"].get("title") or ""
        parent = _PARENT.search(f"{title} {(lead['article'].get('abstract') or '')[:600]}")
        edition_docs.append({
            "id": f"ed{i + 1}", "members": ed, "lead": lead, "year": _year(lead),
            "topic": _topic(title), "pediatric": bool(_PEDIATRIC.search(title)),
            "focused_update": bool(_FOCUSED.search(title)),
            "parent_year": int(parent.group(1)) if parent else None,
            "parent_quote": parent.group(0) if parent else None,
            "issuers": issuers,
        })

    # 2. Families: same issuer (or both unmatched), same age group, similar topic.
    families: list[list[dict[str, Any]]] = []
    for ed in sorted(edition_docs, key=lambda e: e["year"]):
        ids = {i.id for i in ed["issuers"]}
        placed = False
        if ed["focused_update"] and ed["parent_year"]:
            for fam in families:
                if any(e["year"] == ed["parent_year"] and ({i.id for i in e["issuers"]} & ids)
                       and e["pediatric"] == ed["pediatric"] for e in fam):
                    fam.append(ed)
                    placed = True
                    break
        if not placed:
            for fam in families:
                f_ids = {i.id for f in fam for i in f["issuers"]}
                # Same issuer and a similar topic; with no known issuer on either side only
                # a near-identical title (the same guideline series) — never two bodies.
                need = 0.45 if ids & f_ids else 0.7 if not ids and not f_ids else None
                if need is not None and fam[0]["pediatric"] == ed["pediatric"] and \
                        max(_jaccard(ed["topic"], f["topic"]) for f in fam) >= need:
                    fam.append(ed)
                    placed = True
                    break
        if not placed:
            families.append([ed])

    # 3. Status within each family; 4. currency window across families.
    newest_applicable = max((e["year"] for e in edition_docs
                             if e["lead"]["admission"]["state"] in ("direct", "broad_answering")), default=None)

    def applicable(e: dict[str, Any]) -> bool:
        return e["lead"]["admission"]["state"] in ("direct", "broad_answering")

    def specific(e: dict[str, Any]) -> bool:  # its title names the asked treatment / test / exposure
        return bool(((e["lead"].get("admission") or {}).get("signals") or {}).get("names_intervention"))
    family_out = []
    for fi, fam in enumerate(families):
        full = sorted([e for e in fam if not e["focused_update"]], key=lambda e: e["year"])
        updates = sorted([e for e in fam if e["focused_update"]], key=lambda e: e["year"])
        current_full = full[-1] if full else None
        label = (current_full or fam[-1])["lead"]["article"].get("title") or ""
        issuers = sorted({i.id for e in fam for i in e["issuers"]})
        for e in fam:
            status, note = "current", None
            if e["focused_update"]:
                status = "focused_update"
                parent = next((f for f in full if f["year"] == e["parent_year"]), None)
                note = (f"Focused update of the {e['parent_year']} edition (“{e['parent_quote']}”) — it "
                        "updates some recommendations; it does not replace the whole guideline."
                        if parent else "Focused update — the edition it updates is not in these results.")
            elif current_full is not None and e is not current_full:
                status = "superseded"
                note = f"Superseded by the {current_full['year']} edition of the same guideline."
            currency = "current"
            if status == "superseded":
                currency = "superseded"
            elif newest_applicable and e["year"] < newest_applicable - 5:
                updated_by = [u for u in updates if u["parent_year"] == e["year"]]
                most_specific = specific(e) and not any(
                    x["year"] > e["year"] and applicable(x) and specific(x) for x in edition_docs)
                if updated_by:
                    currency = "current"
                    note = f"Kept: updated by the {updated_by[-1]['year']} focused update."
                elif most_specific:
                    currency = "current"
                    note = ("Kept despite its age: no newer guideline in these results addresses the asked "
                            "treatment specifically.")
                else:
                    currency = "older_than_window"
                    note = (f"More than five years older than the newest comparable guideline "
                            f"({newest_applicable}).")
            issuer_objs = e["issuers"]
            feed = registry.FEED_LABELS[issuer_objs[0].official_feed] if issuer_objs else \
                "Issuer not in the registry — found through PubMed / Europe PMC indexing"
            for k, m in enumerate(e["members"]):
                m["lineage"] = {
                    "edition_id": e["id"], "family_id": f"fam{fi + 1}", "family_label": label[:200],
                    "edition_year": e["year"], "lead": k == 0,
                    "kind": ("summary" if _SUMMARY.search(m["article"].get("title") or "")
                             else "co_publication" if k else ("focused_update" if e["focused_update"] else "full")),
                    "status": status, "currency": currency, "note": note,
                    "issuers": [{"id": i.id, "name": i.name} for i in issuer_objs], "source_label": feed,
                    "co_publications": [{"pmid": x["article"].get("pmid"), "journal": x["article"].get("journal"),
                                         "publication_id": x["id"]} for x in e["members"] if x is not m],
                    "cutoff": search_cutoff(m["article"].get("abstract")),
                }
        family_out.append({
            "id": f"fam{fi + 1}", "label": label[:200], "issuers": issuers,
            "editions": [{"year": e["year"], "publication_id": e["lead"]["id"],
                          "status": e["lead"]["lineage"]["status"], "currency": e["lead"]["lineage"]["currency"],
                          "focused_update": e["focused_update"], "records": len(e["members"])}
                         for e in sorted(fam, key=lambda x: x["year"])],
        })

    # Reviews: cutoffs (the five-year window is not applied to reviews).
    for p in admitted:
        if p["admission"].get("role") in ("systematic_review", "meta_analysis") and "lineage" not in p:
            p["lineage"] = {"kind": "review", "cutoff": search_cutoff(p["article"].get("abstract"))}
    return {"families": family_out, "newest_comparable_guideline_year": newest_applicable,
            "registry": {"version": registry.REGISTRY_VERSION, "status": registry.REGISTRY_STATUS}}


def update_anchor(pubs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The latest verified search cutoff among admitted guidelines and reviews that
    answer the question (direct or broad) — the anchor for the update search."""
    best: Optional[dict[str, Any]] = None
    for p in pubs:
        lin = p.get("lineage") or {}
        cut = lin.get("cutoff")
        if not cut or (p.get("admission") or {}).get("state") not in ("direct", "broad_answering"):
            continue
        if lin.get("currency") in ("superseded", "older_than_window"):
            continue
        if best is None or cut["date"] > best["date"]:
            best = {**cut, "publication_id": p["id"], "title": (p.get("article") or {}).get("title"),
                    "pmid": (p.get("article") or {}).get("pmid"), "role": p["admission"].get("role")}
    return best
