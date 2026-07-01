"""Phase K5.2C-0 — read-only live PubMed smoke (no-key path, rate-limited).

Validates discovery + labeling against REAL PubMed data:
  * builds the exact ``cochrane_review`` query the compiler emits,
  * runs esearch/esummary against NCBI E-utilities (no key; polite UA header),
  * confirms returned records are detected as Cochrane reviews,
  * confirms a non-applicable lane (mechanism/background) emits NO Cochrane query.

NEVER prints/logs any API key. Read-only. No writes. No scraping.
"""
import sys
import time
import httpx

sys.path.insert(0, "/app/backend")

from services.query_compiler import boolean_query_compiler  # noqa: E402
from services.cochrane_lane import article_cochrane_provenance  # noqa: E402
from services.cochrane_lane import cochrane_query_applicable  # noqa: E402

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": "ClinEvidenceSearch/preview (mailto:dev@example.org)"}


def _therapy_picott():
    return {
        "type_of_question": "therapy",
        "question_subtype": "",
        "normalized_question": "anticoagulation for atrial fibrillation stroke prevention",
        "must_have_concepts": [
            {"concept": "atrial fibrillation", "role": "condition",
             "synonyms": ["af"], "required_for_high_relevance": True},
            {"concept": "anticoagulation", "role": "intervention",
             "synonyms": ["anticoagulant"], "required_for_high_relevance": True},
        ],
        "should_have_concepts": [], "ranking_concepts": [],
        "negative_or_penalty_concepts": [], "pico": {},
    }


def main():
    p = _therapy_picott()
    fam = [q for q in boolean_query_compiler.compile(p, sources=["pubmed"])
           if q.query_family == "cochrane_review"]
    assert fam, "compiler did not emit a cochrane_review family for therapy"
    q = fam[0]
    print("THERAPY cochrane query:", q.query_string)

    with httpx.Client(timeout=30, headers=HEADERS) as cli:
        r = cli.get(f"{EUTILS}/esearch.fcgi", params={
            "db": "pubmed", "term": q.query_string, "retmax": "8",
            "retmode": "json",
        })
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        print("esearch hits:", len(ids), "pmids:", ids)
        assert ids, "no Cochrane reviews returned for a common therapy query"
        time.sleep(0.4)  # no-key pacing (≤3 req/s)
        s = cli.get(f"{EUTILS}/esummary.fcgi", params={
            "db": "pubmed", "id": ",".join(ids), "retmode": "json",
        })
        s.raise_for_status()
        result = s.json().get("result", {})
        detected = 0
        for pmid in ids:
            rec = result.get(pmid) or {}
            journal = rec.get("fulljournalname") or rec.get("source") or ""
            prov = article_cochrane_provenance({"journal": journal})
            mark = "COCHRANE" if prov else "other"
            print(f"  {pmid}: {journal[:50]!r} -> {mark}")
            if prov:
                detected += 1
        print("detected_as_cochrane:", detected, "/", len(ids))
        assert detected >= max(1, len(ids) // 2), \
            "journal-scoped query should return mostly Cochrane reviews"

    # Control: mechanism/background lane must NOT actively emit a Cochrane query.
    pm = _therapy_picott(); pm["type_of_question"] = "background"
    assert not cochrane_query_applicable(pm)
    bg_fam = [x for x in boolean_query_compiler.compile(pm, sources=["pubmed"])
              if x.query_family == "cochrane_review"]
    assert not bg_fam, "background lane must not emit a Cochrane query family"
    print("CONTROL background: no cochrane_review family emitted (correct)")
    print("LIVE SMOKE PASS")


if __name__ == "__main__":
    main()
