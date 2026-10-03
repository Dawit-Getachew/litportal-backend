"""Week-2 engine retrieval: portfolio + PRESS (W2.1), retrieval + union (W2.2),
lineage + update (W2.3), admission (W2.4), access (W2.5), and a full engine run
through the API. All sources and the model are faked — no network."""
from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

from clinical_engine import store
from clinical_engine.config import settings
from clinical_engine.retrieval import access, admission, http, lineage, portfolio, press, projection, union
from clinical_engine.retrieval import pubmed as pubmed_src
from clinical_engine.retrieval.run import run_portfolio
from services import engine_concept_plan as ecp
from tests.engine_sources_fixture import FakeModel, FakeSources, Rec, dims
from tests.test_clinical_engine_search_terms import _signup, engine_api  # noqa: F401 — shared fixture

PLAN = ecp.validate_plan({
    "plan_version": ecp.PLAN_VERSION, "question_type": "treatment", "use_synonyms": True,
    "filters": {"publication_years": None, "exclude_animal_only": True},
    "concepts": [
        {"id": "c1", "role": "condition", "term": "septic shock", "required": True, "mesh": "Shock, Septic",
         "synonyms": []},
        {"id": "c2", "role": "intervention", "term": "corticosteroids", "required": True,
         "mesh": "Adrenal Cortex Hormones", "synonyms": ["hydrocortisone"]},
        {"id": "c3", "role": "outcome", "term": "mortality", "required": False, "mesh": None, "synonyms": []},
    ],
})


@pytest.fixture(autouse=True)
def _fast_sources(monkeypatch):
    monkeypatch.setattr(http, "_INTERVALS", {k: 0.0 for k in http._INTERVALS})
    yield
    http.TEST_TRANSPORT = None


# ── W2.1 portfolio + PRESS ──────────────────────────────────────────────────


def test_portfolio_families_and_no_cross_concept_or():
    pf = portfolio.build(PLAN, "In adults with septic shock, do corticosteroids reduce mortality?")
    fams = {(s.family, s.source) for s in pf["specs"]}
    for fam in ("main", "original_wording", "narrow", "broad", "guidelines", "condition_guidelines",
                "systematic_reviews", "cochrane", "recent"):
        assert (fam, "pubmed") in fams, fam
    assert ("main", "europepmc") in fams and ("cochrane", "europepmc") not in fams
    main = next(s for s in pf["specs"] if s.family == "main" and s.source == "pubmed")
    # Condition AND treatment: each required concept is its own AND-ed block.
    assert '"Shock, Septic"[MeSH Terms]' in main.query and ") AND (" in main.query
    assert '"mortality"' not in main.query  # ranking-only concepts never restrict
    narrow = next(s for s in pf["specs"] if s.family == "narrow" and s.source == "pubmed")
    assert "randomized controlled trial[Publication Type]" in narrow.query  # NLM therapy narrow filter
    assert all(not press.syntax_problems(s.query, s.source) for s in pf["specs"])
    assert pf["alternative"] is None  # nothing justifies a relaxation here


def test_original_wording_skips_negation_and_alternatives_are_bounded():
    words, why = portfolio.original_wording("In adults without diabetes, does metformin reduce mortality?")
    assert words == [] and "negation" in why
    phrase_plan = ecp.validate_plan({**PLAN, "concepts": [
        PLAN["concepts"][0],
        {"id": "c2", "role": "intervention", "term": "early corticosteroid treatment", "required": True,
         "mesh": None, "synonyms": ["early corticosteroid therapy"]},
    ]})
    pf = portfolio.build(phrase_plan, "Does early corticosteroid treatment help in septic shock?")
    alt = [s for s in pf["specs"] if s.family == "alternative"]
    assert alt and all(s.condition == "if_sparse" for s in alt)
    pm = next(s for s in alt if s.source == "pubmed")
    assert '"corticosteroid"[tiab]' in pm.query and '"corticosteroids"[tiab]' in pm.query
    assert '"Shock, Septic"[MeSH Terms]' in pm.query and ") AND (" in pm.query  # still AND-ed
    assert pf["alternative"]["edits"][0] == {"op": "add_term", "concept_id": "c2", "text": "corticosteroid"}
    report = press.check(phrase_plan, pf, pubmed_main={"count": 4, "translation": "x", "warnings": [], "errors": []})
    by_key = {d["key"]: d for d in report["domains"]}
    assert by_key["text_words"]["status"] == "warn" and by_key["translation"]["status"] == "warn"
    assert "not a human PRESS peer review" in report["disclaimer"]
    outcome_plan = ecp.validate_plan({**PLAN, "concepts": [*PLAN["concepts"][:2], {**PLAN["concepts"][2], "required": True}]})
    relaxed = portfolio.build(outcome_plan, "q")["alternative"]
    assert relaxed["edits"] == [{"op": "set_required", "concept_id": "c3", "enabled": False}]
    years = portfolio.build(ecp.validate_plan({**PLAN, "filters": {"publication_years": 5}}), "q")["alternative"]
    assert years["edits"] == [{"op": "set_publication_years", "years": 0}]


def test_press_syntax_problems():
    assert press.syntax_problems('("a"[tiab] AND "b"[tiab]', "pubmed") == ["unclosed parenthesis"]
    assert "unbalanced quotation marks" in press.syntax_problems('("a[tiab])', "pubmed")
    assert press.syntax_problems('"a"[foo]', "pubmed") == ["unknown PubMed field tag [foo]"]
    assert press.syntax_problems("(a AND )", "pubmed") == ["an operator with nothing on one side"]
    assert press.syntax_problems('BOGUS:"x"', "europepmc") == ["unknown Europe PMC field BOGUS:"]


def test_update_specs_anchor_on_cutoff_not_publication_date():
    specs = portfolio.update_specs(PLAN, "2024-11-30")
    pm = next(s for s in specs if s.source == "pubmed")
    ep = next(s for s in specs if s.source == "europepmc")
    assert '("2024/11/30"[crdt] : "3000"[crdt])' in pm.query
    assert "FIRST_IDATE:[2024-11-30 TO" in ep.query


# ── W2.2 retrieval + union ──────────────────────────────────────────────────

RECORDS = [
    Rec("100", "Hydrocortisone for septic shock: a randomized trial", "Septic shock; corticosteroids "
        "(hydrocortisone) vs placebo; mortality at 28 days.", year=2018, doi="10.1/main", nct="NCT00000001",
        mesh=["Shock, Septic"]),
    Rec("101", "Hydrocortisone in septic shock: a post hoc subgroup analysis by severity",
        "Septic shock post hoc subgroup; corticosteroids (hydrocortisone).", year=2019, doi="10.1/sub",
        nct="NCT00000001",
        mesh=["Shock, Septic"]),
    Rec("102", "Corticosteroids in cardiac surgery: a randomized trial", "Septic shock was not studied. "
        "Corticosteroids after cardiac surgery; hydrocortisone.", year=2016, doi="10.1/cardiac"),
    Rec("103", "Hydrocortisone in patients without septic shock", "Patients without septic shock received "
        "hydrocortisone.", year=2017, doi="10.1/neg"),
    Rec("104", "Corticosteroids for septic shock: systematic review and meta-analysis",
        "We searched MEDLINE and Embase from inception to March 15, 2023. Septic shock; corticosteroids; "
        "mortality.", year=2023, doi="10.1/sr", types=["Systematic Review", "Meta-Analysis"],
        pmcid="PMC1000104", oa_licence="by"),
    Rec("105", "Surviving Sepsis Campaign: international guidelines for management of septic shock 2021",
        "Surviving Sepsis Campaign guideline on septic shock including corticosteroids.", year=2021,
        doi="10.1/ssc21", types=["Practice Guideline"]),
    Rec("106", "Surviving Sepsis Campaign: international guidelines for management of septic shock 2021",
        "Surviving Sepsis Campaign guideline on septic shock including corticosteroids.", year=2021,
        doi="10.1/ssc21b", journal="Intensive Care Med", types=["Practice Guideline"]),
    Rec("107", "Surviving Sepsis Campaign: international guidelines for management of septic shock 2016",
        "Surviving Sepsis Campaign guideline on septic shock including corticosteroids.", year=2016,
        doi="10.1/ssc16", types=["Practice Guideline"]),
    Rec("108", "Hydrocortisone versus placebo in septic shock: a newer randomized trial",
        "Septic shock; corticosteroids (hydrocortisone); mortality.", year=2025, doi="10.1/new", entrez="2025/02/01"),
    Rec("109", "Corticosteroids in septic shock: protocol for a randomized trial",
        "Study protocol. Septic shock; hydrocortisone.", year=2024, doi="10.1/protocol"),
    Rec("110", "Hydrocortisone dosing in septic shock: wrong DOI in Europe PMC", "Septic shock; hydrocortisone.",
        year=2020, doi="10.1/right", epmc_doi="10.1/wrong-elsewhere"),
    Rec("111", "Low-dose hydrocortisone in septic shock with open access", "Septic shock; corticosteroids "
        "(hydrocortisone); mortality.",
        year=2021, doi="10.1/oa", pmcid="PMC1000111", oa_licence="by"),
    Rec("112", "Hydrocortisone in septic shock: full text belongs to another paper", "Septic shock; hydrocortisone.",
        year=2021, doi="10.1/mismatch", pmcid="PMC1000112", oa_licence="by",
        jats_title="An unrelated paper about knee surgery outcomes"),
]


def _decide(title: str) -> dict:
    t = title.lower()
    if "cardiac surgery" in t:
        return dims(condition="different", state="direct")  # the model is wrong; the rule is not
    if "without septic shock" in t:
        return dims(state="direct")                          # negation signal must win
    if "subgroup" in t:
        return dims(state="direct")                          # trial link must cap it
    if "guidelines" in t:
        return dims(role="guideline", condition="broader", state="broad_answering")
    if "systematic review" in t:
        return dims(role="systematic_review")
    return dims()


async def _run_retrieval(sources: FakeSources, run_id: str):
    pf = portfolio.build(PLAN, "In adults with septic shock, do corticosteroids reduce mortality?")
    states = await run_portfolio(run_id, [s for s in pf["specs"] if s.condition is None], transport=sources.transport)
    return pf, states


async def test_retrieval_ledger_union_and_identity(client):
    await store.ensure_indexes()
    run_id = "t-" + uuid.uuid4().hex[:8]
    fake = FakeSources(RECORDS)
    pf, states = await _run_retrieval(fake, run_id)
    assert states["pubmed"]["status"] == "complete" and states["europepmc"]["status"] == "complete"
    ledger = await store.queries_for_run(run_id)
    main = next(q for q in ledger if q["family"] == "main" and q["source"] == "pubmed")
    assert main["status"] == "ok" and main["count"] >= 8 and main["translation"].startswith("T(")
    assert main["records_read"] == min(main["count"], main["max_records"]) and main["truncated"] is False
    summary = await union.build(run_id)
    pubs = {p["article"]["pmid"]: p for p in await store.publications_for_run(run_id)}
    # PubMed + Europe PMC copies of the same paper merge into one verified publication.
    assert pubs["100"]["identity_status"] == "verified" and pubs["100"]["sources"] == ["europepmc", "pubmed"]
    assert {"main", "narrow"} <= set(pubs["100"]["families"]) and pubs["100"]["best_rank"]["pubmed"] >= 1
    # A record whose DOI disagrees across sources is quarantined, never merged.
    assert pubs["110"]["identity_status"] == "quarantined" and summary["quarantined_records"] >= 2
    # The article's own ids — never its reference list's.
    assert pubs["100"]["article"]["doi"] == "10.1/main" and pubs["100"]["article"]["pmcid"] is None
    # Separate reports of one trial are linked by NCT, not merged.
    assert pubs["101"]["trial_ids"] == ["NCT00000001"] and pubs["101"]["id"] != pubs["100"]["id"]


async def test_source_outage_is_partial_not_zero(client):
    run_id = "t-" + uuid.uuid4().hex[:8]
    fake = FakeSources(RECORDS, fail={"pubmed"})
    _, states = await _run_retrieval(fake, run_id)
    assert states["pubmed"]["status"] == "failed" and states["europepmc"]["status"] == "complete"
    ledger = [q for q in await store.queries_for_run(run_id) if q["source"] == "pubmed"]
    assert ledger and all(q["status"] == "failed" and q["error"] == "http_500" for q in ledger)
    assert not any(q["status"] == "zero" for q in ledger)  # an outage never looks like "no results"


async def test_rate_limit_retry_after_is_honoured(client):
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "API rate limit exceeded"})
        return httpx.Response(200, json={"esearchresult": {"count": "1", "idlist": ["1"]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        page = await pubmed_src.esearch(c, "x")
    assert page.count == 1 and seen["n"] == 2


# ── W2.4 admission (wrong condition, negation, main trial vs subgroup) ──────


def _cqm():
    from clinical_engine.contracts.question import ClinicalQuestionMap, QuestionElement

    return ClinicalQuestionMap(
        question_text="In adults with septic shock, do corticosteroids reduce mortality?", question_type="treatment",
        framework="PICO", summary="Corticosteroids for mortality in adults with septic shock.",
        elements=[QuestionElement(role="condition", value="septic shock", source="user_text", active=True),
                  QuestionElement(role="intervention", value="corticosteroids", source="user_text", active=True),
                  QuestionElement(role="outcome", value="mortality", source="user_text", active=True),
                  QuestionElement(role="age_group", value="adult", source="user_text", active=True)])


async def test_admission_fixtures_and_sections(client):
    run_id = "t-" + uuid.uuid4().hex[:8]
    await _run_retrieval(FakeSources(RECORDS), run_id)
    await union.build(run_id)
    pubs = await store.publications_for_run(run_id)
    model = FakeModel(_decide)
    usage = await admission.admit(pubs, _cqm(), PLAN, gateway=model)
    by = {p["article"]["pmid"]: p["admission"] for p in pubs}
    assert usage["method"] == "model" and model.calls >= 1
    assert by["100"]["state"] == "direct"
    assert by["102"]["state"] == "excluded" and by["102"]["exclusion"] == "wrong_condition"   # wrong condition
    assert by["103"]["state"] == "excluded" and by["103"]["exclusion"] == "wrong_population"  # negation
    assert by["101"]["state"] == "supportive" and "NCT00000001" in by["101"]["reason"]          # subgroup of main trial
    assert by["109"]["state"] == "excluded" and by["109"]["exclusion"] == "no_results"         # protocol, no model
    assert by["110"]["state"] == "held_back"                                                    # identity conflict
    lineage.build(pubs)
    sections = admission.assign_sections(pubs)
    main_ids = [p["article"]["pmid"] for p in sections["main"]]
    assert "100" in main_ids and "102" not in main_ids and "103" not in main_ids
    assert len(sections["main"]) <= admission.MAIN_LIMIT and len(sections["other"]) <= admission.OTHER_LIMIT
    # No filler: only qualifying papers are listed as the answer.
    assert all(p["admission"]["state"] in admission.QUALIFYING for p in sections["main"])
    # One entry per guideline edition, current edition first, superseded never in the main list.
    guide_main = [p for p in sections["main"] if p["admission"]["role"] == "guideline"]
    assert [p["article"]["year"] if "year" in p["article"] else p["article"]["publication_year"]
            for p in guide_main] == [2021]
    assert sum(1 for p in sections["main"] + sections["other"] + sections["more"]
               if p["article"]["pmid"] in ("105", "106")) == 1


async def test_admission_without_a_model_never_admits(client):
    run_id = "t-" + uuid.uuid4().hex[:8]
    await _run_retrieval(FakeSources(RECORDS), run_id)
    await union.build(run_id)
    pubs = await store.publications_for_run(run_id)

    class NoModel(FakeModel):
        def is_configured(self) -> bool:
            return False

    usage = await admission.admit(pubs, _cqm(), PLAN, gateway=NoModel(_decide))
    assert usage["method"] == "rules_only"
    assert not any(p["admission"]["state"] in admission.QUALIFYING for p in pubs)


def test_policy_state_rules():
    ps = admission.policy_state
    assert ps(dims(condition="different"), "direct", None)[0] == "excluded"
    assert ps(dims(intervention="narrower"), "supportive", None) == ("direct", None)        # class member
    assert ps(dims(population="narrower"), "direct", None) == ("supportive", None)          # subgroup
    assert ps(dims(comparison="narrower"), "direct", None, comparator_stated=False) == ("supportive", None)
    assert ps(dims(comparison="narrower"), "direct", None, comparator_stated=True) == ("direct", None)
    assert ps(dims(condition="broader"), "direct", None) == ("broad_answering", None)
    assert ps(dims(intervention="na"), "direct", None) == ("indirect", None)
    assert ps(dims(intervention="na"), "direct", None, na_intervention_ok=True) == ("direct", None)
    assert ps(dims(), "excluded", "not_a_study") == ("excluded", "not_a_study")


# ── W2.3 lineage ────────────────────────────────────────────────────────────


def _pub(pid, title, year, *, state="broad_answering", role="guideline", journal="Crit Care Med", abstract=""):
    return {"id": pid, "article": {"title": title, "publication_year": year, "journal": journal, "pmid": pid,
                                   "abstract": abstract, "publication_types_json": ["practice guideline"]},
            "admission": {"state": state, "role": role}}


def test_lineage_editions_focused_update_currency_and_cutoffs():
    pubs = [
        _pub("1", "Surviving Sepsis Campaign: international guidelines for management of sepsis and septic shock", 2016),
        _pub("2", "Surviving Sepsis Campaign: international guidelines for management of sepsis and septic shock", 2021),
        _pub("3", "Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock", 2021,
             journal="Intensive Care Med"),
        _pub("4", "Surviving Sepsis Campaign international guidelines for septic shock in children", 2020),
        _pub("5", "Guidelines for the diagnosis and management of critical illness-related corticosteroid "
                  "insufficiency: Society of Critical Care Medicine", 2017, state="direct"),
        _pub("6", "2024 Focused Update: Guidelines on Use of Corticosteroids in Sepsis", 2024, state="direct",
             abstract="Society of Critical Care Medicine. New evidence warrants a focused update of the 2017 guideline."),
        _pub("7", "European Society of Intensive Care Medicine guidelines on septic shock fluids", 2008),
        _pub("8", "Corticosteroids in septic shock: a systematic review", 2024, role="systematic_review", state="direct",
             abstract="We searched PubMed and Embase from inception to 30 June 2023. Twelve trials were included."),
    ]
    out = lineage.build(pubs)
    lin = {p["id"]: p.get("lineage") or {} for p in pubs}
    assert lin["2"]["edition_id"] == lin["3"]["edition_id"] and lin["3"]["kind"] == "co_publication"
    assert lin["1"]["status"] == "superseded" and lin["2"]["status"] == "current"
    assert lin["4"]["family_id"] != lin["2"]["family_id"]  # paediatric guideline is another family
    assert lin["6"]["status"] == "focused_update" and lin["6"]["family_id"] == lin["5"]["family_id"]
    assert "does not replace the whole guideline" in lin["6"]["note"]
    assert lin["5"]["status"] == "current"
    assert lin["7"]["currency"] == "current" and "only guideline" in (lin["7"]["note"] or "")  # justified exception
    assert out["newest_comparable_guideline_year"] == 2024
    assert lin["8"]["cutoff"]["date"] == "2023-06-30"
    anchor = lineage.update_anchor(pubs)
    assert anchor["publication_id"] == "8" and anchor["date"] == "2023-06-30"


def test_cutoff_reader_refuses_other_works_dates():
    assert lineage.search_cutoff("The commission compared the DSG Guideline 2018 (literature search until December "
                                 "2018) with the SSC Guideline 2021 (literature search until July 2020).") is None
    assert lineage.search_cutoff("Databases were searched until 31 January 2024.")["date"] == "2024-01-31"


# ── W2.5 access ─────────────────────────────────────────────────────────────


async def test_access_routes_identity_permission_and_blocking(client):
    fake = FakeSources(RECORDS)
    base = {"identity_status": "verified", "sources": ["pubmed"]}

    def pub(pmid):
        r = fake.by_pmid(pmid)
        return {**base, "id": pmid, "article": {"pmid": r.pmid, "doi": r.doi, "pmcid": r.pmcid, "title": r.title,
                                                "abstract": r.abstract, "journal": r.journal},
                "epmc": {"in_epmc": bool(r.oa_licence), "is_open_access": bool(r.oa_licence)}}

    async with httpx.AsyncClient(transport=fake.transport) as c:
        ok = await access.resolve(pub("111"), c)
        mismatch = await access.resolve(pub("112"), c)
        closed = await access.resolve(pub("100"), c)
    assert ok["acquired"]["status"] == "full_text" and ok["acquired"]["route"] == "europepmc"
    assert ok["acquired"]["completeness"]["status"] == "complete" and ok["permitted"]["text_scope"] == "full_text"
    assert ok["acquired"]["identity"]["ok"] and ok["readable"]["status"] == "free_full_text"
    # A full text whose title belongs to another paper is rejected on every route.
    assert mismatch["acquired"]["status"] != "full_text"
    assert any(a["status"] == "identity_mismatch" for a in mismatch["attempts"])
    # Paywalled: readable via the publisher, abstract permitted, no full text — and no ranking effect.
    assert closed["acquired"]["status"] == "abstract_only" and closed["readable"]["status"] == "publisher_link"
    assert closed["permitted"]["text_scope"] == "abstract" and closed["readable"]["unpaywall"] == "not_configured"
    # Cochrane (via PubMed) fails closed.
    cochrane = {**pub("100"), "article": {**pub("100")["article"], "journal": "Cochrane Database of Systematic Reviews"}}
    async with httpx.AsyncClient(transport=fake.transport) as c:
        assert (await access.resolve(cochrane, c))["permitted"]["text_scope"] == "none"
    # A known wrong link is never offered.
    blocked = {**base, "id": "34605781", "article": {"pmid": "34605781", "doi": "10.1155/2014/698265", "title": "SSC"}}
    async with httpx.AsyncClient(transport=fake.transport) as c:
        res = await access.resolve(blocked, c)
    assert all("10.1155/2014/698265" not in link["url"] for link in res["readable"]["links"])
    assert res["readable"]["blocked_links"]


async def test_access_stage_never_stalls(client, monkeypatch):
    async def slow(*a, **k):
        await asyncio.sleep(5)

    monkeypatch.setattr(access, "resolve", slow)
    pubs = [{"id": "x", "identity_status": "verified", "sources": ["pubmed"], "article": {"pmid": "1"}}]
    out = await access.resolve_many(pubs, stage_seconds=0.2)
    assert out["not_checked"] == 1


# ── Projection: identity-safe article upsert ────────────────────────────────


async def test_projection_never_grafts_a_borrowed_doi(client):
    from core.db import Collections, get_db

    db = get_db()
    doi = "10.7/" + uuid.uuid4().hex[:6]
    await db[Collections.ARTICLES].insert_one({"id": "legacy-" + doi, "canonical_key": "legacy:" + doi,
                                              "pmid": "9" + doi[-5:], "doi": doi, "doi_normalized": doi,
                                              "title": "The paper that really owns this DOI"})
    new = {"canonical_key": "pmid:77" + doi[-4:], "pmid": "77" + doi[-4:], "doi": doi, "doi_normalized": doi,
           "title": "A different paper that was given the same DOI"}
    aid, notes = await projection.upsert_article(new)
    owner = await db[Collections.ARTICLES].find_one({"id": "legacy-" + doi}, {"_id": 0})
    created = await db[Collections.ARTICLES].find_one({"id": aid}, {"_id": 0})
    assert aid != "legacy-" + doi and owner["title"] == "The paper that really owns this DOI"
    assert "doi_normalized_held_by_another_record" in notes and "doi_normalized" not in created


# ── Full engine run through the API ─────────────────────────────────────────

BASE = "/api/v2/engine"


async def test_full_engine_run_through_the_api(client, engine_api, monkeypatch):  # noqa: F811
    from clinical_engine.orchestrator import runner

    fake = FakeSources(RECORDS)
    http.TEST_TRANSPORT = fake.transport
    monkeypatch.setattr(settings, "RETRIEVAL_ADAPTER", "engine")
    monkeypatch.setattr(admission, "admission_gateway", lambda: FakeModel(_decide))
    h = await _signup(client)
    v = (await client.post(f"{BASE}/questions", headers=h,
                           json={"text": "In adults with septic shock, do corticosteroids reduce mortality?"})).json()
    qid = v["question_id"]
    v = (await client.post(f"{BASE}/questions/{qid}/approve-interpretation", headers=h, json={"version": 1})).json()
    tv = v["search_terms"]["version"]
    plan = (await client.get(f"{BASE}/questions/{qid}/search-plan", headers=h,
                             params={"version": 1, "terms_version": tv})).json()
    assert plan["searches"] and plan["press"]["label"] == "Automated PRESS-informed check"
    assert {d["key"] for d in plan["press"]["domains"]} == {"translation", "boolean", "subject_headings",
                                                            "text_words", "spelling_syntax", "limits"}
    r = await client.post(f"{BASE}/questions/{qid}/confirm", headers=h,
                          json={"version": 1, "mode": "answer_for_me", "search_terms_version": tv})
    assert r.status_code == 200, r.text
    run_id = r.json()["run"]["id"]
    run = None
    for _ in range(400):
        run = await store.get_run(run_id)
        if run["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.05)
    assert run["retrieval_adapter"] == "engine" and run["status"] == "completed", run.get("error")
    sid = run["links"]["search_session_id"]
    ev = (await client.get(f"{BASE}/sessions/{sid}/evidence", headers=h)).json()
    assert ev["stage"] == "done" and ev["listed"] and not ev["partial"]
    assert ev["sources"]["pubmed"]["status"] == "complete"
    main = ev["sections"]["main"]
    assert main and all(c["state"] in ("direct", "broad_answering") for c in main)
    assert {c["pmid"] for c in main} >= {"100"}
    assert any(c["role"] == "guideline" and c["lineage"]["currency"] == "current" for c in ev["guidelines"])
    assert ev["portfolio"]["press"]["summary"] and ev["portfolio"]["queries"][0]["translation"]
    assert all(c["access"]["status"] != "checking" for c in main)
    # The existing session API keeps working on the projected rows.
    legacy = await client.get(f"/api/search-sessions/{sid}", headers=h)
    assert legacy.status_code == 200 and legacy.json()["status"] == "awaiting_selection"
    # Report reads the engine ledger; no padding.
    rep = (await client.get(f"{BASE}/sessions/{sid}/report", headers=h)).json()
    assert rep["engine"] and rep["results"]["broader_shown"] == 0
    # "Answer for me" selected from the engine list, every item's AI use checked.
    es = (await client.get(f"{BASE}/evidence-sets/{run['links']['evidence_set_id']}", headers=h)).json()
    assert es["items"] and all(i["ai_processing_allowed"] for i in es["items"])
    # Retry access for one listed paper; owner only.
    one = main[0]["id"]
    assert (await client.post(f"{BASE}/sessions/{sid}/evidence/{one}/access/retry", headers=h)).status_code == 200
    other = await _signup(client)
    assert (await client.get(f"{BASE}/sessions/{sid}/evidence", headers=other)).status_code == 404
    assert (await client.post(f"{BASE}/sessions/{sid}/evidence/{one}/access/retry", headers=other)).status_code == 404
    await runner.shutdown()
