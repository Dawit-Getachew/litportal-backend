"""Second review step: search terms (MeSH + wordings) the clinician approves, and
the search plan built from them. Also the whole-blood → red-cell regression and
the Clinical Question Map V2 frameworks. No network: NCBI and the model are faked."""
from __future__ import annotations

import json
import uuid

import httpx
import pytest

from clinical_engine import lifecycle, questions, store
from clinical_engine.config import settings
from clinical_engine.contracts.question import ClinicalQuestionMap, QuestionElement, framework_for, map_framework
from clinical_engine.frameworks import INTENTS, TYPE_LABELS, field_view, type_options
from clinical_engine.gateway import ModelCallResult
from clinical_engine.interpretation.model_reading import QUESTION_TYPES
from clinical_engine.interpretation.present import ChipEdit, EngineInputError, apply_edits, build_summary
from clinical_engine.orchestrator import runner
from clinical_engine.search_terms import builder, mesh
from clinical_engine.search_terms.contracts import SearchTerms, TermConcept, TermEdit, terms_hash
from clinical_engine.search_terms.plan import query_preview, to_concept_plan
from clinical_engine.search_terms.synonyms import (
    ConceptInput,
    clean_term,
    default_on,
    fallback_terms,
    is_head_preserving_shortening,
    validate_output,
)
from services import engine_concept_plan as ecp
from services.query_compiler import validate_query_syntax

pytestmark = pytest.mark.xdist_group("clinical_engine")


def el(role: str, value: str, **kw) -> QuestionElement:
    return QuestionElement(role=role, value=value, source="user_text", **kw)  # type: ignore[arg-type]


def cqm_of(qtype: str, *elements: QuestionElement, text: str = "question?") -> ClinicalQuestionMap:
    m = ClinicalQuestionMap(question_text=text, question_type=qtype, framework=framework_for(qtype),  # type: ignore[arg-type]
                            elements=list(elements))
    m.framework = map_framework(m)  # type: ignore[assignment]
    m.summary = build_summary(m)
    return m


WHOLE_BLOOD = cqm_of(
    "treatment", el("intervention", "whole blood transfusion"), el("condition", "hemorrhagic shock"),
    el("population", "trauma patients"), el("outcome", "survival"),
    text="Does whole blood transfusion improve survival in trauma patients with hemorrhagic shock?",
)


# ── Fakes ────────────────────────────────────────────────────────────────────

_MESH_DB = {
    "hemorrhagic shock": ("68012771", "D012771", ["Shock, Hemorrhagic", "Hemorrhagic Shock"]),
    "trauma": ("68014947", "D014947", ["Wounds and Injuries", "Injuries", "Trauma"]),
    "survival": ("68013534", "D013534", ["Survival"]),
    "covid-19": ("2052179", "D000086382", ["COVID-19", "COVID 19", "SARS-CoV-2 Infection"]),
}


def ncbi_transport(calls: list | None = None, fail: bool = False) -> httpx.MockTransport:
    """E-utilities db=mesh: esearch "<term>"[mh] → ids; esummary → records."""
    by_uid = {uid: (ui, terms) for uid, ui, terms in _MESH_DB.values()}

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        if fail:
            return httpx.Response(503)
        params = dict(request.url.params)
        if request.url.path.endswith("esearch.fcgi"):
            term = params["term"].replace('"', "").replace("[mh]", "").strip().lower()
            hit = _MESH_DB.get(term)
            return httpx.Response(200, json={"esearchresult": {"idlist": [hit[0]] if hit else []}})
        result = {}
        for uid in params["id"].split(","):
            ui, terms = by_uid[uid]
            result[uid] = {"ds_recordtype": "descriptor", "ds_meshui": ui, "ds_meshterms": terms,
                           "ds_scopenote": f"note for {terms[0]}"}
        return httpx.Response(200, json={"result": result})

    return httpx.MockTransport(handler)


class FakeGateway:
    """Returns a fixed synonym classification (by concept term)."""

    def __init__(self, by_term: dict[str, list[tuple[str, str]]] | None, error: str | None = None):
        self.by_term = by_term or {}
        self.error = error
        self.model = "fake-model"
        self.calls = 0

    def is_configured(self) -> bool:
        return True

    async def complete_json(self, system: str, user: str, *, max_tokens: int, timeout_s: float) -> ModelCallResult:
        self.calls += 1
        if self.error:
            return ModelCallResult(None, self.model, error=self.error)
        payload = json.loads(user)
        concepts = [{"id": c["id"], "terms": [{"text": t, "kind": k} for t, k in self.by_term.get(c["term"], [])]}
                    for c in payload["concepts"]]
        return ModelCallResult({"concepts": concepts}, self.model, 100, 50, 10)


WHOLE_BLOOD_TERMS = {
    "whole blood transfusion": [
        ("whole-blood transfusion", "equivalent"), ("LTOWB", "equivalent"), ("low titer group O whole blood", "narrower"),
        ("massive transfusion", "narrower"),            # sibling the model mislabels — must stay OFF
        ("red blood cell transfusion", "equivalent"),   # sibling mislabelled equivalent — must stay OFF
        ("blood transfusion", "narrower"),              # head-preserving shortening → broader, OFF
        ("whole blood", "equivalent"),                  # bare product (also a lab specimen) — OFF
        ("RBC", "equivalent"),                          # unrelated acronym — OFF
        ("component therapy", "related"),
    ],
    "hemorrhagic shock": [("haemorrhagic shock", "equivalent"), ("hypovolemic shock", "narrower"),
                          ("traumatic hemorrhagic shock", "narrower"), ("shock, hemorrhagic", "equivalent")],
    "trauma": [("injuries", "equivalent"), ("trauma patients", "equivalent"), ("hemorrhagic shock", "equivalent")],
}


@pytest.fixture(autouse=True)
def fresh_mesh_cache():
    mesh.clear_cache()
    yield
    mesh.clear_cache()


async def build(cqm=WHOLE_BLOOD, by_term=WHOLE_BLOOD_TERMS, **kw) -> SearchTerms:
    return await builder.build_search_terms(cqm, question_version=1, gateway=FakeGateway(by_term, kw.pop("error", None)),
                                            mesh_transport=ncbi_transport(**kw))


def concept(terms: SearchTerms, term: str) -> TermConcept:
    return next(c for c in terms.concepts if c.term == term)


def on(c: TermConcept) -> set[str]:
    return {t.text for t in c.terms if t.enabled}


# ── Whole blood → red cells regression (legacy pipeline) ────────────────────


def test_blood_transfusion_no_longer_maps_to_red_cell_transfusion():
    from services.clinical_ontology_mapper import clinical_ontology_mapper

    m = clinical_ontology_mapper.map_term("blood transfusion")
    assert m.canonical_concept.lower() != "red blood cell transfusion"
    assert "packed red blood cells" not in [s.lower() for s in m.synonyms]


def test_lift_skips_a_fragment_of_an_existing_concept():
    from services.clinical_context_engine import _is_fragment_of_existing, _lift_ontology_into_picott

    picott = {"must_have_concepts": [{"concept": "whole blood transfusion", "role": "intervention", "synonyms": []}],
              "pico": {"population": ["trauma patients with hemorrhagic shock"]}}
    red_cell_hit = {"original_term": "blood transfusion", "normalized_term": "blood transfusion",
                    "canonical_concept": "red blood cell transfusion", "role_hint": "intervention",
                    "confidence": 0.99, "synonyms": ["RBC transfusion"], "mesh_terms": ["Blood Transfusion"]}
    lifts = _lift_ontology_into_picott(picott, [red_cell_hit], role_weights_dict={"intervention": 100})
    assert lifts == []
    assert [m["concept"] for m in picott["must_have_concepts"]] == ["whole blood transfusion"]
    # A glued phrase ("X with Y") can still yield a separate concept.
    existing = [(["trauma", "patients", "with", "hemorrhagic", "shock"], "population")]
    assert not _is_fragment_of_existing("hemorrhagic shock", "condition", existing)


# ── Clinical Question Map V2 ─────────────────────────────────────────────────


def test_every_question_type_has_an_intent_label_and_framework():
    types_in_intents = [t for i in INTENTS for t in i.types]
    assert sorted(types_in_intents) == sorted(QUESTION_TYPES)
    assert set(TYPE_LABELS) == set(QUESTION_TYPES)
    assert len(INTENTS) == 14  # 13 V2 intents + "other"
    options = type_options()
    assert {o["value"] for o in options} == set(QUESTION_TYPES)
    assert next(o for o in options if o["value"] == "qualitative")["framework"] == "PICo / SPIDER"


def test_peo_becomes_peco_when_a_comparator_is_added():
    m = cqm_of("harm", el("exposure", "NSAIDs"), el("outcome", "acute kidney injury"))
    assert m.framework == "PEO"
    edited = apply_edits(m, [ChipEdit(op="add", role="comparator", value="paracetamol")])
    assert edited.framework == "PECO"


def test_field_view_follows_the_framework_and_hides_nothing():
    m = cqm_of("prognostic_factor", el("prognostic_factor", "frailty"), el("outcome", "1-year mortality"),
               el("condition", "heart failure"), el("comparator", "robust patients"))
    fields = field_view(m, age_placeholder=True)
    labels = [f["label"] for f in fields]
    assert labels[:5] == ["Population", "Prognostic factor", "Reference level", "Outcome", "Horizon"]
    placed = [cid for f in fields for cid in f["chip_ids"]]
    assert sorted(placed) == sorted([e.id for e in m.elements] + ["add:age_group"])
    assert fields[-1]["label"] == "Other details"  # the comparator is not a PFO field


def test_new_intents_have_readable_summaries():
    q = cqm_of("qualitative", el("phenomenon", "patients' experiences"), el("intervention", "home hemodialysis"))
    assert q.summary == "Experiences of home hemodialysis."
    p = cqm_of("prediction_model", el("index_test", "qSOFA"), el("outcome", "in-hospital mortality"),
               el("condition", "suspected infection"))
    assert p.summary.startswith("How well qSOFA predicts in-hospital mortality")


# ── Term validation ─────────────────────────────────────────────────────────


def test_terms_must_be_plain_words():
    assert clean_term("  whole-blood transfusion ") == "whole-blood transfusion"
    for bad in ('"x"[tiab]', "a OR b", "blood*", "shock, septic", "(steroids)", "see doi 10.1000/x"):
        assert clean_term(bad) is None


def test_default_on_is_conservative():
    wb = "whole blood transfusion"
    assert default_on("equivalent", "whole-blood transfusion", wb, "model")
    assert default_on("equivalent", "WBT", wb, "model")
    assert default_on("narrower", "low titer group O whole blood", wb, "model")
    assert not default_on("narrower", "massive transfusion", wb, "model")
    assert not default_on("equivalent", "red blood cell transfusion", wb, "model")
    assert not default_on("equivalent", "RBC", wb, "model")
    assert not default_on("equivalent", "whole blood", wb, "model")
    assert not default_on("broader", "blood transfusion", wb, "model")
    assert not default_on("narrower", "hypovolemic shock", "hemorrhagic shock", "model")
    assert default_on("equivalent", "quick Sequential Organ Failure Assessment", "qSOFA", "model")
    assert not default_on("equivalent", "SOFA score", "qSOFA", "model")
    assert is_head_preserving_shortening("blood transfusion", wb)


def test_model_output_is_validated():
    concepts = [ConceptInput("c1", "intervention", "whole blood transfusion"),
                ConceptInput("c2", "condition", "hemorrhagic shock", candidates=["Hemorrhagic Shock"])]
    raw = {"concepts": [
        {"id": "c1", "terms": [{"text": "blood transfusion", "kind": "equivalent"},
                               {"text": "hemorrhagic shock", "kind": "related"},   # another concept
                               {"text": "X[tiab]", "kind": "equivalent"},
                               {"text": "fresh whole blood", "kind": "weird"}]},
        {"id": "c9", "terms": []},
        {"id": "c2", "terms": [{"text": "Hemorrhagic Shock", "kind": "exclude"}]},
    ]}
    reading = validate_output(raw, concepts)
    assert [(t.text, t.kind) for t in reading.terms["c1"]] == [("blood transfusion", "broader")]
    assert reading.terms["c2"] == []
    assert {"is_another_concept", "not_a_plain_term", "unknown_kind", "unknown_concept"} <= set(reading.rejected)
    assert validate_output("nonsense", concepts).rejected == ["schema_invalid"]


# ── Building the terms ──────────────────────────────────────────────────────


async def test_build_maps_mesh_exactly_and_switches_on_only_safe_terms():
    terms = await build()
    assert [c.term for c in terms.concepts] == ["hemorrhagic shock", "trauma", "whole blood transfusion", "survival"]
    wb, shock, trauma = concept(terms, "whole blood transfusion"), concept(terms, "hemorrhagic shock"), concept(terms, "trauma")
    assert wb.mesh is None  # no exact heading → never the broad "Blood Transfusion"
    assert shock.mesh and shock.mesh.descriptor == "Shock, Hemorrhagic" and shock.mesh.match == "exact"
    # "whole-blood transfusion" is the concept itself (hyphen = space); "LTOWB" is not
    # an acronym of "whole blood transfusion", so it stays off as an "equivalent".
    assert "whole-blood transfusion" not in {t.text for t in wb.terms}
    assert on(wb) == {"low titer group O whole blood"}
    off = {t.text for t in wb.terms if not t.enabled}
    assert {"LTOWB", "massive transfusion", "red blood cell transfusion", "blood transfusion", "whole blood",
            "RBC"} <= off
    assert on(shock) == {"haemorrhagic shock", "traumatic hemorrhagic shock"}
    # population: people words are not searched, and it ranks rather than restricts
    assert trauma.required is False and trauma.mesh and trauma.mesh.descriptor == "Wounds and Injuries"
    assert "hemorrhagic shock" not in {t.text for t in trauma.terms}  # another concept is never a synonym
    assert concept(terms, "survival").required is False
    assert terms.mapping.mesh == "ncbi" and terms.mapping.synonyms == "model"


async def test_population_that_restates_the_condition_is_dropped():
    m = cqm_of("prediction_model", el("index_test", "qSOFA"), el("condition", "suspected infection"),
               el("population", "adults with suspected infection"), el("outcome", "in-hospital mortality"))
    terms = await build(m, {})
    assert [c.term for c in terms.concepts] == ["suspected infection", "qSOFA", "in-hospital mortality"]


async def test_broader_mesh_is_offered_switched_off():
    m = cqm_of("treatment", el("intervention", "dexamethasone"), el("condition", "severe COVID-19"))
    terms = await build(m, {})
    covid = concept(terms, "severe COVID-19")
    assert covid.mesh and covid.mesh.descriptor == "COVID-19"
    assert covid.mesh.match == "broader" and covid.mesh.enabled is False
    assert '"COVID-19"[MeSH Terms]' not in (query_preview(terms, m)["pubmed"] or "")


async def test_qualitative_phenomenon_gets_the_qualitative_vocabulary():
    m = cqm_of("qualitative", el("phenomenon", "patients' experiences"), el("intervention", "home hemodialysis"))
    terms = await build(m, {})
    phen = concept(terms, "patients' experiences")
    assert {"qualitative", "interviews", "lived experience", "focus groups"} <= on(phen)


async def test_mesh_or_model_outage_degrades_honestly():
    terms = await build(fail=True, error="timeout")
    assert terms.mapping.mesh == "unavailable" and terms.mapping.synonyms == "rules_only"
    assert terms.mapping.fallback_reason == "model_timeout"
    assert all(c.mesh is None for c in terms.concepts)
    assert len(terms.warnings) == 2
    assert all(not t.enabled for c in terms.concepts for t in c.terms)  # nothing unchecked is switched on


def test_fallback_offers_mesh_entry_terms_switched_off():
    out = fallback_terms([ConceptInput("c", "condition", "sepsis", candidates=["Septicemia", "Sepsis"])])
    assert [t.text for t in out["c"]] == ["Septicemia"]


async def test_mesh_lookup_is_cached():
    calls: list = []
    await mesh.lookup_many(["hemorrhagic shock"], transport=ncbi_transport(calls))
    n = len(calls)
    again = await mesh.lookup_many(["hemorrhagic shock"], transport=ncbi_transport(calls))
    assert len(calls) == n and again.records["hemorrhagic shock"].ui == "D012771"


# ── Edits ────────────────────────────────────────────────────────────────────


async def test_term_edits():
    terms = await build()
    wb = concept(terms, "whole blood transfusion")
    massive = next(t for t in wb.terms if t.text == "massive transfusion")
    edited = builder.apply_term_edits(terms, [
        TermEdit(op="set_term_enabled", concept_id=wb.id, term_id=massive.id, enabled=True),
        TermEdit(op="add_term", concept_id=wb.id, text="warm fresh whole blood"),
        TermEdit(op="add_term", concept_id=wb.id, text="blood transfusion"),  # user may add a broader term
        TermEdit(op="set_publication_years", years=10),
        TermEdit(op="set_exclude_animal_only", enabled=False),
        TermEdit(op="set_required", concept_id=concept(terms, "trauma").id, enabled=True),
    ])
    wb2 = concept(edited, "whole blood transfusion")
    assert {"massive transfusion", "warm fresh whole blood", "blood transfusion"} <= on(wb2)
    assert next(t for t in wb2.terms if t.text == "blood transfusion").kind == "broader"
    assert edited.version == terms.version + 1 and edited.filters.publication_years == 10
    assert concept(edited, "trauma").required is True
    assert terms_hash(edited) != terms_hash(terms)
    with pytest.raises(EngineInputError):
        builder.apply_term_edits(terms, [TermEdit(op="add_term", concept_id=wb.id, text="x OR y")])
    with pytest.raises(EngineInputError):
        builder.apply_term_edits(terms, [TermEdit(op="set_mesh_enabled", concept_id=wb.id, enabled=True)])
    none_required = builder.apply_term_edits(
        terms, [TermEdit(op="set_required", concept_id=c.id, enabled=False) for c in terms.concepts])
    assert builder.blocking_problems(none_required)


# ── Plan → queries (services side, shipped in the worker images) ─────────────


async def test_plan_queries_use_only_approved_terms():
    terms = await build()
    plan = ecp.validate_plan(to_concept_plan(terms, WHOLE_BLOOD))
    assert plan is not None and plan["legacy_type_of_question"] == "therapy"
    assert plan["framework_label"] == "PICO" and plan["intent_label"] == "Treatment or prevention effects"
    from services.interpreted_question_display import framework_display_fields

    shown = framework_display_fields({"engine_concept_plan": plan,
                                      "clinical_question_map_json": {"frameworks": ["PFO"]}})
    assert shown["question_framework"] == "PICO"  # the approved framework, not the pipeline's guess
    queries = ecp.build_queries(plan, ["pubmed", "europe_pmc", "openalex"])
    assert {q["source"] for q in queries} == {"pubmed", "europe_pmc"}
    families = {(q["source"], q["query_family"]) for q in queries}
    assert ("pubmed", "mesh_keyword") in families and ("pubmed", "cochrane_review") in families
    for q in queries:
        assert validate_query_syntax(q["query_string"], source=q["source"]).valid
        assert "red blood cell" not in q["query_string"].lower() and "massive transfusion" not in q["query_string"]
        assert q["planner_source"] == ecp.PLANNER_SOURCE
    pubmed = next(q["query_string"] for q in queries if q["source"] == "pubmed" and q["query_family"] == "mesh_keyword")
    assert '"Shock, Hemorrhagic"[MeSH Terms]' in pubmed and '"low titer group O whole blood"[tiab]' in pubmed
    # the population concept ranks rather than restricts: not AND-ed into the query
    assert '"trauma"[tiab]' not in pubmed and "Wounds and Injuries" not in pubmed
    assert 'NOT ("animals"[MeSH Terms] NOT "humans"[MeSH Terms])' in pubmed
    epmc = next(q["query_string"] for q in queries if q["source"] == "europe_pmc" and q["query_family"] == "mesh_keyword")
    assert "MESH:" not in epmc and 'TITLE:"porcine"' in epmc


async def test_synonyms_switch_off_searches_only_the_concepts():
    terms = await build()
    off = builder.apply_term_edits(terms, [TermEdit(op="set_use_synonyms", enabled=False)])
    q = query_preview(off, WHOLE_BLOOD)["pubmed"]
    assert '"whole blood transfusion"[tiab]' in q and "LTOWB" not in q and "haemorrhagic" not in q
    assert '"Shock, Hemorrhagic"[MeSH Terms]' in q  # MeSH is controlled vocabulary, toggled separately


def test_plan_makes_the_approved_concepts_the_pipeline_concepts():
    plan = ecp.validate_plan({
        "plan_version": ecp.PLAN_VERSION, "question_type": "treatment", "legacy_type_of_question": "therapy",
        "legacy_subtype": None, "use_synonyms": True, "filters": {},
        "concepts": [
            {"role": "intervention", "term": "whole blood transfusion", "required": True, "mesh": None,
             "synonyms": ["LTOWB"]},
            {"role": "comparator", "term": "component therapy", "required": False, "mesh": None, "synonyms": []},
            {"role": "outcome", "term": "survival", "required": False, "mesh": None, "synonyms": []},
        ]})
    picott = {"must_have_concepts": [{"concept": "red blood cell transfusion", "role": "intervention"}],
              "pico": {"population": ["glued phrase"], "setting": ["ED"]}, "type_of_question": "prognosis"}
    anchors = {"primary_anchors": [{"canonical_concept": "red blood cell transfusion"}],
               "anchor_failure_reason": "no_primary_clinical_entity"}
    ecp.apply_to_picott(picott, anchors, plan)
    assert [m["concept"] for m in picott["must_have_concepts"]] == ["whole blood transfusion"]
    assert picott["must_have_concepts"][0]["synonyms"] == ["LTOWB"]
    assert picott["should_have_concepts"][0]["concept"] == "component therapy"
    assert picott["ranking_concepts"][0]["concept"] == "survival"
    assert picott["type_of_question"] == "therapy" and picott["pico"]["setting"] == ["ED"]
    assert "population" not in picott["pico"]
    assert [a["canonical_concept"] for a in anchors["primary_anchors"]] == ["whole blood transfusion"]
    assert anchors["anchor_failure_reason"] is None
    assert ecp.validate_plan({"plan_version": "other"}) is None
    assert ecp.validate_plan({**plan, "concepts": [{**plan["concepts"][1]}]}) is None  # nothing required


# ── Over HTTP: the two-step review ──────────────────────────────────────────

BASE = "/api/v2/engine"


@pytest.fixture
async def engine_api(client, monkeypatch):
    await lifecycle.stop()
    monkeypatch.setattr(settings, "ENABLED", True)
    monkeypatch.setattr(settings, "RETRIEVAL_ADAPTER", "mock")
    monkeypatch.setattr(settings, "POLL_INTERVAL_S", 0.05)
    monkeypatch.setenv("LLM_PROVIDER", "none")  # interpretation: rules only; terms: MeSH only
    real_lookup = mesh.lookup_many

    async def offline_lookup(terms, **kw):
        kw["transport"] = ncbi_transport()
        return await real_lookup(terms, **kw)

    monkeypatch.setattr(mesh, "lookup_many", offline_lookup)

    async def fake_counts(preview, **kw):
        return {"pubmed": 42 if preview.get("pubmed") else None, "europe_pmc": 7}

    monkeypatch.setattr(questions, "hit_counts", fake_counts)
    await store.ensure_indexes()
    yield
    await runner.shutdown()


async def _signup(client) -> dict:
    email = f"terms-{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post("/api/auth/signup", json={"email": email, "password": "Terms9!Password", "name": "T"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def test_two_step_review_over_http(client, engine_api):
    h = await _signup(client)
    r = await client.post(f"{BASE}/questions", headers=h,
                          json={"text": "Is whole blood transfusion effective for survival in hemorrhagic shock?"})
    view = r.json()
    qid, v = view["question_id"], view["version"]
    assert view["step"] == 1 and view["can_approve_interpretation"]
    assert view["interpretation"]["framework_label"] == "PICO"
    assert [f["label"] for f in view["interpretation"]["fields"]][:4] == ["Patients", "Treatment", "Comparison", "Outcomes"]
    assert "approve_interpretation" in view["allowed_actions"] and view["search_terms"] is None

    # Step 1 → 2: terms are built, nothing is searched.
    r = await client.post(f"{BASE}/questions/{qid}/approve-interpretation", headers=h, json={"version": v})
    assert r.status_code == 200, r.text
    view = r.json()
    st = view["search_terms"]
    assert view["status"] == "awaiting_search_terms" and view["step"] == 2 and view["interpretation_approval"]
    assert st["version"] == 1 and st["can_run"] and st["query_preview"]["pubmed"]
    assert {s["id"] for s in st["sources"]} >= {"pubmed", "europe_pmc", "reviews", "cochrane"}
    assert view["run"] is None
    shock = next(c for c in st["concepts"] if c["term"] == "hemorrhagic shock")
    assert shock["mesh"]["descriptor"] == "Shock, Hemorrhagic"
    # Approving again is idempotent.
    again = (await client.post(f"{BASE}/questions/{qid}/approve-interpretation", headers=h, json={"version": v})).json()
    assert again["search_terms"]["version"] == 1

    # Edit the terms; a stale edit is refused.
    edits = [{"op": "set_mesh_enabled", "concept_id": shock["id"], "enabled": False},
             {"op": "add_term", "concept_id": shock["id"], "text": "haemorrhagic shock"},
             {"op": "set_publication_years", "years": 10}]
    r = await client.post(f"{BASE}/questions/{qid}/search-terms/revise", headers=h,
                          json={"version": v, "base_terms_version": 1, "edits": edits})
    assert r.status_code == 200, r.text
    st = r.json()["search_terms"]
    assert st["version"] == 2 and "MeSH Terms" not in st["query_preview"]["pubmed"].split("NOT")[0]
    assert '"haemorrhagic shock"[tiab]' in st["query_preview"]["pubmed"]
    stale = await client.post(f"{BASE}/questions/{qid}/search-terms/revise", headers=h,
                              json={"version": v, "base_terms_version": 1, "edits": edits})
    assert stale.status_code == 409 and stale.json()["detail"]["code"] == "stale_terms"
    bad = await client.post(f"{BASE}/questions/{qid}/search-terms/revise", headers=h,
                            json={"version": v, "base_terms_version": 2,
                                  "edits": [{"op": "add_term", "concept_id": shock["id"], "text": "a AND b"}]})
    assert bad.status_code == 422

    counts = (await client.get(f"{BASE}/questions/{qid}/search-terms/counts?version={v}&terms_version=2",
                               headers=h)).json()
    assert counts["counts"] == {"pubmed": 42, "europe_pmc": 7}

    # Confirming needs the exact terms version the clinician saw.
    r = await client.post(f"{BASE}/questions/{qid}/confirm", headers=h,
                          json={"version": v, "mode": "review_papers", "search_terms_version": 1})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "stale_terms"
    r = await client.post(f"{BASE}/questions/{qid}/confirm", headers=h,
                          json={"version": v, "mode": "review_papers", "search_terms_version": 2})
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["status"] == "confirmed" and view["approval"]["search_terms_version"] == 2
    assert view["approval"]["search_terms_hash"]
    # Terms can no longer change once the search started.
    late = await client.post(f"{BASE}/questions/{qid}/search-terms/revise", headers=h,
                             json={"version": v, "base_terms_version": 2, "edits": edits})
    assert late.status_code == 409

    trace = [e["event"] for e in (await client.get(f"{BASE}/questions/{qid}/trace", headers=h)).json()["events"]]
    assert trace.index("interpretation_approved") < trace.index("search_terms_revised") < trace.index("approval_recorded")

    # Feedback on the results.
    fb = await client.post(f"{BASE}/questions/{qid}/feedback", headers=h,
                           json={"helpful": True, "comment": "Found the RCT I needed"})
    assert fb.status_code == 201 and fb.json()["recorded"]
    other = await _signup(client)
    assert (await client.post(f"{BASE}/questions/{qid}/feedback", headers=other,
                              json={"helpful": False})).status_code == 404


async def test_revising_the_question_during_step_two_goes_back_to_step_one(client, engine_api):
    h = await _signup(client)
    view = (await client.post(f"{BASE}/questions", headers=h,
                              json={"text": "Does dexamethasone reduce mortality in severe COVID-19?"})).json()
    qid = view["question_id"]
    await client.post(f"{BASE}/questions/{qid}/approve-interpretation", headers=h, json={"version": 1})
    r = await client.post(f"{BASE}/questions/{qid}/revise", headers=h,
                          json={"base_version": 1, "edits": [{"op": "add", "role": "age_group", "value": "adult"}]})
    view = r.json()
    assert view["version"] == 2 and view["status"] == "awaiting_confirmation" and view["step"] == 1
    assert view["search_terms"] is None
    # confirm with a terms version is refused before the new interpretation is approved
    r = await client.post(f"{BASE}/questions/{qid}/confirm", headers=h,
                          json={"version": 2, "mode": "review_papers", "search_terms_version": 1})
    assert r.status_code == 409


async def test_feedback_comment_with_phi_is_refused(client, engine_api):
    h = await _signup(client)
    view = (await client.post(f"{BASE}/questions", headers=h,
                              json={"text": "Does dexamethasone reduce mortality in severe COVID-19?"})).json()
    qid = view["question_id"]
    assert (await client.post(f"{BASE}/questions/{qid}/feedback", headers=h, json={"helpful": True})).status_code == 409
    await client.post(f"{BASE}/questions/{qid}/confirm", headers=h, json={"version": 1, "mode": "review_papers"})
    r = await client.post(f"{BASE}/questions/{qid}/feedback", headers=h,
                          json={"helpful": False, "comment": "Patient John Smith MRN 12345678 was not helped"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "phi_detected"


# ── The run hands the approved plan to the search pipeline ──────────────────


async def test_run_passes_the_approved_plan_to_the_search(client, monkeypatch):
    from clinical_engine.orchestrator import tasks as task_mod
    from core.db import Collections, get_db

    await lifecycle.stop()
    monkeypatch.setattr(settings, "ENABLED", True)
    monkeypatch.setattr(settings, "RETRIEVAL_ADAPTER", "legacy_pipeline")
    monkeypatch.setattr(settings, "POLL_INTERVAL_S", 0.05)
    monkeypatch.setenv("LLM_PROVIDER", "none")
    real_lookup = mesh.lookup_many

    async def offline_lookup(terms, **kw):
        kw["transport"] = ncbi_transport()
        return await real_lookup(terms, **kw)

    monkeypatch.setattr(mesh, "lookup_many", offline_lookup)
    seen: list = []

    async def start(user_id, inputs, concept_plan=None):
        seen.append(concept_plan)
        sid = f"fake-{uuid.uuid4().hex}"
        await get_db()[Collections.SEARCH_SESSIONS].insert_one(
            {"id": sid, "user_id": user_id, "status": "awaiting_selection", "original_question": inputs["question"],
             "deleted_at": None, "created_at": store.now()})
        return sid

    monkeypatch.setattr(task_mod, "_start_legacy_session", start)
    await store.ensure_indexes()
    u = {"id": f"plan-user-{uuid.uuid4().hex[:6]}"}
    view = await questions.create_question(u, "Is whole blood transfusion effective for survival in hemorrhagic shock?")
    view = await questions.approve_interpretation(u, view["question_id"], version=1)
    view = await questions.confirm_question(u, view["question_id"], version=1, mode="review_papers",
                                            search_terms_version=view["search_terms"]["version"])
    run_id = view["run"]["id"]
    for _ in range(200):
        run = await store.get_run(run_id)
        if run["status"] in ("completed", "failed"):
            break
        import asyncio
        await asyncio.sleep(0.05)
    assert run["status"] == "completed", run
    assert seen and seen[0]["plan_version"] == ecp.PLAN_VERSION
    assert {c["term"] for c in seen[0]["concepts"] if c["required"]} >= {"whole blood transfusion", "hemorrhagic shock"}
    full = await questions.get_run(u, run_id)
    assert full["search_plan"]["approved_search"]["terms_hash"] == seen[0]["terms_hash"]
    assert full["search_plan"]["approved_search"]["queries"]["pubmed"]
    await runner.shutdown()


async def test_start_session_stores_the_plan_on_the_session(client, monkeypatch):
    """The internal session start keeps the public route's contract and stores extras."""
    from core.db import Collections, get_db
    from models.sessions import SearchSessionCreate
    from routers import search_sessions

    scheduled: list = []
    monkeypatch.setattr(search_sessions, "schedule_pipeline", lambda *a, **k: scheduled.append(a))
    plan = {"plan_version": ecp.PLAN_VERSION, "concepts": []}
    resp = await search_sessions.start_session(
        SearchSessionCreate(question="Does whole blood improve survival in hemorrhagic shock?"),
        "start-session-user", extras={"engine_concept_plan": plan},
    )
    doc = await get_db()[Collections.SEARCH_SESSIONS].find_one({"id": resp.id}, {"_id": 0})
    assert doc["engine_concept_plan"] == plan and scheduled
    # PHI: nothing stored, extras included
    blocked = await search_sessions.start_session(
        SearchSessionCreate(question="Patient John Smith MRN 12345678 whole blood?"),
        "start-session-user", extras={"engine_concept_plan": plan},
    )
    assert getattr(blocked, "status_code", None) == 200
    body = json.loads(blocked.body)
    stored = await get_db()[Collections.SEARCH_SESSIONS].find_one({"id": body["session_id"]}, {"_id": 0})
    assert stored["status"] == "blocked_phi" and "engine_concept_plan" not in stored


async def test_step_one_exact_terms_only_starts_step_two_with_synonyms_off(client, engine_api, monkeypatch):
    fake = FakeGateway({"hemorrhagic shock": [("haemorrhagic shock", "equivalent")]})
    monkeypatch.setattr(builder, "_terms_gateways", lambda: [fake])
    h = await _signup(client)
    view = (await client.post(f"{BASE}/questions", headers=h,
                              json={"text": "Is whole blood transfusion effective for survival in hemorrhagic shock?"})).json()
    qid, v = view["question_id"], view["version"]
    r = await client.post(f"{BASE}/questions/{qid}/approve-interpretation", headers=h,
                          json={"version": v, "use_synonyms": False})
    assert r.status_code == 200, r.text
    st = r.json()["search_terms"]
    assert st["use_synonyms"] is False and st["version"] == 1
    shock = next(c for c in st["concepts"] if c["term"] == "hemorrhagic shock")
    assert any(t["text"] == "haemorrhagic shock" for t in shock["terms"])  # still listed, to switch back on
    assert "haemorrhagic" not in st["query_preview"]["pubmed"]
    assert '"Shock, Hemorrhagic"[MeSH Terms]' in st["query_preview"]["pubmed"]
    # Back to step 1 and approve again with synonyms on → applied as a terms edit.
    st2 = (await client.post(f"{BASE}/questions/{qid}/approve-interpretation", headers=h,
                             json={"version": v, "use_synonyms": True})).json()["search_terms"]
    assert st2["use_synonyms"] is True and st2["version"] == 2 and "haemorrhagic" in st2["query_preview"]["pubmed"]
    # Same choice again (or no choice) is a no-op.
    st3 = (await client.post(f"{BASE}/questions/{qid}/approve-interpretation", headers=h,
                             json={"version": v})).json()["search_terms"]
    assert st3["version"] == 2
    trace = (await client.get(f"{BASE}/questions/{qid}/trace", headers=h)).json()["events"]
    approved = next(e for e in trace if e["event"] == "interpretation_approved")
    assert approved["details"]["use_synonyms"] is False
