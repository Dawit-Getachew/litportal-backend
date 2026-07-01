"""Phase L1 — Explicit Evidence Lens + Population Focus preference tests.

Covers the pure preference engine (``services/evidence_lens.py``):
matching, bounded boosting, the cap-dominates safety invariant, the
shadow vs soft_boost modes, normalization, and the no-op defaults.
"""
from __future__ import annotations

import services.evidence_lens as el


# --- Normalization -----------------------------------------------------

def test_normalize_lens_defaults_and_passthrough():
    assert el.normalize_lens("mechanism") == "mechanism"
    assert el.normalize_lens("MECHANISM") == "mechanism"
    assert el.normalize_lens("nonsense") == el.DEFAULT_EVIDENCE_LENS
    assert el.normalize_lens(None) == "clinically_direct"


def test_normalize_population_defaults_and_passthrough():
    assert el.normalize_population("pediatrics") == "pediatrics"
    assert el.normalize_population("garbage") == el.DEFAULT_POPULATION_FOCUS
    assert el.normalize_population(None) == "any"


def test_resolve_mode_default(monkeypatch):
    monkeypatch.delenv("L1_PREFERENCE_MODE", raising=False)
    monkeypatch.setenv("APP_ENV", "development")
    assert el.resolve_preference_mode() == "soft_boost"


def test_resolve_mode_prod_downgrades_to_shadow(monkeypatch):
    monkeypatch.setenv("L1_PREFERENCE_MODE", "soft_boost")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("L1_PREFERENCE_PROD_SOFT_BOOST_ALLOWED", raising=False)
    assert el.resolve_preference_mode() == "shadow"


def test_resolve_mode_prod_allowed_keeps_soft_boost(monkeypatch):
    monkeypatch.setenv("L1_PREFERENCE_MODE", "soft_boost")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("L1_PREFERENCE_PROD_SOFT_BOOST_ALLOWED", "true")
    assert el.resolve_preference_mode() == "soft_boost"


# --- L1.0.1 production UI visibility gate -----------------------------

def test_ui_visible_in_preview_dev(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    assert el.l1_ui_visible() is True
    monkeypatch.setenv("APP_ENV", "preview")
    assert el.l1_ui_visible() is True


def test_ui_hidden_in_production_by_default(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("L1_PREFERENCE_PROD_UI_VISIBLE", raising=False)
    assert el.l1_ui_visible() is False


def test_ui_visible_in_production_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("L1_PREFERENCE_PROD_UI_VISIBLE", "true")
    assert el.l1_ui_visible() is True


# --- Default lens / population are no-ops -------------------------------

def test_default_lens_and_population_is_noop():
    art = {"title": "direct colon cancer prognosis", "abstract": "survival",
           "publication_year": 2024}
    pref = el.compute_article_preference(
        art, {}, lens="clinically_direct", population="any", current_year=2026,
    )
    assert pref["boost"] == 0.0
    assert pref["matched"] is False
    assert pref["signals"] == []


# --- Lens matching -----------------------------------------------------

def test_mechanism_lens_boosts_mechanism_article():
    art = {"title": "Mechanism of pruritus", "abstract": "molecular pathway pathophysiology"}
    pref = el.compute_article_preference(art, {}, lens="mechanism", population="any")
    assert pref["matched"] is True
    assert "mechanism_focus" in pref["signals"]
    assert pref["boost"] == el._LENS_BOOST


def test_mechanism_lens_does_not_boost_treatment_article():
    art = {"title": "Randomized trial of drug X", "abstract": "efficacy outcome"}
    pref = el.compute_article_preference(art, {}, lens="mechanism", population="any")
    assert pref["matched"] is False
    assert pref["boost"] == 0.0


def test_risk_incidence_lens():
    art = {"title": "Incidence of pneumothorax after central line",
           "abstract": "complication rate in a cohort"}
    pref = el.compute_article_preference(art, {}, lens="risk_incidence", population="any")
    assert "risk_incidence_focus" in pref["signals"]


def test_drug_safety_lens():
    art = {"title": "Adverse effects of dexmedetomidine",
           "abstract": "safety and toxicity in ICU"}
    pref = el.compute_article_preference(art, {}, lens="drug_safety", population="any")
    assert "drug_safety_focus" in pref["signals"]


def test_guidelines_lens_pub_type():
    art = {"title": "CRC screening", "abstract": "average risk adults",
           "publication_types": ["Practice Guideline"]}
    pref = el.compute_article_preference(art, {}, lens="guidelines", population="any")
    assert "guideline_publication_type" in pref["signals"]


def test_systematic_reviews_lens():
    art = {"title": "Systematic review", "abstract": "pooled",
           "publication_types": ["Systematic Review", "Meta-Analysis"]}
    pref = el.compute_article_preference(
        art, {}, lens="systematic_reviews", population="any")
    assert "systematic_review_or_meta_analysis" in pref["signals"]


def test_most_recent_lens_recency_tiers():
    recent = el.compute_article_preference(
        {"publication_year": 2025}, {}, lens="most_recent", population="any",
        current_year=2026)
    assert recent["boost"] == el._LENS_BOOST
    midrange = el.compute_article_preference(
        {"publication_year": 2022}, {}, lens="most_recent", population="any",
        current_year=2026)
    assert midrange["boost"] == el._LENS_BOOST_WEAK
    old = el.compute_article_preference(
        {"publication_year": 2010}, {}, lens="most_recent", population="any",
        current_year=2026)
    assert old["boost"] == 0.0


# --- Population matching ------------------------------------------------

def test_pediatrics_population_boost():
    art = {"title": "Outcomes in children", "abstract": "pediatric cohort"}
    pref = el.compute_article_preference(art, {}, lens="clinically_direct", population="pediatrics")
    assert "pediatric_population" in pref["signals"]
    assert pref["boost"] == el._POP_BOOST


def test_pediatrics_does_not_match_adult_only_article():
    art = {"title": "Outcomes in adults", "abstract": "adult cohort"}
    pref = el.compute_article_preference(art, {}, lens="clinically_direct", population="pediatrics")
    assert pref["matched"] is False


def test_lens_plus_population_boost_clamped():
    art = {"title": "Mechanism in children",
           "abstract": "pathophysiology pathway pediatric infant"}
    pref = el.compute_article_preference(art, {}, lens="mechanism", population="pediatrics")
    # 5 (mechanism) + 3 (pediatric) = 8 == MAX_L1_BOOST
    assert pref["boost"] == el.MAX_L1_BOOST
    assert set(pref["signals"]) == {"mechanism_focus", "pediatric_population"}


# --- apply_evidence_lens: boosting + safety + modes -------------------

def _row(aid, score, cap=None, **art):
    art.setdefault("title", "")
    art.setdefault("abstract", "")
    return {"_article": {"id": aid, **art}, "overall_rank_score": float(score),
            "relevance_cap_applied": cap}


def test_soft_boost_reorders_matched_article_up():
    rows = [
        _row("treat", 72, title="treatment trial", abstract="therapy"),
        _row("mech", 70, title="mechanism", abstract="pathophysiology pathway"),
    ]
    diag = el.apply_evidence_lens(rows, lens="mechanism", population="any", mode="soft_boost")
    assert rows[0]["_article"]["id"] == "mech"  # 70+5=75 > 72
    assert diag["rank_changed_count"] == 2
    assert diag["preference_affected_ranking"] is True
    assert diag["articles_boosted"] == 1


def test_caps_dominate_recent_wrong_lane_cannot_outrank_direct():
    rows = [
        _row("direct", 70, cap=None, title="colon cancer cohort", abstract="survival",
             publication_year=2018),
        _row("recent_capped", 28, cap=30, title="unrelated", abstract="x",
             publication_year=2025),
    ]
    rows.sort(key=lambda r: -r["overall_rank_score"])
    diag = el.apply_evidence_lens(rows, lens="most_recent", population="any",
                                  mode="soft_boost", current_year=2026)
    assert rows[0]["_article"]["id"] == "direct"
    # recent_capped got a boost signal but was clamped back to its cap.
    capped = next(r for r in rows if r["_article"]["id"] == "recent_capped")
    assert capped["overall_rank_score"] == 30.0
    assert capped["l1_preference_boost"] == el._LENS_BOOST
    assert diag["safety_caps_dominated_count"] == 1
    assert diag["safety_caps_dominant"] is True


def test_shadow_mode_does_not_change_order_or_scores():
    rows = [
        _row("treat", 72, title="treatment", abstract="therapy"),
        _row("mech", 70, title="mechanism", abstract="pathophysiology"),
    ]
    diag = el.apply_evidence_lens(rows, lens="mechanism", population="any", mode="shadow")
    # order unchanged, scores unchanged
    assert [r["_article"]["id"] for r in rows] == ["treat", "mech"]
    assert rows[1]["overall_rank_score"] == 70.0
    assert diag["rank_changed_count"] == 0
    # but the would-be change is still recorded
    assert diag["would_change_count"] == 1
    assert diag["articles_matched"] == 1


def test_apply_never_hides_or_filters():
    rows = [_row(f"a{i}", 50 + i) for i in range(5)]
    before = {r["_article"]["id"] for r in rows}
    el.apply_evidence_lens(rows, lens="mechanism", population="pediatrics", mode="soft_boost")
    after = {r["_article"]["id"] for r in rows}
    assert before == after  # nothing removed
    assert len(rows) == 5


def test_packet_metadata_projection():
    diag = el.apply_evidence_lens(
        [_row("a", 60, title="mechanism", abstract="pathway")],
        lens="mechanism", population="any", mode="soft_boost")
    meta = el.preference_packet_metadata(diag)
    assert meta["evidence_lens"] == "mechanism"
    assert meta["population_focus"] == "any"
    assert meta["provenance"] == "user_explicit_preference"
    assert meta["safety_caps_dominant"] is True
    assert meta["is_hard_filter"] is False


def test_option_lists():
    lenses = {o["value"] for o in el.list_lens_options()}
    assert lenses == set(el.EVIDENCE_LENSES)
    pops = {o["value"] for o in el.list_population_options()}
    assert pops == set(el.POPULATION_FOCUS_OPTIONS)
