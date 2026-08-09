"""Phase 4A — section parser (Task 11 / D)."""
from __future__ import annotations

from xml.etree import ElementTree as ET


from services.oa_fetcher.section_parser import (
    extract_bioc_license,
    extract_jats_license,
    parse_bioc,
    parse_jats,
)
from tests.fixtures_oa_xml import JATS_SAMPLE, JATS_TABLES_FIGS, BIOC_SAMPLE


def _section(sections, name):
    for s in sections:
        if s.section_name == name:
            return s
    return None


def test_jats_extracts_abstract_methods_results_conclusion():
    root = ET.fromstring(JATS_SAMPLE)
    sections = parse_jats(root)
    names = {s.section_name for s in sections}
    assert "abstract" in names
    assert "methods" in names
    assert "results" in names
    assert "conclusion" in names


def test_jats_extracts_tables_with_captions():
    root = ET.fromstring(JATS_TABLES_FIGS)
    sections = parse_jats(root)
    tables = _section(sections, "tables")
    assert tables is not None
    assert "Primary outcome by group" in tables.section_text
    assert "0.79" in tables.section_text  # cell value present


def test_jats_excludes_references_from_include_in_summary():
    root = ET.fromstring(JATS_SAMPLE)
    sections = parse_jats(root)
    refs = _section(sections, "references")
    # Either absent or include_in_summary=False — both pass the safety bar.
    if refs is not None:
        assert refs.include_in_summary is False


def test_jats_excludes_author_affiliations():
    root = ET.fromstring(JATS_SAMPLE)
    sections = parse_jats(root)
    for s in sections:
        if s.section_name in {"author_info", "acknowledgments"}:
            assert s.include_in_summary is False


def test_jats_excludes_acknowledgments():
    root = ET.fromstring(JATS_SAMPLE)
    sections = parse_jats(root)
    ack = _section(sections, "acknowledgments")
    if ack is not None:
        assert ack.include_in_summary is False


def test_bioc_extracts_sections():
    root = ET.fromstring(BIOC_SAMPLE)
    sections = parse_bioc(root)
    names = {s.section_name for s in sections}
    assert {"abstract", "methods", "results", "discussion", "conclusion"}.issubset(names)


def test_bioc_excludes_references():
    root = ET.fromstring(BIOC_SAMPLE)
    sections = parse_bioc(root)
    refs = _section(sections, "references")
    if refs is not None:
        assert refs.include_in_summary is False


def test_token_count_populated():
    root = ET.fromstring(JATS_SAMPLE)
    sections = parse_jats(root)
    assert any(s.token_count > 0 for s in sections)


def test_evidence_text_level_oa_full_text():
    """The parser returns generic ParsedSection rows; the *fetcher* attaches the
    `evidence_text_level=oa_full_text` label. Verify the level is applied by
    the service path indirectly: the parser DOES emit include_in_summary
    flags consistent with the oa_full_text level (abstract/methods/results
    all True).
    """
    root = ET.fromstring(JATS_SAMPLE)
    sections = parse_jats(root)
    for name in ("abstract", "methods", "results", "conclusion"):
        s = _section(sections, name)
        if s is not None:
            assert s.include_in_summary is True


def test_dropped_short_sections_under_50_chars():
    """Sections with text shorter than 50 chars after cleaning are dropped."""
    tiny = """<article>
      <body>
        <sec sec-type="methods"><title>Methods</title><p>too short.</p></sec>
      </body>
    </article>"""
    root = ET.fromstring(tiny)
    sections = parse_jats(root)
    assert _section(sections, "methods") is None


def test_inline_citation_markers_stripped():
    root = ET.fromstring(JATS_SAMPLE)
    sections = parse_jats(root)
    abs_sec = _section(sections, "abstract")
    assert abs_sec is not None
    # The "[1]" marker in the JATS_SAMPLE abstract should be stripped.
    assert "[1]" not in abs_sec.section_text
    # The "(Smith et al, 2011)" marker in the results section should be stripped.
    results = _section(sections, "results")
    if results:
        assert "(Smith et al, 2011)" not in results.section_text


def test_extract_jats_license_from_permissions():
    root = ET.fromstring(JATS_SAMPLE)
    lic = extract_jats_license(root)
    assert lic is not None
    assert "creativecommons.org/licenses/by" in lic.lower()


def test_extract_bioc_license_from_infon():
    root = ET.fromstring(BIOC_SAMPLE)
    lic = extract_bioc_license(root)
    assert lic is not None
    assert "creativecommons.org/licenses/by" in lic.lower()
