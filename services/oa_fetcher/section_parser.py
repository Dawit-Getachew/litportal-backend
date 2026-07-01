"""JATS XML + BioC XML → ParsedSection list.

Shared by `europe_pmc_oa.py` (JATS) and `pmc_oa.py` (BioC). Pure-Python, no
network, no LLM. Deterministic.

`include_in_summary` defaults follow Phase 2D conventions:
  TRUE  : abstract, methods, results, discussion, limitations, conclusion,
          tables, figure_captions, body
  FALSE : title, introduction, background, supplementary, references,
          acknowledgments, author_info, copyright, other
"""
from __future__ import annotations

import os
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict
from xml.etree import ElementTree as ET


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


MAX_EXTRACT_CHARS_PER_ARTICLE = _env_int("MAX_EXTRACT_CHARS_PER_ARTICLE", 8000)
MIN_SECTION_CHARS = 50


class ParsedSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    section_name: str
    section_text: str
    include_in_summary: bool
    token_count: int


_INCLUDE_TRUE = {
    "abstract", "methods", "results", "discussion", "limitations",
    "conclusion", "tables", "figure_captions", "body",
}


def _include_default(section_name: str) -> bool:
    return section_name in _INCLUDE_TRUE


# ----------------------------------------------------------------------------
# JATS (Europe PMC fullTextXML)
# ----------------------------------------------------------------------------


_JATS_TYPE_MAP = [
    (("intro", "introduction", "background"), "introduction"),
    (("methods", "materials and methods", "study design", "patients and methods"), "methods"),
    (("results", "findings"), "results"),
    (("discussion",), "discussion"),
    (("conclusion", "conclusions"), "conclusion"),
    (("limitation", "limitations"), "limitations"),
    (("supplementary",), "supplementary"),
]


def _jats_section_name_from(sec: ET.Element) -> str:
    sec_type = (sec.get("sec-type") or "").strip().lower()
    title_text = ""
    title_el = sec.find("title")
    if title_el is not None:
        title_text = _text_only(title_el).strip().lower()
    for keys, mapped in _JATS_TYPE_MAP:
        for k in keys:
            if k in sec_type or k == title_text:
                return mapped
    # Heuristic: a 'background' section is sometimes labeled introduction.
    return "body"


def _text_only(el: ET.Element) -> str:
    """Recursively gather visible text from an element, skipping refs/xrefs."""
    if el is None:
        return ""
    parts: list[str] = []

    def walk(e: ET.Element) -> None:
        tag = (e.tag or "").lower().rsplit("}", 1)[-1]
        if tag in {"xref", "ref", "label", "fn", "graphic", "media"}:
            return
        if e.text:
            parts.append(e.text)
        for child in list(e):
            walk(child)
            if child.tail:
                parts.append(child.tail)
    walk(el)
    return _clean(" ".join(parts))


_INLINE_CITATION_RX = [
    re.compile(r"\[\s*\d+(\s*[,–-]\s*\d+)*\s*\]"),       # [1], [1-3], [1,2]
    re.compile(r"\(\s*[A-Z][a-zA-Z'\-]+\s*(et al\.?)?,?\s+\d{4}[a-z]?\s*\)"),
    re.compile(r"\^\d+(\^|\b)"),
]


def _clean(text: str) -> str:
    if not text:
        return ""
    s = text
    for rx in _INLINE_CITATION_RX:
        s = rx.sub("", s)
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_ns(root: ET.Element) -> None:
    """Strip XML namespaces in place (BioC/JATS both use unprefixed tags after this)."""
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag.startswith("{"):
            el.tag = el.tag.split("}", 1)[1]


def _cap_text(text: str) -> str:
    cap = MAX_EXTRACT_CHARS_PER_ARTICLE // 3
    if len(text) <= cap:
        return text
    snip = text[:cap]
    # back off to last sentence end where possible
    cut = max(snip.rfind(". "), snip.rfind("? "), snip.rfind("! "))
    if cut > cap // 2:
        return snip[: cut + 1]
    return snip


def _token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()) * 4 // 3)


def _add_section(out: list[ParsedSection], section_name: str, text: str) -> None:
    text = _clean(text)
    if len(text) < MIN_SECTION_CHARS:
        return
    text = _cap_text(text)
    out.append(ParsedSection(
        section_name=section_name,
        section_text=text,
        include_in_summary=_include_default(section_name),
        token_count=_token_count(text),
    ))


def parse_jats(xml_root: ET.Element) -> list[ParsedSection]:
    """Parse Europe PMC fullTextXML JATS into a list of ParsedSection records."""
    _strip_ns(xml_root)
    out: list[ParsedSection] = []

    # Title
    article_title = xml_root.find(".//front/article-meta/title-group/article-title")
    if article_title is not None:
        _add_section(out, "title", _text_only(article_title))

    # Abstract (top-level — concatenate paragraphs)
    for abs_el in xml_root.findall(".//front/article-meta/abstract"):
        text = " ".join(_text_only(p) for p in abs_el.findall(".//p")) or _text_only(abs_el)
        _add_section(out, "abstract", text)

    # Body sections
    body = xml_root.find("body")
    if body is not None:
        # Iterate strictly top-level <sec> children to avoid duplicating nested
        # sections.
        for sec in list(body):
            if (sec.tag or "").lower() != "sec":
                continue
            name = _jats_section_name_from(sec)
            text = " ".join(_text_only(p) for p in sec.findall(".//p")) or _text_only(sec)
            _add_section(out, name, text)
        # Fallback: if body had no sec children, collect <p> directly.
        if not any((c.tag or "").lower() == "sec" for c in list(body)):
            text = " ".join(_text_only(p) for p in body.findall(".//p"))
            _add_section(out, "body", text)

    # Tables (caption + flattened cells)
    tables_text_parts: list[str] = []
    for tw in xml_root.findall(".//table-wrap"):
        cap = tw.find("caption")
        if cap is not None:
            tables_text_parts.append(_text_only(cap))
        for cell in tw.findall(".//th") + tw.findall(".//td"):
            cell_text = _text_only(cell)
            if cell_text:
                tables_text_parts.append(cell_text)
    if tables_text_parts:
        _add_section(out, "tables", " | ".join(tables_text_parts))

    # Figure captions
    fig_caps: list[str] = []
    for fig in xml_root.findall(".//fig"):
        cap = fig.find("caption")
        if cap is not None:
            fig_caps.append(_text_only(cap))
    if fig_caps:
        _add_section(out, "figure_captions", " ".join(fig_caps))

    # Acknowledgments + author info — emit with include_in_summary=False
    for ack in xml_root.findall(".//ack"):
        _add_section(out, "acknowledgments", _text_only(ack))
    for au in xml_root.findall(".//author-notes"):
        _add_section(out, "author_info", _text_only(au))

    # References — emit as a single block with include=False
    refs = xml_root.findall(".//ref-list")
    if refs:
        # Just emit a tiny placeholder so we exercise the "exclude from summary" path.
        text = " ".join(_text_only(r) for r in refs)
        _add_section(out, "references", text[:300])

    return out


# ----------------------------------------------------------------------------
# BioC (NCBI BioC_xml/{PMCID}/unicode)
# ----------------------------------------------------------------------------


_BIOC_TYPE_MAP = {
    "title": "title", "title_1": "title", "article-title": "title",
    "abstract": "abstract", "abstract_title_1": "abstract",
    "intro": "introduction", "introduction": "introduction",
    "methods": "methods", "materials and methods": "methods",
    "results": "results",
    "discuss": "discussion", "discussion": "discussion",
    "concl": "conclusion", "conclusion": "conclusion",
    "table_caption": "tables", "table": "tables",
    "fig_caption": "figure_captions", "fig": "figure_captions",
    "ref": "references", "references": "references", "ref_list": "references",
    "auth_cont": "acknowledgments", "ack": "acknowledgments",
    "acknowledgments": "acknowledgments", "footnote": "acknowledgments",
    "author_info": "author_info",
}


def _bioc_section_name(passage: ET.Element) -> str:
    type_val, sect_val = "", ""
    for infon in passage.findall("infon"):
        key = (infon.get("key") or "").lower()
        val = (infon.text or "").strip().lower()
        if key == "type":
            type_val = val
        elif key == "section_type":
            sect_val = val
    for v in (sect_val, type_val):
        if not v:
            continue
        if v in _BIOC_TYPE_MAP:
            return _BIOC_TYPE_MAP[v]
        # fuzzy: front matter / paragraph in body
        if v in {"paragraph", "para"}:
            return "body"
    return "body"


def parse_bioc(xml_root: ET.Element) -> list[ParsedSection]:
    _strip_ns(xml_root)
    # Group passages by mapped section name and concatenate their text.
    grouped: dict[str, list[str]] = {}
    for passage in xml_root.findall(".//passage"):
        name = _bioc_section_name(passage)
        text_el = passage.find("text")
        text = _text_only(text_el) if text_el is not None else ""
        if not text:
            continue
        grouped.setdefault(name, []).append(text)
    out: list[ParsedSection] = []
    # Maintain a stable, summary-friendly ordering.
    order = [
        "title", "abstract", "introduction", "methods", "results",
        "discussion", "limitations", "conclusion", "tables",
        "figure_captions", "body", "references", "acknowledgments",
        "author_info", "supplementary",
    ]
    for name in order:
        if name in grouped:
            _add_section(out, name, " ".join(grouped[name]))
    # Any leftovers we did not anticipate.
    for name, parts in grouped.items():
        if name not in order:
            _add_section(out, name, " ".join(parts))
    return out


# ----------------------------------------------------------------------------
# License extraction helpers (called by the fetcher modules)
# ----------------------------------------------------------------------------


def extract_jats_license(xml_root: ET.Element) -> Optional[str]:
    """Pull a CC license string out of JATS `<permissions>` block."""
    _strip_ns(xml_root)
    perms = xml_root.find(".//permissions")
    if perms is None:
        return None
    lic = perms.find("license")
    if lic is not None:
        ltype = lic.get("license-type") or ""
        href = ""
        # Look for <license-p> or <ext-link xlink:href> or a direct attribute.
        for el in lic.iter():
            for attr_name, attr_val in el.attrib.items():
                if "href" in attr_name.lower() and "creativecommons" in str(attr_val).lower():
                    href = str(attr_val)
                    break
            if href:
                break
        if href:
            return href
        if ltype:
            return ltype
    cs = perms.find("copyright-statement")
    if cs is not None:
        return _text_only(cs)
    return None


def extract_bioc_license(xml_root: ET.Element) -> Optional[str]:
    _strip_ns(xml_root)
    for infon in xml_root.findall(".//infon"):
        if (infon.get("key") or "").lower() in {"license", "license_url"}:
            v = (infon.text or "").strip()
            if v:
                return v
    return None
