"""Tiny helpers for Phase 2D upload tests — synthesize small PDFs in memory."""
from __future__ import annotations

import io

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


def make_text_pdf(
    *,
    title: str = "Sample Article",
    body_lines: list[str] | None = None,
    metadata: dict | None = None,
) -> bytes:
    """Build a small text-only PDF using reportlab.

    Pages contain the body lines. PDF /Title metadata is set to ``title``.
    Optional ``metadata`` extras (author/subject/keywords) are merged in.
    """
    body_lines = body_lines or ["This is the body of the article."]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.setTitle(title)
    if metadata:
        for k, v in metadata.items():
            if k == "author":
                c.setAuthor(v)
            elif k == "subject":
                c.setSubject(v)
            elif k == "keywords":
                c.setKeywords(v)
            elif k == "creator":
                c.setCreator(v)
    y = 720
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, title[:120])
    y -= 28
    c.setFont("Helvetica", 11)
    for line in body_lines:
        if y < 72:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = 720
        # ReportLab can't render unicode easily without fonts; ASCII-only is fine
        # for these tests.
        c.drawString(72, y, line[:110])
        y -= 16
    c.save()
    return buf.getvalue()


def make_pdf_with_annotation_text(text: str) -> bytes:
    """Build a 1-page PDF and stamp a free-text annotation with the given text."""
    base = make_text_pdf(title="Annotated", body_lines=["Body content."])
    reader = PdfReader(io.BytesIO(base))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    page = writer.pages[0]
    annot = DictionaryObject({
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/FreeText"),
        NameObject("/Rect"): ArrayObject([
            NumberObject(50), NumberObject(50), NumberObject(300), NumberObject(80),
        ]),
        NameObject("/Contents"): TextStringObject(text),
        NameObject("/F"): NumberObject(4),
    })
    annots = page.get("/Annots") or ArrayObject()
    annots.append(annot)
    page[NameObject("/Annots")] = annots
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def make_pdf_with_javascript() -> bytes:
    """Build a PDF whose catalog has /OpenAction with /JS."""
    base = make_text_pdf(title="Active", body_lines=["a"])
    reader = PdfReader(io.BytesIO(base))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    root = writer._root_object
    js_action = DictionaryObject({
        NameObject("/Type"): NameObject("/Action"),
        NameObject("/S"): NameObject("/JavaScript"),
        NameObject("/JS"): TextStringObject("app.alert('hi');"),
    })
    root[NameObject("/OpenAction")] = js_action
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def make_pdf_with_embedded_attachment() -> bytes:
    """Build a PDF whose Names tree has /EmbeddedFiles."""
    base = make_text_pdf(title="EmbAtt", body_lines=["a"])
    reader = PdfReader(io.BytesIO(base))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_attachment("evil.txt", b"private patient data")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def make_encrypted_pdf() -> bytes:
    """Build a tiny pdf and encrypt with a password using pypdf.PdfWriter.encrypt."""
    base = make_text_pdf(title="Enc", body_lines=["secret"])
    reader = PdfReader(io.BytesIO(base))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password="x", owner_password="y", use_128bit=True)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def make_apixaban_pdf() -> bytes:
    """Realistic-ish article PDF for matching tests."""
    body = [
        "Apixaban for stroke prevention in non-valvular atrial fibrillation",
        "ABSTRACT",
        "BACKGROUND: We compared apixaban with warfarin for stroke prevention.",
        "METHODS: In this randomized double-blind trial we randomly assigned 18,201 patients with atrial fibrillation.",
        "RESULTS: Apixaban was superior to warfarin in reducing the rate of stroke or systemic embolism.",
        "DISCUSSION: Apixaban reduced major bleeding and mortality.",
        "CONCLUSIONS: Apixaban is superior to warfarin for stroke prevention in non-valvular atrial fibrillation.",
        "REFERENCES: doi: 10.1056/NEJMoa1107039",
        "PMID: 21870978",
        "Published in NEJM 2011 365 981-992",
    ]
    return make_text_pdf(
        title="Apixaban versus warfarin in patients with atrial fibrillation",
        body_lines=body,
        metadata={
            "subject": "doi: 10.1056/NEJMoa1107039",
            "author": "Granger CB et al.",
        },
    )
