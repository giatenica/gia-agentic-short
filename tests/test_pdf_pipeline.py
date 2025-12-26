"""Tests for PDF parsing with page locators.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import pytest
from pathlib import Path

from pypdf import PdfWriter

from src.evidence.pdf_parser import parse_pdf_to_parsed_payload
from src.evidence.pipeline import _choose_raw_pdf_path
from src.evidence.pipeline import run_pdf_evidence_pipeline_for_source
from src.evidence.store import EvidenceStore
from src.utils.schema_validation import validate_evidence_item


def _escape_pdf_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", " ")
    )


def _build_synthetic_pdf(page_texts: list[str]) -> bytes:
    """Build a small PDF with N pages and simple text.

    This is a deterministic fixture generator for unit tests.
    """

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

    n_pages = len(page_texts)
    if n_pages < 1:
        raise ValueError("need at least one page")

    catalog_id = 1
    pages_id = 2
    first_page_id = 3
    first_content_id = first_page_id + n_pages
    font_id = first_content_id + n_pages

    objects: list[bytes] = []

    # 1: Catalog
    objects.append(f"{catalog_id} 0 obj\n<< /Type /Catalog /Pages {pages_id} 0 R >>\nendobj\n".encode("utf-8"))

    # 2: Pages
    kids = " ".join(f"{first_page_id + i} 0 R" for i in range(n_pages))
    objects.append(
        f"{pages_id} 0 obj\n<< /Type /Pages /Kids [ {kids} ] /Count {n_pages} >>\nendobj\n".encode("utf-8")
    )

    # Page + content objects
    for i, text in enumerate(page_texts):
        page_id = first_page_id + i
        content_id = first_content_id + i

        page_obj = (
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>\n"
            f"endobj\n"
        ).encode("utf-8")
        objects.append(page_obj)

        escaped = _escape_pdf_text(text)
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET\n".encode("utf-8")
        content_obj = (
            f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("utf-8")
            + stream
            + b"endstream\nendobj\n"
        )
        objects.append(content_obj)

    # Font
    objects.append(
        f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("utf-8")
    )

    # Assemble with xref
    offsets: list[int] = [0]
    body = bytearray()
    body.extend(header)

    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)

    xref_offset = len(body)
    size = len(offsets)

    body.extend(f"xref\n0 {size}\n".encode("utf-8"))
    body.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        body.extend(f"{off:010d} 00000 n \n".encode("utf-8"))

    body.extend(
        (
            f"trailer\n<< /Size {size} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("utf-8")
    )

    return bytes(body)


@pytest.mark.unit
def test_pdf_pipeline_writes_parsed_with_page_locators_and_evidence_items(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    source_id = "src_pdf"
    sp = store.ensure_source_layout(source_id)

    pdf_bytes = _build_synthetic_pdf(
        [
            "Table 1: Summary statistics for the sample. This is long enough.",
            "Figure 2: Another caption-like line with 42% in it.",
        ]
    )

    raw_pdf_path = sp.raw_dir / "paper.pdf"
    raw_pdf_path.write_bytes(pdf_bytes)

    summary = run_pdf_evidence_pipeline_for_source(
        project_folder=str(temp_project_folder),
        source_id=source_id,
        raw_pdf_filename="paper.pdf",
        max_items=25,
        created_at="2025-01-01T00:00:00+00:00",
    )

    assert summary["source_id"] == source_id
    assert summary["parsed_blocks_count"] > 0

    parsed = store.read_parsed(source_id)
    assert isinstance(parsed, dict)
    blocks = parsed.get("blocks")
    assert isinstance(blocks, list)
    assert blocks

    # parsed.json blocks must include page locators for both pages.
    pages = {
        b.get("span", {}).get("start_page")
        for b in blocks
        if isinstance(b, dict) and isinstance(b.get("span"), dict)
    }
    assert 1 in pages
    assert 2 in pages

    items = store.read_evidence_items(source_id, validate=True)
    assert isinstance(items, list)
    assert items

    # At least one EvidenceItem must carry a page locator.
    assert any(
        isinstance(i.get("locator"), dict)
        and isinstance(i["locator"].get("span"), dict)
        and isinstance(i["locator"]["span"].get("start_page"), int)
        for i in items
    )

    for item in items:
        validate_evidence_item(item)


@pytest.mark.unit
def test_choose_raw_pdf_path_prefers_matching_filename(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    sp = store.ensure_source_layout("src_pdf")

    a = sp.raw_dir / "a.pdf"
    b = sp.raw_dir / "b.pdf"
    a.write_bytes(b"%PDF-1.4\n%fixture\n")
    b.write_bytes(b"%PDF-1.4\n%fixture\n")

    chosen = _choose_raw_pdf_path(sp.raw_dir, preferred_filename="b.pdf")
    assert chosen.name == "b.pdf"


@pytest.mark.unit
def test_choose_raw_pdf_path_falls_back_when_preferred_missing_or_not_pdf(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    sp = store.ensure_source_layout("src_pdf")

    (sp.raw_dir / "a.pdf").write_bytes(b"%PDF-1.4\n%fixture\n")
    (sp.raw_dir / "b.pdf").write_bytes(b"%PDF-1.4\n%fixture\n")
    (sp.raw_dir / "not_pdf.txt").write_text("nope", encoding="utf-8")

    chosen_missing = _choose_raw_pdf_path(sp.raw_dir, preferred_filename="missing.pdf")
    assert chosen_missing.name == "a.pdf"

    chosen_wrong_ext = _choose_raw_pdf_path(sp.raw_dir, preferred_filename="not_pdf.txt")
    assert chosen_wrong_ext.name == "a.pdf"


@pytest.mark.unit
def test_choose_raw_pdf_path_rejects_path_traversal(temp_project_folder, tmp_path):
    store = EvidenceStore(str(temp_project_folder))
    sp = store.ensure_source_layout("src_pdf")

    (sp.raw_dir / "a.pdf").write_bytes(b"%PDF-1.4\n%fixture\n")
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4\n%fixture\n")

    chosen = _choose_raw_pdf_path(sp.raw_dir, preferred_filename="../outside.pdf")
    assert chosen.is_relative_to(sp.raw_dir.resolve())
    assert chosen.name == "a.pdf"


@pytest.mark.unit
def test_parse_pdf_to_parsed_payload_raises_on_corrupt_pdf(tmp_path):
    pdf_path = tmp_path / "bad.pdf"
    pdf_path.write_bytes(b"not a pdf")

    with pytest.raises(ValueError):
        parse_pdf_to_parsed_payload(pdf_path)


@pytest.mark.unit
def test_parse_pdf_to_parsed_payload_raises_on_encrypted_pdf(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt(user_password="secret")

    pdf_path = tmp_path / "encrypted.pdf"
    with pdf_path.open("wb") as f:
        writer.write(f)

    with pytest.raises(ValueError):
        parse_pdf_to_parsed_payload(pdf_path)
