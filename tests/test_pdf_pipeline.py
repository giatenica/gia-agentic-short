"""Tests for PDF parsing with page locators.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json

import pytest

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

    # parsed.json blocks must include page locators.
    assert any(
        isinstance(b, dict)
        and isinstance(b.get("span"), dict)
        and b["span"].get("start_page") == 1
        and b["span"].get("end_page") == 1
        for b in blocks
    )

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
