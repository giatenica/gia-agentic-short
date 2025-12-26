"""PDF parsing for the evidence pipeline.

This module parses a PDF stored under `sources/<source_id>/raw/` into a
`parsed.json` payload with page-indexed spans.

Design goals:
- Pure-Python parsing via `pypdf`
- Deterministic output for the same PDF input (within library stability limits)
- No network access

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.evidence.parser import MVPLineBlockParser


@dataclass(frozen=True)
class PdfParseResult:
    parsed_payload: Dict[str, Any]
    page_count: int


def parse_pdf_to_parsed_payload(pdf_path: Path) -> PdfParseResult:
    """Parse a PDF into a parsed payload compatible with `extract_evidence_items`.

    The returned payload is a dict with:
    - blocks: list of {kind, span, text}
      where span includes start_page/end_page and start_line/end_line
    - parser_name, parser_version
    - page_count

    Notes:
    - Line numbers are global across the entire PDF (monotonic) and are primarily
            intended for stable ordering. Empty pages still advance the global line
            counter by 1 to preserve monotonic line numbering across pages.
    - Page numbers are 1-based.

        Raises:
                FileNotFoundError: If the PDF path does not exist.
                ValueError: If the PDF cannot be parsed or is encrypted.
    """
    if not isinstance(pdf_path, Path):
        raise ValueError("pdf_path must be a Path")
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as e:
        raise ValueError(f"Failed to read PDF: {pdf_path}") from e
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {pdf_path}") from e

    if getattr(reader, "is_encrypted", False):
        try:
            decrypted = reader.decrypt("")
        except Exception as e:
            raise ValueError(f"Encrypted PDF is not supported: {pdf_path}") from e
        if not decrypted:
            raise ValueError(f"Encrypted PDF is not supported: {pdf_path}")
    parser = MVPLineBlockParser()

    blocks: List[Dict[str, Any]] = []

    global_line_offset = 1
    page_count = len(reader.pages)

    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.replace("\r\n", "\n").replace("\r", "\n")

        parsed = parser.parse(page_text)

        for b in parsed.blocks:
            start_line = global_line_offset + b.span.start_line - 1
            end_line = global_line_offset + b.span.end_line - 1
            blocks.append(
                {
                    "kind": b.kind,
                    "span": {
                        "start_page": page_idx,
                        "end_page": page_idx,
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                    "text": b.text,
                }
            )

        page_lines = page_text.splitlines()
        # Ensure monotonic line offsets even for empty pages.
        global_line_offset += max(len(page_lines), 1)

    payload: Dict[str, Any] = {
        "blocks": blocks,
        "parser_name": "pypdf_page_parser",
        "parser_version": "mvp-1",
        "page_count": page_count,
    }

    return PdfParseResult(parsed_payload=payload, page_count=page_count)
