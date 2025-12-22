"""
Tests for Parser Interface and MVP Parser
========================================

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import sys
from pathlib import Path
import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.evidence.parser import MVPLineBlockParser


@pytest.mark.unit
def test_parser_splits_paragraphs_and_tracks_line_spans():
    text = "Line 1\nLine 2\n\nLine 4\n"
    doc = MVPLineBlockParser().parse(text)
    assert len(doc.blocks) == 2

    b1, b2 = doc.blocks
    assert b1.kind == "paragraph"
    assert b1.span.start_line == 1
    assert b1.span.end_line == 2
    assert "Line 1" in b1.text

    assert b2.kind == "paragraph"
    assert b2.span.start_line == 4
    assert b2.span.end_line == 4
    assert b2.text.strip() == "Line 4"


@pytest.mark.unit
def test_parser_detects_markdown_heading_and_code_blocks():
    text = "# Title\n\nPara\n\n```python\nprint('x')\n```\n"
    doc = MVPLineBlockParser().parse(text)
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["heading", "paragraph", "code"]

    heading = doc.blocks[0]
    assert heading.span.start_line == 1
    assert heading.span.end_line == 1

    code = doc.blocks[-1]
    assert code.span.start_line == 5
    assert code.span.end_line == 7
    assert "print('x')" in code.text


@pytest.mark.unit
def test_parser_detects_tex_section_headings():
    text = "\\section{Intro}\nLine\n\n\\subsection{Details}\nMore\n"
    doc = MVPLineBlockParser().parse(text)
    kinds = [b.kind for b in doc.blocks]
    # heading, paragraph, heading, paragraph
    assert kinds == ["heading", "paragraph", "heading", "paragraph"]
