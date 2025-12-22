"""
Tests for Parser Interface and MVP Parser
========================================

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import pytest

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


@pytest.mark.unit
def test_parser_handles_empty_input():
    doc = MVPLineBlockParser().parse("")
    assert doc.blocks == []


@pytest.mark.unit
def test_parser_handles_unclosed_code_block_as_code_to_eof():
    text = "Para\n\n```python\nprint('x')\n"
    doc = MVPLineBlockParser().parse(text)
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["paragraph", "code"]
    code = doc.blocks[-1]
    assert "print('x')" in code.text
    assert code.span.start_line == 3
    assert code.span.end_line == 4


@pytest.mark.unit
def test_parser_consecutive_headings_do_not_create_empty_paragraphs():
    text = "# A\n# B\n\nPara\n"
    doc = MVPLineBlockParser().parse(text)
    kinds = [b.kind for b in doc.blocks]
    assert kinds == ["heading", "heading", "paragraph"]


@pytest.mark.unit
def test_parser_requires_space_after_hash_for_markdown_heading():
    text = "####NoSpace\nPara\n"
    doc = MVPLineBlockParser().parse(text)
    assert [b.kind for b in doc.blocks] == ["paragraph"]
    assert "####NoSpace" in doc.blocks[0].text


@pytest.mark.unit
def test_parser_detects_additional_tex_headings():
    text = "\\subsubsection{A}\nX\n\\paragraph{B}\nY\n"
    doc = MVPLineBlockParser().parse(text)
    assert [b.kind for b in doc.blocks] == ["heading", "paragraph", "heading", "paragraph"]
