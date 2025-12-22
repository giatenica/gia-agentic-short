""" 
Parser Interface and MVP Parser
==============================
Transforms raw source text into location-indexed blocks.

This module is intentionally minimal for Sprint 1:
- A small interface for parsers
- A basic parser that splits text into blocks and assigns line spans

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class TextSpan:
    """1-based line span in the original document."""

    start_line: int
    end_line: int


@dataclass(frozen=True)
class TextBlock:
    """A location-indexed block of text."""

    kind: str
    span: TextSpan
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    """Parsed representation of a document."""

    blocks: List[TextBlock]
    parser_name: str
    parser_version: str = "mvp-1"


class DocumentParser(Protocol):
    """Parser interface used by the Evidence pipeline."""

    name: str
    version: str

    def parse(self, text: str) -> ParsedDocument:
        ...


class MVPLineBlockParser:
    """MVP parser that creates blocks with line spans.

    Heuristics:
    - Markdown fenced code blocks (``` ... ```) become kind='code'
    - Lines that look like headings (#, ##, \section{...}, \subsection{...}) become kind='heading'
    - Other content is grouped into paragraph blocks split by blank lines
    """

    name = "mvp_line_block_parser"
    version = "mvp-1"

    def parse(self, text: str) -> ParsedDocument:
        lines = text.splitlines()
        blocks: List[TextBlock] = []

        def flush_paragraph(paragraph_lines: List[str], start_line: Optional[int], end_line: Optional[int]):
            if not paragraph_lines or start_line is None or end_line is None:
                return
            if end_line < start_line:
                return
            paragraph_text = "\n".join(paragraph_lines).strip("\n")
            if not paragraph_text.strip():
                return
            blocks.append(
                TextBlock(kind="paragraph", span=TextSpan(start_line=start_line, end_line=end_line), text=paragraph_text)
            )

        def flush_paragraph_until(end_line: int) -> None:
            nonlocal paragraph_lines, paragraph_start
            if paragraph_start is None or not paragraph_lines:
                paragraph_lines = []
                paragraph_start = None
                return
            flush_paragraph(paragraph_lines, paragraph_start, end_line)
            paragraph_lines = []
            paragraph_start = None

        def is_tex_heading_line(stripped: str) -> bool:
            tex_prefixes = (
                "\\chapter{",
                "\\section{",
                "\\subsection{",
                "\\subsubsection{",
                "\\paragraph{",
                "\\subparagraph{",
            )
            return stripped.startswith(tex_prefixes)

        in_code = False
        code_start_line: Optional[int] = None
        code_lines: List[str] = []

        paragraph_start: Optional[int] = None
        paragraph_lines: List[str] = []

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Markdown fenced code blocks
            if stripped.startswith("```"):
                if not in_code:
                    # Starting code block
                    flush_paragraph_until(idx - 1)

                    in_code = True
                    code_start_line = idx
                    code_lines = [line]
                else:
                    # Ending code block
                    code_lines.append(line)
                    blocks.append(
                        TextBlock(
                            kind="code",
                            span=TextSpan(start_line=code_start_line or idx, end_line=idx),
                            text="\n".join(code_lines),
                        )
                    )
                    in_code = False
                    code_start_line = None
                    code_lines = []
                continue

            if in_code:
                code_lines.append(line)
                continue

            # Blank line splits paragraphs
            if stripped == "":
                flush_paragraph_until(idx - 1)
                continue

            # Heading detection
            is_md_heading = stripped.startswith("#") and stripped.lstrip("#").startswith(" ")
            is_tex_heading = is_tex_heading_line(stripped)
            if is_md_heading or is_tex_heading:
                flush_paragraph_until(idx - 1)

                blocks.append(TextBlock(kind="heading", span=TextSpan(start_line=idx, end_line=idx), text=line))
                continue

            # Normal paragraph line
            if paragraph_start is None:
                paragraph_start = idx
            paragraph_lines.append(line)

        # Flush tail
        if in_code and code_start_line is not None and code_lines:
            blocks.append(
                TextBlock(
                    kind="code",
                    span=TextSpan(start_line=code_start_line, end_line=len(lines) if lines else 1),
                    text="\n".join(code_lines),
                )
            )
        else:
            flush_paragraph(paragraph_lines, paragraph_start, len(lines) if lines else None)

        return ParsedDocument(blocks=blocks, parser_name=self.name, parser_version=self.version)


def parse_to_blocks(text: str) -> List[TextBlock]:
    """Convenience helper returning blocks using the MVP parser."""
    return MVPLineBlockParser().parse(text).blocks
