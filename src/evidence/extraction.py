"""Evidence extraction helpers.

This module provides deterministic, offline extraction of schema-valid EvidenceItem
objects from a parsed document representation.

Design goals:
- Deterministic output for the same input
- No external dependencies and no network calls
- Strict adherence to the EvidenceItem JSON schema

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.utils.schema_validation import validate_evidence_item


@dataclass(frozen=True)
class NormalizedBlock:
    """Normalized representation of a parsed text block."""

    kind: str
    start_line: Optional[int]
    end_line: Optional[int]
    start_page: Optional[int]
    end_page: Optional[int]
    text: str


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def normalize_parsed_blocks(parsed: Any) -> List[NormalizedBlock]:
    """Normalize a parsed representation into blocks.

    Supported shapes:
    - {"blocks": [ {"kind": ..., "span": {"start_line": .., "end_line": ..}, "text": ...}, ... ]}
    - [ {"kind": ..., "span": {...}, "text": ...}, ... ]

    Args:
        parsed: Parsed document representation.

    Returns:
        List of NormalizedBlock objects.

    Raises:
        ValueError: If parsed cannot be interpreted.
    """

    if isinstance(parsed, dict) and "blocks" in parsed:
        blocks = parsed.get("blocks")
    else:
        blocks = parsed

    if not isinstance(blocks, list):
        raise ValueError("parsed must be a list of blocks or an object with a 'blocks' list")

    normalized: List[NormalizedBlock] = []
    for raw in blocks:
        if not isinstance(raw, dict):
            raise ValueError("each block must be an object")

        kind = str(raw.get("kind") or "")
        text = raw.get("text")
        if not isinstance(text, str):
            raise ValueError("block.text must be a string")

        span = raw.get("span")
        start_line: Optional[int] = None
        end_line: Optional[int] = None
        start_page: Optional[int] = None
        end_page: Optional[int] = None
        if isinstance(span, dict):
            start_line = _coerce_int(span.get("start_line"))
            end_line = _coerce_int(span.get("end_line"))
            start_page = _coerce_int(span.get("start_page"))
            end_page = _coerce_int(span.get("end_page"))

        normalized.append(
            NormalizedBlock(
                kind=kind or "unknown",
                start_line=start_line,
                end_line=end_line,
                start_page=start_page,
                end_page=end_page,
                text=text,
            )
        )

    return normalized


def _stable_evidence_id(parts: Iterable[str]) -> str:
    joined = "\u241f".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"ev_{digest[:16]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


_QUOTE_RE = re.compile(r"(\"[^\"]{5,}\"|“[^”]{5,}”)", re.DOTALL)
_METRIC_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|\b)")
_TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:table)\s+(\d+|[ivxlcdm]+)\s*(?::|\.|-|–|—)\s*",
    re.IGNORECASE,
)
_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:figure|fig\.)\s+(\d+|[ivxlcdm]+)\s*(?::|\.|-|–|—)\s*",
    re.IGNORECASE,
)


def _choose_kind(block_text: str) -> str:
    if _TABLE_CAPTION_RE.search(block_text):
        return "table"
    if _FIGURE_CAPTION_RE.search(block_text):
        return "figure"
    if _QUOTE_RE.search(block_text):
        return "quote"
    if _METRIC_RE.search(block_text):
        return "metric"
    return "claim"


def _default_locator(source_id: str) -> Dict[str, Any]:
    return {"type": "other", "value": source_id}


def _span_dict(
    start_line: Optional[int],
    end_line: Optional[int],
    start_page: Optional[int],
    end_page: Optional[int],
) -> Optional[Dict[str, int]]:
    """Build a schema-compatible span dict from optional line and page pairs.

    Rules:
    - If a pair is provided (both values non-None), it must be valid and ordered.
    - A span may include lines, pages, or both.
    - If neither pair is present, returns None.

    Returns:
        Dict with some of: start_line/end_line and start_page/end_page, or None.
        Returns None when no valid span keys can be produced.
    """
    out: Dict[str, int] = {}

    if start_line is not None and end_line is not None:
        if start_line < 1 or end_line < 1:
            return None
        if end_line < start_line:
            return None
        out.update({"start_line": start_line, "end_line": end_line})

    if start_page is not None and end_page is not None:
        if start_page < 1 or end_page < 1:
            return None
        if end_page < start_page:
            return None
        out.update({"start_page": start_page, "end_page": end_page})

    return out or None


def extract_evidence_items(
    *,
    parsed: Any,
    source_id: str,
    locator: Optional[Dict[str, Any]] = None,
    parser_name: str = "deterministic_extractor",
    parser_version: str = "mvp-1",
    parser_method: str = "heuristic",
    created_at: Optional[str] = None,
    max_items: int = 25,
    min_excerpt_chars: int = 20,
    max_excerpt_chars: int = 500,
) -> List[Dict[str, Any]]:
    """Extract EvidenceItem objects from parsed blocks.

    The extractor is deterministic for the same inputs. It prefers:
    - quote: blocks containing quoted spans
    - metric: blocks containing numeric metrics
    - claim: otherwise

    Args:
        parsed: Parsed representation to extract from.
        source_id: Source identifier.
        locator: Optional EvidenceItem locator; defaults to type=other, value=source_id.
        parser_name: Value for EvidenceItem.parser.name.
        parser_version: Value for EvidenceItem.parser.version.
        parser_method: Value for EvidenceItem.parser.method.
        created_at: Optional ISO 8601 timestamp for EvidenceItem.created_at; if None,
            the current UTC time is used. Pass a fixed value when you need reproducible
            evidence timestamps across runs.
        max_items: Max number of evidence items to emit.
        min_excerpt_chars: Skip excerpts shorter than this.
        max_excerpt_chars: Truncate excerpts to this length.

    Returns:
        List of EvidenceItem dicts.

    Raises:
        ValueError: If inputs are invalid.
    """

    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a non-empty string")

    if max_items < 0:
        raise ValueError("max_items must be >= 0")

    if min_excerpt_chars < 0:
        raise ValueError("min_excerpt_chars must be >= 0")

    if max_excerpt_chars <= 0:
        raise ValueError("max_excerpt_chars must be positive")

    if locator is None:
        locator = _default_locator(source_id)

    blocks = normalize_parsed_blocks(parsed)

    items: List[Dict[str, Any]] = []
    created_at_value = created_at or _utc_now_iso()

    for idx, block in enumerate(blocks):
        if len(items) >= max_items:
            break

        if block.kind in {"heading", "code"}:
            continue

        excerpt = block.text.strip()
        if len(excerpt) < min_excerpt_chars:
            continue

        excerpt = _truncate(excerpt, max_excerpt_chars)

        kind = _choose_kind(excerpt)

        span = _span_dict(block.start_line, block.end_line, block.start_page, block.end_page)
        loc = dict(locator)
        if span is not None:
            loc = {**loc, "span": span}

        evidence_id = _stable_evidence_id(
            [
                source_id,
                str(idx),
                kind,
                str(block.start_line or ""),
                str(block.end_line or ""),
                str(block.start_page or ""),
                str(block.end_page or ""),
                excerpt,
            ]
        )

        item: Dict[str, Any] = {
            "schema_version": "1.0",
            "evidence_id": evidence_id,
            "source_id": source_id,
            "kind": kind,
            "locator": loc,
            "excerpt": excerpt,
            "created_at": created_at_value,
            "parser": {
                "name": parser_name,
                "version": parser_version,
                "method": parser_method,
            },
            "metadata": {
                "block_kind": block.kind,
            },
        }

        validate_evidence_item(item)
        items.append(item)

    return items
