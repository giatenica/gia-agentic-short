"""Local evidence pipeline.

This module provides an optional, offline stage to:
- discover local project sources
- optionally ingest/copy them into sources/<source_id>/raw/
- parse text into location-indexed blocks
- write sources/<source_id>/parsed.json
- extract schema-valid EvidenceItems
- write sources/<source_id>/evidence.json (and optionally append to the JSONL ledger)

It is designed to be used by workflows as a best-effort pre-processing step.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from loguru import logger

from src.evidence.parser import MVPLineBlockParser
from src.evidence.source_fetcher import SourceFetcherTool, LocalSource
from src.evidence.store import EvidenceStore
from src.evidence.extraction import extract_evidence_items
from src.tracing import get_tracer, safe_set_span_attributes


_tracer = get_tracer("evidence-pipeline")


@dataclass(frozen=True)
class EvidencePipelineConfig:
    enabled: bool = False
    max_sources: int = 50
    max_chars_per_source: int = 200_000
    ingest_sources: bool = True
    append_to_ledger: bool = False
    max_items_per_source: int = 25

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "EvidencePipelineConfig":
        raw = context.get("evidence_pipeline")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", False))
        max_sources = int(raw.get("max_sources", 50))
        max_chars_per_source = int(raw.get("max_chars_per_source", 200_000))
        ingest_sources = bool(raw.get("ingest_sources", True))
        append_to_ledger = bool(raw.get("append_to_ledger", False))
        max_items_per_source = int(raw.get("max_items_per_source", 25))

        if max_sources < 0:
            max_sources = 0
        if max_chars_per_source < 0:
            max_chars_per_source = 0
        if max_items_per_source < 0:
            max_items_per_source = 0

        return cls(
            enabled=enabled,
            max_sources=max_sources,
            max_chars_per_source=max_chars_per_source,
            ingest_sources=ingest_sources,
            append_to_ledger=append_to_ledger,
            max_items_per_source=max_items_per_source,
        )


def _parsed_payload_from_text(text: str) -> Dict[str, Any]:
    parser = MVPLineBlockParser()
    parsed = parser.parse(text)

    blocks = []
    for b in parsed.blocks:
        blocks.append(
            {
                "kind": b.kind,
                "span": {"start_line": b.span.start_line, "end_line": b.span.end_line},
                "text": b.text,
            }
        )

    return {
        "blocks": blocks,
        "parser_name": parsed.parser_name,
        "parser_version": parsed.parser_version,
    }


def run_local_evidence_pipeline(
    *,
    project_folder: str,
    config: EvidencePipelineConfig,
) -> Dict[str, Any]:
    """Run local evidence pre-processing.

    Returns:
        Summary dict with keys:
        - source_ids
        - discovered_count
        - processed_count
        - per_source (list)
        - errors
    """

    store = EvidenceStore(project_folder)
    fetcher = SourceFetcherTool(project_folder)

    discovered: List[LocalSource] = fetcher.discover_sources()
    discovered = (
        discovered[: config.max_sources]
        if config.max_sources > 0
        else discovered
    )

    source_ids: List[str] = []
    errors: List[str] = []
    processed = 0
    per_source: List[Dict[str, Any]] = []

    for src in discovered:
        with _tracer.start_as_current_span("evidence_source") as span:
            safe_set_span_attributes(
                span,
                {
                    "source_id": src.source_id,
                    "relative_path": src.relative_path,
                    "ingest_sources": bool(config.ingest_sources),
                    "max_chars_per_source": int(config.max_chars_per_source),
                    "max_items_per_source": int(config.max_items_per_source),
                },
            )

            try:
                if config.ingest_sources:
                    fetcher.ingest_source(src)

                text = fetcher.load_text(src, max_chars=config.max_chars_per_source)
                parsed_payload = _parsed_payload_from_text(text)

                parsed_blocks_count = 0
                blocks = parsed_payload.get("blocks")
                if isinstance(blocks, list):
                    parsed_blocks_count = len(blocks)

                store.write_parsed(src.source_id, parsed_payload)

                items = extract_evidence_items(
                    parsed=parsed_payload,
                    source_id=src.source_id,
                    created_at=src.created_at,
                    max_items=config.max_items_per_source,
                )

                store.write_evidence_items(src.source_id, items)

                evidence_items_count = len(items) if isinstance(items, list) else 0
                if config.append_to_ledger and items:
                    store.append_many(items)

                source_ids.append(src.source_id)
                processed += 1

                per_source.append(
                    {
                        "source_id": src.source_id,
                        "relative_path": src.relative_path,
                        "parsed_blocks_count": parsed_blocks_count,
                        "evidence_items_count": evidence_items_count,
                    }
                )

                safe_set_span_attributes(
                    span,
                    {
                        "parsed_blocks_count": parsed_blocks_count,
                        "evidence_items_count": evidence_items_count,
                        "success": True,
                    },
                )

            except ValueError as e:
                # Unsupported formats, path validation errors, schema errors.
                msg = f"Evidence pipeline skipped/failed for {src.relative_path}: {e}"
                logger.warning(msg)
                errors.append(msg)
                safe_set_span_attributes(span, {"success": False, "error_type": "ValueError"})
            except OSError as e:
                msg = f"Evidence pipeline IO error for {src.relative_path}: {e}"
                logger.warning(msg)
                errors.append(msg)
                safe_set_span_attributes(span, {"success": False, "error_type": "OSError"})
            except Exception as e:
                msg = f"Evidence pipeline unexpected error for {src.relative_path}: {e}"
                logger.warning(msg)
                errors.append(msg)
                safe_set_span_attributes(span, {"success": False, "error_type": type(e).__name__})

    return {
        "source_ids": source_ids,
        "discovered_count": len(discovered),
        "processed_count": processed,
        "per_source": per_source,
        "errors": errors,
    }
