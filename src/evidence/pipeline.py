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

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from src.evidence.parser import MVPLineBlockParser
from src.evidence.pdf_parser import parse_pdf_to_parsed_payload
from src.evidence.source_fetcher import SourceFetcherTool, LocalSource
from src.evidence.store import EvidenceStore
from src.evidence.extraction import extract_evidence_items
from src.tracing import get_tracer, safe_set_span_attributes


_tracer = get_tracer("evidence-pipeline")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_evidence_coverage_artifact(*, project_folder: str, summary: Dict[str, Any]) -> None:
    pf = Path(project_folder).expanduser().resolve()
    outputs_dir = pf / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "1.0",
        "created_at": _utc_now_iso(),
        "summary": summary,
    }

    (outputs_dir / "evidence_coverage.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _choose_raw_pdf_path(raw_dir: Path, preferred_filename: str | None = None) -> Path:
    raw_dir_resolved = raw_dir.resolve()

    if preferred_filename:
        candidate = raw_dir_resolved / preferred_filename
        candidate_resolved = candidate.resolve()
        if (
            candidate_resolved.exists()
            and candidate_resolved.is_file()
            and candidate_resolved.suffix.lower() == ".pdf"
            and candidate_resolved.is_relative_to(raw_dir_resolved)
        ):
            return candidate_resolved

    pdfs = [p for p in raw_dir_resolved.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    pdfs.sort(key=lambda p: p.name)
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {raw_dir_resolved}")
    return pdfs[0]


def run_pdf_evidence_pipeline_for_source(
    *,
    project_folder: str,
    source_id: str,
    raw_pdf_filename: str | None = None,
    max_items: int = 25,
    created_at: str | None = None,
) -> Dict[str, Any]:
    """Parse a stored PDF and extract evidence items.

    The `sources/<source_id>/raw/` directory is created if it does not exist, and
    this function expects at least one `.pdf` file to already be present there.
    If multiple PDFs are present, `raw_pdf_filename` can be used to select a
    specific file; otherwise the first PDF in sorted name order is used.

    It writes:
    - sources/<source_id>/parsed.json
    - sources/<source_id>/evidence.json

    Args:
        project_folder: Root project folder that contains the `sources/` directory.
        source_id: Identifier of the source under `sources/<source_id>/`.
        raw_pdf_filename: Optional filename of the PDF under `sources/<source_id>/raw/`
            to parse. If provided and the file does not exist, is not a PDF, or
            would escape the raw folder, it is ignored and the first PDF in name
            order is used instead.
        max_items: Maximum number of evidence items to extract.
        created_at: Optional timestamp string used when creating evidence items.

    Returns:
        Summary dict with keys: source_id, raw_pdf_path, parsed_blocks_count,
        evidence_items_count.

    Raises:
        FileNotFoundError: If no PDF is found under `sources/<source_id>/raw/`.
    """

    store = EvidenceStore(project_folder)
    sp = store.ensure_source_layout(source_id)

    pdf_path = _choose_raw_pdf_path(sp.raw_dir, preferred_filename=raw_pdf_filename)
    parse_result = parse_pdf_to_parsed_payload(pdf_path)
    parsed_payload = parse_result.parsed_payload

    blocks = parsed_payload.get("blocks")
    parsed_blocks_count = len(blocks) if isinstance(blocks, list) else 0
    store.write_parsed(source_id, parsed_payload)

    items = extract_evidence_items(
        parsed=parsed_payload,
        source_id=source_id,
        created_at=created_at,
        max_items=max_items,
        min_excerpt_chars=5,
    )
    store.write_evidence_items(source_id, items)

    return {
        "source_id": source_id,
        "raw_pdf_path": str(pdf_path.relative_to(store.project_folder)),
        "parsed_blocks_count": parsed_blocks_count,
        "evidence_items_count": len(items),
    }


def discover_acquired_sources(project_folder: str) -> List[str]:
    """Discover source_ids for already-acquired sources in sources/ directory.
    
    This finds sources that were downloaded by `acquire_sources_from_citations`
    or manually placed in the `sources/<source_id>/raw/` structure.
    
    Args:
        project_folder: Root project folder containing `sources/` directory.
    
    Returns:
        List of source_id strings for sources that have raw files.
    """
    pf = Path(project_folder).expanduser().resolve()
    sources_dir = pf / "sources"
    
    if not sources_dir.exists() or not sources_dir.is_dir():
        return []
    
    source_ids: List[str] = []
    for entry in sources_dir.iterdir():
        if not entry.is_dir():
            continue
        
        raw_dir = entry / "raw"
        if not raw_dir.exists() or not raw_dir.is_dir():
            continue
        
        # Check if there are any files in raw/
        has_files = any(f.is_file() for f in raw_dir.iterdir())
        if not has_files:
            continue
        
        # Prefer the canonical source_id stored in retrieval.json (if present).
        # This avoids leaking the internal directory encoding (colons/slashes -> underscores)
        # and keeps IDs consistent with citations and Step 0 outputs.
        retrieval_meta_path = raw_dir / "retrieval.json"
        source_id = entry.name
        if retrieval_meta_path.exists() and retrieval_meta_path.is_file():
            try:
                meta = json.loads(retrieval_meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    meta_source_id = meta.get("source_id")
                    if isinstance(meta_source_id, str) and meta_source_id:
                        source_id = meta_source_id
            except Exception:
                # If metadata is missing or malformed, fall back to directory name.
                pass
        source_ids.append(source_id)
    
    return sorted(source_ids)


def run_evidence_pipeline_for_acquired_sources(
    *,
    project_folder: str,
    config: EvidencePipelineConfig,
    source_ids: List[str] | None = None,
) -> Dict[str, Any]:
    """Run evidence pipeline on already-acquired sources.
    
    This processes sources that were downloaded by `acquire_sources_from_citations`
    and are already in the `sources/<source_id>/raw/` structure.
    
    Args:
        project_folder: Root project folder.
        config: Pipeline configuration.
        source_ids: Optional list of specific source_ids to process.
            If None, discovers all sources in sources/ directory.
    
    Returns:
        Summary dict similar to run_local_evidence_pipeline.
    """
    if source_ids is None:
        source_ids = discover_acquired_sources(project_folder)
    
    if config.max_sources > 0:
        source_ids = source_ids[:config.max_sources]
    
    store = EvidenceStore(project_folder)
    processed_ids: List[str] = []
    errors: List[str] = []
    per_source: List[Dict[str, Any]] = []
    
    for source_id in source_ids:
        with _tracer.start_as_current_span("evidence_acquired_source") as span:
            safe_set_span_attributes(span, {"source_id": source_id})
            
            try:
                sp = store.source_paths(source_id)
                
                if not sp.raw_dir.exists():
                    continue
                
                # Find raw files
                raw_files = list(sp.raw_dir.iterdir())
                raw_files = [f for f in raw_files if f.is_file()]
                
                if not raw_files:
                    continue
                
                # Determine file type and process accordingly
                pdf_files = [f for f in raw_files if f.suffix.lower() == ".pdf"]
                txt_files = [f for f in raw_files if f.suffix.lower() in {".txt", ".md"}]
                html_files = [f for f in raw_files if f.suffix.lower() == ".html"]
                
                parsed_payload: Dict[str, Any] = {}
                parsed_blocks_count = 0
                evidence_items_count = 0
                extra_info: Dict[str, Any] = {}
                
                if pdf_files:
                    # Process PDF
                    pdf_path = sorted(pdf_files)[0]
                    parse_result = parse_pdf_to_parsed_payload(pdf_path)
                    parsed_payload = parse_result.parsed_payload
                    extra_info["page_count"] = int(parse_result.page_count)
                    extra_info["file_type"] = "pdf"
                elif txt_files:
                    # Process text file
                    txt_path = sorted(txt_files)[0]
                    text = txt_path.read_text(encoding="utf-8", errors="replace")
                    if config.max_chars_per_source > 0:
                        text = text[:config.max_chars_per_source]
                    parsed_payload = _parsed_payload_from_text(text)
                    extra_info["file_type"] = "text"
                elif html_files:
                    # Process HTML file - strip tags to get plain text
                    html_path = sorted(html_files)[0]
                    raw_html = html_path.read_text(encoding="utf-8", errors="replace")
                    # Check if there's a pre-extracted text file (source.txt)
                    txt_from_html = html_path.parent / "source.txt"
                    if txt_from_html.exists() and txt_from_html.is_file():
                        text = txt_from_html.read_text(encoding="utf-8", errors="replace")
                    else:
                        # Strip HTML tags using the extractor
                        from src.evidence.acquisition import _HTMLTextExtractor
                        extractor = _HTMLTextExtractor()
                        extractor.feed(raw_html)
                        text = extractor.get_text()
                    if config.max_chars_per_source > 0:
                        text = text[:config.max_chars_per_source]
                    parsed_payload = _parsed_payload_from_text(text)
                    extra_info["file_type"] = "html"
                else:
                    # Skip unsupported file types
                    continue
                
                blocks = parsed_payload.get("blocks")
                if isinstance(blocks, list):
                    parsed_blocks_count = len(blocks)
                
                store.write_parsed(source_id, parsed_payload)
                
                items = extract_evidence_items(
                    parsed=parsed_payload,
                    source_id=source_id,
                    created_at=_utc_now_iso(),
                    max_items=config.max_items_per_source,
                )
                store.write_evidence_items(source_id, items)
                evidence_items_count = len(items) if isinstance(items, list) else 0
                
                if config.append_to_ledger and items:
                    store.append_many(items)
                
                processed_ids.append(source_id)
                per_source.append({
                    "source_id": source_id,
                    "parsed_blocks_count": parsed_blocks_count,
                    "evidence_items_count": evidence_items_count,
                    **extra_info,
                })
                
                safe_set_span_attributes(span, {
                    "parsed_blocks_count": parsed_blocks_count,
                    "evidence_items_count": evidence_items_count,
                    "success": True,
                })
                
            except Exception as e:
                msg = f"Evidence pipeline failed for acquired source {source_id}: {e}"
                logger.warning(msg)
                errors.append(msg)
                safe_set_span_attributes(span, {"success": False, "error": str(e)})
    
    summary = {
        "source_ids": processed_ids,
        "discovered_count": len(source_ids) if source_ids else 0,
        "processed_count": len(processed_ids),
        "per_source": per_source,
        "errors": errors,
    }
    
    _write_evidence_coverage_artifact(project_folder=project_folder, summary=summary)
    return summary


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
                rel_path = Path(src.relative_path)
                ext = rel_path.suffix.lower()

                parsed_payload: Dict[str, Any]
                parsed_blocks_count = 0
                evidence_items_count = 0
                extra_source_info: Dict[str, Any] = {}

                if ext == ".pdf":
                    pdf_path = fetcher.project_folder / rel_path
                    if config.ingest_sources:
                        ingest_info = fetcher.ingest_source(src)
                        raw_path = ingest_info.get("raw_path")
                        if isinstance(raw_path, str) and raw_path:
                            pdf_path = fetcher.project_folder / raw_path

                    parse_result = parse_pdf_to_parsed_payload(pdf_path)
                    parsed_payload = parse_result.parsed_payload
                    extra_source_info["page_count"] = int(parse_result.page_count)
                else:
                    if config.ingest_sources:
                        fetcher.ingest_source(src)
                    text = fetcher.load_text(src, max_chars=config.max_chars_per_source)
                    parsed_payload = _parsed_payload_from_text(text)

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
                        **extra_source_info,
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

    summary = {
        "source_ids": source_ids,
        "discovered_count": len(discovered),
        "processed_count": processed,
        "per_source": per_source,
        "errors": errors,
    }

    _write_evidence_coverage_artifact(project_folder=project_folder, summary=summary)
    return summary



