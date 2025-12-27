"""Source-to-citation mapping utilities.

Deterministic section writers consume a `source_citation_map` that links
EvidenceItem.source_id values to canonical citation keys from
bibliography/citations.json.

This module provides best-effort heuristics to build that mapping by inspecting
stored source files under sources/<source_id>/raw/.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from src.citations.crossref import normalize_doi
from src.citations.registry import load_citations
from src.utils.validation import validate_project_folder


_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_DOI_SANITIZED_RE = re.compile(r"10\.\d{4,9}_[ -._;()A-Z0-9]+", re.IGNORECASE)
_ARXIV_RE = re.compile(r"(?:(?:arxiv:)?)(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


def _normalize_arxiv(value: str) -> str:
    s = value.strip().lower()
    s = s.removeprefix("arxiv:")
    return s


def _extract_doi(text: str) -> Optional[str]:
    m = _DOI_RE.search(text)
    if m:
        raw = m.group(0)
    else:
        # Many downloaders sanitize the DOI slash into an underscore in filenames.
        m2 = _DOI_SANITIZED_RE.search(text)
        if not m2:
            return None
        raw = m2.group(0)
        if "/" not in raw and "_" in raw:
            prefix, suffix = raw.split("_", 1)
            raw = f"{prefix}/{suffix}"

    # Trim common filename cruft.
    raw = raw.strip().strip(').,;]}>"\'')
    for ext in (".pdf", ".html", ".htm", ".txt", ".json", ".bib", ".bibtex"):
        if raw.lower().endswith(ext):
            raw = raw[: -len(ext)]
            break

    try:
        return normalize_doi(raw)
    except ValueError as exc:
        logger.debug(f"Failed to normalize DOI candidate '{raw}': {exc}")
        return raw


def _extract_arxiv(text: str) -> Optional[str]:
    m = _ARXIV_RE.search(text)
    if not m:
        return None
    return _normalize_arxiv(m.group(1))


def _index_citations(records: list[Dict[str, Any]]) -> tuple[Dict[str, str], Dict[str, str]]:
    by_doi: Dict[str, str] = {}
    by_arxiv: Dict[str, str] = {}

    for r in records:
        if not isinstance(r, dict):
            continue
        key = str(r.get("citation_key") or "").strip()
        if not key:
            continue

        ident = r.get("identifiers")
        if isinstance(ident, dict):
            doi = ident.get("doi")
            if isinstance(doi, str) and doi.strip():
                try:
                    norm_doi = normalize_doi(doi)
                except ValueError as exc:
                    logger.debug(f"Failed to normalize DOI identifier '{doi}': {exc}")
                    norm_doi = doi.strip()

                if norm_doi in by_doi and by_doi[norm_doi] != key:
                    logger.warning(
                        "Duplicate DOI '{doi}' for citation keys '{existing}' and '{new}'. Keeping first occurrence.",
                        doi=norm_doi,
                        existing=by_doi[norm_doi],
                        new=key,
                    )
                elif norm_doi not in by_doi:
                    by_doi[norm_doi] = key

            arxiv = ident.get("arxiv")
            if isinstance(arxiv, str) and arxiv.strip():
                norm_arxiv = _normalize_arxiv(arxiv)
                if norm_arxiv in by_arxiv and by_arxiv[norm_arxiv] != key:
                    logger.warning(
                        "Duplicate arXiv ID '{arxiv}' for citation keys '{existing}' and '{new}'. Keeping first occurrence.",
                        arxiv=norm_arxiv,
                        existing=by_arxiv[norm_arxiv],
                        new=key,
                    )
                elif norm_arxiv not in by_arxiv:
                    by_arxiv[norm_arxiv] = key

    return by_doi, by_arxiv


def build_source_citation_map(project_folder: str | Path) -> Dict[str, str]:
    """Build a best-effort mapping of source_id -> citation_key.
    
    This function builds a mapping by:
    1. Reading retrieval.json metadata from sources (preferred, contains DOI/source_id)
    2. Falling back to filename-based DOI/arXiv extraction
    3. Matching against citation registry by DOI or arXiv ID
    """

    pf = validate_project_folder(project_folder)
    records = load_citations(pf, validate=True)
    by_doi, by_arxiv = _index_citations(records)

    sources_dir = pf / "sources"
    if not sources_dir.exists() or not sources_dir.is_dir():
        return {}

    mapping: Dict[str, str] = {}

    for source_dir in sorted([p for p in sources_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        raw_dir = source_dir / "raw"
        if not raw_dir.exists() or not raw_dir.is_dir():
            continue

        # Get source_id from retrieval.json if available (canonical), else use dirname
        retrieval_path = raw_dir / "retrieval.json"
        source_id = source_dir.name
        metadata_doi: Optional[str] = None
        metadata_arxiv: Optional[str] = None
        
        if retrieval_path.exists() and retrieval_path.is_file():
            try:
                meta = json.loads(retrieval_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    # Get canonical source_id
                    meta_source_id = meta.get("source_id")
                    if isinstance(meta_source_id, str) and meta_source_id.strip():
                        source_id = meta_source_id.strip()
                    
                    # Extract DOI from metadata
                    requested = meta.get("requested")
                    if isinstance(requested, dict):
                        # Check for DOI in requested params
                        req_doi = requested.get("doi")
                        if isinstance(req_doi, str) and req_doi.strip():
                            try:
                                metadata_doi = normalize_doi(req_doi)
                            except ValueError:
                                metadata_doi = req_doi.strip()
                        
                        # Check for arXiv in requested params
                        req_arxiv = requested.get("arxiv_id") or requested.get("id")
                        if isinstance(req_arxiv, str) and req_arxiv.strip():
                            metadata_arxiv = _normalize_arxiv(req_arxiv)
                        
                        # Check for DOI in URL (e.g., doi.org URLs)
                        req_url = requested.get("url") or requested.get("arxiv_url")
                        if isinstance(req_url, str) and "doi.org" in req_url:
                            extracted = _extract_doi(req_url)
                            if extracted and not metadata_doi:
                                metadata_doi = extracted
                        
                        # Check for arXiv in URL
                        if isinstance(req_url, str) and "arxiv.org" in req_url.lower():
                            extracted = _extract_arxiv(req_url)
                            if extracted and not metadata_arxiv:
                                metadata_arxiv = extracted
                    
                    # Also check for DOI stored directly in provider field for DOI-based acquisitions
                    if source_id.startswith("doi:") and not metadata_doi:
                        # Extract DOI from source_id like "doi:abc123..."
                        metadata_doi = _extract_doi(source_id)
                    
                    # Check for arXiv ID in source_id
                    if source_id.startswith("arxiv:") and not metadata_arxiv:
                        metadata_arxiv = _normalize_arxiv(source_id.replace("arxiv:", ""))
                        
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug(f"Failed to read retrieval.json for {source_dir.name}: {exc}")

        # Try to match using metadata first (preferred)
        matched_key: Optional[str] = None
        
        if metadata_doi and metadata_doi in by_doi:
            matched_key = by_doi[metadata_doi]
        elif metadata_arxiv and metadata_arxiv in by_arxiv:
            matched_key = by_arxiv[metadata_arxiv]
        
        # Fall back to filename-based extraction if metadata didn't match
        if not matched_key:
            candidates = []
            for fp in raw_dir.iterdir():
                if fp.is_file():
                    candidates.append(fp.name)

            for c in candidates:
                doi = _extract_doi(c)
                if doi and doi in by_doi:
                    matched_key = by_doi[doi]
                    break

                arxiv = _extract_arxiv(c)
                if arxiv and arxiv in by_arxiv:
                    matched_key = by_arxiv[arxiv]
                    break

        if matched_key:
            mapping[source_id] = matched_key

    return mapping


def source_citation_map_path(project_folder: str | Path) -> Path:
    pf = validate_project_folder(project_folder)
    return pf / "bibliography" / "source_citation_map.json"


def load_source_citation_map(project_folder: str | Path) -> Dict[str, str]:
    path = source_citation_map_path(project_folder)
    if not path.exists() or not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.exception(f"Failed to load source citation map from {path}: {exc}")
        return {}

    if not isinstance(payload, dict):
        return {}

    out: Dict[str, str] = {}
    for k, v in payload.items():
        if isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip():
            out[k.strip()] = v.strip()
    return out


def write_source_citation_map(project_folder: str | Path, mapping: Dict[str, str]) -> Path:
    pf = validate_project_folder(project_folder)
    out_path = source_citation_map_path(pf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    safe: Dict[str, str] = {}
    for k, v in mapping.items():
        if isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip():
            safe[k.strip()] = v.strip()

    out_path.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(f"Wrote source citation map to {out_path}")
    return out_path
