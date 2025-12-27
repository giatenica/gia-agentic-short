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
    """Build a best-effort mapping of source_id -> citation_key."""

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

        # evidence.json uses source_id, which in this repo corresponds to the directory name
        # under sources/ (already sanitized via source_id_to_dirname).
        source_id = source_dir.name

        candidates = []
        for fp in raw_dir.iterdir():
            if fp.is_file():
                candidates.append(fp.name)

        matched_key: Optional[str] = None
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
