"""Citation registry utilities.

Stores per-project citation metadata under:
- bibliography/citations.json

This module is offline and filesystem-first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.utils.schema_validation import validate_citation_record
from src.utils.validation import validate_project_folder


@dataclass(frozen=True)
class CitationRegistryPaths:
    project_folder: Path
    bibliography_dir: Path
    citations_path: Path


def citation_registry_paths(project_folder: str | Path) -> CitationRegistryPaths:
    pf = validate_project_folder(project_folder)
    bib_dir = pf / "bibliography"
    return CitationRegistryPaths(
        project_folder=pf,
        bibliography_dir=bib_dir,
        citations_path=bib_dir / "citations.json",
    )


def ensure_bibliography_layout(project_folder: str | Path) -> CitationRegistryPaths:
    paths = citation_registry_paths(project_folder)
    paths.bibliography_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_citations(project_folder: str | Path, *, validate: bool = True) -> List[Dict[str, Any]]:
    """Load citations.json from a project.

    Returns an empty list if the registry does not exist.
    """
    paths = citation_registry_paths(project_folder)
    if not paths.citations_path.exists():
        return []

    with open(paths.citations_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"citations.json must be a list at {paths.citations_path}")

    if validate:
        seen_keys: set[str] = set()
        for record in payload:
            if not isinstance(record, dict):
                raise ValueError(f"CitationRecord must be an object at {paths.citations_path}")
            validate_citation_record(record)
            key = str(record.get("citation_key") or "")
            if not key:
                raise ValueError("CitationRecord.citation_key must be a non-empty string")
            if key in seen_keys:
                raise ValueError(f"Duplicate citation_key in citations.json: {key}")
            seen_keys.add(key)

    return payload


def save_citations(project_folder: str | Path, records: List[Dict[str, Any]], *, validate: bool = True) -> CitationRegistryPaths:
    """Write citations.json for a project.

    The registry is written as a list of CitationRecord objects.
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list")

    if validate:
        seen_keys: set[str] = set()
        for record in records:
            validate_citation_record(record)
            key = str(record.get("citation_key") or "")
            if key in seen_keys:
                raise ValueError(f"Duplicate citation_key in records: {key}")
            seen_keys.add(key)

    paths = ensure_bibliography_layout(project_folder)
    paths.citations_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def upsert_citation(project_folder: str | Path, record: Dict[str, Any], *, validate: bool = True) -> CitationRegistryPaths:
    """Insert or replace a citation record by citation_key."""
    if validate:
        validate_citation_record(record)

    key = str(record.get("citation_key") or "").strip()
    if not key:
        raise ValueError("CitationRecord.citation_key must be a non-empty string")

    records = load_citations(project_folder, validate=validate)
    updated: list[Dict[str, Any]] = []
    replaced = False

    for existing in records:
        if str(existing.get("citation_key") or "") == key:
            updated.append(record)
            replaced = True
        else:
            updated.append(existing)

    if not replaced:
        updated.append(record)

    return save_citations(project_folder, updated, validate=validate)


def make_minimal_citation_record(
    *,
    citation_key: str,
    title: str,
    authors: List[str],
    year: int,
    status: str = "unverified",
    created_at: Optional[str] = None,
    identifiers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a minimal schema-valid CitationRecord dict."""
    record: Dict[str, Any] = {
        "schema_version": "1.0",
        "citation_key": citation_key,
        "status": status,
        "title": title,
        "authors": authors,
        "year": year,
        "created_at": created_at or _utc_now_iso(),
    }
    if identifiers is not None:
        record["identifiers"] = identifiers

    validate_citation_record(record)
    return record


def citation_keys(project_folder: str | Path) -> List[str]:
    """Return a sorted list of citation keys in the registry."""
    records = load_citations(project_folder, validate=True)
    keys = [str(r.get("citation_key")) for r in records if isinstance(r.get("citation_key"), str)]
    return sorted(set(keys))


def has_verified_citations(project_folder: str | Path) -> bool:
    """Return True if any citation record has status=verified."""
    records = load_citations(project_folder, validate=True)
    for r in records:
        if r.get("status") == "verified":
            return True
    return False


def ensure_citations_registry_exists(project_folder: str | Path) -> CitationRegistryPaths:
    """Create an empty citations.json if missing."""
    paths = ensure_bibliography_layout(project_folder)
    if not paths.citations_path.exists():
        try:
            save_citations(project_folder, [], validate=False)
        except Exception as e:
            logger.warning(f"Failed to initialize citations.json: {e}")
    return paths
