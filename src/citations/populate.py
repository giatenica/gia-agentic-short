"""Populate bibliography artifacts from literature search results.

This module converts Edison-style citations data into schema-valid CitationRecords
and writes them to the canonical per-project registry under bibliography/.

It is designed to be best-effort:
- If Crossref resolution succeeds for a DOI, records are status=verified.
- If Crossref resolution fails, records are status=error but still schema-valid.
- If no DOI exists, records are status=unverified.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.citations.bibliography import build_bibliography, mint_stable_citation_key
from src.citations.crossref import resolve_crossref_doi_to_record
from src.citations.registry import make_minimal_citation_record


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_citation_records_from_citations_data(
    citations_data: List[dict],
    *,
    created_at: Optional[str] = None,
    resolve_doi_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Convert citations_data into CitationRecord list.

    Returns a dict containing:
    - records: list[dict]
    - verification: counts and overall status
    """

    created_at_val = created_at or _utc_now_iso_z()
    resolver = resolve_doi_fn or resolve_crossref_doi_to_record

    used_keys: set[str] = set()
    records: List[Dict[str, Any]] = []
    verified = 0
    unverified = 0
    error = 0

    for c in citations_data:
        if not isinstance(c, dict):
            continue

        raw_title = c.get("title")
        title = raw_title.strip() if isinstance(raw_title, str) else ""
        if not title:
            title = "(missing title)"

        raw_authors = c.get("authors")
        authors: List[str] = []
        if isinstance(raw_authors, list):
            authors = [a.strip() for a in raw_authors if isinstance(a, str) and a.strip()]
        if not authors:
            authors = ["(unknown)"]

        year_val = c.get("year")
        year = year_val if isinstance(year_val, int) else 1900
        if year < 1000 or year > 2100:
            year = 1900

        citation_key = mint_stable_citation_key(
            authors=authors,
            year=year,
            title=title,
            existing_keys=used_keys,
        )
        used_keys.add(citation_key)

        raw_doi = c.get("doi")
        doi = raw_doi.strip() if isinstance(raw_doi, str) else ""

        if doi:
            try:
                record = resolver(
                    doi=doi,
                    citation_key=citation_key,
                    created_at=created_at_val,
                )
                verified += 1
            except Exception as e:
                logger.exception("Crossref resolution failed")
                record = make_minimal_citation_record(
                    citation_key=citation_key,
                    title=title,
                    authors=authors,
                    year=year,
                    status="error",
                    created_at=created_at_val,
                    identifiers={"doi": doi},
                )
                record["notes"] = f"Crossref resolution failed: {type(e).__name__}"
                error += 1
        else:
            record = make_minimal_citation_record(
                citation_key=citation_key,
                title=title,
                authors=authors,
                year=year,
                status="unverified",
                created_at=created_at_val,
                identifiers=None,
            )
            unverified += 1

        journal = c.get("journal")
        if isinstance(journal, str) and journal.strip() and not record.get("venue"):
            record["venue"] = journal.strip()

        url = c.get("url")
        if isinstance(url, str) and url.strip() and not record.get("url"):
            record["url"] = url.strip()

        records.append(record)

    status = "verified"
    if error > 0:
        status = "error"
    elif unverified > 0:
        status = "unverified"

    return {
        "records": records,
        "verification": {
            "status": status,
            "total": int(verified + unverified + error),
            "verified": int(verified),
            "unverified": int(unverified),
            "error": int(error),
        },
    }


def build_and_write_bibliography_from_citations_data(
    *,
    project_folder: str,
    citations_data: List[dict],
    created_at: Optional[str] = None,
    resolve_doi_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Best-effort creation of bibliography artifacts from citations_data."""

    built = build_citation_records_from_citations_data(
        citations_data,
        created_at=created_at,
        resolve_doi_fn=resolve_doi_fn,
    )
    records = built.get("records")
    if not isinstance(records, list):
        records = []

    paths = build_bibliography(project_folder, records=records, validate=True)

    return {
        "citations_json": str(paths.citations_path),
        "references_bib": str(paths.references_bib_path),
        "verification": built.get("verification"),
    }
