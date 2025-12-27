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
from src.citations.crossref import normalize_doi
from src.citations.registry import make_minimal_citation_record
from src.citations.registry import load_citations
from src.citations.verification import CitationVerificationPolicy, is_verification_stale, resolve_doi_to_record_with_fallback


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_citation_records_from_citations_data(
    citations_data: List[dict],
    *,
    created_at: Optional[str] = None,
    resolve_doi_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    existing_by_key: Optional[Dict[str, Dict[str, Any]]] = None,
    verification_policy: Optional[CitationVerificationPolicy] = None,
) -> Dict[str, Any]:
    """Convert citations_data into CitationRecord list.

    Returns a dict containing:
    - records: list[dict]
    - verification: counts and overall status
    """

    created_at_val = created_at or _utc_now_iso_z()
    resolver = resolve_doi_fn or resolve_doi_to_record_with_fallback
    policy = verification_policy or CitationVerificationPolicy()

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
                doi = normalize_doi(doi)
            except ValueError:
                doi = doi.strip()

            existing = existing_by_key.get(citation_key) if isinstance(existing_by_key, dict) else None
            if isinstance(existing, dict) and str(existing.get("status") or "") == "verified":
                existing_doi = None
                identifiers = existing.get("identifiers")
                if isinstance(identifiers, dict) and isinstance(identifiers.get("doi"), str):
                    existing_doi = identifiers.get("doi")
                try:
                    existing_doi_norm = normalize_doi(existing_doi) if existing_doi else None
                except ValueError:
                    existing_doi_norm = existing_doi

                last_checked = None
                verification = existing.get("verification")
                if isinstance(verification, dict):
                    last_checked = verification.get("last_checked")

                if existing_doi_norm == doi and not is_verification_stale(
                    last_checked=last_checked,
                    now=created_at_val,
                    policy=policy,
                ):
                    records.append(existing)
                    verified += 1
                    continue

            try:
                try:
                    record = resolver(
                        doi=doi,
                        citation_key=citation_key,
                        created_at=created_at_val,
                        existing_record=existing,
                        policy=policy,
                        now=created_at_val,
                    )
                except TypeError as te:
                    # Backwards compatibility: custom resolvers may not accept
                    # the extended kwargs used by the fallback chain.
                    msg = str(te)
                    if "unexpected keyword argument" not in msg:
                        raise
                    record = resolver(
                        doi=doi,
                        citation_key=citation_key,
                        created_at=created_at_val,
                    )
                if str(record.get("status") or "") == "verified":
                    verified += 1
                elif bool(record.get("manual_verification_required")):
                    unverified += 1
                else:
                    unverified += 1
            except Exception as e:
                logger.exception("Citation verification failed")
                record = make_minimal_citation_record(
                    citation_key=citation_key,
                    title=title,
                    authors=authors,
                    year=year,
                    status="error",
                    created_at=created_at_val,
                    identifiers={"doi": doi},
                )
                record["verification"] = {
                    "status": "error",
                    "provider_used": "verification_chain",
                    "last_checked": created_at_val,
                    "attempts": [
                        {
                            "provider": "verification_chain",
                            "ok": False,
                            "checked_at": created_at_val,
                            "error": {"type": type(e).__name__, "message": str(e)},
                        }
                    ],
                }
                record["manual_verification_required"] = True
                record["notes"] = f"Verification failed: {type(e).__name__}"
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
            record["verification"] = {
                "status": "manual",
                "provider_used": None,
                "last_checked": created_at_val,
                "attempts": [],
            }
            record["manual_verification_required"] = True
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

    existing_by_key: Dict[str, Dict[str, Any]] = {}
    try:
        existing = load_citations(project_folder, validate=True)
        for r in existing:
            if isinstance(r, dict) and isinstance(r.get("citation_key"), str):
                existing_by_key[str(r["citation_key"])] = r
    except Exception:
        existing_by_key = {}

    built = build_citation_records_from_citations_data(
        citations_data,
        created_at=created_at,
        resolve_doi_fn=resolve_doi_fn,
        existing_by_key=existing_by_key,
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
