"""OpenAlex metadata resolver.

This module implements a minimal client for the OpenAlex API and helpers to
normalize OpenAlex responses into the project's CitationRecord shape.

It is intentionally small and filesystem-agnostic; persistence is handled by
`src.citations.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from src.citations.crossref import CITATION_FALLBACK_YEAR, normalize_doi
from src.citations.registry import make_minimal_citation_record
from src.config import TIMEOUTS


OPENALEX_API_BASE = "https://api.openalex.org"


class OpenAlexError(RuntimeError):
    pass


class OpenAlexNotFoundError(OpenAlexError):
    pass


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class OpenAlexClientConfig:
    base_url: str = OPENALEX_API_BASE
    user_agent: str = "gia-agentic-short/1.0"


class OpenAlexClient:
    """Small sync client for OpenAlex."""

    def __init__(self, config: Optional[OpenAlexClientConfig] = None):
        self.config = config or OpenAlexClientConfig()
        self._timeout = httpx.Timeout(
            timeout=float(TIMEOUTS.EXTERNAL_API),
            connect=float(TIMEOUTS.LLM_CONNECT),
        )

    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": self.config.user_agent}

    def fetch_work_by_doi(self, doi: str) -> Dict[str, Any]:
        """Fetch a work object from OpenAlex by DOI.

        Args:
            doi: The Digital Object Identifier to look up.

        Returns:
            The raw OpenAlex work object as a dict.

        Raises:
            OpenAlexNotFoundError: If the DOI is not present in OpenAlex.
            OpenAlexError: For network/HTTP/JSON errors.
        """
        normalized = normalize_doi(doi)
        # OpenAlex expects the DOI as a URL in the works path: /works/https://doi.org/<doi>
        work_id = f"https://doi.org/{normalized}"
        work_path = quote(work_id, safe="")
        url = f"{self.config.base_url}/works/{work_path}"

        try:
            with httpx.Client(timeout=self._timeout, headers=self._headers()) as client:
                resp = client.get(url)
        except httpx.HTTPError as e:
            raise OpenAlexError(f"OpenAlex request failed: {e}")

        if resp.status_code == 404:
            raise OpenAlexNotFoundError(f"DOI not found in OpenAlex: {normalized}")
        if resp.status_code >= 400:
            raise OpenAlexError(f"OpenAlex returned HTTP {resp.status_code}")

        try:
            payload = resp.json()
        except ValueError as e:
            raise OpenAlexError(f"Failed to parse OpenAlex response as JSON: {e}")

        if not isinstance(payload, dict):
            raise OpenAlexError("OpenAlex response must be an object")

        return payload


def _author_names(work: Dict[str, Any]) -> List[str]:
    raw = work.get("authorships")
    if not isinstance(raw, list):
        return []

    out: List[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        author = entry.get("author")
        if not isinstance(author, dict):
            continue
        name = author.get("display_name")
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def openalex_work_to_citation_record(
    *,
    work: Dict[str, Any],
    citation_key: str,
    status: str,
    doi: str,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert an OpenAlex work object into a schema-valid CitationRecord.

    Args:
        work: Raw OpenAlex work object.
        citation_key: Stable key assigned by the bibliography builder.
        status: CitationRecord status (usually "verified").
        doi: DOI string to attach to the record.
        created_at: Optional stable timestamp to use for the record.

    Returns:
        A dictionary matching the project's CitationRecord schema.
    """
    title_val = work.get("display_name")
    title = title_val.strip() if isinstance(title_val, str) and title_val.strip() else "(missing title)"

    authors = _author_names(work)
    if not authors:
        authors = ["(unknown)"]

    year_val = work.get("publication_year")
    year = year_val if isinstance(year_val, int) else CITATION_FALLBACK_YEAR
    if year < 1000 or year > 2100:
        year = CITATION_FALLBACK_YEAR

    record = make_minimal_citation_record(
        citation_key=citation_key,
        title=title,
        authors=authors,
        year=year,
        status=status,
        created_at=created_at or _utc_now_iso_z(),
        identifiers={"doi": normalize_doi(doi)},
    )

    host_venue = work.get("host_venue")
    if isinstance(host_venue, dict):
        venue_name = host_venue.get("display_name")
        if isinstance(venue_name, str) and venue_name.strip():
            record["venue"] = venue_name.strip()

    biblio = work.get("biblio")
    if isinstance(biblio, dict):
        for key, dest in (("volume", "volume"), ("issue", "issue"), ("first_page", "pages")):
            value = biblio.get(key)
            if isinstance(value, str) and value.strip():
                if dest == "pages":
                    # OpenAlex provides first_page only; keep as-is.
                    record[dest] = value.strip()
                else:
                    record[dest] = value.strip()

    primary_location = work.get("primary_location")
    if isinstance(primary_location, dict):
        landing = primary_location.get("landing_page_url")
        if isinstance(landing, str) and landing.strip():
            record["url"] = landing.strip()

    record.setdefault("metadata", {})
    if isinstance(record.get("metadata"), dict):
        record["metadata"]["openalex"] = {
            "id": work.get("id"),
            "source": "openalex",
        }

    return record


def resolve_openalex_doi_to_record(
    *,
    doi: str,
    citation_key: str,
    client: Optional[OpenAlexClient] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a DOI via OpenAlex and return a schema-valid CitationRecord.

    This helper looks up the given DOI using the OpenAlex API, then converts
    the returned work metadata into the project's CitationRecord shape.

    Args:
        doi: The Digital Object Identifier to resolve.
        citation_key: The internal citation key to assign to the record.
        client: Optional preconfigured OpenAlexClient instance to use.
        created_at: Optional stable timestamp to use for the record.

    Returns:
        A dictionary representing a schema-valid CitationRecord.
    """
    oa_client = client or OpenAlexClient()
    work = oa_client.fetch_work_by_doi(doi)
    return openalex_work_to_citation_record(
        work=work,
        citation_key=citation_key,
        status="verified",
        doi=doi,
        created_at=created_at,
    )
