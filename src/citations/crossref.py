"""Crossref metadata resolver.

This module provides a small client for the Crossref REST API and helpers to
normalize Crossref responses into the project's CitationRecord shape.

It is intentionally minimal and filesystem-agnostic; persistence is handled by
`src.citations.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from src.citations.registry import make_minimal_citation_record, upsert_citation
from src.config import TIMEOUTS


CROSSREF_API_BASE = "https://api.crossref.org"

# Fallback year used when Crossref does not provide any publication date.
# This should be treated as "unknown" in downstream logic.
CITATION_FALLBACK_YEAR = 1900


class CrossrefError(RuntimeError):
    pass


class CrossrefNotFoundError(CrossrefError):
    pass


def normalize_doi(doi: str) -> str:
    """Normalize a DOI string.

    Accepts bare DOIs and common URL forms like https://doi.org/<doi>.
    """
    if not isinstance(doi, str):
        raise ValueError("doi must be a string")
    value = doi.strip()
    if not value:
        raise ValueError("doi must be non-empty")

    lower = value.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if lower.startswith(prefix):
            value = value[len(prefix) :].strip()
            break

    return value.strip()


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CrossrefClientConfig:
    """Configuration for Crossref client."""

    base_url: str = CROSSREF_API_BASE
    mailto: Optional[str] = None
    user_agent: str = "gia-agentic-short/1.0"


class CrossrefClient:
    """Small sync client for Crossref."""

    def __init__(self, config: Optional[CrossrefClientConfig] = None):
        self.config = config or CrossrefClientConfig()

        # Use centralized timeouts; avoid hardcoding.
        self._timeout = httpx.Timeout(
            timeout=float(TIMEOUTS.EXTERNAL_API),
            connect=float(TIMEOUTS.LLM_CONNECT),
        )

    def _headers(self) -> Dict[str, str]:
        ua = self.config.user_agent
        if self.config.mailto:
            ua = f"{ua} (mailto:{self.config.mailto})"
        return {"User-Agent": ua}

    def fetch_work_by_doi(self, doi: str) -> Dict[str, Any]:
        """Fetch a Crossref work payload by DOI.

        Returns the `message` object from Crossref.
        """
        normalized = normalize_doi(doi)
        doi_path = quote(normalized, safe="")
        url = f"{self.config.base_url}/works/{doi_path}"

        try:
            with httpx.Client(timeout=self._timeout, headers=self._headers()) as client:
                resp = client.get(url)
        except httpx.HTTPError as e:
            raise CrossrefError(f"Crossref request failed: {e}")

        if resp.status_code == 404:
            raise CrossrefNotFoundError(f"DOI not found in Crossref: {normalized}")
        if resp.status_code >= 400:
            raise CrossrefError(f"Crossref returned HTTP {resp.status_code}")

        try:
            data = resp.json()
        except ValueError as e:
            raise CrossrefError(f"Failed to parse Crossref response as JSON: {e}")
        message = data.get("message")
        if not isinstance(message, dict):
            raise CrossrefError("Crossref response missing 'message' object")
        return message

    def search_by_title(
        self,
        *,
        title: str,
        author: Optional[str] = None,
        year: Optional[int] = None,
        rows: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search Crossref works using bibliographic query.

        Returns the list of work items (each item is a Crossref work summary).
        """
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if rows <= 0:
            raise ValueError("rows must be > 0")

        params: Dict[str, Any] = {
            "query.bibliographic": title.strip(),
            "rows": int(rows),
        }
        if author and author.strip():
            params["query.author"] = author.strip()
        if year is not None:
            if year < 1000 or year > 2100:
                raise ValueError("year out of supported range")
            params["filter"] = f"from-pub-date:{year},until-pub-date:{year}"

        url = f"{self.config.base_url}/works"

        try:
            with httpx.Client(timeout=self._timeout, headers=self._headers()) as client:
                resp = client.get(url, params=params)
        except httpx.HTTPError as e:
            raise CrossrefError(f"Crossref request failed: {e}")

        if resp.status_code >= 400:
            raise CrossrefError(f"Crossref returned HTTP {resp.status_code}")

        try:
            data = resp.json()
        except ValueError as e:
            raise CrossrefError(f"Failed to parse Crossref response as JSON: {e}")
        message = data.get("message")
        if not isinstance(message, dict):
            raise CrossrefError("Crossref response missing 'message' object")

        items = message.get("items")
        if items is None:
            return []
        if not isinstance(items, list):
            raise CrossrefError("Crossref response message.items must be a list")

        return [it for it in items if isinstance(it, dict)]


def _pick_first_str(values: Any) -> Optional[str]:
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    if isinstance(values, str) and values.strip():
        return values.strip()
    return None


def _format_authors(work: Dict[str, Any]) -> List[str]:
    authors_raw = work.get("author")
    if not isinstance(authors_raw, list):
        return []

    authors: List[str] = []
    for a in authors_raw:
        if not isinstance(a, dict):
            continue
        given = a.get("given")
        family = a.get("family")
        if isinstance(given, str) and isinstance(family, str) and given.strip() and family.strip():
            authors.append(f"{given.strip()} {family.strip()}")
        elif isinstance(family, str) and family.strip():
            authors.append(family.strip())
    return authors


def _extract_year(work: Dict[str, Any]) -> Optional[int]:
    for key in ("published-print", "published-online", "issued", "created"):
        part = work.get(key)
        if not isinstance(part, dict):
            continue
        date_parts = part.get("date-parts")
        if isinstance(date_parts, list) and date_parts:
            first = date_parts[0]
            if isinstance(first, list) and first:
                year = first[0]
                if isinstance(year, int):
                    return year
    return None


def _coerce_scalar_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return None


def _extract_issue(work: Dict[str, Any]) -> Optional[str]:
    direct = _coerce_scalar_str(work.get("issue"))
    if direct:
        return direct

    journal_issue = work.get("journal-issue")
    if isinstance(journal_issue, dict):
        return _coerce_scalar_str(journal_issue.get("issue"))
    return None


def _version_type_from_crossref_work_type(work: Dict[str, Any]) -> str:
    """Map Crossref work types to project-level version types.

    Crossref uses a variety of work types; we collapse them into a small set
    compatible with CitationRecord.version.type.

    Returns:
        One of: published, preprint, working_paper, unknown.
    """
    work_type = work.get("type")
    if not isinstance(work_type, str):
        return "unknown"
    t = work_type.strip().lower()
    if not t:
        return "unknown"

    if t == "posted-content":
        return "preprint"
    if t in ("working-paper", "report"):
        return "working_paper"
    if t in ("journal-article", "proceedings-article", "book-chapter"):
        return "published"

    return "unknown"


def _extract_related_dois_from_relation(work: Dict[str, Any], relation_keys: List[str]) -> List[str]:
    """Extract related DOIs from Crossref relation metadata.

    Args:
        work: Crossref work payload.
        relation_keys: Relation keys to scan (for example: is-preprint-of,
            has-preprint, is-version-of, has-version).

    Returns:
        A deterministic (sorted, unique) list of normalized DOIs.

    Notes:
        Crossref relation entries may contain invalid DOIs; those are skipped.
    """
    relation = work.get("relation")
    if not isinstance(relation, dict):
        return []

    found: List[str] = []
    for key in relation_keys:
        items = relation.get(key)
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            id_type = it.get("id-type")
            if id_type != "doi":
                continue
            raw = it.get("id")
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                found.append(normalize_doi(raw))
            except Exception:
                continue

    # Deterministic: unique + sort.
    return sorted({d for d in found if isinstance(d, str) and d.strip()})


def _extract_version_object(work: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a CitationRecord.version object from Crossref metadata.

    Returns None when there is no meaningful version information to record
    (unknown type and no related links).
    """
    vtype = _version_type_from_crossref_work_type(work)

    published_links = _extract_related_dois_from_relation(work, ["is-preprint-of", "is-version-of"])
    working_links = _extract_related_dois_from_relation(work, ["has-preprint", "has-version"])

    related_published = published_links[0] if published_links else None
    related_working = working_links[0] if working_links else None

    if vtype == "unknown" and not related_published and not related_working:
        return None

    version: Dict[str, Any] = {"type": vtype}
    if related_working:
        version["related_working_paper"] = related_working
    if related_published:
        version["related_published"] = related_published
    return version


def crossref_work_to_citation_record(
    *,
    work: Dict[str, Any],
    citation_key: str,
    status: str = "verified",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a Crossref work object to a CitationRecord."""
    title = _pick_first_str(work.get("title")) or "(missing title)"
    authors = _format_authors(work)
    year = _extract_year(work) or CITATION_FALLBACK_YEAR

    doi = work.get("DOI")
    identifiers: Dict[str, Any] = {}
    if isinstance(doi, str) and doi.strip():
        identifiers["doi"] = normalize_doi(doi)

    record = make_minimal_citation_record(
        citation_key=citation_key,
        title=title,
        authors=authors or ["(unknown)"],
        year=year,
        status=status,
        created_at=created_at or _utc_now_iso_z(),
        identifiers=identifiers or None,
    )

    venue = _pick_first_str(work.get("container-title"))
    if venue:
        record["venue"] = venue

    volume = _coerce_scalar_str(work.get("volume"))
    if volume:
        record["volume"] = volume

    issue = _extract_issue(work)
    if issue:
        record["issue"] = issue

    pages = _coerce_scalar_str(work.get("page"))
    if pages:
        record["pages"] = pages

    publisher = work.get("publisher")
    if isinstance(publisher, str) and publisher.strip():
        record["publisher"] = publisher.strip()

    url = work.get("URL")
    if isinstance(url, str) and url.strip():
        record["url"] = url.strip()

    version = _extract_version_object(work)
    if version:
        record["version"] = version

    return record


def resolve_crossref_doi_to_record(
    *,
    doi: str,
    citation_key: str,
    client: Optional[CrossrefClient] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a DOI via Crossref and return a schema-valid CitationRecord."""
    c = client or CrossrefClient()
    work = c.fetch_work_by_doi(doi)
    return crossref_work_to_citation_record(
        work=work,
        citation_key=citation_key,
        status="verified",
        created_at=created_at,
    )


def resolve_crossref_doi_and_upsert(
    *,
    project_folder: str,
    doi: str,
    citation_key: str,
    client: Optional[CrossrefClient] = None,
) -> Dict[str, Any]:
    """Resolve Crossref metadata for a DOI and upsert into citations.json."""
    record = resolve_crossref_doi_to_record(doi=doi, citation_key=citation_key, client=client)
    upsert_citation(project_folder, record)
    return record
