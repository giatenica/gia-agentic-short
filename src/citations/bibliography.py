"""Bibliography builder.

Writes per-project bibliography artifacts under:
- bibliography/citations.json (registry)
- bibliography/references.bib (BibTeX)

This module is offline and deterministic: it does not call any external APIs and
avoids timestamps in outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from src.citations.crossref import normalize_doi
from src.citations.registry import ensure_bibliography_layout, load_citations, save_citations
from src.utils.schema_validation import validate_citation_record
from src.utils.validation import validate_project_folder


@dataclass(frozen=True)
class BibliographyPaths:
    """Resolved bibliography paths for a project."""

    project_folder: Path
    bibliography_dir: Path
    citations_path: Path
    references_bib_path: Path


def bibliography_paths(project_folder: str | Path) -> BibliographyPaths:
    """Return canonical bibliography paths for a project folder."""

    pf = validate_project_folder(project_folder)
    bib_dir = pf / "bibliography"
    return BibliographyPaths(
        project_folder=pf,
        bibliography_dir=bib_dir,
        citations_path=bib_dir / "citations.json",
        references_bib_path=bib_dir / "references.bib",
    )


def _strip_or_none(value: Any) -> Optional[str]:
    if isinstance(value, str):
        s = value.strip()
        return s or None
    return None


def _record_doi(record: Dict[str, Any]) -> Optional[str]:
    identifiers = record.get("identifiers")
    if not isinstance(identifiers, dict):
        return None
    raw = identifiers.get("doi")
    doi = _strip_or_none(raw)
    if not doi:
        return None
    try:
        return normalize_doi(doi)
    except Exception:
        return doi


def _status_rank(status: Any) -> int:
    if status == "verified":
        return 2
    if status == "unverified":
        return 1
    return 0


def _non_empty_optional_field_count(record: Dict[str, Any]) -> int:
    count = 0
    for key in (
        "venue",
        "volume",
        "issue",
        "pages",
        "publisher",
        "url",
        "abstract",
        "notes",
    ):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            count += 1

    identifiers = record.get("identifiers")
    if isinstance(identifiers, dict):
        if _strip_or_none(identifiers.get("doi")):
            count += 1
        if _strip_or_none(identifiers.get("arxiv")):
            count += 1

    version = record.get("version")
    if isinstance(version, dict) and version:
        count += 1

    keywords = record.get("keywords")
    if isinstance(keywords, list) and any(isinstance(k, str) and k.strip() for k in keywords):
        count += 1

    return count


def _merge_records_prefer_primary(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    """Merge secondary into primary by filling empty fields.

    The citation key from the primary record is preserved.
    """

    merged = dict(primary)

    for key in (
        "venue",
        "volume",
        "issue",
        "pages",
        "publisher",
        "url",
        "abstract",
        "keywords",
        "version",
        "notes",
        "metadata",
    ):
        if key not in merged or merged.get(key) in (None, "", [], {}):
            if key in secondary and secondary.get(key) not in (None, "", [], {}):
                merged[key] = secondary.get(key)

    p_ident = merged.get("identifiers")
    s_ident = secondary.get("identifiers")
    if not isinstance(p_ident, dict):
        if isinstance(s_ident, dict):
            merged["identifiers"] = dict(s_ident)
    else:
        if isinstance(s_ident, dict):
            p2 = dict(p_ident)
            for ik in ("doi", "arxiv"):
                if p2.get(ik) in (None, "") and s_ident.get(ik) not in (None, ""):
                    p2[ik] = s_ident.get(ik)
            merged["identifiers"] = p2

    return merged


def dedupe_citation_records_by_doi(
    records: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Deduplicate CitationRecords by normalized DOI.

    Returns:
        - Deduped list of records.
        - Mapping from dropped citation_key -> kept citation_key.

    Notes:
        - Only records with a DOI participate in dedupe.
        - If multiple records share a DOI, the kept record is chosen
          deterministically and then filled with any missing fields from
          dropped records.
    """

    by_doi: Dict[str, List[Dict[str, Any]]] = {}
    no_doi: List[Dict[str, Any]] = []

    for r in records:
        if not isinstance(r, dict):
            continue
        doi = _record_doi(r)
        if doi:
            by_doi.setdefault(doi, []).append(r)
        else:
            no_doi.append(r)

    dropped_to_kept: Dict[str, str] = {}
    deduped: List[Dict[str, Any]] = []

    for doi, group in sorted(by_doi.items(), key=lambda kv: kv[0]):
        sorted_group = sorted(
            group,
            key=lambda rec: (
                -_status_rank(rec.get("status")),
                -_non_empty_optional_field_count(rec),
                str(rec.get("created_at") or ""),
                str(rec.get("citation_key") or ""),
            ),
        )
        kept = dict(sorted_group[0])
        kept_key = str(kept.get("citation_key") or "")

        for other in sorted_group[1:]:
            other_key = str(other.get("citation_key") or "")
            if other_key:
                dropped_to_kept[other_key] = kept_key
            kept = _merge_records_prefer_primary(kept, other)

        deduped.append(kept)

    # Preserve non-DOI records as-is.
    deduped.extend(no_doi)

    # Ensure schema validity (best effort);
    # callers can validate strictly when saving.
    ok: List[Dict[str, Any]] = []
    for r in deduped:
        try:
            validate_citation_record(r)
            ok.append(r)
        except Exception as e:
            key = r.get("citation_key")
            logger.warning(f"Dropping invalid CitationRecord during DOI dedupe: key={key} err={e}")

    return ok, dropped_to_kept


def mint_stable_citation_key(
    *,
    authors: List[str],
    year: int,
    title: str,
    existing_keys: Iterable[str] = (),
) -> str:
    """Mint a stable citation key.

    Strategy:
        - Base: <first-author-lastname><year>
                - If collision, append alphabetic suffixes: a..z, aa..az, ba.. etc.
                - If the first author is missing, incorporate a stable short hash of the title
                    into the base key to reduce collisions.

    This helper is intended for importing records that do not yet have a key.
    """

    def _index_to_suffix(index: int) -> str:
        """Convert 0-based index to suffix: 0 -> a, 25 -> z, 26 -> aa."""
        n = index + 1
        chars: List[str] = []
        while n > 0:
            n -= 1
            n, rem = divmod(n, 26)
            chars.append(chr(ord("a") + rem))
        return "".join(reversed(chars))

    first_author = authors[0] if authors and isinstance(authors[0], str) else ""
    first_author = first_author.strip()

    last_name = "Unknown"
    if first_author:
        parts = first_author.split()
        last_name = parts[-1]

    base = "".join(ch for ch in last_name if ch.isalnum())
    if not base:
        base = "Unknown"

    base_key = f"{base}{int(year)}"

    # If author info is missing, use a short stable title hash to reduce collisions.
    # Keep it alphanumeric for compatibility with downstream citation key parsing.
    if base == "Unknown":
        normalized_title = (title or "").strip().lower().encode("utf-8")
        if normalized_title:
            digest = hashlib.sha1(normalized_title).hexdigest()[:6]
            base_key = f"{base_key}{digest}"

    taken = {k for k in existing_keys if isinstance(k, str) and k.strip()}
    if base_key not in taken:
        return base_key

    # Collision: choose suffix based on deterministic sequence.
    suffix_index = 0
    while True:
        suffix = _index_to_suffix(suffix_index)
        key = f"{base_key}{suffix}"
        if key not in taken:
            return key
        suffix_index += 1


def citation_record_to_bibtex(record: Dict[str, Any]) -> str:
    """Convert a CitationRecord into a BibTeX entry string.

    Note:
        Field values are inserted mostly verbatim (aside from replacing newlines
        with spaces). Callers should ensure values are BibTeX-safe if they may
        contain characters that could break BibTeX syntax.
    """

    key = str(record.get("citation_key") or "").strip()
    if not key:
        raise ValueError("CitationRecord.citation_key must be a non-empty string")

    title = str(record.get("title") or "").strip()
    authors = record.get("authors")
    year = record.get("year")

    if not isinstance(authors, list) or not all(isinstance(a, str) and a.strip() for a in authors):
        raise ValueError(f"Invalid authors list for citation_key={key}")
    if not isinstance(year, int):
        raise ValueError(f"Invalid year for citation_key={key}")

    author_value = " and ".join(a.strip() for a in authors)

    venue = _strip_or_none(record.get("venue"))
    volume = _strip_or_none(record.get("volume"))
    issue = _strip_or_none(record.get("issue"))
    pages = _strip_or_none(record.get("pages"))
    publisher = _strip_or_none(record.get("publisher"))
    url = _strip_or_none(record.get("url"))

    doi: Optional[str] = None
    identifiers = record.get("identifiers")
    if isinstance(identifiers, dict):
        doi = _strip_or_none(identifiers.get("doi"))
        if doi:
            doi = _record_doi(record) or doi

    entry_type = "article" if venue else "misc"

    fields: List[Tuple[str, str]] = [
        ("title", title),
        ("author", author_value),
        ("year", str(year)),
    ]
    if venue:
        fields.append(("journal", venue))
    if volume:
        fields.append(("volume", volume))
    if issue:
        fields.append(("number", issue))
    if pages:
        fields.append(("pages", pages))
    if publisher:
        fields.append(("publisher", publisher))
    if doi:
        fields.append(("doi", doi))
    if url:
        fields.append(("url", url))

    lines = [f"@{entry_type}{{{key},"]
    for field, value in fields:
        v = value.replace("\n", " ").strip()
        if not v:
            continue
        lines.append(f"  {field} = {{{v}}},")
    lines.append("}")

    return "\n".join(lines)


def build_bibliography(
    project_folder: str | Path,
    *,
    records: Optional[List[Dict[str, Any]]] = None,
    validate: bool = True,
) -> BibliographyPaths:
    """Build canonical bibliography artifacts for a project.

    Writes:
        - bibliography/citations.json (deduped by DOI)
        - bibliography/references.bib

    Returns:
        Paths to the written artifacts.
    """

    paths = bibliography_paths(project_folder)
    ensure_bibliography_layout(project_folder)

    source_records = records if records is not None else load_citations(project_folder, validate=validate)

    deduped, dropped_to_kept = dedupe_citation_records_by_doi(source_records)
    if dropped_to_kept:
        logger.info(f"DOI dedupe collapsed {len(dropped_to_kept)} citation records")

    # Sort for deterministic registry output.
    deduped_sorted = sorted(deduped, key=lambda r: str(r.get("citation_key") or ""))

    # Save updated citations registry.
    save_citations(project_folder, deduped_sorted, validate=validate)

    # Write BibTeX.
    entries = [citation_record_to_bibtex(r) for r in deduped_sorted]
    content = "% Canonical bibliography\n% Generated by bibliography builder\n\n" + "\n\n".join(entries) + "\n"
    paths.references_bib_path.write_text(content, encoding="utf-8")

    return paths
