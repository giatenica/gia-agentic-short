"""Source acquisition automation.

This module extends the evidence tooling with a small, deterministic source
acquisition layer that can ingest a project "sources list" into the evidence
layout under `sources/<source_id>/`.

Supported source kinds:
- arxiv: downloads the PDF and stores it under sources/<source_id>/raw/
- pdf_url: downloads a remote PDF URL into sources/<source_id>/raw/
- html_url: downloads HTML, stores raw HTML, and extracts evidence from the text

Design goals:
- Stable `source_id` mapping from identifiers (URL hash or arXiv id)
- Dedup: repeated entries produce a single source folder
- Safety: only https URLs; enforce centralized max bytes and timeouts via src/config.py
- Graceful degradation: failures return structured error records

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

from src.config import RETRIEVAL, TIMEOUTS
from src.evidence.pdf_retrieval import PdfRetrievalTool, parse_arxiv_id
from src.evidence.store import EvidenceStore
from src.utils.filesystem import validate_source_id


def _utc_now_iso_z() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _is_https_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme.lower() == "https" and bool(parsed.netloc)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._chunks.append(data.strip())

    def get_text(self) -> str:
        return "\n".join(self._chunks).strip()


def _stable_id_from_url(prefix: str, url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"{prefix}:{h[:12]}"


@dataclass(frozen=True)
class SourceAcquisitionConfig:
    enabled: bool = True
    max_attempts: int = 3
    max_download_bytes: int = RETRIEVAL.MAX_PDF_BYTES

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "SourceAcquisitionConfig":
        raw = context.get("source_acquisition")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", True))
        max_attempts = int(raw.get("max_attempts", 3))
        max_download_bytes = int(raw.get("max_download_bytes", RETRIEVAL.MAX_PDF_BYTES))

        if max_attempts < 1:
            max_attempts = 1
        if max_download_bytes < 0:
            max_download_bytes = 0

        return cls(
            enabled=enabled,
            max_attempts=max_attempts,
            max_download_bytes=max_download_bytes,
        )


def _read_sources_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Sources list not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        out = [p for p in payload if isinstance(p, dict)]
        return out
    if isinstance(payload, dict) and isinstance(payload.get("sources"), list):
        return [p for p in payload["sources"] if isinstance(p, dict)]

    raise ValueError("Sources list must be a list[object] or {sources: [...]} dict")


def _compute_source_id(spec: Dict[str, Any]) -> str:
    kind = str(spec.get("kind", "")).strip()
    explicit = spec.get("source_id")
    if isinstance(explicit, str) and explicit.strip():
        validate_source_id(explicit)
        return explicit

    if kind == "arxiv":
        value = spec.get("id") or spec.get("url")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("arxiv source requires 'id' or 'url'")
        # Normalize so URLs and plain IDs map to the same stable source_id.
        arxiv_id = parse_arxiv_id(value.strip())
        safe = arxiv_id.replace("/", "_")
        return f"arxiv:{safe}"

    if kind == "pdf_url":
        url = spec.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("pdf_url source requires 'url'")
        return _stable_id_from_url("pdf", url.strip())

    if kind == "html_url":
        url = spec.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("html_url source requires 'url'")
        return _stable_id_from_url("html", url.strip())

    raise ValueError(f"Unsupported source kind: {kind}")


def ingest_sources_list(
    *,
    project_folder: str,
    sources_list_path: str,
    config: Optional[SourceAcquisitionConfig] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Ingest a project sources list into the evidence layout.

    Returns a structured summary; this function never raises for individual
    source failures.
    """

    cfg = config or SourceAcquisitionConfig()
    if not cfg.enabled:
        return {
            "ok": True,
            "enabled": False,
            "created_source_ids": [],
            "per_source": [],
            "errors": [],
        }

    pf = Path(project_folder).expanduser().resolve()
    list_path = (pf / sources_list_path).resolve()

    specs = _read_sources_list(list_path)

    store = EvidenceStore(str(pf))

    per_source: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # Dedup by computed source_id.
    seen: set[str] = set()

    pdf_tool = PdfRetrievalTool(
        project_folder=str(pf),
        client=client,
        max_pdf_bytes=int(cfg.max_download_bytes),
        download_timeout_seconds=int(TIMEOUTS.PDF_DOWNLOAD),
        connect_timeout_seconds=int(TIMEOUTS.LLM_CONNECT),
    )

    close_client = False
    http_client = client
    if http_client is None:
        timeout = httpx.Timeout(timeout=float(TIMEOUTS.EXTERNAL_API), connect=float(TIMEOUTS.LLM_CONNECT))
        http_client = httpx.Client(timeout=timeout, follow_redirects=True)
        close_client = True

    try:
        for idx, spec in enumerate(specs):
            kind = str(spec.get("kind", "")).strip()
            try:
                source_id = _compute_source_id(spec)
                if source_id in seen:
                    per_source.append(
                        {
                            "index": idx,
                            "kind": kind,
                            "source_id": source_id,
                            "ok": True,
                            "deduped": True,
                        }
                    )
                    continue
                seen.add(source_id)

                store.ensure_source_layout(source_id)

                if kind == "arxiv":
                    value = spec.get("id") or spec.get("url")
                    retrieved = pdf_tool.retrieve_arxiv_pdf(
                        arxiv_id_or_url=str(value),
                        source_id=source_id,
                        use_semantic_scholar_fallback=bool(spec.get("use_semantic_scholar_fallback", True)),
                    )
                    per_source.append(
                        {
                            "index": idx,
                            "kind": kind,
                            "source_id": retrieved.source_id,
                            "ok": True,
                            "raw_pdf_path": retrieved.raw_pdf_path,
                            "retrieved_from": retrieved.retrieved_from,
                        }
                    )
                    continue

                if kind == "pdf_url":
                    url = str(spec.get("url")).strip()
                    retrieved = pdf_tool.retrieve_pdf_url(
                        url=url,
                        source_id=source_id,
                        filename=spec.get("filename") if isinstance(spec.get("filename"), str) else None,
                        max_attempts=int(cfg.max_attempts),
                    )
                    per_source.append(
                        {
                            "index": idx,
                            "kind": kind,
                            "source_id": retrieved.source_id,
                            "ok": True,
                            "raw_pdf_path": retrieved.raw_pdf_path,
                            "retrieved_from": retrieved.retrieved_from,
                        }
                    )
                    continue

                if kind == "html_url":
                    url = spec.get("url")
                    if not isinstance(url, str) or not url.strip():
                        raise ValueError("html_url source requires 'url'")
                    url = url.strip()
                    if not _is_https_url(url):
                        raise ValueError("Only https URLs are allowed")

                    sp = store.ensure_source_layout(source_id)
                    raw_html_path = sp.raw_dir / "source.html"
                    meta_path = sp.raw_dir / "retrieval.json"

                    # Cache: if we already retrieved the same URL, do not fetch again.
                    if meta_path.exists() and raw_html_path.exists():
                        try:
                            cached = json.loads(meta_path.read_text(encoding="utf-8"))
                        except Exception:
                            cached = None
                        if isinstance(cached, dict) and cached.get("retrieved_from") == url:
                            per_source.append(
                                {
                                    "index": idx,
                                    "kind": kind,
                                    "source_id": source_id,
                                    "ok": True,
                                    "deduped": True,
                                    "retrieved_from": url,
                                }
                            )
                            continue

                    resp = http_client.get(url)
                    resp.raise_for_status()

                    declared = resp.headers.get("content-length")
                    if declared is not None:
                        try:
                            declared_bytes = int(declared)
                        except (TypeError, ValueError):
                            declared_bytes = None
                        if declared_bytes is not None and declared_bytes > cfg.max_download_bytes:
                            raise ValueError("HTML exceeds max size")

                    content = resp.content
                    if len(content) > cfg.max_download_bytes:
                        raise ValueError("HTML exceeds max size")

                    raw_html_path.write_bytes(content)

                    extractor = _HTMLTextExtractor()
                    extractor.feed(content.decode("utf-8", errors="replace"))
                    text = extractor.get_text()
                    raw_text_path = sp.raw_dir / "source.txt"
                    raw_text_path.write_text(text + "\n" if text else "", encoding="utf-8")

                    content_type = resp.headers.get("content-type")
                    sha256 = hashlib.sha256(content).hexdigest()

                    meta = {
                        "source_id": source_id,
                        "provider": "html_url",
                        "requested": {"url": url},
                        "retrieved_from": url,
                        "retrieved_at": _utc_now_iso_z(),
                        "sha256": sha256,
                        "size_bytes": len(content),
                        "content_type": content_type,
                    }
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

                    per_source.append(
                        {
                            "index": idx,
                            "kind": kind,
                            "source_id": source_id,
                            "ok": True,
                            "retrieved_from": url,
                            "raw_html_path": str(raw_html_path.relative_to(store.project_folder)),
                            "raw_text_path": str(raw_text_path.relative_to(store.project_folder)),
                        }
                    )
                    continue

                raise ValueError(f"Unsupported source kind: {kind}")

            except Exception as e:
                logger.warning(f"Source acquisition failed (index={idx}, kind={kind}): {type(e).__name__}: {e}")
                err = {
                    "index": idx,
                    "kind": kind,
                    "ok": False,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
                errors.append(err)
                per_source.append(err)

    finally:
        if close_client and http_client is not None:
            http_client.close()

    created_source_ids: List[str] = [
        str(p.get("source_id"))
        for p in per_source
        if p.get("ok") is True and isinstance(p.get("source_id"), str)
    ]

    return {
        "ok": not errors,
        "enabled": True,
        "sources_list_path": str(list_path),
        "created_source_ids": sorted(set(created_source_ids)),
        "per_source": per_source,
        "errors": errors,
    }


def find_default_sources_list_path(project_folder: str) -> Optional[str]:
    pf = Path(project_folder).expanduser().resolve()
    candidates = [
        pf / "sources_list.json",
        pf / "sources.json",
        pf / "inputs" / "sources.json",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return str(p.relative_to(pf))
    return None


def ingest_sources_list_if_present(
    *, project_folder: str, context: Optional[Dict[str, Any]] = None, client: Optional[httpx.Client] = None
) -> Dict[str, Any]:
    ctx = context if isinstance(context, dict) else {}
    cfg = SourceAcquisitionConfig.from_context(ctx)
    rel = find_default_sources_list_path(project_folder)
    if not rel:
        return {"ok": True, "enabled": bool(cfg.enabled), "skipped": True, "reason": "no_sources_list"}
    return ingest_sources_list(project_folder=project_folder, sources_list_path=rel, config=cfg, client=client)


def build_sources_list_from_citations(
    citations_data: List[Dict[str, Any]],
    *,
    include_arxiv: bool = True,
    include_doi_urls: bool = True,
    include_direct_urls: bool = True,
) -> List[Dict[str, Any]]:
    """Convert citations_data into a sources_list for source acquisition.
    
    This function extracts downloadable sources from citation metadata,
    converting DOIs to unpaywall/sci-hub URLs and detecting arXiv links.
    
    Args:
        citations_data: List of citation dicts with doi/url/title/authors fields
        include_arxiv: Whether to include arXiv papers (default: True)
        include_doi_urls: Whether to include DOI-based URLs (default: True)
        include_direct_urls: Whether to include direct PDF/HTML URLs (default: True)
    
    Returns:
        List of source specs compatible with ingest_sources_list()
    """
    sources: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    
    for citation in citations_data:
        if not isinstance(citation, dict):
            continue
        
        title = citation.get("title", "")
        doi = citation.get("doi")
        url = citation.get("url")
        
        # Try to extract arXiv ID from URL or DOI
        arxiv_id = None
        arxiv_url_source_id = None
        if isinstance(url, str) and url.strip():
            url = url.strip()
            # Check for arXiv URL patterns
            if "arxiv.org" in url.lower():
                try:
                    arxiv_id = parse_arxiv_id(url)
                    # Also compute the URL-based source_id to mark as seen
                    arxiv_url_source_id = _stable_id_from_url("htm", url)
                except Exception:
                    pass
        
        # arXiv sources - highest priority, free full text
        if include_arxiv and arxiv_id:
            source_id = f"arxiv:{arxiv_id.replace('/', '_')}"
            # Mark both the arXiv source_id and the URL source_id as seen
            # to prevent duplicate extraction via direct URL path
            if arxiv_url_source_id:
                seen_ids.add(arxiv_url_source_id)
            if source_id not in seen_ids:
                seen_ids.add(source_id)
                sources.append({
                    "kind": "arxiv",
                    "id": arxiv_id,
                    "source_id": source_id,
                    "title": title,
                    "use_semantic_scholar_fallback": True,
                })
            continue  # Skip other extraction for this citation (even if duplicate)
        
        # DOI-based sources - try to construct Unpaywall/Open Access URLs
        if include_doi_urls and isinstance(doi, str) and doi.strip():
            doi = doi.strip()
            # Normalize DOI
            if doi.startswith("https://doi.org/"):
                doi = doi[len("https://doi.org/"):]
            elif doi.startswith("http://doi.org/"):
                doi = doi[len("http://doi.org/"):]
            elif doi.startswith("doi:"):
                doi = doi[4:]
            
            if doi:
                # Use DOI to construct Unpaywall API URL for open access lookup
                # For now, construct a doi.org URL which may redirect to publisher
                # The acquisition system will try to download the PDF
                doi_url = f"https://doi.org/{doi}"
                source_id = _stable_id_from_url("doi", doi_url)
                if source_id not in seen_ids:
                    seen_ids.add(source_id)
                    sources.append({
                        "kind": "pdf_url",
                        "url": doi_url,
                        "source_id": source_id,
                        "title": title,
                        "doi": doi,
                    })
                    continue
        
        # Direct URL sources - PDF or HTML
        if include_direct_urls and isinstance(url, str) and url.strip():
            url = url.strip()
            if not url.startswith("https://"):
                continue
            
            # Determine if this looks like a PDF URL
            is_pdf_like = (
                url.lower().endswith(".pdf") or
                "/pdf/" in url.lower() or
                "pdf" in url.lower().split("/")[-1]
            )
            
            kind = "pdf_url" if is_pdf_like else "html_url"
            source_id = _stable_id_from_url(kind[:3], url)
            
            if source_id not in seen_ids:
                seen_ids.add(source_id)
                sources.append({
                    "kind": kind,
                    "url": url,
                    "source_id": source_id,
                    "title": title,
                })
    
    return sources


def write_sources_list(project_folder: str, sources: List[Dict[str, Any]]) -> Path:
    """Write a sources_list.json file to the project folder.
    
    Args:
        project_folder: Path to the project folder
        sources: List of source specs
    
    Returns:
        Path to the written sources_list.json file
    """
    pf = Path(project_folder).expanduser().resolve()
    sources_list_path = pf / "sources_list.json"
    
    payload = {
        "sources": sources,
        "generated_at": _utc_now_iso_z(),
        "total": len(sources),
    }
    
    sources_list_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    
    logger.info(f"Wrote sources_list.json with {len(sources)} sources to {sources_list_path}")
    return sources_list_path


def acquire_sources_from_citations(
    *,
    project_folder: str,
    citations_data: List[Dict[str, Any]],
    config: Optional[SourceAcquisitionConfig] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """End-to-end source acquisition from citations data.
    
    This is the main integration function that:
    1. Builds a sources_list from citations_data
    2. Writes sources_list.json to the project folder
    3. Runs source acquisition to download PDFs/HTMLs
    
    Args:
        project_folder: Path to the project folder
        citations_data: List of citation dicts from Edison/literature search
        config: Optional acquisition config
        client: Optional httpx client for reuse
    
    Returns:
        Acquisition result dict with created_source_ids and errors
    """
    cfg = config or SourceAcquisitionConfig()
    if not cfg.enabled:
        return {
            "ok": True,
            "enabled": False,
            "skipped": True,
            "reason": "source_acquisition_disabled",
        }
    
    # Build sources list from citations
    sources = build_sources_list_from_citations(citations_data)
    
    if not sources:
        return {
            "ok": True,
            "enabled": True,
            "skipped": True,
            "reason": "no_downloadable_sources_in_citations",
            "citations_count": len(citations_data),
        }
    
    # Write sources list
    sources_list_path = write_sources_list(project_folder, sources)
    
    # Run acquisition
    result = ingest_sources_list(
        project_folder=project_folder,
        sources_list_path=str(sources_list_path.name),
        config=cfg,
        client=client,
    )
    
    return result
