"""PDF retrieval helpers.

Implements deterministic, time-bounded retrieval of open-access PDFs into the
project evidence layout:

- `sources/<source_id>/raw/<filename>.pdf`
- `sources/<source_id>/raw/retrieval.json`

The primary target is arXiv PDFs. Optional Semantic Scholar discovery can be
used as a fallback when the arXiv download URL fails.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from src.config import RETRIEVAL, TIMEOUTS
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


def _safe_filename(stem: str) -> str:
    safe = []
    for ch in stem:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "document"


def parse_arxiv_id(value: str) -> str:
    """Parse an arXiv identifier from a plain id, arXiv URL, or 'arXiv:' prefixed string."""
    v = (value or "").strip()
    if not v:
        raise ValueError("arXiv id must be non-empty")

    if v.lower().startswith("arxiv:"):
        v = v.split(":", 1)[1].strip()

    if v.lower().startswith("http://") or v.lower().startswith("https://"):
        parsed = urlparse(v)
        if parsed.netloc.lower() not in {"arxiv.org", "www.arxiv.org"}:
            raise ValueError("Only arxiv.org URLs are supported")
        parts = [p for p in parsed.path.split("/") if p]
        # /abs/<id>
        if len(parts) >= 2 and parts[0] in {"abs", "pdf"}:
            v = parts[1]
        else:
            raise ValueError("Unrecognized arXiv URL format")

    v = v.strip()
    if v.lower().endswith(".pdf"):
        v = v[: -len(".pdf")]

    if ".." in v:
        raise ValueError("Invalid arXiv id")

    # Keep slashes for URL construction (old-style ids), but do not allow whitespace.
    if any(ch.isspace() for ch in v):
        raise ValueError("Invalid arXiv id")

    return v


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


@dataclass(frozen=True)
class RetrievedPdf:
    source_id: str
    raw_pdf_path: str
    metadata_path: str
    sha256: str
    size_bytes: int
    retrieved_from: str


class PdfRetrievalTool:
    """Retrieve PDFs into the project evidence layout."""

    def __init__(
        self,
        project_folder: str,
        client: Optional[httpx.Client] = None,
        max_pdf_bytes: int = RETRIEVAL.MAX_PDF_BYTES,
        download_timeout_seconds: int = TIMEOUTS.PDF_DOWNLOAD,
        connect_timeout_seconds: int = TIMEOUTS.LLM_CONNECT,
    ):
        self.project_folder = project_folder
        self.max_pdf_bytes = max_pdf_bytes
        self.download_timeout_seconds = download_timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self._client = client

    def _client_or_default(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        timeout = httpx.Timeout(
            timeout=self.download_timeout_seconds,
            connect=self.connect_timeout_seconds,
        )
        return httpx.Client(timeout=timeout, follow_redirects=True)

    def _download_to_path(self, url: str, dest_path: Path) -> tuple[str, int, Optional[str]]:
        if not _is_https_url(url):
            raise ValueError("Only https URLs are allowed")

        tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        h = hashlib.sha256()
        total = 0
        content_type: Optional[str] = None

        client = self._client_or_default()
        close_client = self._client is None
        try:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type")
                declared_len = resp.headers.get("content-length")
                if declared_len is not None:
                    try:
                        declared_bytes = int(declared_len)
                    except (TypeError, ValueError):
                        declared_bytes = None
                    if declared_bytes is not None and declared_bytes > self.max_pdf_bytes:
                        raise ValueError("PDF exceeds max size")

                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > self.max_pdf_bytes:
                            raise ValueError("PDF exceeds max size")
                        h.update(chunk)
                        f.write(chunk)
                    f.flush()

            tmp_path.replace(dest_path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                # Best-effort cleanup; failure to remove the temporary file is non-fatal.
                pass
            raise
        finally:
            if close_client:
                client.close()

        return h.hexdigest(), total, content_type

    def _semantic_scholar_open_access_pdf_url(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Return OA PDF discovery info via Semantic Scholar Graph API.

        Returns None when not available.
        """
        client = self._client_or_default()
        close_client = self._client is None
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
            params = {
                "fields": "isOpenAccess,openAccessPdf,license,externalIds",
            }
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                return None
            raise
        except httpx.RequestError:
            return None
        finally:
            if close_client:
                client.close()

        if not isinstance(payload, dict):
            return None

        oa = payload.get("openAccessPdf")
        if not isinstance(oa, dict):
            return None
        pdf_url = oa.get("url")
        if not isinstance(pdf_url, str) or not pdf_url:
            return None
        if not _is_https_url(pdf_url):
            return None

        return {
            "paper_id": payload.get("paperId"),
            "is_open_access": payload.get("isOpenAccess"),
            "license": payload.get("license"),
            "open_access_pdf": oa,
        }

    def retrieve_arxiv_pdf(
        self,
        arxiv_id_or_url: str,
        source_id: Optional[str] = None,
        use_semantic_scholar_fallback: bool = True,
    ) -> RetrievedPdf:
        """Retrieve an arXiv PDF into `sources/<source_id>/raw/`.

        Writes two files:
        - The downloaded PDF under `sources/<source_id>/raw/<filename>.pdf`
        - A metadata JSON file under `sources/<source_id>/raw/retrieval.json`

        When `use_semantic_scholar_fallback=True`, a failed arXiv download will
        trigger an OA PDF lookup via Semantic Scholar. If the fallback succeeds,
        metadata will include the original arXiv error context.

        Raises:
            ValueError: Invalid arXiv id/URL, invalid computed source_id, non-HTTPS URL,
                or PDF size exceeding the configured maximum.
            httpx.HTTPError: Network/HTTP failures from httpx when downloading.
            RuntimeError: When both arXiv download and fallback fail.
        """
        arxiv_id = parse_arxiv_id(arxiv_id_or_url)

        safe_id_for_source = arxiv_id.replace("/", "_")
        sid = source_id or f"arxiv:{safe_id_for_source}"
        validate_source_id(sid)

        store = EvidenceStore(self.project_folder)
        sp = store.ensure_source_layout(sid)

        pdf_name = _safe_filename(f"{safe_id_for_source}.pdf")
        pdf_path = sp.raw_dir / pdf_name
        metadata_path = sp.raw_dir / "retrieval.json"

        primary_url = arxiv_pdf_url(arxiv_id)
        retrieval_url = primary_url
        semantic_info: Optional[Dict[str, Any]] = None
        arxiv_error: Optional[Dict[str, str]] = None

        try:
            sha256, size_bytes, content_type = self._download_to_path(primary_url, pdf_path)
            retrieval_used = "arxiv"
        except Exception as arxiv_exc:
            if not use_semantic_scholar_fallback:
                raise

            arxiv_error = {
                "type": type(arxiv_exc).__name__,
                "message": str(arxiv_exc),
            }

            try:
                semantic_info = self._semantic_scholar_open_access_pdf_url(arxiv_id)
                if not semantic_info:
                    raise RuntimeError("Semantic Scholar fallback did not return OA PDF metadata")

                fallback_url = semantic_info.get("open_access_pdf", {}).get("url")
                if not isinstance(fallback_url, str) or not fallback_url:
                    raise RuntimeError("Semantic Scholar fallback metadata missing OA PDF URL")

                retrieval_url = fallback_url
                sha256, size_bytes, content_type = self._download_to_path(fallback_url, pdf_path)
                retrieval_used = "semantic_scholar"
            except Exception as fallback_exc:
                raise RuntimeError(f"Semantic Scholar fallback failed: {fallback_exc}") from arxiv_exc

        record: Dict[str, Any] = {
            "source_id": sid,
            "provider": retrieval_used,
            "requested": {
                "arxiv_id": arxiv_id,
                "input": arxiv_id_or_url,
            },
            "retrieved_from": retrieval_url,
            "retrieved_at": _utc_now_iso_z(),
            "sha256": sha256,
            "size_bytes": size_bytes,
            "content_type": content_type,
        }
        if semantic_info is not None:
            record["semantic_scholar"] = semantic_info
        if arxiv_error is not None:
            record["arxiv_error"] = arxiv_error

        metadata_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return RetrievedPdf(
            source_id=sid,
            raw_pdf_path=str(pdf_path.relative_to(store.project_folder)),
            metadata_path=str(metadata_path.relative_to(store.project_folder)),
            sha256=sha256,
            size_bytes=size_bytes,
            retrieved_from=retrieval_url,
        )


def retrieve_arxiv_pdf(
    project_folder: str,
    arxiv_id_or_url: str,
    source_id: Optional[str] = None,
    use_semantic_scholar_fallback: bool = True,
) -> RetrievedPdf:
    """Convenience wrapper for `PdfRetrievalTool.retrieve_arxiv_pdf`."""
    return PdfRetrievalTool(project_folder).retrieve_arxiv_pdf(
        arxiv_id_or_url=arxiv_id_or_url,
        source_id=source_id,
        use_semantic_scholar_fallback=use_semantic_scholar_fallback,
    )
