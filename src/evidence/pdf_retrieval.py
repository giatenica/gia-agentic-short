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
import time
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


def _stable_id_from_url(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"pdf:{h[:12]}"


def _try_read_cached_retrieval(metadata_path: Path) -> Optional[Dict[str, Any]]:
    if not metadata_path.exists() or not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _retry_after_seconds(headers: Any) -> Optional[float]:
    """Parse Retry-After header as seconds.

    Keep parsing intentionally minimal and deterministic. We only honor integer
    seconds and clamp to a small maximum so CLI workflows remain responsive.
    """
    try:
        value = headers.get("retry-after") if headers is not None else None
    except Exception:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        seconds = float(int(value.strip()))
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(2.0, seconds)


def _error_dict(exc: BaseException) -> Dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _write_retrieval_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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

    def _sleep_backoff(self, attempt: int) -> None:
        # Simple exponential backoff; keep it short because this is used in CLI workflows.
        delay = min(2.0, 0.5 * (2 ** (attempt - 1)))
        time.sleep(delay)

    def _sleep_retry_after(self, seconds: float) -> None:
        time.sleep(seconds)

    def _download_with_retries(self, url: str, dest_path: Path, *, max_attempts: int = 3) -> tuple[str, int, Optional[str]]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._download_to_path(url, dest_path)
            except ValueError as exc:
                # Deterministic failures (e.g., max size exceeded) should not be retried.
                raise
            except httpx.HTTPStatusError as exc:
                last_exc = exc

                status = exc.response.status_code if exc.response is not None else None
                # Do not retry "not found".
                if status == 404:
                    break

                if attempt >= max_attempts:
                    break

                # Respect Retry-After when provided (common for 429).
                retry_after = _retry_after_seconds(exc.response.headers if exc.response is not None else None)
                if retry_after is not None and status in {429, 503}:
                    self._sleep_retry_after(retry_after)
                else:
                    self._sleep_backoff(attempt)
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    break
                self._sleep_backoff(attempt)

        assert last_exc is not None
        raise last_exc

    def retrieve_pdf_url(
        self,
        url: str,
        source_id: Optional[str] = None,
        filename: Optional[str] = None,
        max_attempts: int = 3,
        doi: Optional[str] = None,
    ) -> RetrievedPdf:
        """Retrieve a remote PDF into `sources/<source_id>/raw/`.

        The source_id defaults to a stable hash of the URL.

        Caching:
        - If `sources/<source_id>/raw/retrieval.json` exists and its `retrieved_from`
          matches the requested URL, this will skip re-downloading.

        Args:
            url: HTTPS URL to download PDF from.
            source_id: Custom source ID (defaults to URL hash).
            filename: Custom filename for the PDF.
            max_attempts: Number of download retries.
            doi: Optional DOI associated with this source for citation mapping.

        Raises:
            ValueError: For non-HTTPS URLs, invalid source_id, or PDF size exceeding limit.
            httpx.HTTPError: For network errors.
        """
        if not _is_https_url(url):
            raise ValueError("Only https URLs are allowed")

        sid = source_id or _stable_id_from_url(url)
        validate_source_id(sid)

        store = EvidenceStore(self.project_folder)
        sp = store.ensure_source_layout(sid)

        metadata_path = sp.raw_dir / "retrieval.json"
        cached = _try_read_cached_retrieval(metadata_path)
        if (
            isinstance(cached, dict)
            and cached.get("ok") is True
            and cached.get("retrieved_from") == url
        ):
            raw_pdf_path = cached.get("raw_pdf_path")
            pdf_path: Optional[Path] = None
            if isinstance(raw_pdf_path, str) and raw_pdf_path:
                try:
                    candidate = (store.project_folder / raw_pdf_path).resolve()
                    if candidate.is_file() and sp.raw_dir.resolve() in candidate.parents:
                        pdf_path = candidate
                except OSError:
                    pdf_path = None

            if pdf_path is None:
                pdf_candidates = sorted(sp.raw_dir.glob("*.pdf"))
                if pdf_candidates:
                    pdf_path = pdf_candidates[0]

            if pdf_path is not None:
                sha256 = cached.get("sha256")
                size_bytes = cached.get("size_bytes")
                if isinstance(sha256, str) and isinstance(size_bytes, int):
                    return RetrievedPdf(
                        source_id=sid,
                        raw_pdf_path=str(pdf_path.relative_to(store.project_folder)),
                        metadata_path=str(metadata_path.relative_to(store.project_folder)),
                        sha256=sha256,
                        size_bytes=size_bytes,
                        retrieved_from=url,
                    )

        name = filename
        if not name:
            parsed = urlparse(url)
            name = Path(parsed.path).name or "document.pdf"
        if not name.lower().endswith(".pdf"):
            name = f"{name}.pdf"
        name = _safe_filename(name)

        pdf_path = sp.raw_dir / name
        attempts: list[Dict[str, Any]] = []
        
        # Build requested params dict with optional DOI
        requested_params: Dict[str, Any] = {"url": url, "filename": name}
        if doi:
            requested_params["doi"] = doi
        
        try:
            sha256, size_bytes, content_type = self._download_with_retries(url, pdf_path, max_attempts=max_attempts)
            attempts.append({"provider": "pdf_url", "ok": True, "retrieved_from": url})
        except Exception as exc:
            attempts.append({"provider": "pdf_url", "ok": False, "retrieved_from": url, "error": _error_dict(exc)})
            record: Dict[str, Any] = {
                "ok": False,
                "source_id": sid,
                "provider": "pdf_url",
                "requested": requested_params,
                "retrieved_from": url,
                "retrieved_at": _utc_now_iso_z(),
                "attempts": attempts,
                "error": _error_dict(exc),
            }
            _write_retrieval_json(metadata_path, record)
            raise

        record: Dict[str, Any] = {
            "ok": True,
            "source_id": sid,
            "provider": "pdf_url",
            "requested": requested_params,
            "retrieved_from": url,
            "retrieved_at": _utc_now_iso_z(),
            "sha256": sha256,
            "size_bytes": size_bytes,
            "content_type": content_type,
            "raw_pdf_path": str(pdf_path.relative_to(store.project_folder)),
            "attempts": attempts,
        }
        _write_retrieval_json(metadata_path, record)

        return RetrievedPdf(
            source_id=sid,
            raw_pdf_path=str(pdf_path.relative_to(store.project_folder)),
            metadata_path=str(metadata_path.relative_to(store.project_folder)),
            sha256=sha256,
            size_bytes=size_bytes,
            retrieved_from=url,
        )

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

        cached = _try_read_cached_retrieval(metadata_path)
        if isinstance(cached, dict):
            cached_arxiv = cached.get("requested", {}).get("arxiv_id") if isinstance(cached.get("requested"), dict) else None
            cached_url = cached.get("retrieved_from")
            if cached_arxiv == arxiv_id and isinstance(cached_url, str) and cached_url:
                if pdf_path.exists() and pdf_path.is_file():
                    sha256 = cached.get("sha256")
                    size_bytes = cached.get("size_bytes")
                    if isinstance(sha256, str) and isinstance(size_bytes, int):
                        return RetrievedPdf(
                            source_id=sid,
                            raw_pdf_path=str(pdf_path.relative_to(store.project_folder)),
                            metadata_path=str(metadata_path.relative_to(store.project_folder)),
                            sha256=sha256,
                            size_bytes=size_bytes,
                            retrieved_from=cached_url,
                        )

        primary_url = arxiv_pdf_url(arxiv_id)
        retrieval_url = primary_url
        semantic_info: Optional[Dict[str, Any]] = None
        arxiv_error: Optional[Dict[str, str]] = None
        attempts: list[Dict[str, Any]] = []

        try:
            sha256, size_bytes, content_type = self._download_with_retries(primary_url, pdf_path)
            retrieval_used = "arxiv"
            attempts.append({"provider": "arxiv", "ok": True, "retrieved_from": primary_url})
        except Exception as arxiv_exc:
            attempts.append({"provider": "arxiv", "ok": False, "retrieved_from": primary_url, "error": _error_dict(arxiv_exc)})
            if not use_semantic_scholar_fallback:
                record: Dict[str, Any] = {
                    "ok": False,
                    "source_id": sid,
                    "provider": "arxiv",
                    "requested": {
                        "arxiv_id": arxiv_id,
                        "input": arxiv_id_or_url,
                    },
                    "retrieved_from": primary_url,
                    "retrieved_at": _utc_now_iso_z(),
                    "attempts": attempts,
                    "error": _error_dict(arxiv_exc),
                }
                if pdf_path.exists() and pdf_path.is_file():
                    record["raw_pdf_path"] = str(pdf_path.relative_to(store.project_folder))
                _write_retrieval_json(metadata_path, record)
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
                sha256, size_bytes, content_type = self._download_with_retries(fallback_url, pdf_path)
                retrieval_used = "semantic_scholar"
                attempts.append({"provider": "semantic_scholar", "ok": True, "retrieved_from": fallback_url})
            except Exception as fallback_exc:
                attempts.append(
                    {
                        "provider": "semantic_scholar",
                        "ok": False,
                        "retrieved_from": retrieval_url,
                        "error": _error_dict(fallback_exc),
                    }
                )
                record: Dict[str, Any] = {
                    "ok": False,
                    "source_id": sid,
                    "provider": "arxiv",
                    "requested": {
                        "arxiv_id": arxiv_id,
                        "input": arxiv_id_or_url,
                    },
                    "retrieved_from": primary_url,
                    "retrieved_at": _utc_now_iso_z(),
                    "attempts": attempts,
                    "error": {
                        "type": "RuntimeError",
                        "message": f"Semantic Scholar fallback failed: {fallback_exc}",
                    },
                    "arxiv_error": arxiv_error,
                    "semantic_scholar": semantic_info,
                }
                if pdf_path.exists() and pdf_path.is_file():
                    record["raw_pdf_path"] = str(pdf_path.relative_to(store.project_folder))
                _write_retrieval_json(metadata_path, record)
                raise RuntimeError(f"Semantic Scholar fallback failed: {fallback_exc}") from arxiv_exc

        record: Dict[str, Any] = {
            "ok": True,
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
            "raw_pdf_path": str(pdf_path.relative_to(store.project_folder)),
            "attempts": attempts,
        }
        if semantic_info is not None:
            record["semantic_scholar"] = semantic_info
        if arxiv_error is not None:
            record["arxiv_error"] = arxiv_error

        _write_retrieval_json(metadata_path, record)

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
