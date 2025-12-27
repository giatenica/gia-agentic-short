from __future__ import annotations

import json

import httpx
import pytest

from src.evidence.acquisition import ingest_sources_list
from src.evidence.store import EvidenceStore


def _write_sources_list(tmp_project, payload) -> str:
    path = tmp_project / "sources_list.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return "sources_list.json"


@pytest.mark.unit
def test_source_acquisition_dedups_duplicate_pdf_urls(temp_project_folder):
    # Arrange: two identical URLs should map to the same stable source_id.
    url = "https://example.com/paper.pdf"
    rel = _write_sources_list(
        temp_project_folder,
        [
            {"kind": "pdf_url", "url": url},
            {"kind": "pdf_url", "url": url},
        ],
    )

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4\nabc")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, follow_redirects=True)

    # Act
    summary = ingest_sources_list(project_folder=str(temp_project_folder), sources_list_path=rel, client=client)

    # Assert
    assert summary["ok"] is True
    assert calls["count"] == 1
    assert len(summary["created_source_ids"]) == 1


@pytest.mark.unit
def test_source_acquisition_rejects_non_https_urls(temp_project_folder):
    rel = _write_sources_list(
        temp_project_folder,
        [
            {"kind": "pdf_url", "url": "http://example.com/paper.pdf"},
        ],
    )

    summary = ingest_sources_list(project_folder=str(temp_project_folder), sources_list_path=rel, client=httpx.Client())
    assert summary["ok"] is False
    assert summary["errors"][0]["error_type"] == "ValueError"


@pytest.mark.unit
def test_source_acquisition_enforces_max_bytes_on_pdf(temp_project_folder):
    url = "https://example.com/too-big.pdf"
    rel = _write_sources_list(temp_project_folder, [{"kind": "pdf_url", "url": url}])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf", "content-length": str(9999)},
            content=b"%PDF-1.4\nabc",
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, follow_redirects=True)

    from src.evidence.acquisition import SourceAcquisitionConfig

    cfg = SourceAcquisitionConfig(max_download_bytes=10)
    summary = ingest_sources_list(
        project_folder=str(temp_project_folder),
        sources_list_path=rel,
        config=cfg,
        client=client,
    )

    assert summary["ok"] is False
    assert summary["errors"][0]["error_type"] == "ValueError"


@pytest.mark.unit
def test_source_acquisition_html_url_writes_artifacts(temp_project_folder):
    url = "https://example.com/page"
    rel = _write_sources_list(temp_project_folder, [{"kind": "html_url", "url": url}])

    def handler(request: httpx.Request) -> httpx.Response:
        html = b"<html><body><p>This is a sufficiently long paragraph to extract evidence from.</p></body></html>"
        return httpx.Response(200, headers={"content-type": "text/html"}, content=html)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, follow_redirects=True)

    summary = ingest_sources_list(project_folder=str(temp_project_folder), sources_list_path=rel, client=client)
    assert summary["ok"] is True
    assert len(summary["created_source_ids"]) == 1

    store = EvidenceStore(str(temp_project_folder))
    source_id = summary["created_source_ids"][0]

    sp = store.source_paths(source_id)
    assert sp.raw_dir.exists()
    assert (sp.raw_dir / "source.html").exists()
    assert (sp.raw_dir / "source.txt").exists()
    assert (sp.raw_dir / "retrieval.json").exists()
