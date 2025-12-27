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


@pytest.mark.unit
def test_source_acquisition_dedups_arxiv_id_and_arxiv_url(temp_project_folder):
    rel = _write_sources_list(
        temp_project_folder,
        [
            {"kind": "arxiv", "id": "1234.56789"},
            {"kind": "arxiv", "url": "https://arxiv.org/abs/1234.56789"},
        ],
    )

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        assert str(request.url) == "https://arxiv.org/pdf/1234.56789.pdf"
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4\nabc")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, follow_redirects=True)

    summary = ingest_sources_list(project_folder=str(temp_project_folder), sources_list_path=rel, client=client)

    assert summary["ok"] is True
    assert calls["count"] == 1
    assert summary["created_source_ids"] == ["arxiv:1234.56789"]


# ==============================================================================
# Tests for build_sources_list_from_citations
# ==============================================================================

@pytest.mark.unit
def test_build_sources_list_extracts_arxiv_from_url():
    """Test that arXiv URLs are correctly extracted and converted to arxiv source kind."""
    from src.evidence.acquisition import build_sources_list_from_citations
    
    citations = [
        {
            "title": "Deep Learning Survey",
            "url": "https://arxiv.org/abs/2301.12345",
            "doi": None,
        },
        {
            "title": "Transformer Paper",
            "url": "https://arxiv.org/pdf/1706.03762.pdf",
            "doi": None,
        },
    ]
    
    sources = build_sources_list_from_citations(citations)
    
    assert len(sources) == 2
    assert all(s["kind"] == "arxiv" for s in sources)
    assert sources[0]["id"] == "2301.12345"
    assert sources[1]["id"] == "1706.03762"


@pytest.mark.unit
def test_build_sources_list_extracts_doi():
    """Test that DOIs are correctly extracted and converted to pdf_url source kind."""
    from src.evidence.acquisition import build_sources_list_from_citations
    
    citations = [
        {
            "title": "Nature Paper",
            "doi": "10.1038/s41586-024-07487-w",
            "url": None,
        },
        {
            "title": "DOI with prefix",
            "doi": "https://doi.org/10.1126/science.abc1234",
            "url": None,
        },
    ]
    
    sources = build_sources_list_from_citations(citations)
    
    assert len(sources) == 2
    assert all(s["kind"] == "pdf_url" for s in sources)
    assert sources[0]["doi"] == "10.1038/s41586-024-07487-w"
    assert sources[1]["doi"] == "10.1126/science.abc1234"
    assert "doi.org" in sources[0]["url"]


@pytest.mark.unit
def test_build_sources_list_extracts_direct_urls():
    """Test that direct PDF/HTML URLs are correctly extracted."""
    from src.evidence.acquisition import build_sources_list_from_citations
    
    citations = [
        {
            "title": "PDF Paper",
            "url": "https://example.com/paper.pdf",
            "doi": None,
        },
        {
            "title": "HTML Paper",
            "url": "https://example.com/article/12345",
            "doi": None,
        },
    ]
    
    sources = build_sources_list_from_citations(citations)
    
    assert len(sources) == 2
    assert sources[0]["kind"] == "pdf_url"
    assert sources[1]["kind"] == "html_url"


@pytest.mark.unit
def test_build_sources_list_deduplicates():
    """Test that duplicate URLs are deduplicated."""
    from src.evidence.acquisition import build_sources_list_from_citations
    
    citations = [
        {"title": "Paper 1", "url": "https://arxiv.org/abs/2301.12345"},
        {"title": "Paper 1 (copy)", "url": "https://arxiv.org/abs/2301.12345"},
    ]
    
    sources = build_sources_list_from_citations(citations)
    
    assert len(sources) == 1


@pytest.mark.unit
def test_build_sources_list_handles_empty_citations():
    """Test that empty citations list returns empty sources."""
    from src.evidence.acquisition import build_sources_list_from_citations
    
    sources = build_sources_list_from_citations([])
    assert sources == []


@pytest.mark.unit
def test_build_sources_list_skips_invalid_citations():
    """Test that invalid citations are skipped."""
    from src.evidence.acquisition import build_sources_list_from_citations
    
    citations = [
        None,
        "not a dict",
        {"title": "No URL or DOI"},  # Missing both url and doi
        {"title": "Valid", "url": "https://arxiv.org/abs/2301.12345"},
    ]
    
    sources = build_sources_list_from_citations(citations)
    
    assert len(sources) == 1
    assert sources[0]["kind"] == "arxiv"


@pytest.mark.unit
def test_build_sources_list_respects_include_flags():
    """Test that include flags control which sources are extracted."""
    from src.evidence.acquisition import build_sources_list_from_citations
    
    citations = [
        {"title": "arXiv", "url": "https://arxiv.org/abs/2301.12345"},
        {"title": "DOI", "doi": "10.1038/s41586-024-07487-w"},
        {"title": "Direct", "url": "https://example.com/paper.pdf"},
    ]
    
    # Only arXiv
    sources = build_sources_list_from_citations(
        citations,
        include_arxiv=True,
        include_doi_urls=False,
        include_direct_urls=False,
    )
    assert len(sources) == 1
    assert sources[0]["kind"] == "arxiv"
    
    # Only DOI
    sources = build_sources_list_from_citations(
        citations,
        include_arxiv=False,
        include_doi_urls=True,
        include_direct_urls=False,
    )
    assert len(sources) == 1
    assert sources[0]["kind"] == "pdf_url"
    assert "doi" in sources[0]


# ==============================================================================
# Tests for write_sources_list
# ==============================================================================

@pytest.mark.unit
def test_write_sources_list_creates_file(temp_project_folder):
    """Test that write_sources_list creates a valid JSON file."""
    from src.evidence.acquisition import write_sources_list
    
    sources = [
        {"kind": "arxiv", "id": "2301.12345", "source_id": "arxiv:2301.12345", "title": "Test"},
    ]
    
    path = write_sources_list(str(temp_project_folder), sources)
    
    assert path.exists()
    assert path.name == "sources_list.json"
    
    content = json.loads(path.read_text())
    assert "sources" in content
    assert content["total"] == 1
    assert "generated_at" in content


# ==============================================================================
# Tests for acquire_sources_from_citations
# ==============================================================================

@pytest.mark.unit
def test_acquire_sources_from_citations_disabled_returns_skip(temp_project_folder):
    """Test that disabled config returns skip result."""
    from src.evidence.acquisition import acquire_sources_from_citations, SourceAcquisitionConfig
    
    cfg = SourceAcquisitionConfig(enabled=False)
    result = acquire_sources_from_citations(
        project_folder=str(temp_project_folder),
        citations_data=[{"title": "Test", "url": "https://arxiv.org/abs/2301.12345"}],
        config=cfg,
    )
    
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "source_acquisition_disabled"


@pytest.mark.unit
def test_acquire_sources_from_citations_empty_citations(temp_project_folder):
    """Test that empty citations returns skip result."""
    from src.evidence.acquisition import acquire_sources_from_citations, SourceAcquisitionConfig
    
    cfg = SourceAcquisitionConfig(enabled=True)
    result = acquire_sources_from_citations(
        project_folder=str(temp_project_folder),
        citations_data=[],
        config=cfg,
    )
    
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "no_downloadable_sources_in_citations"


@pytest.mark.unit
def test_acquire_sources_from_citations_writes_sources_list(temp_project_folder):
    """Test that acquire_sources_from_citations writes sources_list.json."""
    from src.evidence.acquisition import acquire_sources_from_citations, SourceAcquisitionConfig
    
    citations = [{"title": "Test Paper", "url": "https://arxiv.org/abs/2301.12345"}]
    
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4\nabc")
    
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, follow_redirects=True)
    
    cfg = SourceAcquisitionConfig(enabled=True)
    result = acquire_sources_from_citations(
        project_folder=str(temp_project_folder),
        citations_data=citations,
        config=cfg,
        client=client,
    )
    
    # Check sources_list.json was created
    sources_list_path = temp_project_folder / "sources_list.json"
    assert sources_list_path.exists()
    
    content = json.loads(sources_list_path.read_text())
    assert content["total"] == 1
    assert content["sources"][0]["kind"] == "arxiv"
