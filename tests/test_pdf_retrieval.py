"""Tests for PDF retrieval.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import json

import httpx
import pytest

from src.evidence.pdf_retrieval import PdfRetrievalTool, _safe_filename, parse_arxiv_id


def _mock_transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.unit
def test_pdf_retrieval_downloads_arxiv_pdf_and_writes_metadata(temp_project_folder):
    pdf_bytes = b"%PDF-1.7\n1 0 obj\nendobj\n%%EOF\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://arxiv.org/pdf/1234.56789.pdf"
        return httpx.Response(
            200,
            content=pdf_bytes,
            headers={"content-type": "application/pdf"},
        )

    client = httpx.Client(transport=_mock_transport(handler))
    tool = PdfRetrievalTool(str(temp_project_folder), client=client, max_pdf_bytes=10_000)

    result = tool.retrieve_arxiv_pdf("1234.56789", use_semantic_scholar_fallback=False)

    raw_pdf = temp_project_folder / result.raw_pdf_path
    meta_path = temp_project_folder / result.metadata_path

    assert raw_pdf.exists()
    assert raw_pdf.read_bytes() == pdf_bytes
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["source_id"].startswith("arxiv:")
    assert meta["provider"] == "arxiv"
    assert meta["retrieved_from"] == "https://arxiv.org/pdf/1234.56789.pdf"
    assert meta["sha256"] == result.sha256
    assert meta["size_bytes"] == len(pdf_bytes)


@pytest.mark.unit
def test_pdf_retrieval_rejects_non_https_url(temp_project_folder):
    tool = PdfRetrievalTool(str(temp_project_folder))
    with pytest.raises(ValueError, match="Only https URLs are allowed"):
        tool._download_to_path("http://example.com/a.pdf", temp_project_folder / "x.pdf")


@pytest.mark.unit
def test_pdf_retrieval_enforces_max_size(temp_project_folder):
    pdf_bytes = b"%PDF-" + (b"x" * 200)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=pdf_bytes, headers={"content-type": "application/pdf"})

    client = httpx.Client(transport=_mock_transport(handler))
    tool = PdfRetrievalTool(str(temp_project_folder), client=client, max_pdf_bytes=50)

    with pytest.raises(ValueError, match="PDF exceeds max size"):
        tool.retrieve_arxiv_pdf("1234.56789", use_semantic_scholar_fallback=False)

    # Should not leave partial files in the evidence layout
    sources_dir = temp_project_folder / "sources"
    if sources_dir.exists():
        leftovers = list(sources_dir.rglob("*.tmp"))
        assert leftovers == []


@pytest.mark.unit
def test_pdf_retrieval_uses_semantic_scholar_fallback_when_arxiv_fails(temp_project_folder):
    pdf_bytes = b"%PDF-1.7\n%%EOF\n"
    s2_pdf = "https://example.org/open.pdf"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://arxiv.org/pdf/1234.56789.pdf":
            return httpx.Response(404, content=b"not found")
        if url.startswith("https://api.semanticscholar.org/graph/v1/paper/arXiv:1234.56789"):
            return httpx.Response(
                200,
                json={
                    "paperId": "abc",
                    "isOpenAccess": True,
                    "license": "cc-by",
                    "openAccessPdf": {"url": s2_pdf, "status": "OPEN"},
                    "externalIds": {"ArXiv": "1234.56789"},
                },
            )
        if url == s2_pdf:
            return httpx.Response(200, content=pdf_bytes, headers={"content-type": "application/pdf"})
        raise AssertionError(f"Unexpected URL: {url}")

    client = httpx.Client(transport=_mock_transport(handler))
    tool = PdfRetrievalTool(str(temp_project_folder), client=client, max_pdf_bytes=10_000)

    result = tool.retrieve_arxiv_pdf("1234.56789", use_semantic_scholar_fallback=True)

    raw_pdf = temp_project_folder / result.raw_pdf_path
    meta_path = temp_project_folder / result.metadata_path
    assert raw_pdf.exists()
    assert raw_pdf.read_bytes() == pdf_bytes

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["provider"] == "semantic_scholar"
    assert meta["retrieved_from"] == s2_pdf
    assert "semantic_scholar" in meta
    assert meta["semantic_scholar"]["license"] == "cc-by"


@pytest.mark.unit
def test_parse_arxiv_id_accepts_prefix_url_and_pdf_suffix():
    assert parse_arxiv_id("arXiv:1234.56789") == "1234.56789"
    assert parse_arxiv_id("https://arxiv.org/abs/1234.56789") == "1234.56789"
    assert parse_arxiv_id("https://arxiv.org/pdf/1234.56789.pdf") == "1234.56789"
    assert parse_arxiv_id("hep-th/9901001") == "hep-th/9901001"


@pytest.mark.unit
def test_parse_arxiv_id_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        parse_arxiv_id("")
    with pytest.raises(ValueError, match="Only arxiv.org URLs"):
        parse_arxiv_id("https://example.com/abs/1234.56789")
    with pytest.raises(ValueError, match="Invalid arXiv id"):
        parse_arxiv_id("12 34.56789")
    with pytest.raises(ValueError, match="Invalid arXiv id"):
        parse_arxiv_id("../1234.56789")


@pytest.mark.unit
def test_safe_filename_sanitizes_and_falls_back():
    assert _safe_filename("") == "document"
    assert _safe_filename("////") == "document"
    assert "/" not in _safe_filename("a/b")
    assert "\\" not in _safe_filename("a\\b")
    assert _safe_filename("weird☺name")
