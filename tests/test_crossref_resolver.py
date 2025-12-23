"""Unit tests for Crossref citation resolver."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import httpx
import pytest

from src.citations.crossref import (
    CrossrefClient,
    CrossrefError,
    CrossrefNotFoundError,
    CITATION_FALLBACK_YEAR,
    normalize_doi,
    crossref_work_to_citation_record,
    resolve_crossref_doi_and_upsert,
    resolve_crossref_doi_to_record,
)
from src.citations.registry import load_citations
from src.utils.schema_validation import validate_citation_record


@pytest.fixture
def temp_project_folder() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "project.json").write_text(
            json.dumps({"id": "p1", "title": "t", "research_question": "q"}),
            encoding="utf-8",
        )
        yield tmpdir


def _make_crossref_response(payload: dict, *, status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", "https://api.crossref.org/works/10.1000%2F182")
    return httpx.Response(status_code=status_code, json=payload, request=request)


@pytest.mark.unit
def test_normalize_doi_variants():
    assert normalize_doi("10.1000/182") == "10.1000/182"
    assert normalize_doi(" DOI:10.1000/182 ") == "10.1000/182"
    assert normalize_doi("https://doi.org/10.1000/182") == "10.1000/182"


@pytest.mark.unit
def test_resolve_crossref_doi_to_record_minimal_fields():
    payload = {
        "message": {
            "DOI": "10.1000/182",
            "title": ["A Paper"],
            "author": [{"given": "A", "family": "One"}],
            "issued": {"date-parts": [[2020, 1, 1]]},
            "container-title": ["Journal"],
            "publisher": "Pub",
            "URL": "https://example.com/paper",
        }
    }

    client = CrossrefClient()
    with patch.object(httpx.Client, "get", return_value=_make_crossref_response(payload)):
        rec = resolve_crossref_doi_to_record(doi="10.1000/182", citation_key="Key2020", client=client)

    validate_citation_record(rec)
    assert rec["citation_key"] == "Key2020"
    assert rec["status"] == "verified"
    assert rec["title"] == "A Paper"
    assert rec["year"] == 2020
    assert rec["venue"] == "Journal"
    assert rec["publisher"] == "Pub"
    assert rec["url"] == "https://example.com/paper"
    assert rec["identifiers"]["doi"] == "10.1000/182"


@pytest.mark.unit
def test_resolve_crossref_doi_and_upsert_writes_registry(temp_project_folder: str):
    payload = {
        "message": {
            "DOI": "10.1000/182",
            "title": ["A Paper"],
            "author": [{"given": "A", "family": "One"}],
            "issued": {"date-parts": [[2020, 1, 1]]},
            "URL": "https://example.com/paper",
        }
    }

    client = CrossrefClient()
    with patch.object(httpx.Client, "get", return_value=_make_crossref_response(payload)):
        resolve_crossref_doi_and_upsert(
            project_folder=temp_project_folder,
            doi="https://doi.org/10.1000/182",
            citation_key="Key2020",
            client=client,
        )

    records = load_citations(temp_project_folder)
    assert len(records) == 1
    assert records[0]["citation_key"] == "Key2020"
    assert records[0]["identifiers"]["doi"] == "10.1000/182"


@pytest.mark.unit
def test_resolve_crossref_doi_not_found():
    payload = {"message": {"status": "resource not found"}}
    client = CrossrefClient()

    with patch.object(httpx.Client, "get", return_value=_make_crossref_response(payload, status_code=404)):
        with pytest.raises(CrossrefNotFoundError):
            resolve_crossref_doi_to_record(doi="10.5555/notfound", citation_key="Key", client=client)


@pytest.mark.unit
def test_search_by_title_empty_results():
    payload = {"message": {"items": []}}
    client = CrossrefClient()

    with patch.object(httpx.Client, "get", return_value=_make_crossref_response(payload)):
        items = client.search_by_title(title="Some title", rows=3)

    assert items == []


@pytest.mark.unit
def test_search_by_title_invalid_params():
    client = CrossrefClient()
    with pytest.raises(ValueError):
        client.search_by_title(title="", rows=5)
    with pytest.raises(ValueError):
        client.search_by_title(title="t", rows=0)
    with pytest.raises(ValueError):
        client.search_by_title(title="t", year=999)


@pytest.mark.unit
def test_crossref_work_to_citation_record_fallbacks():
    work = {
        "DOI": "10.1000/182",
        "title": [],
        "author": [],
        "issued": {"date-parts": []},
    }
    rec = crossref_work_to_citation_record(work=work, citation_key="Key", status="verified")

    validate_citation_record(rec)
    assert rec["title"] == "(missing title)"
    assert rec["authors"] == ["(unknown)"]
    assert rec["year"] == CITATION_FALLBACK_YEAR
    assert rec["identifiers"]["doi"] == "10.1000/182"


@pytest.mark.unit
def test_fetch_work_by_doi_invalid_json_raises_crossref_error():
    # Create a response with invalid JSON content so resp.json() raises ValueError.
    request = httpx.Request("GET", "https://api.crossref.org/works/10.1000%2F182")
    bad_json_response = httpx.Response(status_code=200, content=b"not-json", request=request)

    client = CrossrefClient()
    with patch.object(httpx.Client, "get", return_value=bad_json_response):
        with pytest.raises(CrossrefError):
            client.fetch_work_by_doi("10.1000/182")
