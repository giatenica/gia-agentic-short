"""Unit tests for OpenAlex citation resolver."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.citations.openalex import (
    OpenAlexClient,
    OpenAlexError,
    OpenAlexNotFoundError,
    openalex_work_to_citation_record,
    resolve_openalex_doi_to_record,
)
from src.utils.schema_validation import validate_citation_record


def _make_openalex_response(*, status_code: int, payload) -> httpx.Response:
    request = httpx.Request("GET", "https://api.openalex.org/works/x")
    return httpx.Response(status_code=status_code, json=payload, request=request)


@pytest.mark.unit
def test_fetch_work_by_doi_success():
    payload = {"id": "https://openalex.org/W1", "display_name": "A Paper", "publication_year": 2020}

    client = OpenAlexClient()
    with patch.object(httpx.Client, "get", return_value=_make_openalex_response(status_code=200, payload=payload)):
        work = client.fetch_work_by_doi("10.1000/182")

    assert work["display_name"] == "A Paper"


@pytest.mark.unit
def test_fetch_work_by_doi_404_raises_not_found():
    client = OpenAlexClient()
    with patch.object(httpx.Client, "get", return_value=_make_openalex_response(status_code=404, payload={"error": "no"})):
        with pytest.raises(OpenAlexNotFoundError):
            client.fetch_work_by_doi("10.5555/notfound")


@pytest.mark.unit
def test_fetch_work_by_doi_http_error_raises():
    client = OpenAlexClient()
    with patch.object(httpx.Client, "get", return_value=_make_openalex_response(status_code=500, payload={"error": "bad"})):
        with pytest.raises(OpenAlexError):
            client.fetch_work_by_doi("10.1000/182")


@pytest.mark.unit
def test_fetch_work_by_doi_invalid_json_raises():
    client = OpenAlexClient()

    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("nope")

    with patch.object(httpx.Client, "get", return_value=resp):
        with pytest.raises(OpenAlexError):
            client.fetch_work_by_doi("10.1000/182")


@pytest.mark.unit
def test_fetch_work_by_doi_non_object_json_raises():
    client = OpenAlexClient()
    with patch.object(httpx.Client, "get", return_value=_make_openalex_response(status_code=200, payload=[1, 2, 3])):
        with pytest.raises(OpenAlexError):
            client.fetch_work_by_doi("10.1000/182")


@pytest.mark.unit
def test_openalex_work_to_citation_record_minimal_fields_and_created_at():
    work = {
        "id": "https://openalex.org/W1",
        "display_name": "A Paper",
        "publication_year": 2020,
        "authorships": [{"author": {"display_name": "A One"}}],
        "host_venue": {"display_name": "Journal"},
        "primary_location": {"landing_page_url": "https://example.com/paper"},
        "biblio": {"volume": "1", "issue": "2", "first_page": "10"},
    }

    rec = openalex_work_to_citation_record(
        work=work,
        citation_key="Key2020",
        status="verified",
        doi="https://doi.org/10.1000/182",
        created_at="2020-01-01T00:00:00Z",
    )

    validate_citation_record(rec)
    assert rec["citation_key"] == "Key2020"
    assert rec["status"] == "verified"
    assert rec["title"] == "A Paper"
    assert rec["year"] == 2020
    assert rec["venue"] == "Journal"
    assert rec["volume"] == "1"
    assert rec["issue"] == "2"
    assert rec["pages"] == "10"
    assert rec["url"] == "https://example.com/paper"
    assert rec["created_at"] == "2020-01-01T00:00:00Z"
    assert rec["identifiers"]["doi"] == "10.1000/182"


@pytest.mark.unit
def test_resolve_openalex_doi_to_record_calls_client_and_validates_schema():
    payload = {
        "id": "https://openalex.org/W1",
        "display_name": "A Paper",
        "publication_year": 2020,
        "authorships": [{"author": {"display_name": "A One"}}],
    }

    with patch("src.citations.openalex.OpenAlexClient.fetch_work_by_doi", return_value=payload):
        rec = resolve_openalex_doi_to_record(doi="10.1000/182", citation_key="Key", created_at="2020-01-01T00:00:00Z")

    validate_citation_record(rec)
    assert rec["created_at"] == "2020-01-01T00:00:00Z"
