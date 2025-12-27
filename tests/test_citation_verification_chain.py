"""Unit tests for citation verification chain (Crossref -> OpenAlex) and stale detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.citations.registry import make_minimal_citation_record
from src.citations.verification import (
    CitationVerificationPolicy,
    is_verification_stale,
    resolve_doi_to_record_with_fallback,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.mark.unit
def test_is_verification_stale_true_when_missing_or_unparseable():
    policy = CitationVerificationPolicy(max_age_hours=24)
    assert is_verification_stale(last_checked=None, now=_iso(datetime.now(timezone.utc)), policy=policy) is True
    assert is_verification_stale(last_checked="not-a-date", now=_iso(datetime.now(timezone.utc)), policy=policy) is True


@pytest.mark.unit
def test_is_verification_stale_respects_max_age_hours():
    now = datetime.now(timezone.utc)
    policy = CitationVerificationPolicy(max_age_hours=24)

    fresh = _iso(now - timedelta(hours=1))
    stale = _iso(now - timedelta(hours=48))

    assert is_verification_stale(last_checked=fresh, now=_iso(now), policy=policy) is False
    assert is_verification_stale(last_checked=stale, now=_iso(now), policy=policy) is True


@pytest.mark.unit
def test_verification_falls_back_to_openalex_when_crossref_fails():
    doi = "10.1000/182"

    def boom(*args, **kwargs):
        raise RuntimeError("crossref down")

    def ok_openalex(*, doi: str, citation_key: str, **kwargs):
        rec = make_minimal_citation_record(
            citation_key=citation_key,
            title="OA Title",
            authors=["A"],
            year=2020,
            status="verified",
            identifiers={"doi": doi},
        )
        return rec

    with patch("src.citations.verification.resolve_crossref_doi_to_record", boom), patch(
        "src.citations.verification.resolve_openalex_doi_to_record", ok_openalex
    ):
        rec = resolve_doi_to_record_with_fallback(doi=doi, citation_key="K", created_at=_iso(datetime.now(timezone.utc)))

    assert rec["status"] == "verified"
    assert rec["verification"]["provider_used"] == "openalex"
    assert rec["manual_verification_required"] is False
    assert len(rec["verification"]["attempts"]) == 2
    assert rec["verification"]["attempts"][0]["provider"] == "crossref"
    assert rec["verification"]["attempts"][0]["ok"] is False
    assert rec["verification"]["attempts"][1]["provider"] == "openalex"
    assert rec["verification"]["attempts"][1]["ok"] is True


@pytest.mark.unit
def test_verification_uses_crossref_when_it_succeeds():
    doi = "10.1000/182"

    def ok_crossref(*, doi: str, citation_key: str, created_at=None, **kwargs):
        rec = make_minimal_citation_record(
            citation_key=citation_key,
            title="CR Title",
            authors=["A"],
            year=2020,
            status="verified",
            identifiers={"doi": doi},
            created_at=created_at,
        )
        return rec

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("OpenAlex fallback should not be called")

    now = datetime.now(timezone.utc)
    with patch("src.citations.verification.resolve_crossref_doi_to_record", ok_crossref), patch(
        "src.citations.verification.resolve_openalex_doi_to_record", should_not_be_called
    ):
        rec = resolve_doi_to_record_with_fallback(doi=doi, citation_key="K", created_at=_iso(now), now=_iso(now))

    assert rec["status"] == "verified"
    assert rec["verification"]["provider_used"] == "crossref"
    assert rec["manual_verification_required"] is False
    assert len(rec["verification"]["attempts"]) == 1
    assert rec["verification"]["attempts"][0]["provider"] == "crossref"
    assert rec["verification"]["attempts"][0]["ok"] is True


@pytest.mark.unit
def test_verification_returns_manual_record_when_both_providers_fail():
    doi = "10.1000/182"

    def boom(*args, **kwargs):
        raise RuntimeError("down")

    now = datetime.now(timezone.utc)
    with patch("src.citations.verification.resolve_crossref_doi_to_record", boom), patch(
        "src.citations.verification.resolve_openalex_doi_to_record", boom
    ):
        rec = resolve_doi_to_record_with_fallback(doi=doi, citation_key="K", created_at=_iso(now), now=_iso(now))

    assert rec["status"] == "unverified"
    assert rec["manual_verification_required"] is True
    assert rec["verification"]["status"] == "manual"
    assert rec["verification"]["provider_used"] is None
    assert len(rec["verification"]["attempts"]) == 2
    assert rec["verification"]["attempts"][0]["provider"] == "crossref"
    assert rec["verification"]["attempts"][0]["ok"] is False
    assert rec["verification"]["attempts"][1]["provider"] == "openalex"
    assert rec["verification"]["attempts"][1]["ok"] is False


@pytest.mark.unit
def test_verification_reuses_fresh_existing_verified_record():
    now = datetime.now(timezone.utc)
    doi = "10.1000/182"

    existing = make_minimal_citation_record(
        citation_key="K",
        title="Existing",
        authors=["A"],
        year=2020,
        status="verified",
        identifiers={"doi": doi},
        created_at=_iso(now - timedelta(days=10)),
    )
    existing["verification"] = {
        "status": "verified",
        "provider_used": "crossref",
        "last_checked": _iso(now - timedelta(hours=1)),
        "attempts": [{"provider": "crossref", "ok": True, "checked_at": _iso(now - timedelta(hours=1))}],
    }

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("Network resolver should not be called")

    with patch("src.citations.verification.resolve_crossref_doi_to_record", should_not_be_called), patch(
        "src.citations.verification.resolve_openalex_doi_to_record", should_not_be_called
    ):
        rec = resolve_doi_to_record_with_fallback(
            doi=doi,
            citation_key="K",
            created_at=_iso(now),
            existing_record=existing,
            policy=CitationVerificationPolicy(max_age_hours=24),
            now=_iso(now),
        )

    assert rec is existing


@pytest.mark.unit
def test_verification_does_not_reuse_verified_record_with_different_doi():
    now = datetime.now(timezone.utc)
    doi = "10.1000/182"

    existing = make_minimal_citation_record(
        citation_key="K",
        title="Existing",
        authors=["A"],
        year=2020,
        status="verified",
        identifiers={"doi": "10.9999/other"},
        created_at=_iso(now - timedelta(days=10)),
    )
    existing["verification"] = {
        "status": "verified",
        "provider_used": "crossref",
        "last_checked": _iso(now - timedelta(hours=1)),
        "attempts": [{"provider": "crossref", "ok": True, "checked_at": _iso(now - timedelta(hours=1))}],
    }

    def ok_crossref(*, doi: str, citation_key: str, created_at=None, **kwargs):
        return make_minimal_citation_record(
            citation_key=citation_key,
            title="New",
            authors=["A"],
            year=2020,
            status="verified",
            identifiers={"doi": doi},
            created_at=created_at,
        )

    cr_mock = MagicMock(side_effect=ok_crossref)
    with patch("src.citations.verification.resolve_crossref_doi_to_record", cr_mock):
        rec = resolve_doi_to_record_with_fallback(
            doi=doi,
            citation_key="K",
            created_at=_iso(now),
            existing_record=existing,
            policy=CitationVerificationPolicy(max_age_hours=24),
            now=_iso(now),
        )

    assert rec is not existing
    assert cr_mock.called
