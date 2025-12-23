"""Unit tests for bibliography builder."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from src.citations.bibliography import (
    build_bibliography,
    dedupe_citation_records_by_doi,
    mint_stable_citation_key,
)
from src.citations.registry import load_citations, make_minimal_citation_record, save_citations


@pytest.fixture
def temp_project_folder() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "project.json").write_text(
            json.dumps({"id": "p1", "title": "t", "research_question": "q"}),
            encoding="utf-8",
        )
        yield tmpdir


@pytest.mark.unit
def test_mint_stable_citation_key_suffixes_deterministic():
    existing = {"Smith2020"}
    k1 = mint_stable_citation_key(authors=["John Smith"], year=2020, title="A", existing_keys=existing)
    assert k1 == "Smith2020a"
    existing.add(k1)

    k2 = mint_stable_citation_key(authors=["John Smith"], year=2020, title="B", existing_keys=existing)
    assert k2 == "Smith2020b"


@pytest.mark.unit
def test_mint_stable_citation_key_suffixes_beyond_z():
    base = "Smith2020"
    existing = {base}
    # Exhaust a..z
    for ch in "abcdefghijklmnopqrstuvwxyz":
        existing.add(f"{base}{ch}")

    k = mint_stable_citation_key(authors=["Jane Smith"], year=2020, title="X", existing_keys=existing)
    assert k == "Smith2020aa"


@pytest.mark.unit
def test_dedupe_by_doi_keeps_one_record_and_maps_dropped():
    rec1 = make_minimal_citation_record(
        citation_key="Key1",
        title="T1",
        authors=["A"],
        year=2001,
        identifiers={"doi": "10.1234/abcd"},
        status="verified",
    )
    rec2 = make_minimal_citation_record(
        citation_key="Key2",
        title="T2",
        authors=["B"],
        year=2002,
        identifiers={"doi": "https://doi.org/10.1234/abcd"},
        status="unverified",
    )

    deduped, dropped = dedupe_citation_records_by_doi([rec1, rec2])
    assert len(deduped) == 1
    assert dropped == {"Key2": "Key1"}


@pytest.mark.unit
def test_dedupe_by_doi_merges_missing_fields_into_kept_record():
    # Kept record should be the verified one, but it should receive missing fields
    # from the dropped record.
    kept = make_minimal_citation_record(
        citation_key="Keep",
        title="T",
        authors=["A"],
        year=2001,
        identifiers={"doi": "10.1000/merge"},
        status="verified",
    )
    dropped = make_minimal_citation_record(
        citation_key="Drop",
        title="T",
        authors=["A"],
        year=2001,
        identifiers={"doi": "https://doi.org/10.1000/merge"},
        status="unverified",
    )
    dropped["venue"] = "Journal of Testing"
    dropped["pages"] = "1-10"

    deduped, mapping = dedupe_citation_records_by_doi([kept, dropped])
    assert mapping == {"Drop": "Keep"}
    assert len(deduped) == 1
    assert deduped[0]["citation_key"] == "Keep"
    assert deduped[0]["venue"] == "Journal of Testing"
    assert deduped[0]["pages"] == "1-10"


@pytest.mark.unit
def test_build_bibliography_writes_bib_and_updates_registry(temp_project_folder: str):
    rec1 = make_minimal_citation_record(
        citation_key="KeyA",
        title="T1",
        authors=["Alice A"],
        year=2001,
        identifiers={"doi": "10.5555/xyz"},
        status="unverified",
    )
    rec2 = make_minimal_citation_record(
        citation_key="KeyB",
        title="T2",
        authors=["Bob B"],
        year=2002,
        identifiers={"doi": "doi:10.5555/xyz"},
        status="verified",
    )
    rec2["venue"] = "Journal of Finance"
    rec2["pages"] = "3-56"

    save_citations(temp_project_folder, [rec1, rec2])

    paths = build_bibliography(temp_project_folder)
    assert paths.citations_path.exists()
    assert paths.references_bib_path.exists()

    loaded = load_citations(temp_project_folder)
    assert len(loaded) == 1
    assert loaded[0]["citation_key"] == "KeyB"

    bib = paths.references_bib_path.read_text(encoding="utf-8")
    assert "@" in bib
    assert "10.5555/xyz" in bib
    assert "@article{KeyB," in bib
    assert "title = {T2}" in bib
    assert "author = {Bob B}" in bib
    assert "year = {2002}" in bib
    assert "journal = {Journal of Finance}" in bib
    assert "pages = {3-56}" in bib
