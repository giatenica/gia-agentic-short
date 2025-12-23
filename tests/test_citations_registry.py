"""Tests for citation registry and schema validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.citations.registry import (
    citation_keys,
    ensure_citations_registry_exists,
    load_citations,
    make_minimal_citation_record,
    save_citations,
    upsert_citation,
)
from src.utils.schema_validation import is_valid_citation_record, validate_citation_record


@pytest.fixture
def temp_project_folder() -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "project.json").write_text(
            json.dumps({"id": "p1", "title": "t", "research_question": "q"}),
            encoding="utf-8",
        )
        yield tmpdir


@pytest.mark.unit
def test_validate_citation_record_minimal_ok():
    rec = make_minimal_citation_record(
        citation_key="Zingales1994",
        title="A Title",
        authors=["Author One", "Author Two"],
        year=1994,
    )
    assert is_valid_citation_record(rec) is True


@pytest.mark.unit
def test_validate_citation_record_missing_required_fails():
    with pytest.raises(ValueError):
        validate_citation_record({"schema_version": "1.0"})


@pytest.mark.unit
def test_registry_create_empty(temp_project_folder: str):
    paths = ensure_citations_registry_exists(temp_project_folder)
    assert paths.citations_path.exists()
    assert load_citations(temp_project_folder) == []


@pytest.mark.unit
def test_save_and_load_roundtrip(temp_project_folder: str):
    rec1 = make_minimal_citation_record(
        citation_key="Key1999",
        title="T1",
        authors=["A"],
        year=1999,
        status="unverified",
    )
    rec2 = make_minimal_citation_record(
        citation_key="Key2000",
        title="T2",
        authors=["B"],
        year=2000,
        status="verified",
    )

    save_citations(temp_project_folder, [rec1, rec2])
    loaded = load_citations(temp_project_folder)
    assert {r["citation_key"] for r in loaded} == {"Key1999", "Key2000"}


@pytest.mark.unit
def test_upsert_replaces_by_key(temp_project_folder: str):
    rec1 = make_minimal_citation_record(
        citation_key="Key2001",
        title="T1",
        authors=["A"],
        year=2001,
        status="unverified",
    )
    rec2 = make_minimal_citation_record(
        citation_key="Key2001",
        title="T2",
        authors=["A"],
        year=2001,
        status="verified",
    )

    upsert_citation(temp_project_folder, rec1)
    upsert_citation(temp_project_folder, rec2)

    loaded = load_citations(temp_project_folder)
    assert len(loaded) == 1
    assert loaded[0]["title"] == "T2"
    assert loaded[0]["status"] == "verified"


@pytest.mark.unit
def test_duplicate_key_in_save_fails(temp_project_folder: str):
    rec = make_minimal_citation_record(
        citation_key="Dup",
        title="T",
        authors=["A"],
        year=2002,
    )

    with pytest.raises(ValueError):
        save_citations(temp_project_folder, [rec, rec])


@pytest.mark.unit
def test_citation_keys_sorted_unique(temp_project_folder: str):
    rec1 = make_minimal_citation_record(
        citation_key="BKey",
        title="T1",
        authors=["A"],
        year=2001,
    )
    rec2 = make_minimal_citation_record(
        citation_key="AKey",
        title="T2",
        authors=["B"],
        year=2002,
    )

    save_citations(temp_project_folder, [rec1, rec2])
    assert citation_keys(temp_project_folder) == ["AKey", "BKey"]
