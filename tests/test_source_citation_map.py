import json

import pytest

from src.citations.registry import save_citations
from src.citations.source_map import (
    build_source_citation_map,
    load_source_citation_map,
    write_source_citation_map,
)


@pytest.mark.unit
def test_source_citation_map_matches_doi_in_raw_filenames(tmp_path):
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "project.json").write_text(json.dumps({"id": "p1"}), encoding="utf-8")
    (project / "bibliography").mkdir(parents=True)
    (project / "sources" / "src1" / "raw").mkdir(parents=True)

    # Create a citation record with a DOI.
    save_citations(
        project,
        [
            {
                "schema_version": "1.0",
                "citation_key": "Smith2020Example",
                "status": "verified",
                "title": "Example",
                "authors": ["Smith"],
                "year": 2020,
                "created_at": "2020-01-01T00:00:00Z",
                "identifiers": {"doi": "10.1000/xyz123"},
            }
        ],
        validate=True,
    )

    # Raw filename includes DOI.
    (project / "sources" / "src1" / "raw" / "paper_10.1000_xyz123.pdf").write_text("", encoding="utf-8")

    mapping = build_source_citation_map(project)
    assert mapping == {"src1": "Smith2020Example"}


@pytest.mark.unit
def test_source_citation_map_roundtrip_json(tmp_path):
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "project.json").write_text(json.dumps({"id": "p1"}), encoding="utf-8")
    (project / "bibliography").mkdir(parents=True)

    path = write_source_citation_map(project, {"src1": "Key1"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"src1": "Key1"}

    loaded = load_source_citation_map(project)
    assert loaded == {"src1": "Key1"}


@pytest.mark.unit
def test_source_citation_map_reads_doi_from_retrieval_json(tmp_path):
    """Test that DOI from retrieval.json metadata takes precedence over filename."""
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "project.json").write_text(json.dumps({"id": "p1"}), encoding="utf-8")
    (project / "bibliography").mkdir(parents=True)
    (project / "sources" / "hashid123" / "raw").mkdir(parents=True)

    # Create a citation record with a DOI.
    save_citations(
        project,
        [
            {
                "schema_version": "1.0",
                "citation_key": "Jones2021Meta",
                "status": "verified",
                "title": "Metadata Test",
                "authors": ["Jones"],
                "year": 2021,
                "created_at": "2021-01-01T00:00:00Z",
                "identifiers": {"doi": "10.1234/meta-test"},
            }
        ],
        validate=True,
    )

    # Filename is generic (no DOI), but retrieval.json has the DOI.
    (project / "sources" / "hashid123" / "raw" / "source.pdf").write_text("", encoding="utf-8")
    retrieval_meta = {
        "ok": True,
        "source_id": "hashid123",
        "provider": "pdf_url",
        "requested": {
            "url": "https://doi.org/10.1234/meta-test",
            "doi": "10.1234/meta-test",
        },
        "retrieved_from": "https://example.com/paper.pdf",
    }
    (project / "sources" / "hashid123" / "raw" / "retrieval.json").write_text(
        json.dumps(retrieval_meta), encoding="utf-8"
    )

    mapping = build_source_citation_map(project)
    assert mapping == {"hashid123": "Jones2021Meta"}


@pytest.mark.unit
def test_source_citation_map_reads_arxiv_from_retrieval_json(tmp_path):
    """Test that arXiv ID from retrieval.json metadata is used for matching."""
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "project.json").write_text(json.dumps({"id": "p1"}), encoding="utf-8")
    (project / "bibliography").mkdir(parents=True)
    (project / "sources" / "arxiv_2301.12345" / "raw").mkdir(parents=True)

    # Create a citation record with an arXiv ID.
    save_citations(
        project,
        [
            {
                "schema_version": "1.0",
                "citation_key": "ArxivPaper2023",
                "status": "verified",
                "title": "ArXiv Test",
                "authors": ["Author"],
                "year": 2023,
                "created_at": "2023-01-01T00:00:00Z",
                "identifiers": {"arxiv": "2301.12345"},
            }
        ],
        validate=True,
    )

    # Generic filename, retrieval.json has arXiv ID.
    (project / "sources" / "arxiv_2301.12345" / "raw" / "paper.pdf").write_text("", encoding="utf-8")
    retrieval_meta = {
        "ok": True,
        "source_id": "arxiv:2301_12345",
        "provider": "arxiv",
        "requested": {
            "arxiv_id": "2301.12345",
            "input": "2301.12345",
        },
        "retrieved_from": "https://arxiv.org/pdf/2301.12345.pdf",
    }
    (project / "sources" / "arxiv_2301.12345" / "raw" / "retrieval.json").write_text(
        json.dumps(retrieval_meta), encoding="utf-8"
    )

    mapping = build_source_citation_map(project)
    # The canonical source_id from retrieval.json should be used as the key.
    assert "arxiv:2301_12345" in mapping
    assert mapping["arxiv:2301_12345"] == "ArxivPaper2023"


@pytest.mark.unit
def test_source_citation_map_fallback_to_filename_when_no_metadata(tmp_path):
    """Test that filename-based extraction still works when retrieval.json is absent."""
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "project.json").write_text(json.dumps({"id": "p1"}), encoding="utf-8")
    (project / "bibliography").mkdir(parents=True)
    (project / "sources" / "src_old" / "raw").mkdir(parents=True)

    # Create a citation record with a DOI.
    save_citations(
        project,
        [
            {
                "schema_version": "1.0",
                "citation_key": "OldStyle2019",
                "status": "verified",
                "title": "Old Style Paper",
                "authors": ["Old"],
                "year": 2019,
                "created_at": "2019-01-01T00:00:00Z",
                "identifiers": {"doi": "10.5555/oldstyle"},
            }
        ],
        validate=True,
    )

    # No retrieval.json, but DOI is in filename.
    (project / "sources" / "src_old" / "raw" / "10.5555_oldstyle.pdf").write_text("", encoding="utf-8")

    mapping = build_source_citation_map(project)
    assert mapping == {"src_old": "OldStyle2019"}
