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
