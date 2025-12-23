import pytest

from src.citations.gates import CitationGateConfig, CitationGateError, check_citation_gate, enforce_citation_gate
from src.citations.registry import save_citations


@pytest.mark.unit
def test_citation_gate_passes_when_all_referenced_keys_verified(temp_project_folder):
    (temp_project_folder / "RESEARCH_OVERVIEW.md").write_text(
        "This cites [@smith2020].\n",
        encoding="utf-8",
    )

    save_citations(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "citation_key": "smith2020",
                "status": "verified",
                "title": "A Paper",
                "authors": ["Smith"],
                "year": 2020,
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
        validate=True,
    )

    cfg = CitationGateConfig(enabled=True, on_missing="block", on_unverified="downgrade")
    result = check_citation_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "pass"
    assert result["missing_keys"] == []
    assert result["unverified_keys"] == []


@pytest.mark.unit
def test_citation_gate_blocks_when_missing_key_and_configured_to_block(temp_project_folder):
    (temp_project_folder / "RESEARCH_OVERVIEW.md").write_text(
        "Missing cite [@missing2021].\n",
        encoding="utf-8",
    )
    save_citations(temp_project_folder, [], validate=False)

    cfg = CitationGateConfig(enabled=True, on_missing="block", on_unverified="downgrade")
    result = check_citation_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is False
    assert result["action"] == "block"
    assert result["missing_keys"] == ["missing2021"]

    with pytest.raises(CitationGateError):
        enforce_citation_gate(project_folder=str(temp_project_folder), config=cfg)


@pytest.mark.unit
def test_citation_gate_downgrades_when_unverified_key_and_configured_to_downgrade(temp_project_folder):
    (temp_project_folder / "RESEARCH_OVERVIEW.md").write_text(
        "Unverified cite [@doe2019].\n",
        encoding="utf-8",
    )
    save_citations(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "citation_key": "doe2019",
                "status": "unverified",
                "title": "Another Paper",
                "authors": ["Doe"],
                "year": 2019,
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
        validate=True,
    )

    cfg = CitationGateConfig(enabled=True, on_missing="block", on_unverified="downgrade")
    result = check_citation_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "downgrade"
    assert result["missing_keys"] == []
    assert result["unverified_keys"] == ["doe2019"]
