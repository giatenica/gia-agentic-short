import pytest

from src.citations.gates import (
    CitationGateConfig,
    CitationGateError,
    check_citation_gate,
    enforce_citation_gate,
    find_referenced_citation_keys,
)
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
def test_citation_gate_config_from_context_parses_and_sanitizes_values():
    cfg = CitationGateConfig.from_context(
        {
            "citation_gate": {
                "enabled": True,
                "on_missing": "downgrade",
                "on_unverified": "block",
            }
        }
    )
    assert cfg.enabled is True
    assert cfg.on_missing == "downgrade"
    assert cfg.on_unverified == "block"

    bad = CitationGateConfig.from_context(
        {
            "citation_gate": {
                "enabled": True,
                "on_missing": "nope",
                "on_unverified": "also_nope",
            }
        }
    )
    assert bad.enabled is True
    assert bad.on_missing == "block"
    assert bad.on_unverified == "downgrade"


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
def test_citation_gate_blocks_when_missing_and_unverified_and_missing_is_block(temp_project_folder):
    (temp_project_folder / "RESEARCH_OVERVIEW.md").write_text(
        "Both cites [@missing2021] and [@doe2019].\n",
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
    assert result["ok"] is False
    assert result["action"] == "block"
    assert "missing2021" in (result["missing_keys"] or [])
    assert "doe2019" in (result["unverified_keys"] or [])


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


@pytest.mark.unit
def test_citation_gate_disabled_is_permissive_even_with_missing(temp_project_folder):
    (temp_project_folder / "RESEARCH_OVERVIEW.md").write_text(
        "Missing cite [@missing2021].\n",
        encoding="utf-8",
    )

    cfg = CitationGateConfig(enabled=False, on_missing="block", on_unverified="block")
    result = check_citation_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "disabled"


@pytest.mark.unit
def test_find_referenced_citation_keys_extracts_markdown_multiple_and_latex(temp_project_folder):
    (temp_project_folder / "RESEARCH_OVERVIEW.md").write_text(
        "Multi cite [@KeyOne; @keyTwo] and also @keyThree.\n",
        encoding="utf-8",
    )
    (temp_project_folder / "paper").mkdir(exist_ok=True)
    (temp_project_folder / "paper" / "main.tex").write_text(
        "\\citet{Smith2020} and \\citep{Doe2019, Roe2018}.\n",
        encoding="utf-8",
    )

    # Excluded directories should not be scanned.
    (temp_project_folder / "sources").mkdir(exist_ok=True)
    (temp_project_folder / "sources" / "ignored.md").write_text("[@ignoredKey]", encoding="utf-8")

    # Hidden directories should not be scanned.
    (temp_project_folder / ".vscode").mkdir(exist_ok=True)
    (temp_project_folder / ".vscode" / "ignored.md").write_text("[@alsoIgnored]", encoding="utf-8")

    keys, docs = find_referenced_citation_keys(str(temp_project_folder))
    assert {"keyone", "keytwo", "keythree", "smith2020", "doe2019", "roe2018"}.issubset(keys)
    assert "ignoredkey" not in keys
    assert "alsoignored" not in keys

    assert "RESEARCH_OVERVIEW.md" in docs
    assert "paper/main.tex" in docs
