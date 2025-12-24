import pytest

from src.citations.registry import make_minimal_citation_record, save_citations
from src.evidence.store import EvidenceStore
from src.literature.gates import LiteratureGateConfig, LiteratureGateError, check_literature_gate, enforce_literature_gate


def _minimal_evidence_item(*, evidence_id: str, source_id: str):
    return {
        "schema_version": "1.0",
        "evidence_id": evidence_id,
        "source_id": source_id,
        "kind": "quote",
        "locator": {"type": "file", "value": "dummy.txt", "span": {"start_line": 1, "end_line": 1}},
        "excerpt": "A supporting quote.",
        "created_at": "2025-01-01T00:00:00Z",
        "parser": {"name": "mvp"},
    }


@pytest.mark.unit
def test_literature_gate_disabled_is_permissive(temp_project_folder):
    cfg = LiteratureGateConfig(enabled=False)
    result = check_literature_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "disabled"


@pytest.mark.unit
def test_literature_gate_blocks_when_thresholds_not_met(temp_project_folder):
    save_citations(temp_project_folder, [], validate=False)

    cfg = LiteratureGateConfig(
        enabled=True,
        on_failure="block",
        min_verified_citations=1,
        min_evidence_items_total=1,
        min_evidence_items_per_source=0,
    )

    result = check_literature_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is False
    assert result["action"] == "block"

    with pytest.raises(LiteratureGateError):
        enforce_literature_gate(project_folder=str(temp_project_folder), config=cfg)


@pytest.mark.unit
def test_literature_gate_passes_when_thresholds_met(temp_project_folder):
    save_citations(
        temp_project_folder,
        [
            make_minimal_citation_record(
                citation_key="Known2020",
                title="Known",
                authors=["A"],
                year=2020,
                status="verified",
            ),
            make_minimal_citation_record(
                citation_key="Known2021",
                title="Known2",
                authors=["B"],
                year=2021,
                status="verified",
            ),
        ],
        validate=True,
    )

    store = EvidenceStore(str(temp_project_folder))
    store.write_evidence_items(
        "source1",
        [
            _minimal_evidence_item(evidence_id="ev1", source_id="source1"),
            _minimal_evidence_item(evidence_id="ev2", source_id="source1"),
        ],
    )

    cfg = LiteratureGateConfig(
        enabled=True,
        on_failure="block",
        min_verified_citations=2,
        min_evidence_items_total=2,
        min_evidence_items_per_source=2,
    )

    result = check_literature_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "pass"
    assert result["verified_citations"] == 2
    assert result["evidence_items_total"] == 2
    assert result["sources_below_min"] == []


@pytest.mark.unit
def test_literature_gate_downgrades_when_configured(temp_project_folder):
    save_citations(temp_project_folder, [], validate=False)

    cfg = LiteratureGateConfig(enabled=True, on_failure="downgrade", min_verified_citations=1, min_evidence_items_total=1)
    result = check_literature_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "downgrade"


@pytest.mark.unit
def test_literature_gate_config_from_context_parses_and_sanitizes_values():
    cfg = LiteratureGateConfig.from_context(
        {
            "literature_gate": {
                "enabled": True,
                "on_failure": "downgrade",
                "min_verified_citations": "2",
                "min_evidence_items_total": 3,
                "min_evidence_items_per_source": "1",
            }
        }
    )
    assert cfg.enabled is True
    assert cfg.on_failure == "downgrade"
    assert cfg.min_verified_citations == 2
    assert cfg.min_evidence_items_total == 3
    assert cfg.min_evidence_items_per_source == 1

    bad = LiteratureGateConfig.from_context({"literature_gate": {"enabled": True, "on_failure": "nope"}})
    assert bad.enabled is True
    assert bad.on_failure == "block"

    nonint = LiteratureGateConfig.from_context({"literature_gate": {"enabled": True, "min_verified_citations": "x"}})
    assert nonint.min_verified_citations >= 0
