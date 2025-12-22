import pytest

from src.evidence.extraction import extract_evidence_items
from src.evidence.gates import EvidenceGateConfig, EvidenceGateError, check_evidence_gate, enforce_evidence_gate
from src.evidence.store import EvidenceStore


@pytest.mark.unit
def test_evidence_gate_default_is_permissive(temp_project_folder):
    result = check_evidence_gate(project_folder=str(temp_project_folder))
    assert result["ok"] is True
    assert result["require_evidence"] is False


@pytest.mark.unit
def test_evidence_gate_blocks_when_required_and_missing(temp_project_folder):
    cfg = EvidenceGateConfig(require_evidence=True, min_items_per_source=1)

    result = check_evidence_gate(
        project_folder=str(temp_project_folder),
        source_ids=["src1"],
        config=cfg,
    )
    assert result["ok"] is False
    assert result["per_source"]["src1"]["count"] == 0

    with pytest.raises(EvidenceGateError):
        enforce_evidence_gate(
            project_folder=str(temp_project_folder),
            source_ids=["src1"],
            config=cfg,
        )


@pytest.mark.unit
def test_evidence_gate_passes_when_required_and_present(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))

    parsed = {
        "blocks": [
            {
                "kind": "paragraph",
                "span": {"start_line": 1, "end_line": 1},
                "text": '"Quoted statement" with enough length for extraction.',
            }
        ]
    }
    items = extract_evidence_items(
        parsed=parsed,
        source_id="src1",
        created_at="2025-01-01T00:00:00+00:00",
        max_items=5,
    )
    store.write_evidence_items("src1", items)

    cfg = EvidenceGateConfig(require_evidence=True, min_items_per_source=1)
    result = enforce_evidence_gate(
        project_folder=str(temp_project_folder),
        source_ids=["src1"],
        config=cfg,
    )

    assert result["ok"] is True
    assert result["per_source"]["src1"]["count"] == 1
