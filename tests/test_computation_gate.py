import json

import pytest

from src.claims.gates import ComputationGateConfig, check_computation_gate, enforce_computation_gate, ComputationGateError


def _write_claims(project_folder, claims):
    claims_dir = project_folder / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    (claims_dir / "claims.json").write_text(json.dumps(claims, indent=2) + "\n")


def _write_metrics(project_folder, metrics):
    outputs_dir = project_folder / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


@pytest.mark.unit
def test_computation_gate_config_from_context_parses_and_sanitizes_values():
    cfg = ComputationGateConfig.from_context(
        {
            "computation_gate": {
                "enabled": True,
                "on_missing_metrics": "downgrade",
            }
        }
    )
    assert cfg.enabled is True
    assert cfg.on_missing_metrics == "downgrade"

    bad = ComputationGateConfig.from_context(
        {
            "computation_gate": {
                "enabled": True,
                "on_missing_metrics": "nope",
            }
        }
    )
    assert bad.enabled is True
    assert bad.on_missing_metrics == "block"

    defaulted = ComputationGateConfig.from_context({"computation_gate": "not-a-dict"})
    assert defaulted == ComputationGateConfig()


@pytest.mark.unit
def test_computation_gate_default_downgrades_when_metric_missing(temp_project_folder):
    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "computed",
                "statement": "Test claim",
                "metric_keys": ["m1"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    result = check_computation_gate(project_folder=temp_project_folder)
    assert result["ok"] is True
    assert result["enabled"] is True
    assert result["action"] == "downgrade"
    assert result["missing_metric_keys"] == ["m1"]


@pytest.mark.unit
def test_computation_gate_blocks_when_metric_missing(temp_project_folder):
    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "computed",
                "statement": "Test claim",
                "metric_keys": ["m_missing"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    cfg = ComputationGateConfig(enabled=True, on_missing_metrics="block")
    result = check_computation_gate(project_folder=temp_project_folder, config=cfg)

    assert result["enabled"] is True
    assert result["ok"] is False
    assert result["action"] == "block"
    assert result["missing_metric_keys"] == ["m_missing"]

    with pytest.raises(ComputationGateError):
        enforce_computation_gate(project_folder=temp_project_folder, config=cfg)


@pytest.mark.unit
def test_computation_gate_passes_when_metrics_present(temp_project_folder):
    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "computed",
                "statement": "Test claim",
                "metric_keys": ["m1"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    _write_metrics(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "metric_key": "m1",
                "name": "Metric 1",
                "value": 1.23,
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    cfg = ComputationGateConfig(enabled=True, on_missing_metrics="block")
    result = check_computation_gate(project_folder=temp_project_folder, config=cfg)

    assert result["ok"] is True
    assert result["action"] == "pass"
    assert result["missing_metric_keys"] == []


@pytest.mark.unit
def test_computation_gate_downgrades_when_metric_missing(temp_project_folder):
    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "computed",
                "statement": "Test claim",
                "metric_keys": ["m_missing"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    cfg = ComputationGateConfig(enabled=True, on_missing_metrics="downgrade")
    result = check_computation_gate(project_folder=temp_project_folder, config=cfg)

    assert result["ok"] is True
    assert result["action"] == "downgrade"
    assert result["missing_metric_keys"] == ["m_missing"]


@pytest.mark.unit
def test_computation_gate_ignores_non_computed_claims(temp_project_folder):
    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "source_backed",
                "statement": "Test claim",
                "citation_keys": ["smith2020"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    cfg = ComputationGateConfig(enabled=True, on_missing_metrics="block")
    result = check_computation_gate(project_folder=temp_project_folder, config=cfg)

    assert result["ok"] is True
    assert result["action"] == "pass"
    assert result["missing_metric_keys"] == []
