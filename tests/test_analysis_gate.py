import json

import pytest

from src.analysis.gates import AnalysisGateConfig, AnalysisGateError, check_analysis_gate, enforce_analysis_gate


def _write_metrics(project_folder, records):
    outputs_dir = project_folder / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _minimal_metric(metric_key: str = "m1", value: float = 1.23):
    return {
        "schema_version": "1.0",
        "metric_key": metric_key,
        "name": "Metric",
        "value": value,
        "created_at": "2025-01-01T00:00:00Z",
    }


def _touch(project_folder, relpath: str, content: str = "x"):
    p = project_folder / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


@pytest.mark.unit
def test_analysis_gate_disabled_is_permissive(temp_project_folder):
    cfg = AnalysisGateConfig(enabled=False)
    result = check_analysis_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "disabled"


@pytest.mark.unit
def test_analysis_gate_blocks_when_metrics_missing(temp_project_folder):
    cfg = AnalysisGateConfig(enabled=True, on_failure="block", min_metrics=1)
    result = check_analysis_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is False
    assert result["action"] == "block"

    with pytest.raises(AnalysisGateError):
        enforce_analysis_gate(project_folder=str(temp_project_folder), config=cfg)


@pytest.mark.unit
def test_analysis_gate_passes_when_metrics_present_and_valid(temp_project_folder):
    _write_metrics(temp_project_folder, [_minimal_metric()])

    cfg = AnalysisGateConfig(enabled=True, on_failure="block", min_metrics=1)
    result = check_analysis_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "pass"
    assert result["metrics_valid_items"] == 1


@pytest.mark.unit
def test_analysis_gate_blocks_when_require_tables_but_none_exist(temp_project_folder):
    _write_metrics(temp_project_folder, [_minimal_metric()])

    cfg = AnalysisGateConfig(enabled=True, on_failure="block", require_tables=True)
    result = check_analysis_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is False
    assert result["action"] == "block"


@pytest.mark.unit
def test_analysis_gate_passes_when_require_tables_and_figures_artifacts_present(temp_project_folder):
    _write_metrics(temp_project_folder, [_minimal_metric()])
    _touch(temp_project_folder, "outputs/tables/table1.tex", "\\begin{tabular}{}\\end{tabular}")
    _touch(temp_project_folder, "outputs/figures/fig1.pdf", "%PDF-1.4")

    cfg = AnalysisGateConfig(enabled=True, on_failure="block", require_tables=True, require_figures=True)
    result = check_analysis_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "pass"
    assert result["tables_count"] >= 1
    assert result["figures_count"] >= 1


@pytest.mark.unit
def test_analysis_gate_downgrades_when_configured(temp_project_folder):
    cfg = AnalysisGateConfig(enabled=True, on_failure="downgrade", min_metrics=1)
    result = check_analysis_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "downgrade"


@pytest.mark.unit
def test_analysis_gate_config_from_context_parses_and_sanitizes_values():
    cfg = AnalysisGateConfig.from_context(
        {
            "analysis_gate": {
                "enabled": True,
                "on_failure": "downgrade",
                "min_metrics": "2",
                "require_tables": True,
                "require_figures": True,
            }
        }
    )

    assert cfg.enabled is True
    assert cfg.on_failure == "downgrade"
    assert cfg.min_metrics == 2
    assert cfg.require_tables is True
    assert cfg.require_figures is True

    bad = AnalysisGateConfig.from_context({"analysis_gate": {"enabled": True, "on_failure": "nope"}})
    assert bad.enabled is True
    assert bad.on_failure == "block"
