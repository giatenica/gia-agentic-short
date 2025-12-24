import json

import pytest

from src.agents.results_writer import ResultsWriterAgent


def _write_claims(project_folder, *, metric_keys):
    claims_dir = project_folder / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "schema_version": "1.0",
            "claim_id": "claim_alpha",
            "kind": "computed",
            "statement": "Computed result for alpha",
            "metric_keys": list(metric_keys),
            "created_at": "2025-01-01T00:00:00Z",
        }
    ]
    (claims_dir / "claims.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_metrics(project_folder, *, metrics):
    out_dir = project_folder / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = []
    for m in metrics:
        payload.append(
            {
                "schema_version": "1.0",
                "metric_key": m["metric_key"],
                "name": m.get("name") or m["metric_key"],
                "value": m["value"],
                "unit": m.get("unit"),
                "created_at": "2025-01-01T00:00:00Z",
            }
        )

    (out_dir / "metrics.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_results_writer_emits_only_metric_numbers(temp_project_folder):
    _write_claims(temp_project_folder, metric_keys=["alpha"])
    _write_metrics(
        temp_project_folder,
        metrics=[{"metric_key": "alpha", "name": "Alpha", "value": 0.5, "unit": ""}],
    )

    agent = ResultsWriterAgent(client=None)
    result = await agent.execute({"project_folder": str(temp_project_folder)})

    assert result.success is True
    assert result.structured_data["output_relpath"] == "outputs/sections/results.tex"

    tex = (temp_project_folder / "outputs/sections/results.tex").read_text(encoding="utf-8")
    assert "\\section{Results}" in tex
    assert "0.5" in tex


@pytest.mark.unit
@pytest.mark.asyncio
async def test_results_writer_blocks_on_missing_metric(temp_project_folder):
    _write_claims(temp_project_folder, metric_keys=["alpha"])
    _write_metrics(temp_project_folder, metrics=[])

    agent = ResultsWriterAgent(client=None)
    ctx = {"project_folder": str(temp_project_folder), "results_writer": {"on_missing_metrics": "block"}}
    result = await agent.execute(ctx)

    assert result.success is False
    assert (temp_project_folder / "outputs/sections/results.tex").exists() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_results_writer_downgrades_on_missing_metric(temp_project_folder):
    _write_claims(temp_project_folder, metric_keys=["alpha"])
    _write_metrics(temp_project_folder, metrics=[])

    agent = ResultsWriterAgent(client=None)
    ctx = {"project_folder": str(temp_project_folder), "results_writer": {"on_missing_metrics": "downgrade"}}
    result = await agent.execute(ctx)

    assert result.success is True
    tex = (temp_project_folder / "outputs/sections/results.tex").read_text(encoding="utf-8")
    assert "\\section{Results}" in tex
    assert "0.5" not in tex
    assert result.structured_data["metadata"]["action"] == "downgrade"
