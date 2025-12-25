import json

import pytest

from src.agents.methods_writer import MethodsWriterAgent

from src.citations.registry import make_minimal_citation_record, save_citations


def _write_evidence_marker(project_folder, source_id: str):
    sources_dir = project_folder / "sources" / source_id
    sources_dir.mkdir(parents=True, exist_ok=True)
    (sources_dir / "evidence.json").write_text("[]\n", encoding="utf-8")


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
async def test_methods_writer_blocks_on_missing_metric(temp_project_folder):
    _write_claims(temp_project_folder, metric_keys=["alpha"])
    _write_metrics(temp_project_folder, metrics=[])

    agent = MethodsWriterAgent(client=None)
    ctx = {"project_folder": str(temp_project_folder), "methods_writer": {"on_missing_metrics": "block"}}
    result = await agent.execute(ctx)

    assert result.success is False
    assert (temp_project_folder / "outputs/sections/methods.tex").exists() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_methods_writer_writes_output_on_downgrade(temp_project_folder):
    _write_claims(temp_project_folder, metric_keys=["alpha"])
    _write_metrics(temp_project_folder, metrics=[])

    agent = MethodsWriterAgent(client=None)
    ctx = {"project_folder": str(temp_project_folder), "methods_writer": {"on_missing_metrics": "downgrade"}}
    result = await agent.execute(ctx)

    assert result.success is True
    out_path = temp_project_folder / "outputs/sections/methods.tex"
    assert out_path.exists()

    tex = out_path.read_text(encoding="utf-8")
    assert "\\section{Data and Methodology}" in tex
    assert result.structured_data["metadata"]["action"] == "downgrade"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_methods_writer_disabled_writes_section(temp_project_folder):
    agent = MethodsWriterAgent(client=None)
    ctx = {"project_folder": str(temp_project_folder), "methods_writer": {"enabled": False}}
    result = await agent.execute(ctx)

    assert result.success is True
    out_path = temp_project_folder / "outputs/sections/methods.tex"
    assert out_path.exists()
    assert result.structured_data["metadata"]["enabled"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_methods_writer_blocks_on_missing_evidence_when_configured(temp_project_folder):
    agent = MethodsWriterAgent(client=None)
    ctx = {"project_folder": str(temp_project_folder), "methods_writer": {"on_missing_evidence": "block"}}
    result = await agent.execute(ctx)

    assert result.success is False
    assert "evidence.json" in (result.error or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_methods_writer_blocks_on_missing_citation_when_configured(temp_project_folder):
    _write_evidence_marker(temp_project_folder, "src_1")
    save_citations(
        temp_project_folder,
        [
            make_minimal_citation_record(
                citation_key="Known2020",
                title="Known",
                authors=["A"],
                year=2020,
                status="verified",
            )
        ],
        validate=True,
    )

    agent = MethodsWriterAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "methods_writer": {"on_missing_citation": "block"},
        "source_citation_map": {},
    }
    result = await agent.execute(ctx)

    assert result.success is False
    assert "Missing canonical citation key" in (result.error or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_methods_writer_success_path_includes_metric_and_citation(temp_project_folder):
    _write_evidence_marker(temp_project_folder, "src_1")
    save_citations(
        temp_project_folder,
        [
            make_minimal_citation_record(
                citation_key="Known2020",
                title="Known",
                authors=["A"],
                year=2020,
                status="verified",
            )
        ],
        validate=True,
    )
    _write_claims(temp_project_folder, metric_keys=["alpha"])
    _write_metrics(temp_project_folder, metrics=[{"metric_key": "alpha", "value": 1.5, "unit": "pct"}])

    agent = MethodsWriterAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "source_citation_map": {"src_1": "Known2020"},
    }
    result = await agent.execute(ctx)

    assert result.success is True
    tex = (temp_project_folder / "outputs/sections/methods.tex").read_text(encoding="utf-8")
    assert "\\cite{Known2020}" in tex
    assert "alpha" in tex
    assert "1.5" in tex
