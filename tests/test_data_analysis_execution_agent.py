import json

import pytest

from src.agents.data_analysis_execution import DataAnalysisExecutionAgent
from src.agents.results_writer import ResultsWriterAgent
from src.utils.schema_validation import is_valid_metric_record


def _write_claims(project_folder, *, metric_key: str):
    claims_dir = project_folder / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "schema_version": "1.0",
            "claim_id": "c1",
            "kind": "computed",
            "statement": "A computed statement",
            "metric_keys": [metric_key],
            "created_at": "2025-01-01T00:00:00Z",
        }
    ]
    (claims_dir / "claims.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analysis_execution_agent_runs_script_and_produces_outputs(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    script_text = (
        "import json\n"
        "from pathlib import Path\n"
        "p = Path('.')\n"
        "(p / 'outputs' / 'tables').mkdir(parents=True, exist_ok=True)\n"
        "(p / 'outputs' / 'figures').mkdir(parents=True, exist_ok=True)\n"
        "metrics = [\n"
        "  {\n"
        "    'schema_version': '1.0',\n"
        "    'metric_key': 'alpha',\n"
        "    'name': 'Alpha',\n"
        "    'value': 1.23,\n"
        "    'unit': 'pct',\n"
        "    'created_at': '2025-01-01T00:00:00Z'\n"
        "  }\n"
        "]\n"
        "(p / 'outputs' / 'metrics.json').write_text(json.dumps(metrics) + '\\n', encoding='utf-8')\n"
        "(p / 'outputs' / 'tables' / 't1.tex').write_text('\\\\begin{tabular}{ll}a&b\\\\\\n\\\\end{tabular}\\n', encoding='utf-8')\n"
        "(p / 'outputs' / 'figures' / 'f1.txt').write_text('figure', encoding='utf-8')\n"
        "print('ok')\n"
    )

    (analysis_dir / "run.py").write_text(script_text, encoding="utf-8")

    agent = DataAnalysisExecutionAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "analysis_execution": {
            "scripts": ["analysis/run.py"],
        },
    }

    result = await agent.execute(ctx)
    assert result.success is True

    metrics_path = temp_project_folder / "outputs" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert isinstance(metrics, list)
    assert len(metrics) >= 1
    assert is_valid_metric_record(metrics[0])

    tables_dir = temp_project_folder / "outputs" / "tables"
    assert (tables_dir / "t1.tex").exists()

    artifacts_path = temp_project_folder / "outputs" / "artifacts.json"
    assert artifacts_path.exists()
    payload = json.loads(artifacts_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert "outputs/metrics.json" in payload["created_files"]
    assert "outputs/tables/t1.tex" in payload["created_files"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_results_writer_consumes_metrics_from_analysis(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    script_text = (
        "import json\n"
        "from pathlib import Path\n"
        "p = Path('.')\n"
        "metrics = [{'schema_version':'1.0','metric_key':'alpha','name':'Alpha','value':2.5,'unit':'pct','created_at':'2025-01-01T00:00:00Z'}]\n"
        "(p / 'outputs').mkdir(parents=True, exist_ok=True)\n"
        "(p / 'outputs' / 'tables').mkdir(parents=True, exist_ok=True)\n"
        "(p / 'outputs' / 'metrics.json').write_text(json.dumps(metrics) + '\\n', encoding='utf-8')\n"
        "(p / 'outputs' / 'tables' / 't1.tex').write_text('x', encoding='utf-8')\n"
    )
    (analysis_dir / "run.py").write_text(script_text, encoding="utf-8")

    agent = DataAnalysisExecutionAgent(client=None)
    await agent.execute({"project_folder": str(temp_project_folder), "analysis_execution": {"scripts": ["analysis/run.py"]}})

    _write_claims(temp_project_folder, metric_key="alpha")

    writer = ResultsWriterAgent(client=None)
    r = await writer.execute({"project_folder": str(temp_project_folder)})

    assert r.success is True
    assert len((r.content or "").strip()) > 0
    assert "2.5" in (r.content or "")
