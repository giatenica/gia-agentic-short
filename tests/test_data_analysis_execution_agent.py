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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analysis_execution_agent_fails_on_missing_project_folder(temp_project_folder):
    agent = DataAnalysisExecutionAgent(client=None)
    result = await agent.execute({})
    assert result.success is False
    assert "project_folder" in (result.error or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analysis_execution_agent_disabled_returns_success(temp_project_folder):
    agent = DataAnalysisExecutionAgent(client=None)
    result = await agent.execute({"project_folder": str(temp_project_folder), "analysis_execution": {"enabled": False}})
    assert result.success is True
    assert result.structured_data["metadata"]["enabled"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analysis_execution_agent_blocks_on_script_failure(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    script_text = "raise SystemExit(2)\n"
    (analysis_dir / "fail.py").write_text(script_text, encoding="utf-8")

    agent = DataAnalysisExecutionAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "analysis_execution": {"scripts": ["analysis/fail.py"], "on_script_failure": "block"},
    }
    result = await agent.execute(ctx)
    assert result.success is False
    assert "returncode=2" in (result.error or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analysis_execution_agent_downgrades_on_script_failure_when_configured(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Even if the script fails, the agent can downgrade instead of blocking.
    script_text = "raise SystemExit(3)\n"
    (analysis_dir / "fail.py").write_text(script_text, encoding="utf-8")

    agent = DataAnalysisExecutionAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "analysis_execution": {"scripts": ["analysis/fail.py"], "on_script_failure": "downgrade", "on_missing_outputs": "downgrade"},
    }
    result = await agent.execute(ctx)
    assert result.success is True
    assert result.structured_data["metadata"]["action"] == "downgrade"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analysis_execution_agent_blocks_on_missing_required_outputs(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Writes metrics.json but no tables.
    script_text = (
        "import json\n"
        "from pathlib import Path\n"
        "p = Path('.')\n"
        "(p / 'outputs').mkdir(parents=True, exist_ok=True)\n"
        "metrics = [{'schema_version':'1.0','metric_key':'alpha','name':'Alpha','value':1.0,'unit':'pct','created_at':'2025-01-01T00:00:00Z'}]\n"
        "(p / 'outputs' / 'metrics.json').write_text(json.dumps(metrics) + '\\n', encoding='utf-8')\n"
    )
    (analysis_dir / "no_tables.py").write_text(script_text, encoding="utf-8")

    agent = DataAnalysisExecutionAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "analysis_execution": {"scripts": ["analysis/no_tables.py"], "on_missing_outputs": "block"},
    }
    result = await agent.execute(ctx)
    assert result.success is False
    assert "outputs/tables" in (result.error or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analysis_execution_agent_downgrades_on_missing_required_outputs(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Writes metrics.json but no tables.
    script_text = (
        "import json\n"
        "from pathlib import Path\n"
        "p = Path('.')\n"
        "(p / 'outputs').mkdir(parents=True, exist_ok=True)\n"
        "metrics = [{'schema_version':'1.0','metric_key':'alpha','name':'Alpha','value':1.0,'unit':'pct','created_at':'2025-01-01T00:00:00Z'}]\n"
        "(p / 'outputs' / 'metrics.json').write_text(json.dumps(metrics) + '\\n', encoding='utf-8')\n"
    )
    (analysis_dir / "no_tables.py").write_text(script_text, encoding="utf-8")

    agent = DataAnalysisExecutionAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "analysis_execution": {"scripts": ["analysis/no_tables.py"], "on_missing_outputs": "downgrade"},
    }
    result = await agent.execute(ctx)
    assert result.success is True
    assert result.structured_data["metadata"]["action"] == "downgrade"
    assert "outputs/tables/*.tex" in result.structured_data["metadata"]["missing_outputs"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analysis_execution_agent_blocks_on_invalid_metrics_json(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Writes invalid metrics.json (not a list) and writes a table.
    script_text = (
        "import json\n"
        "from pathlib import Path\n"
        "p = Path('.')\n"
        "(p / 'outputs' / 'tables').mkdir(parents=True, exist_ok=True)\n"
        "(p / 'outputs' / 'metrics.json').write_text(json.dumps({'bad': 1}) + '\\n', encoding='utf-8')\n"
        "(p / 'outputs' / 'tables' / 't1.tex').write_text('x', encoding='utf-8')\n"
    )
    (analysis_dir / "bad_metrics.py").write_text(script_text, encoding="utf-8")

    agent = DataAnalysisExecutionAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "analysis_execution": {"scripts": ["analysis/bad_metrics.py"], "on_missing_outputs": "block"},
    }
    result = await agent.execute(ctx)
    assert result.success is False
    assert "outputs/metrics.json" in (result.error or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analysis_execution_config_defaults_to_downgrade_on_missing_scripts(temp_project_folder):
    """Test that the default config gracefully skips when no analysis scripts exist."""
    from src.agents.data_analysis_execution import AnalysisExecutionConfig
    
    # Verify default is "downgrade"
    config = AnalysisExecutionConfig()
    assert config.on_missing_outputs == "downgrade"
    
    # Test from_context defaults
    config_from_ctx = AnalysisExecutionConfig.from_context({})
    assert config_from_ctx.on_missing_outputs == "downgrade"
    
    # Test agent execution with no analysis folder
    agent = DataAnalysisExecutionAgent(client=None)
    ctx = {"project_folder": str(temp_project_folder)}
    
    # No analysis folder exists
    result = await agent.execute(ctx)
    
    # Should succeed with downgrade action (graceful skip)
    assert result.success is True
    assert result.structured_data.get("metadata", {}).get("action") == "downgrade"
    assert result.structured_data.get("metadata", {}).get("reason") == "no_scripts"

