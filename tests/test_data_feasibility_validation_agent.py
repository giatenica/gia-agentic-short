import json

import pytest

from src.agents.data_feasibility_validation import DataFeasibilityValidationAgent


def _write_csv(project_folder, name: str, csv_text: str) -> None:
    data_dir = project_folder / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / name).write_text(csv_text, encoding="utf-8")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_data_feasibility_agent_pass_case_emits_outputs(temp_project_folder):
    _write_csv(
        temp_project_folder,
        "sample.csv",
        "date,x,y\n2020-01-01,1,10\n2020-12-31,2,20\n",
    )

    agent = DataFeasibilityValidationAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "data_feasibility": {
            "required_columns": ["date", "x", "y"],
            "date_column": "date",
            "sample_period": {"start": "2020-01-01", "end": "2020-12-31"},
            "variables": [
                {"name": "x_plus_y", "requires": ["x", "y"]},
            ],
            "on_failure": "block",
        },
    }

    result = await agent.execute(ctx)
    assert result.success is True

    outputs_dir = temp_project_folder / "outputs"
    json_path = outputs_dir / "data_feasibility.json"
    report_path = outputs_dir / "data_feasibility_report.md"

    assert json_path.exists()
    assert report_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["checks"]["required_columns"]["ok"] is True
    assert payload["checks"]["variables"]["ok"] is True
    assert payload["checks"]["sample_period"]["ok"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_data_feasibility_agent_blocks_on_missing_required_columns(temp_project_folder):
    _write_csv(
        temp_project_folder,
        "sample.csv",
        "date,x\n2020-01-01,1\n2020-12-31,2\n",
    )

    agent = DataFeasibilityValidationAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "data_feasibility": {
            "required_columns": ["date", "x", "y"],
            "date_column": "date",
            "sample_period": {"start": "2020-01-01", "end": "2020-12-31"},
            "on_failure": "block",
        },
    }

    result = await agent.execute(ctx)
    assert result.success is False
    assert "failed" in (result.error or "").lower()

    json_path = temp_project_folder / "outputs" / "data_feasibility.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["checks"]["required_columns"]["ok"] is False
    assert "y" in payload["checks"]["required_columns"]["missing"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_data_feasibility_agent_downgrades_on_insufficient_date_coverage(temp_project_folder):
    _write_csv(
        temp_project_folder,
        "sample.csv",
        "date,x,y\n2020-01-01,1,10\n2020-06-30,2,20\n",
    )

    agent = DataFeasibilityValidationAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "data_feasibility": {
            "required_columns": ["date", "x", "y"],
            "date_column": "date",
            "sample_period": {"start": "2020-01-01", "end": "2020-12-31"},
            "on_failure": "downgrade",
        },
    }

    result = await agent.execute(ctx)
    assert result.success is True
    assert result.structured_data is not None
    assert result.structured_data.get("ok") is False
    assert result.structured_data.get("metadata", {}).get("action") == "downgrade"

    json_path = temp_project_folder / "outputs" / "data_feasibility.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["checks"]["sample_period"]["ok"] is False
