from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _write_minimal_project(project_folder):
    (project_folder / "project.json").write_text(
        json.dumps({"id": "p1", "title": "t"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (project_folder / "RESEARCH_OVERVIEW.md").write_text("overview\n", encoding="utf-8")


def _ok_agent_result(name: str = "x"):
    return SimpleNamespace(
        success=True,
        tokens_used=0,
        error=None,
        structured_data={"files_saved": {}},
        to_dict=lambda: {
            "agent_name": name,
            "success": True,
            "tokens_used": 0,
            "error": None,
            "structured_data": {"files_saved": {}},
        },
    )


@pytest.mark.unit
@pytest.mark.asyncio
@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key", "EDISON_API_KEY": "test-edison"}, clear=True)
@patch("src.llm.claude_client.anthropic.Anthropic")
@patch("src.llm.claude_client.anthropic.AsyncAnthropic")
@patch("src.llm.edison_client.OfficialEdisonClient")
async def test_literature_workflow_runs_evidence_pipeline_by_default(
    mock_official_edison, mock_async_anthropic, mock_anthropic, tmp_path
):
    project_folder = tmp_path / "proj"
    project_folder.mkdir()
    _write_minimal_project(project_folder)

    with patch("src.agents.literature_workflow.run_local_evidence_pipeline") as pipeline:
        pipeline.return_value = {"source_ids": ["file:abc"], "processed_count": 1}

        from src.agents.literature_workflow import LiteratureWorkflow

        wf = LiteratureWorkflow(use_cache=False)
        wf.hypothesis_developer.execute = AsyncMock(return_value=_ok_agent_result("hyp"))
        wf.literature_searcher.execute = AsyncMock(return_value=_ok_agent_result("search"))
        wf.literature_synthesizer.execute = AsyncMock(return_value=_ok_agent_result("synth"))
        wf.paper_structurer.execute = AsyncMock(return_value=_ok_agent_result("paper"))
        wf.project_planner.execute = AsyncMock(return_value=_ok_agent_result("plan"))
        wf.consistency_checker.check_consistency = AsyncMock(return_value=_ok_agent_result("consistency"))
        wf.readiness_assessor.assess_project = AsyncMock(return_value=_ok_agent_result("readiness"))

        result = await wf.run(str(project_folder))

    pipeline.assert_called_once()
    assert result.project_id == "p1"


@pytest.mark.unit
@pytest.mark.asyncio
@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key", "EDISON_API_KEY": "test-edison"}, clear=True)
@patch("src.llm.claude_client.anthropic.Anthropic")
@patch("src.llm.claude_client.anthropic.AsyncAnthropic")
@patch("src.llm.edison_client.OfficialEdisonClient")
async def test_literature_workflow_allows_disabling_evidence_pipeline(
    mock_official_edison, mock_async_anthropic, mock_anthropic, tmp_path
):
    project_folder = tmp_path / "proj"
    project_folder.mkdir()
    _write_minimal_project(project_folder)

    with patch("src.agents.literature_workflow.run_local_evidence_pipeline") as pipeline:
        from src.agents.literature_workflow import LiteratureWorkflow

        wf = LiteratureWorkflow(use_cache=False)
        wf.hypothesis_developer.execute = AsyncMock(return_value=_ok_agent_result("hyp"))
        wf.literature_searcher.execute = AsyncMock(return_value=_ok_agent_result("search"))
        wf.literature_synthesizer.execute = AsyncMock(return_value=_ok_agent_result("synth"))
        wf.paper_structurer.execute = AsyncMock(return_value=_ok_agent_result("paper"))
        wf.project_planner.execute = AsyncMock(return_value=_ok_agent_result("plan"))
        wf.consistency_checker.check_consistency = AsyncMock(return_value=_ok_agent_result("consistency"))
        wf.readiness_assessor.assess_project = AsyncMock(return_value=_ok_agent_result("readiness"))

        result = await wf.run(str(project_folder), workflow_context={"evidence_pipeline": {"enabled": False}})

    pipeline.assert_not_called()
    assert result.project_id == "p1"
