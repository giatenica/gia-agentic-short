from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import AgentResult
from src.agents.literature_workflow import LiteratureWorkflow
from src.citations.registry import save_citations
from src.llm.claude_client import ModelTier, TaskType


class _DummyClaudeClient:
    def get_model_for_task(self, task_type: TaskType) -> ModelTier:
        return ModelTier.SONNET


@pytest.mark.unit
@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key", "EDISON_API_KEY": "test-key"}, clear=True)
async def test_literature_workflow_populates_source_citation_map(tmp_path):
    project = tmp_path / "proj"
    project.mkdir(parents=True)

    # Minimal prerequisites
    (project / "project.json").write_text(json.dumps({"id": "p1"}), encoding="utf-8")
    (project / "RESEARCH_OVERVIEW.md").write_text("overview", encoding="utf-8")

    # Create sources/<source_id>/raw with a DOI in filename.
    (project / "sources" / "src1" / "raw").mkdir(parents=True)
    (project / "sources" / "src1" / "raw" / "paper_10.1000_xyz123.pdf").write_text("", encoding="utf-8")

    # Workflow with injected dummy clients so it doesn't initialize real SDK clients.
    wf = LiteratureWorkflow(client=_DummyClaudeClient(), edison_client=MagicMock(), use_cache=False)

    # Patch agent executes to avoid LLM/Edison.
    ok = AgentResult(
        agent_name="x",
        task_type=TaskType.DATA_EXTRACTION,
        model_tier=ModelTier.HAIKU,
        success=True,
        content="",
        tokens_used=0,
        execution_time=0,
    )
    wf.hypothesis_developer.execute = AsyncMock(return_value=ok)
    wf.literature_searcher.execute = AsyncMock(return_value=ok)

    async def _synth_exec(context):
        save_citations(
            project,
            [
                {
                    "schema_version": "1.0",
                    "citation_key": "Smith2020Example",
                    "status": "verified",
                    "title": "Example",
                    "authors": ["Smith"],
                    "year": 2020,
                    "created_at": "2020-01-01T00:00:00Z",
                    "identifiers": {"doi": "10.1000/xyz123"},
                }
            ],
            validate=True,
        )
        return ok

    wf.literature_synthesizer.execute = AsyncMock(side_effect=_synth_exec)
    wf.paper_structurer.execute = AsyncMock(return_value=ok)
    wf.project_planner.execute = AsyncMock(return_value=ok)
    wf.consistency_checker.check_consistency = AsyncMock(return_value=ok)
    wf.readiness_assessor.assess_project = AsyncMock(return_value=ok)

    captured = {}

    async def _writing_review(context):
        captured.update(context)
        class _R:
            success = True
            needs_revision = False
            def to_payload(self):
                return {
                    "success": True,
                    "needs_revision": False,
                    "written_section_relpaths": [],
                    "gates": {"enabled": False},
                    "review": None,
                    "error": None,
                }
        return _R()

    with patch("src.agents.literature_workflow.run_writing_review_stage", new=AsyncMock(side_effect=_writing_review)):
        result = await wf.run(
            str(project),
            workflow_context={
                "evidence_pipeline": {"enabled": False},
                "writing_review": {"enabled": True, "writers": []},
            },
        )

    assert result.success is True
    assert captured.get("source_citation_map") == {"src1": "Smith2020Example"}

    # Persisted mapping should exist
    assert (project / "bibliography" / "source_citation_map.json").exists()
