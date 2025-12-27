from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.unit
@patch("src.pipeline.runner.LiteratureWorkflow")
@patch("src.pipeline.runner.ResearchWorkflow")
async def test_unified_pipeline_defaults_to_enabled_evidence_pipeline(
    mock_research_workflow, mock_literature_workflow, temp_project_folder
):
    # Arrange: make phase 1 and phase 2 succeed without any external calls.
    mock_research = mock_research_workflow.return_value
    mock_research.run = AsyncMock(return_value={"success": True})

    mock_lit = mock_literature_workflow.return_value
    mock_lit.run = AsyncMock(return_value={"success": True})

    from src.pipeline.runner import run_full_pipeline

    # Act
    await run_full_pipeline(
        str(temp_project_folder),
        enable_gap_resolution=False,
        enable_writing_review=False,
        workflow_overrides=None,
    )

    # Assert: LiteratureWorkflow.run receives workflow_context with evidence_pipeline enabled.
    _, kwargs = mock_lit.run.call_args
    ctx = kwargs.get("workflow_context")
    assert isinstance(ctx, dict)
    assert ctx["evidence_pipeline"]["enabled"] is True
