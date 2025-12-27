from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_full_pipeline_chains_phases(tmp_path):
    project_folder = tmp_path / "proj"
    project_folder.mkdir()

    fake_phase_result = SimpleNamespace(success=True, errors=[], to_dict=lambda: {"success": True, "errors": []})

    with patch("src.pipeline.runner.ResearchWorkflow") as RW, patch(
        "src.pipeline.runner.LiteratureWorkflow"
    ) as LW, patch("src.pipeline.runner.GapResolutionWorkflow") as GW, patch(
        "src.pipeline.runner.run_writing_review_stage"
    ) as WR:
        RW.return_value.run = AsyncMock(return_value=fake_phase_result)
        LW.return_value.run = AsyncMock(return_value=fake_phase_result)
        GW.return_value.run = AsyncMock(return_value=fake_phase_result)
        WR.return_value = SimpleNamespace(to_payload=lambda: {"success": True, "needs_revision": False})

        from src.pipeline.runner import run_full_pipeline

        ctx = await run_full_pipeline(str(project_folder))

    assert ctx.success is True
    assert "phase_1" in ctx.phase_results
    assert "phase_2" in ctx.phase_results
    assert "phase_3" in ctx.phase_results
    assert "phase_4_writing_review" in ctx.phase_results
    assert "start" in ctx.checkpoints
    assert "end" in ctx.checkpoints


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_full_pipeline_stops_on_phase1_failure(tmp_path):
    project_folder = tmp_path / "proj"
    project_folder.mkdir()

    fake_failure = SimpleNamespace(success=False, errors=["nope"], to_dict=lambda: {"success": False, "errors": ["nope"]})

    with patch("src.pipeline.runner.ResearchWorkflow") as RW, patch("src.pipeline.runner.LiteratureWorkflow") as LW:
        RW.return_value.run = AsyncMock(return_value=fake_failure)
        LW.return_value.run = AsyncMock(return_value=SimpleNamespace())

        from src.pipeline.runner import run_full_pipeline

        ctx = await run_full_pipeline(str(project_folder))

    assert ctx.success is False
    assert "phase_1" in ctx.phase_results
    assert "phase_2" not in ctx.phase_results
    assert any("nope" in e for e in ctx.errors)
