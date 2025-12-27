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
        "src.pipeline.runner.run_writing_review_stage", new_callable=AsyncMock
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_full_pipeline_stops_on_phase2_failure(tmp_path):
    project_folder = tmp_path / "proj"
    project_folder.mkdir()

    ok = SimpleNamespace(success=True, errors=[], to_dict=lambda: {"success": True, "errors": []})
    bad = SimpleNamespace(success=False, errors=["phase2"], to_dict=lambda: {"success": False, "errors": ["phase2"]})

    with patch("src.pipeline.runner.ResearchWorkflow") as RW, patch("src.pipeline.runner.LiteratureWorkflow") as LW, patch(
        "src.pipeline.runner.GapResolutionWorkflow"
    ) as GW:
        RW.return_value.run = AsyncMock(return_value=ok)
        LW.return_value.run = AsyncMock(return_value=bad)
        GW.return_value.run = AsyncMock(return_value=ok)

        from src.pipeline.runner import run_full_pipeline

        ctx = await run_full_pipeline(str(project_folder))

    assert ctx.success is False
    assert "phase_1" in ctx.phase_results
    assert "phase_2" in ctx.phase_results
    assert "phase_3" not in ctx.phase_results
    assert any("phase2" in e for e in ctx.errors)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_full_pipeline_stops_on_phase3_failure(tmp_path):
    project_folder = tmp_path / "proj"
    project_folder.mkdir()

    ok = SimpleNamespace(success=True, errors=[], to_dict=lambda: {"success": True, "errors": []})
    bad = SimpleNamespace(success=False, errors=["phase3"], to_dict=lambda: {"success": False, "errors": ["phase3"]})

    with patch("src.pipeline.runner.ResearchWorkflow") as RW, patch("src.pipeline.runner.LiteratureWorkflow") as LW, patch(
        "src.pipeline.runner.GapResolutionWorkflow"
    ) as GW, patch("src.pipeline.runner.run_writing_review_stage", new_callable=AsyncMock) as WR:
        RW.return_value.run = AsyncMock(return_value=ok)
        LW.return_value.run = AsyncMock(return_value=ok)
        GW.return_value.run = AsyncMock(return_value=bad)
        WR.return_value = SimpleNamespace(to_payload=lambda: {"success": True, "needs_revision": False})

        from src.pipeline.runner import run_full_pipeline

        ctx = await run_full_pipeline(str(project_folder))

    assert ctx.success is False
    assert "phase_4_writing_review" not in ctx.phase_results
    assert any("phase3" in e for e in ctx.errors)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_full_pipeline_respects_disable_flags(tmp_path):
    project_folder = tmp_path / "proj"
    project_folder.mkdir()

    ok = SimpleNamespace(success=True, errors=[], to_dict=lambda: {"success": True, "errors": []})

    with patch("src.pipeline.runner.ResearchWorkflow") as RW, patch("src.pipeline.runner.LiteratureWorkflow") as LW, patch(
        "src.pipeline.runner.GapResolutionWorkflow"
    ) as GW, patch("src.pipeline.runner.run_writing_review_stage", new_callable=AsyncMock) as WR:
        RW.return_value.run = AsyncMock(return_value=ok)
        LW.return_value.run = AsyncMock(return_value=ok)
        GW.return_value.run = AsyncMock(return_value=ok)
        WR.return_value = SimpleNamespace(to_payload=lambda: {"success": True, "needs_revision": False})

        from src.pipeline.runner import run_full_pipeline

        ctx = await run_full_pipeline(str(project_folder), enable_gap_resolution=False, enable_writing_review=False)

    assert ctx.success is True
    assert "phase_3" not in ctx.phase_results
    assert "phase_4_writing_review" not in ctx.phase_results


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_full_pipeline_passes_workflow_overrides(tmp_path):
    project_folder = tmp_path / "proj"
    project_folder.mkdir()

    ok = SimpleNamespace(success=True, errors=[], to_dict=lambda: {"success": True, "errors": []})
    overrides = {"evidence_pipeline": {"enabled": True}, "some_key": "some_value"}

    with patch("src.pipeline.runner.ResearchWorkflow") as RW, patch("src.pipeline.runner.LiteratureWorkflow") as LW, patch(
        "src.pipeline.runner.GapResolutionWorkflow"
    ) as GW, patch("src.pipeline.runner.run_writing_review_stage", new_callable=AsyncMock) as WR:
        RW.return_value.run = AsyncMock(return_value=ok)
        LW.return_value.run = AsyncMock(return_value=ok)
        GW.return_value.run = AsyncMock(return_value=ok)
        WR.return_value = SimpleNamespace(to_payload=lambda: {"success": True, "needs_revision": False})

        from src.pipeline.runner import run_full_pipeline

        ctx = await run_full_pipeline(str(project_folder), workflow_overrides=overrides)

        LW.return_value.run.assert_awaited_once()
        _, kwargs = LW.return_value.run.await_args
        assert kwargs.get("workflow_context") == overrides

        WR.assert_awaited_once()
        passed_context = WR.await_args.args[0]
        assert passed_context.get("some_key") == "some_value"
        assert passed_context.get("evidence_pipeline") == {"enabled": True}

    assert ctx.success is True
