from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_full_pipeline_chains_phases(tmp_path):
    project_folder = tmp_path / "proj"
    project_folder.mkdir()

    fake_phase_result = SimpleNamespace(success=True, errors=[], to_dict=lambda: {"success": True, "errors": []})

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), patch("src.pipeline.runner.ResearchWorkflow") as RW, patch(
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

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), patch("src.pipeline.runner.ResearchWorkflow") as RW, patch("src.pipeline.runner.LiteratureWorkflow") as LW:
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

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), patch("src.pipeline.runner.ResearchWorkflow") as RW, patch("src.pipeline.runner.LiteratureWorkflow") as LW, patch(
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

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), patch("src.pipeline.runner.ResearchWorkflow") as RW, patch("src.pipeline.runner.LiteratureWorkflow") as LW, patch(
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

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), patch("src.pipeline.runner.ResearchWorkflow") as RW, patch("src.pipeline.runner.LiteratureWorkflow") as LW, patch(
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

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), patch("src.pipeline.runner.ResearchWorkflow") as RW, patch("src.pipeline.runner.LiteratureWorkflow") as LW, patch(
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


@pytest.mark.unit
def test_default_gate_config_enables_all_gates():
    """Test that default_gate_config enables gates in downgrade mode."""
    from src.pipeline.defaults import default_gate_config

    gates = default_gate_config()

    # All gates should be present.
    assert "evidence_gate" in gates
    assert "citation_gate" in gates
    assert "computation_gate" in gates
    assert "claim_evidence_gate" in gates
    assert "literature_gate" in gates
    assert "analysis_gate" in gates
    assert "citation_accuracy_gate" in gates

    # Evidence gate uses require_evidence instead of enabled.
    assert gates["evidence_gate"]["require_evidence"] is True

    # Others use enabled=True with on_failure=downgrade.
    assert gates["citation_gate"]["enabled"] is True
    assert gates["citation_gate"]["on_missing"] == "downgrade"

    assert gates["computation_gate"]["enabled"] is True
    assert gates["computation_gate"]["on_missing_metrics"] == "downgrade"

    assert gates["claim_evidence_gate"]["enabled"] is True
    assert gates["claim_evidence_gate"]["on_failure"] == "downgrade"

    assert gates["literature_gate"]["enabled"] is True
    assert gates["literature_gate"]["on_failure"] == "downgrade"

    assert gates["analysis_gate"]["enabled"] is True
    assert gates["analysis_gate"]["on_failure"] == "downgrade"

    assert gates["citation_accuracy_gate"]["enabled"] is True
    assert gates["citation_accuracy_gate"]["on_failure"] == "downgrade"


@pytest.mark.unit
def test_build_writing_context_includes_gates(tmp_path):
    """Test that _build_writing_context includes default gate configs."""
    from src.pipeline.runner import _build_writing_context

    project_folder = tmp_path / "proj"
    project_folder.mkdir()

    ctx = _build_writing_context(project_folder)

    # All gate configs should be in context.
    assert "evidence_gate" in ctx
    assert "citation_gate" in ctx
    assert "computation_gate" in ctx
    assert "claim_evidence_gate" in ctx
    assert "literature_gate" in ctx
    assert "analysis_gate" in ctx
    assert "citation_accuracy_gate" in ctx


@pytest.mark.unit
def test_build_writing_context_allows_overrides(tmp_path):
    """Test that extra overrides can disable gates."""
    from src.pipeline.runner import _build_writing_context

    project_folder = tmp_path / "proj"
    project_folder.mkdir()

    ctx = _build_writing_context(
        project_folder,
        extra={
            "citation_gate": {"enabled": False},
            "custom_key": "custom_value",
        },
    )

    # Override should replace the default.
    assert ctx["citation_gate"]["enabled"] is False
    assert ctx["custom_key"] == "custom_value"

    # Others should still be at defaults.
    assert ctx["evidence_gate"]["require_evidence"] is True
