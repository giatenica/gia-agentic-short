"""End-to-end integration tests for full pipeline.

These tests verify data flows correctly through all pipeline phases
using mock data and mocked external API calls.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest


def _create_mock_project(tmp_path: Path, project_name: str = "test_project") -> Path:
    """Create a minimal valid project structure for E2E testing."""
    project = tmp_path / project_name
    project.mkdir(parents=True, exist_ok=True)

    # Create project.json (required by Phase 1)
    project_json = {
        "id": "test_e2e_001",
        "title": "E2E Test Project",
        "research_question": "Does X affect Y in test markets?",
        "has_hypothesis": True,
        "hypothesis": "X positively affects Y.",
        "target_journal": "Test Journal",
        "paper_type": "Short Paper",
        "research_type": "Empirical",
        "has_data": True,
        "data_description": "Mock test data",
        "data_sources": "Mock CSV",
        "key_variables": "X, Y, Z",
        "methodology": "OLS regression",
        "related_literature": "MockAuthor (2024)",
        "expected_contribution": "Test contribution",
        "constraints": "",
        "deadline": "2026-12-31",
    }
    (project / "project.json").write_text(json.dumps(project_json, indent=2), encoding="utf-8")

    # Create analysis folder with mock data
    analysis = project / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "mock_data.csv").write_text("x,y,z\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")

    # Create sources with mock evidence
    sources = project / "sources"
    sources.mkdir(exist_ok=True)
    source1 = sources / "source1"
    source1.mkdir(exist_ok=True)
    (source1 / "evidence.json").write_text(
        json.dumps(
            [
                {
                    "schema_version": "1.0",
                    "item_id": "ev001",
                    "source_id": "source1",
                    "kind": "quote",
                    "text": "Test quote from source",
                    "page_number": 1,
                    "confidence": 0.9,
                    "created_at": "2025-01-01T00:00:00Z",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # Create bibliography with mock citation
    bibliography = project / "bibliography"
    bibliography.mkdir(exist_ok=True)
    (bibliography / "citations.json").write_text(
        json.dumps(
            [
                {
                    "schema_version": "1.0",
                    "citation_key": "mockauthor2024",
                    "source_id": "source1",
                    "title": "Mock Paper Title",
                    "authors": ["Mock Author"],
                    "year": "2024",
                    "status": "verified",
                    "created_at": "2025-01-01T00:00:00Z",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # Create outputs folder structure
    outputs = project / "outputs"
    outputs.mkdir(exist_ok=True)
    (outputs / "metrics.json").write_text(
        json.dumps(
            [
                {
                    "schema_version": "1.0",
                    "metric_key": "test_r2",
                    "name": "R-squared",
                    "value": 0.85,
                    "unit": "",
                    "description": "Model fit",
                    "created_at": "2025-01-01T00:00:00Z",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # Create claims folder
    claims = project / "claims"
    claims.mkdir(exist_ok=True)

    return project


def _mock_phase_result(success: bool = True, errors: list | None = None) -> SimpleNamespace:
    """Create a mock phase result object."""
    errs = errors if errors is not None else []
    return SimpleNamespace(
        success=success,
        errors=errs,
        to_dict=lambda: {"success": success, "errors": errs},
    )


def _mock_writing_result(success: bool = True, needs_revision: bool = False) -> SimpleNamespace:
    """Create a mock writing review result."""
    return SimpleNamespace(
        to_payload=lambda: {
            "success": success,
            "needs_revision": needs_revision,
            "written_section_relpaths": ["outputs/sections/introduction.tex"],
            "gates": {"enabled": True},
        }
    )


@pytest.fixture
def mock_e2e_project(tmp_path):
    """Fixture that creates a mock project for E2E testing."""
    return _create_mock_project(tmp_path)


class TestFullPipelineE2E:
    """End-to-end tests for the full pipeline."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pipeline_produces_output_with_mock_phases(self, mock_e2e_project):
        """Pipeline should complete successfully with mocked phase implementations."""
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            patch("src.pipeline.runner.ResearchWorkflow") as RW,
            patch("src.pipeline.runner.LiteratureWorkflow") as LW,
            patch("src.pipeline.runner.GapResolutionWorkflow") as GW,
            patch("src.pipeline.runner.run_writing_review_stage", new_callable=AsyncMock) as WR,
        ):
            RW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            LW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            GW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            WR.return_value = _mock_writing_result(success=True)

            from src.pipeline.runner import run_full_pipeline

            result = await run_full_pipeline(str(mock_e2e_project))

        # Verify pipeline completed successfully
        assert result.success is True
        assert "phase_1" in result.phase_results
        assert "phase_2" in result.phase_results
        assert "phase_3" in result.phase_results
        assert "phase_4_writing_review" in result.phase_results

        # Verify checkpoints were recorded
        assert "start" in result.checkpoints
        assert "phase_1_complete" in result.checkpoints
        assert "phase_2_complete" in result.checkpoints
        assert "phase_3_complete" in result.checkpoints
        assert "phase_4_complete" in result.checkpoints
        assert "end" in result.checkpoints

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pipeline_generates_claims_from_metrics(self, mock_e2e_project):
        """Pipeline should generate claims.json from metrics.json."""
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            patch("src.pipeline.runner.ResearchWorkflow") as RW,
            patch("src.pipeline.runner.LiteratureWorkflow") as LW,
            patch("src.pipeline.runner.GapResolutionWorkflow") as GW,
            patch("src.pipeline.runner.run_writing_review_stage", new_callable=AsyncMock) as WR,
        ):
            RW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            LW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            GW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            WR.return_value = _mock_writing_result(success=True)

            from src.pipeline.runner import run_full_pipeline

            result = await run_full_pipeline(str(mock_e2e_project))

        # Verify claims were generated
        claims_path = mock_e2e_project / "claims" / "claims.json"
        assert claims_path.exists(), "claims.json should be created"

        claims = json.loads(claims_path.read_text(encoding="utf-8"))
        assert isinstance(claims, list)
        assert len(claims) >= 1
        assert claims[0]["kind"] == "computed"
        assert "test_r2" in claims[0]["claim_id"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pipeline_writes_degradation_summary(self, mock_e2e_project):
        """Pipeline should write degradation summary on completion."""
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            patch("src.pipeline.runner.ResearchWorkflow") as RW,
            patch("src.pipeline.runner.LiteratureWorkflow") as LW,
            patch("src.pipeline.runner.GapResolutionWorkflow") as GW,
            patch("src.pipeline.runner.run_writing_review_stage", new_callable=AsyncMock) as WR,
        ):
            RW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            LW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            GW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            WR.return_value = _mock_writing_result(success=True)

            from src.pipeline.runner import run_full_pipeline

            await run_full_pipeline(str(mock_e2e_project))

        # Verify degradation summary was written
        degradation_path = mock_e2e_project / "outputs" / "degradation_summary.json"
        assert degradation_path.exists(), "degradation_summary.json should be created"

        summary = json.loads(degradation_path.read_text(encoding="utf-8"))
        assert "run_id" in summary
        assert "created_at" in summary
        assert "degradations" in summary

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pipeline_passes_gate_configs_to_writing_stage(self, mock_e2e_project):
        """Pipeline should pass default gate configs to writing review stage."""
        captured_context: Dict[str, Any] = {}

        async def capture_context(ctx):
            captured_context.update(ctx)
            return _mock_writing_result(success=True)

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            patch("src.pipeline.runner.ResearchWorkflow") as RW,
            patch("src.pipeline.runner.LiteratureWorkflow") as LW,
            patch("src.pipeline.runner.GapResolutionWorkflow") as GW,
            patch("src.pipeline.runner.run_writing_review_stage", new_callable=AsyncMock) as WR,
        ):
            RW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            LW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            GW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            WR.side_effect = capture_context

            from src.pipeline.runner import run_full_pipeline

            await run_full_pipeline(str(mock_e2e_project))

        # Verify gate configs were passed
        assert "evidence_gate" in captured_context
        assert "citation_gate" in captured_context
        assert "computation_gate" in captured_context
        assert "claim_evidence_gate" in captured_context
        assert "literature_gate" in captured_context
        assert "analysis_gate" in captured_context
        assert "citation_accuracy_gate" in captured_context

        # Verify gates are in downgrade mode (not blocking)
        assert captured_context["citation_gate"]["enabled"] is True
        assert captured_context["citation_gate"]["on_missing"] == "downgrade"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pipeline_skips_phases_when_disabled(self, mock_e2e_project):
        """Pipeline should skip gap resolution and writing when disabled."""
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            patch("src.pipeline.runner.ResearchWorkflow") as RW,
            patch("src.pipeline.runner.LiteratureWorkflow") as LW,
            patch("src.pipeline.runner.GapResolutionWorkflow") as GW,
            patch("src.pipeline.runner.run_writing_review_stage", new_callable=AsyncMock) as WR,
        ):
            RW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            LW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            GW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            WR.return_value = _mock_writing_result(success=True)

            from src.pipeline.runner import run_full_pipeline

            result = await run_full_pipeline(
                str(mock_e2e_project),
                enable_gap_resolution=False,
                enable_writing_review=False,
            )

        # Verify only Phase 1 and 2 ran
        assert result.success is True
        assert "phase_1" in result.phase_results
        assert "phase_2" in result.phase_results
        assert "phase_3" not in result.phase_results
        assert "phase_4_writing_review" not in result.phase_results

        # Gap resolution workflow should not have been called
        GW.return_value.run.assert_not_awaited()
        WR.assert_not_awaited()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pipeline_halts_on_phase_failure(self, mock_e2e_project):
        """Pipeline should halt and not run subsequent phases on failure."""
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True),
            patch("src.pipeline.runner.ResearchWorkflow") as RW,
            patch("src.pipeline.runner.LiteratureWorkflow") as LW,
            patch("src.pipeline.runner.GapResolutionWorkflow") as GW,
            patch("src.pipeline.runner.run_writing_review_stage", new_callable=AsyncMock) as WR,
        ):
            # Phase 2 fails
            RW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            LW.return_value.run = AsyncMock(
                return_value=_mock_phase_result(success=False, errors=["Literature search failed"])
            )
            GW.return_value.run = AsyncMock(return_value=_mock_phase_result(success=True))
            WR.return_value = _mock_writing_result(success=True)

            from src.pipeline.runner import run_full_pipeline

            result = await run_full_pipeline(str(mock_e2e_project))

        # Verify pipeline reported failure
        assert result.success is False

        # Verify subsequent phases were not run
        assert "phase_3" not in result.phase_results
        assert "phase_4_writing_review" not in result.phase_results

        # Verify error was captured
        assert any("Literature search failed" in e for e in result.errors)


class TestMockProjectFixture:
    """Tests for the mock project fixture itself."""

    @pytest.mark.unit
    def test_mock_project_has_required_structure(self, mock_e2e_project):
        """Mock project should have all required files and folders."""
        # Required files
        assert (mock_e2e_project / "project.json").exists()
        assert (mock_e2e_project / "analysis" / "mock_data.csv").exists()
        assert (mock_e2e_project / "sources" / "source1" / "evidence.json").exists()
        assert (mock_e2e_project / "bibliography" / "citations.json").exists()
        assert (mock_e2e_project / "outputs" / "metrics.json").exists()

        # Folders exist
        assert (mock_e2e_project / "claims").is_dir()
        assert (mock_e2e_project / "outputs").is_dir()

    @pytest.mark.unit
    def test_mock_project_json_is_valid(self, mock_e2e_project):
        """project.json should be valid and parseable."""
        project_json = json.loads((mock_e2e_project / "project.json").read_text(encoding="utf-8"))
        assert "research_question" in project_json
        assert "hypothesis" in project_json
        assert project_json["has_hypothesis"] is True

    @pytest.mark.unit
    def test_mock_evidence_is_valid(self, mock_e2e_project):
        """Evidence file should contain valid evidence items."""
        evidence = json.loads(
            (mock_e2e_project / "sources" / "source1" / "evidence.json").read_text(encoding="utf-8")
        )
        assert isinstance(evidence, list)
        assert len(evidence) >= 1
        assert evidence[0]["schema_version"] == "1.0"
        assert evidence[0]["kind"] == "quote"

    @pytest.mark.unit
    def test_mock_citations_is_valid(self, mock_e2e_project):
        """Citations file should contain valid citation records."""
        citations = json.loads(
            (mock_e2e_project / "bibliography" / "citations.json").read_text(encoding="utf-8")
        )
        assert isinstance(citations, list)
        assert len(citations) >= 1
        assert citations[0]["schema_version"] == "1.0"
        assert citations[0]["status"] == "verified"

    @pytest.mark.unit
    def test_mock_metrics_is_valid(self, mock_e2e_project):
        """Metrics file should contain valid metric records."""
        metrics = json.loads((mock_e2e_project / "outputs" / "metrics.json").read_text(encoding="utf-8"))
        assert isinstance(metrics, list)
        assert len(metrics) >= 1
        assert metrics[0]["schema_version"] == "1.0"
        assert metrics[0]["metric_key"] == "test_r2"
