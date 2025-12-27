"""Tests for evaluation metrics module.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.metrics import (
    EvaluationConfig,
    EvaluationResult,
    MetricResult,
    evaluate_pipeline_output,
    write_evaluation_results,
)


@pytest.fixture
def mock_project(tmp_path: Path) -> Path:
    """Create a mock project with typical output structure."""
    # Create outputs directory structure
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "sections").mkdir()

    # Create section files
    for section in ["introduction", "related_work", "methods", "results", "discussion"]:
        tex = outputs / "sections" / f"{section}.tex"
        tex.write_text(f"% {section}\n" + "Content " * 50, encoding="utf-8")

    # Create metrics and claims
    (outputs / "metrics.json").write_text("[]", encoding="utf-8")
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims" / "claims.json").write_text("[]", encoding="utf-8")

    # Create sources with evidence
    sources = tmp_path / "sources"
    sources.mkdir()
    source1 = sources / "source_001"
    source1.mkdir()
    (source1 / "evidence.json").write_text(
        json.dumps([{"id": "ev_001", "text": "Evidence 1"}]), encoding="utf-8"
    )

    # Create bibliography with citations
    bib = tmp_path / "bibliography"
    bib.mkdir()
    (bib / "citations.json").write_text(
        json.dumps([{"id": "cite_001", "status": "verified"}]), encoding="utf-8"
    )

    return tmp_path


@pytest.mark.unit
class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_default_config(self):
        cfg = EvaluationConfig()
        assert cfg.enabled is True
        assert cfg.min_quality_score == 0.0
        assert cfg.metrics == ["completeness", "evidence_coverage", "citation_coverage"]

    def test_custom_config(self):
        cfg = EvaluationConfig(enabled=False, min_quality_score=0.5, metrics=["completeness"])
        assert cfg.enabled is False
        assert cfg.min_quality_score == 0.5
        assert cfg.metrics == ["completeness"]

    def test_from_context_empty(self):
        cfg = EvaluationConfig.from_context({})
        assert cfg.enabled is True
        assert cfg.min_quality_score == 0.0

    def test_from_context_with_evaluation(self):
        context = {
            "evaluation": {
                "enabled": False,
                "min_quality_score": 0.8,
                "metrics": ["completeness", "claims_coverage"],
            }
        }
        cfg = EvaluationConfig.from_context(context)
        assert cfg.enabled is False
        assert cfg.min_quality_score == 0.8
        assert cfg.metrics == ["completeness", "claims_coverage"]

    def test_from_context_invalid_evaluation(self):
        cfg = EvaluationConfig.from_context({"evaluation": "not a dict"})
        assert cfg.enabled is True  # Falls back to default


@pytest.mark.unit
class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_normalized_score(self):
        result = MetricResult(name="test", score=3.0, max_score=5.0, details={})
        assert result.normalized_score == 0.6

    def test_normalized_score_zero_max(self):
        result = MetricResult(name="test", score=3.0, max_score=0.0, details={})
        assert result.normalized_score == 0.0


@pytest.mark.unit
class TestEvaluatePipelineOutput:
    """Tests for evaluate_pipeline_output function."""

    def test_disabled_evaluation(self, tmp_path: Path):
        cfg = EvaluationConfig(enabled=False)
        result = evaluate_pipeline_output(tmp_path, config=cfg)

        assert result.success is True
        assert result.overall_score == 1.0
        assert result.metrics == []
        assert result.details["skipped"] is True

    def test_completeness_metric_all_sections(self, mock_project: Path):
        cfg = EvaluationConfig(metrics=["completeness"])
        result = evaluate_pipeline_output(mock_project, config=cfg)

        assert len(result.metrics) == 1
        metric = result.metrics[0]
        assert metric.name == "completeness"
        assert metric.score == 5.0
        assert metric.max_score == 5.0
        assert metric.normalized_score == 1.0

    def test_completeness_metric_missing_sections(self, tmp_path: Path):
        outputs = tmp_path / "outputs"
        outputs.mkdir(parents=True)
        (outputs / "sections").mkdir()

        # Only create introduction
        (outputs / "sections" / "introduction.tex").write_text("Content " * 50)

        cfg = EvaluationConfig(metrics=["completeness"])
        result = evaluate_pipeline_output(tmp_path, config=cfg)

        metric = result.metrics[0]
        assert metric.score == 1.0
        assert metric.max_score == 5.0
        assert set(metric.details["missing_sections"]) == {
            "related_work",
            "methods",
            "results",
            "discussion",
        }

    def test_evidence_coverage_metric(self, mock_project: Path):
        cfg = EvaluationConfig(metrics=["evidence_coverage"])
        result = evaluate_pipeline_output(mock_project, config=cfg)

        metric = result.metrics[0]
        assert metric.name == "evidence_coverage"
        assert metric.score == 1.0  # 1 source with evidence / 1 total source

    def test_evidence_coverage_no_sources(self, tmp_path: Path):
        cfg = EvaluationConfig(metrics=["evidence_coverage"])
        result = evaluate_pipeline_output(tmp_path, config=cfg)

        metric = result.metrics[0]
        assert metric.score == 0.0
        assert metric.details["sources_dir_exists"] is False

    def test_citation_coverage_metric(self, mock_project: Path):
        cfg = EvaluationConfig(metrics=["citation_coverage"])
        result = evaluate_pipeline_output(mock_project, config=cfg)

        metric = result.metrics[0]
        assert metric.name == "citation_coverage"
        assert metric.score == 1.0  # 1 verified / 1 total

    def test_citation_coverage_unverified(self, tmp_path: Path):
        bib = tmp_path / "bibliography"
        bib.mkdir()
        (bib / "citations.json").write_text(
            json.dumps([
                {"id": "cite_001", "status": "verified"},
                {"id": "cite_002", "status": "pending"},
            ]),
            encoding="utf-8",
        )

        cfg = EvaluationConfig(metrics=["citation_coverage"])
        result = evaluate_pipeline_output(tmp_path, config=cfg)

        metric = result.metrics[0]
        assert metric.score == 0.5  # 1 verified / 2 total

    def test_claims_coverage_metric(self, mock_project: Path):
        cfg = EvaluationConfig(metrics=["claims_coverage"])
        result = evaluate_pipeline_output(mock_project, config=cfg)

        metric = result.metrics[0]
        assert metric.name == "claims_coverage"
        # With 0 metrics and 0 claims, score should be 1.0
        assert metric.score == 1.0

    def test_overall_score_calculation(self, mock_project: Path):
        cfg = EvaluationConfig(metrics=["completeness", "evidence_coverage", "citation_coverage"])
        result = evaluate_pipeline_output(mock_project, config=cfg)

        # All metrics should have score 1.0, so overall should be 1.0
        assert result.overall_score == 1.0
        assert result.success is True

    def test_quality_threshold_failure(self, tmp_path: Path):
        cfg = EvaluationConfig(min_quality_score=0.9, metrics=["completeness"])
        result = evaluate_pipeline_output(tmp_path, config=cfg)

        # No sections exist, so completeness = 0
        assert result.overall_score == 0.0
        assert result.success is False

    def test_quality_threshold_pass(self, mock_project: Path):
        cfg = EvaluationConfig(min_quality_score=0.5, metrics=["completeness"])
        result = evaluate_pipeline_output(mock_project, config=cfg)

        assert result.overall_score == 1.0
        assert result.success is True


@pytest.mark.unit
class TestWriteEvaluationResults:
    """Tests for write_evaluation_results function."""

    def test_writes_json_file(self, tmp_path: Path):
        result = EvaluationResult(
            success=True,
            overall_score=0.85,
            metrics=[
                MetricResult(name="test", score=8.5, max_score=10.0, details={"key": "value"})
            ],
            created_at="2025-01-01T00:00:00Z",
            details={"enabled": True},
        )

        out_path = write_evaluation_results(tmp_path, result)

        assert out_path.exists()
        assert out_path == tmp_path / "outputs" / "evaluation_results.json"

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["success"] is True
        assert data["overall_score"] == 0.85
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["name"] == "test"

    def test_creates_outputs_directory(self, tmp_path: Path):
        result = EvaluationResult(
            success=True,
            overall_score=1.0,
            metrics=[],
            created_at="2025-01-01T00:00:00Z",
            details={},
        )

        out_path = write_evaluation_results(tmp_path, result)

        assert out_path.parent.exists()
        assert out_path.parent.name == "outputs"
