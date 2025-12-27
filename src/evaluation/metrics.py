"""Post-pipeline evaluation metrics.

Provides lightweight, deterministic evaluation of pipeline output quality.
These metrics are filesystem-first and do not require LLM calls.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for post-pipeline evaluation."""

    enabled: bool = True
    min_quality_score: float = 0.0  # 0.0 means no blocking; set higher to block
    metrics: List[str] = None  # None means all available metrics

    def __post_init__(self):
        if self.metrics is None:
            object.__setattr__(self, "metrics", ["completeness", "evidence_coverage", "citation_coverage"])

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "EvaluationConfig":
        raw = context.get("evaluation")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", True))
        min_score = float(raw.get("min_quality_score", 0.0))
        metrics = raw.get("metrics")
        if not isinstance(metrics, list):
            metrics = None

        return cls(enabled=enabled, min_quality_score=min_score, metrics=metrics)


@dataclass
class MetricResult:
    """Result of a single evaluation metric."""

    name: str
    score: float
    max_score: float
    details: Dict[str, Any]

    @property
    def normalized_score(self) -> float:
        if self.max_score <= 0:
            return 0.0
        return self.score / self.max_score


@dataclass
class EvaluationResult:
    """Result of post-pipeline evaluation."""

    success: bool
    overall_score: float
    metrics: List[MetricResult]
    created_at: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "overall_score": self.overall_score,
            "metrics": [asdict(m) for m in self.metrics],
            "created_at": self.created_at,
            "details": self.details,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json_list(path: Path) -> List[Any]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return []


def _count_tex_files(outputs_dir: Path) -> int:
    """Count .tex section files in outputs/sections/."""
    sections_dir = outputs_dir / "sections"
    if not sections_dir.exists():
        return 0
    return len(list(sections_dir.glob("*.tex")))


def _evaluate_completeness(project_folder: Path) -> MetricResult:
    """Evaluate completeness of generated sections.

    Checks for presence of key output files and sections.
    """
    outputs = project_folder / "outputs"
    sections_dir = outputs / "sections"

    expected_sections = ["introduction", "related_work", "methods", "results", "discussion"]
    found_sections = []
    missing_sections = []

    for section in expected_sections:
        tex_file = sections_dir / f"{section}.tex"
        if tex_file.exists() and tex_file.stat().st_size > 100:
            found_sections.append(section)
        else:
            missing_sections.append(section)

    score = len(found_sections)
    max_score = len(expected_sections)

    return MetricResult(
        name="completeness",
        score=float(score),
        max_score=float(max_score),
        details={
            "found_sections": found_sections,
            "missing_sections": missing_sections,
            "expected_count": max_score,
            "found_count": score,
        },
    )


def _evaluate_evidence_coverage(project_folder: Path) -> MetricResult:
    """Evaluate evidence coverage across sources.

    Checks that evidence items exist for sources in the project.
    """
    sources_dir = project_folder / "sources"
    if not sources_dir.exists():
        return MetricResult(
            name="evidence_coverage",
            score=0.0,
            max_score=1.0,
            details={"sources_dir_exists": False, "sources_with_evidence": 0, "total_sources": 0},
        )

    source_dirs = [d for d in sources_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not source_dirs:
        return MetricResult(
            name="evidence_coverage",
            score=0.0,
            max_score=1.0,
            details={"sources_dir_exists": True, "sources_with_evidence": 0, "total_sources": 0},
        )

    sources_with_evidence = 0
    evidence_items_total = 0

    for source_dir in source_dirs:
        evidence_path = source_dir / "evidence.json"
        if evidence_path.exists():
            items = _load_json_list(evidence_path)
            if items:
                sources_with_evidence += 1
                evidence_items_total += len(items)

    total_sources = len(source_dirs)
    score = sources_with_evidence / total_sources if total_sources > 0 else 0.0

    return MetricResult(
        name="evidence_coverage",
        score=score,
        max_score=1.0,
        details={
            "sources_dir_exists": True,
            "sources_with_evidence": sources_with_evidence,
            "total_sources": total_sources,
            "evidence_items_total": evidence_items_total,
        },
    )


def _evaluate_citation_coverage(project_folder: Path) -> MetricResult:
    """Evaluate citation coverage.

    Checks that citations exist and are verified.
    """
    citations_path = project_folder / "bibliography" / "citations.json"
    citations = _load_json_list(citations_path)

    if not citations:
        return MetricResult(
            name="citation_coverage",
            score=0.0,
            max_score=1.0,
            details={
                "citations_file_exists": citations_path.exists(),
                "total_citations": 0,
                "verified_citations": 0,
            },
        )

    verified = sum(1 for c in citations if isinstance(c, dict) and c.get("status") == "verified")
    total = len(citations)
    score = verified / total if total > 0 else 0.0

    return MetricResult(
        name="citation_coverage",
        score=score,
        max_score=1.0,
        details={
            "citations_file_exists": True,
            "total_citations": total,
            "verified_citations": verified,
        },
    )


def _evaluate_claims_coverage(project_folder: Path) -> MetricResult:
    """Evaluate claims coverage.

    Checks that claims are generated from metrics.
    """
    metrics_path = project_folder / "outputs" / "metrics.json"
    claims_path = project_folder / "claims" / "claims.json"

    metrics = _load_json_list(metrics_path)
    claims = _load_json_list(claims_path)

    metrics_count = len(metrics)
    claims_count = len(claims)

    # Score based on whether claims were generated for metrics
    if metrics_count == 0:
        score = 1.0 if claims_count == 0 else 0.5  # No metrics, so no claims expected
    else:
        score = min(1.0, claims_count / metrics_count)

    return MetricResult(
        name="claims_coverage",
        score=score,
        max_score=1.0,
        details={
            "metrics_count": metrics_count,
            "claims_count": claims_count,
            "metrics_file_exists": metrics_path.exists(),
            "claims_file_exists": claims_path.exists(),
        },
    )


METRIC_EVALUATORS = {
    "completeness": _evaluate_completeness,
    "evidence_coverage": _evaluate_evidence_coverage,
    "citation_coverage": _evaluate_citation_coverage,
    "claims_coverage": _evaluate_claims_coverage,
}


def evaluate_pipeline_output(
    project_folder: str | Path,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationResult:
    """Evaluate the quality of pipeline output.

    This function is deterministic and filesystem-first.
    It does not make any LLM calls.

    Args:
        project_folder: Path to the project folder.
        config: Evaluation configuration.

    Returns:
        EvaluationResult with scores and details.
    """
    pf = Path(project_folder).expanduser().resolve()
    cfg = config or EvaluationConfig()

    if not cfg.enabled:
        return EvaluationResult(
            success=True,
            overall_score=1.0,
            metrics=[],
            created_at=_utc_now_iso(),
            details={"enabled": False, "skipped": True},
        )

    metrics_to_run = cfg.metrics or list(METRIC_EVALUATORS.keys())
    results: List[MetricResult] = []

    for metric_name in metrics_to_run:
        evaluator = METRIC_EVALUATORS.get(metric_name)
        if evaluator is None:
            logger.warning(f"Unknown evaluation metric: {metric_name}")
            continue

        try:
            result = evaluator(pf)
            results.append(result)
        except Exception as e:
            logger.warning(f"Evaluation metric {metric_name} failed: {type(e).__name__}: {e}")
            results.append(
                MetricResult(
                    name=metric_name,
                    score=0.0,
                    max_score=1.0,
                    details={"error": f"{type(e).__name__}: {e}"},
                )
            )

    # Calculate overall score as average of normalized scores
    if results:
        overall_score = sum(r.normalized_score for r in results) / len(results)
    else:
        overall_score = 0.0

    success = overall_score >= cfg.min_quality_score

    return EvaluationResult(
        success=success,
        overall_score=overall_score,
        metrics=results,
        created_at=_utc_now_iso(),
        details={
            "enabled": True,
            "min_quality_score": cfg.min_quality_score,
            "metrics_evaluated": len(results),
        },
    )


def write_evaluation_results(project_folder: Path, result: EvaluationResult) -> Path:
    """Write evaluation results to project folder."""
    out_path = project_folder / "outputs" / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
