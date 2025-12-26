"""Workflow issue tracking.

Persists non-fatal issues detected during autonomous runs so they can be fixed
later without human intervention during execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass(frozen=True)
class WorkflowIssue:
    source: str
    severity: str
    issue_type: str
    title: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "source": self.source,
            "severity": self.severity,
            "type": self.issue_type,
            "title": self.title,
            "details": self.details,
        }
        stable = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        payload["id"] = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:12]
        return payload


def _get_nested(mapping: Dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _severity_rank(value: str) -> int:
    order = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
        "info": 4,
    }
    return order.get((value or "").lower(), 999)


def write_workflow_issue_tracking(
    project_folder: str,
    workflow_results: Dict[str, Any],
    *,
    filename: str = "workflow_issues.json",
    max_consistency_issues: Optional[int] = None,
) -> Optional[Path]:
    """Write a per-project workflow issues file.

    This function is designed to be non-blocking for the main workflow; callers
    should wrap it in try/except and continue even if it fails.
    """
    project_path = Path(project_folder)
    issues: List[WorkflowIssue] = []

    # Consistency issues (A14)
    consistency = _get_nested(workflow_results, "agents", "consistency_check", "structured_data")
    if isinstance(consistency, dict):
        raw_issues = consistency.get("issues")
        if isinstance(raw_issues, list):
            for raw in raw_issues:
                if not isinstance(raw, dict):
                    continue
                issues.append(
                    WorkflowIssue(
                        source="consistency_check",
                        severity=str(raw.get("severity", "")),
                        issue_type="consistency_issue",
                        title=str(raw.get("description") or raw.get("key") or "Consistency issue"),
                        details=raw,
                    )
                )

    # Readiness automation gaps (A15)
    readiness = _get_nested(workflow_results, "agents", "readiness_assessment", "structured_data")
    if isinstance(readiness, dict):
        blocking_gaps = readiness.get("blocking_gaps")
        if isinstance(blocking_gaps, list):
            for gap in blocking_gaps:
                if not isinstance(gap, dict):
                    continue
                priority = str(gap.get("priority", "medium")).lower()
                severity = "high" if priority == "high" else "medium" if priority == "medium" else "low"
                issues.append(
                    WorkflowIssue(
                        source="readiness_assessment",
                        severity=severity,
                        issue_type="automation_gap",
                        title=str(gap.get("description") or "Automation gap"),
                        details=gap,
                    )
                )

    # Missing key files (simple deterministic checks)
    if not (project_path / "PROJECT_PLAN.md").exists():
        issues.append(
            WorkflowIssue(
                source="filesystem",
                severity="medium",
                issue_type="missing_file",
                title="Missing PROJECT_PLAN.md",
                details={
                    "path": "PROJECT_PLAN.md",
                    "description": "Readiness time tracking could not parse estimates without PROJECT_PLAN.md.",
                },
            )
        )

    overview_path = workflow_results.get("overview_path")
    if overview_path and not Path(str(overview_path)).exists():
        issues.append(
            WorkflowIssue(
                source="filesystem",
                severity="high",
                issue_type="missing_output",
                title="Overview output missing on disk",
                details={
                    "path": str(overview_path),
                    "description": "Workflow reported an overview path, but it was not found on disk.",
                },
            )
        )

    # Literature search quality: Edison returned zero citations (Phase 2)
    literature_search = _get_nested(workflow_results, "agents", "literature_search", "structured_data")
    if isinstance(literature_search, dict):
        # Check all possible citation count fields in one pass
        citations = literature_search.get("citations")
        citation_count = literature_search.get("citations_count")
        citation_count_alt = literature_search.get("citation_count")

        has_zero_citations = (
            (isinstance(citations, list) and len(citations) == 0)
            or (isinstance(citation_count, int) and citation_count == 0)
            or (isinstance(citation_count_alt, int) and citation_count_alt == 0)
        )

        if has_zero_citations:
            issues.append(
                WorkflowIssue(
                    source="literature_search",
                    severity="high",
                    issue_type="quality_risk",
                    title="Literature search returned 0 citations",
                    details={
                        "citations_count": 0,
                        "description": "Edison returned no citations; this reduces traceability and may indicate a parsing or prompt issue.",
                    },
                )
            )

    # Sort and optionally cap
    issues_sorted = sorted(issues, key=lambda i: _severity_rank(i.severity))
    if max_consistency_issues is not None:
        kept: List[WorkflowIssue] = []
        consistency_seen = 0
        for issue in issues_sorted:
            if issue.issue_type == "consistency_issue":
                if consistency_seen >= max_consistency_issues:
                    continue
                consistency_seen += 1
            kept.append(issue)
        issues_sorted = kept

    payload = {
        "generated_at": datetime.now().isoformat(),
        "project_id": workflow_results.get("project_id"),
        "project_folder": workflow_results.get("project_folder") or project_folder,
        "workflow": {
            "success": workflow_results.get("success"),
            "error_count": len(workflow_results.get("errors") or []),
            "errors": workflow_results.get("errors") or [],
            "total_tokens": workflow_results.get("total_tokens"),
            "total_time": workflow_results.get("total_time"),
        },
        "summary": {
            "total_issues": len(issues_sorted),
            "critical": sum(1 for i in issues_sorted if i.severity.lower() == "critical"),
            "high": sum(1 for i in issues_sorted if i.severity.lower() == "high"),
            "medium": sum(1 for i in issues_sorted if i.severity.lower() == "medium"),
            "low": sum(1 for i in issues_sorted if i.severity.lower() == "low"),
        },
        "issues": [i.to_dict() for i in issues_sorted],
    }

    output_path = project_path / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved workflow issue tracking to {output_path}")
    return output_path
