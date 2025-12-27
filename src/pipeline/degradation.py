"""Graceful degradation protocol.

This module standardizes how pipeline stages report degraded operation when
prerequisites or external dependencies are missing.

It also provides summary generation used by the unified pipeline runner.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.schema_validation import validate_degradation_event, validate_degradation_summary


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class DegradationEvent:
    stage: str
    reason_code: str
    message: str
    created_at: str
    recommended_action: Optional[str] = None
    severity: str = "warning"
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": "1.0",
            "created_at": self.created_at,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "message": self.message,
            "recommended_action": self.recommended_action,
            "severity": self.severity,
            "details": self.details,
        }
        validate_degradation_event(payload)
        return payload


def make_degradation_event(
    *,
    stage: str,
    reason_code: str,
    message: str,
    recommended_action: Optional[str] = None,
    severity: str = "warning",
    details: Optional[Dict[str, Any]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    evt = DegradationEvent(
        stage=str(stage).strip(),
        reason_code=str(reason_code).strip(),
        message=str(message).strip(),
        created_at=created_at or _utc_now_iso_z(),
        recommended_action=recommended_action,
        severity=severity,
        details=details,
    )
    return evt.to_dict()


def extract_degradations_from_agent_payload(*, agent_payload: Dict[str, Any], stage: str) -> List[Dict[str, Any]]:
    """Best-effort extraction of degradations from AgentResult payloads.

    Looks for existing "degraded"/"degradation_reason" fields, plus the
    literature search fallback_metadata format.
    """

    out: List[Dict[str, Any]] = []

    # Pattern 1: deliberation/consensus
    if agent_payload.get("degraded") is True:
        reason = agent_payload.get("degradation_reason")
        reason_code = str(reason) if isinstance(reason, str) and reason else "degraded"
        out.append(
            make_degradation_event(
                stage=stage,
                reason_code=reason_code,
                message="Stage produced degraded output.",
                recommended_action="Review the stage output and rerun with improved inputs if needed.",
                details={"degradation_reason": reason},
            )
        )

    # Pattern 2: literature search fallback_metadata
    structured = agent_payload.get("structured_data")
    if isinstance(structured, dict):
        fb = structured.get("fallback_metadata")
        if isinstance(fb, dict) and fb.get("degraded") is True:
            used_provider = fb.get("used_provider")
            out.append(
                make_degradation_event(
                    stage=stage,
                    reason_code="literature_search_degraded",
                    message="Literature search fallback chain completed with degraded results.",
                    recommended_action="Configure Edison API access or provide a manual sources list.",
                    details={"used_provider": used_provider, "attempts": fb.get("attempts")},
                )
            )

    return out


def extract_degradations_from_literature_workflow_result(result_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract degradations from LiteratureWorkflowResult.to_dict() payload."""

    out: List[Dict[str, Any]] = []

    # If the workflow already emits standard degradations, trust them.
    emitted = result_payload.get("degradations")
    has_emitted = False
    if isinstance(emitted, list):
        for item in emitted:
            if isinstance(item, dict):
                out.append(item)
                has_emitted = True

    # For backward compatibility, only synthesize degradations from nested agent
    # payloads and evidence pipeline errors when the workflow did not emit any
    # standard degradations itself.
    if has_emitted:
        return out

    agents = result_payload.get("agents")
    if isinstance(agents, dict):
        lit_search = agents.get("literature_search")
        if isinstance(lit_search, dict):
            out.extend(extract_degradations_from_agent_payload(agent_payload=lit_search, stage="literature"))

    # Evidence pipeline result is optional; summarize error state.
    epr = result_payload.get("evidence_pipeline_result")
    if isinstance(epr, dict):
        errors = epr.get("errors")
        if isinstance(errors, list) and errors:
            out.append(
                make_degradation_event(
                    stage="evidence",
                    reason_code="evidence_pipeline_partial_failure",
                    message="Local evidence pipeline encountered errors.",
                    recommended_action="Inspect outputs/evidence_coverage.json and per-source artifacts; rerun with fewer sources or fix inputs.",
                    details={
                        "errors": [str(e) for e in errors][:20],
                        "discovered_count": epr.get("discovered_count"),
                        "processed_count": epr.get("processed_count"),
                    },
                )
            )

    return out


def summarize_degradations(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_stage: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}

    for evt in events:
        if not isinstance(evt, dict):
            continue
        stage = str(evt.get("stage") or "").strip() or "unknown"
        reason = str(evt.get("reason_code") or "").strip() or "unknown"
        by_stage[stage] = by_stage.get(stage, 0) + 1
        by_reason[reason] = by_reason.get(reason, 0) + 1

    return {
        "total": len(events),
        "by_stage": dict(sorted(by_stage.items())),
        "by_reason_code": dict(sorted(by_reason.items())),
    }


def build_degradation_summary(
    *,
    run_id: str,
    project_folder: str,
    degradations: List[Dict[str, Any]],
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "created_at": created_at or _utc_now_iso_z(),
        "run_id": run_id,
        "project_folder": str(project_folder),
        "degradations": list(degradations),
        "counts": summarize_degradations(degradations),
    }
    validate_degradation_summary(payload)
    return payload


def write_degradation_summary(
    *,
    project_folder: Path,
    run_id: str,
    degradations: List[Dict[str, Any]],
    created_at: Optional[str] = None,
) -> Path:
    outputs_dir = project_folder / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summary = build_degradation_summary(
        run_id=run_id,
        project_folder=str(project_folder),
        degradations=degradations,
        created_at=created_at,
    )

    out_path = outputs_dir / "degradation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
