from __future__ import annotations

import pytest

from src.pipeline.degradation import (
    build_degradation_summary,
    extract_degradations_from_literature_workflow_result,
    make_degradation_event,
)
from src.utils.schema_validation import validate_degradation_event, validate_degradation_summary


@pytest.mark.unit
def test_make_degradation_event_is_schema_valid():
    evt = make_degradation_event(
        stage="literature",
        reason_code="literature_search_degraded",
        message="Fallback chain used.",
        recommended_action="Configure Edison.",
        severity="warning",
        details={"used_provider": "manual"},
        created_at="2025-01-01T00:00:00Z",
    )
    validate_degradation_event(evt)


@pytest.mark.unit
def test_build_degradation_summary_counts_and_schema_valid():
    events = [
        make_degradation_event(
            stage="literature",
            reason_code="a",
            message="m",
            created_at="2025-01-01T00:00:00Z",
        ),
        make_degradation_event(
            stage="evidence",
            reason_code="b",
            message="m",
            created_at="2025-01-01T00:00:00Z",
        ),
        make_degradation_event(
            stage="evidence",
            reason_code="b",
            message="m",
            created_at="2025-01-01T00:00:00Z",
        ),
    ]

    summary = build_degradation_summary(
        run_id="run123",
        project_folder="/tmp/project",
        degradations=events,
        created_at="2025-01-01T00:00:00Z",
    )

    validate_degradation_summary(summary)
    assert summary["counts"]["total"] == 3
    assert summary["counts"]["by_stage"]["evidence"] == 2
    assert summary["counts"]["by_reason_code"]["b"] == 2


@pytest.mark.unit
def test_extract_degradations_from_literature_workflow_result_detects_fallback_and_evidence_errors():
    payload = {
        "success": True,
        "project_id": "p",
        "project_folder": "/tmp/p",
        "errors": [],
        "files_created": {},
        "writing_review": None,
        "evidence_pipeline_result": {
            "discovered_count": 2,
            "processed_count": 1,
            "errors": ["parse failed"],
        },
        "degradations": [],
        "agents": {
            "literature_search": {
                "structured_data": {
                    "fallback_metadata": {
                        "degraded": True,
                        "used_provider": "manual",
                        "attempts": [{"provider": "edison", "ok": False}],
                    }
                }
            }
        },
    }

    events = extract_degradations_from_literature_workflow_result(payload)
    assert any(e.get("reason_code") == "literature_search_degraded" for e in events)
    assert any(e.get("reason_code") == "evidence_pipeline_partial_failure" for e in events)
