import json
from pathlib import Path

import pytest

from src.utils.workflow_issue_tracking import write_workflow_issue_tracking


@pytest.mark.unit
def test_write_workflow_issue_tracking_writes_expected_issues(tmp_path: Path) -> None:
    project_folder = tmp_path / "project"
    project_folder.mkdir(parents=True)

    workflow_results = {
        "success": True,
        "project_id": "p1",
        "project_folder": str(project_folder),
        "total_tokens": 123,
        "total_time": 4.56,
        "errors": [],
        "overview_path": None,
        "agents": {
            "consistency_check": {
                "structured_data": {
                    "issues": [
                        {
                            "category": "citation",
                            "severity": "high",
                            "key": "Smith2020",
                            "description": "Citation 'Smith2020' referenced but not defined",
                            "affected_documents": ["RESEARCH_OVERVIEW.md"],
                            "canonical_value": "",
                            "canonical_source": "literature/references.bib",
                            "variants": {"RESEARCH_OVERVIEW.md": "[@Smith2020]"},
                            "suggestion": "Add entry or remove citation",
                        }
                    ],
                    "critical_count": 0,
                    "high_count": 1,
                    "is_consistent": False,
                    "score": 0.85,
                }
            },
            "readiness_assessment": {
                "structured_data": {
                    "blocking_gaps": [
                        {
                            "description": "Missing automated PDF retrieval",
                            "priority": "high",
                            "required_capabilities": ["pdf_retrieval"],
                        }
                    ]
                }
            },
        },
    }

    output_path = write_workflow_issue_tracking(str(project_folder), workflow_results)
    assert output_path is not None
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["project_id"] == "p1"
    assert payload["project_folder"] == str(project_folder)
    assert payload["workflow"]["success"] is True

    issues = payload["issues"]
    assert isinstance(issues, list)

    issue_types = {i["type"] for i in issues}
    assert "consistency_issue" in issue_types
    assert "automation_gap" in issue_types
    assert "missing_file" in issue_types

    missing_plan = [i for i in issues if i["type"] == "missing_file" and i["details"].get("path") == "PROJECT_PLAN.md"]
    assert len(missing_plan) == 1
