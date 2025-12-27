import pytest
import re
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.orchestrator import AgentOrchestrator, OrchestratorConfig, ExecutionMode
from src.agents.task_decomposition import (
    SubtaskRunRecord,
    aggregate_subtask_runs,
    normalize_task_decomposition,
    validate_task_decomposition,
)
from src.llm.claude_client import ModelTier, TaskType
from src.agents.base import AgentResult


@pytest.mark.unit
def test_normalize_task_decomposition_assigns_deterministic_ids():
    payload = {
        "task": {"text": "Do a big thing"},
        "subtasks": [
            {
                "id": "random1",
                "title": "Step one",
                "description": "First step",
                "agent_id": "A01",
                "inputs": {"project_folder": "/tmp"},
            },
            {
                "id": "random2",
                "title": "Step two",
                "description": "Second step",
                "agent_id": "A02",
                "inputs": {"project_folder": "/tmp"},
            },
        ],
    }

    normalized_1 = normalize_task_decomposition(payload)
    normalized_2 = normalize_task_decomposition(payload)

    validate_task_decomposition(normalized_1)
    assert normalized_1 == normalized_2
    assert re.fullmatch(r"ST01_[0-9a-f]{8}", normalized_1["subtasks"][0]["id"])
    assert re.fullmatch(r"ST02_[0-9a-f]{8}", normalized_1["subtasks"][1]["id"])


@pytest.mark.unit
def test_aggregate_subtask_runs_marks_failure_when_any_fails():
    decomposition = normalize_task_decomposition(
        {
            "task": {"text": "Do a big thing"},
            "subtasks": [
                {
                    "id": "x",
                    "title": "Step one",
                    "description": "First step",
                    "agent_id": "A01",
                    "inputs": {},
                },
                {
                    "id": "y",
                    "title": "Step two",
                    "description": "Second step",
                    "agent_id": "A02",
                    "inputs": {},
                },
            ],
        }
    )

    runs = [
        SubtaskRunRecord(subtask_id=decomposition["subtasks"][0]["id"], agent_id="A01", success=True, error=None, result={"ok": True}),
        SubtaskRunRecord(subtask_id=decomposition["subtasks"][1]["id"], agent_id="A02", success=False, error="boom", result=None),
    ]

    agg = aggregate_subtask_runs(decomposition=decomposition, runs=runs)
    assert agg["success"] is False
    assert len(agg["runs"]) == 2


@pytest.mark.unit
@pytest.mark.asyncio
@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
async def test_execute_decomposed_task_isolates_failures(tmp_path):
    (tmp_path / "project.json").write_text(
        "{\"id\": \"p1\", \"title\": \"t\", \"research_question\": \"q\"}\n",
        encoding="utf-8",
    )
    mock_client = MagicMock()

    with patch("src.agents.orchestrator.ClaudeClient", return_value=mock_client):
        with patch("src.agents.orchestrator.CriticalReviewAgent"):
            config = OrchestratorConfig(
                default_mode=ExecutionMode.SINGLE_PASS,
                agent_timeout=10,
                review_timeout=5,
            )
            orch = AgentOrchestrator(str(tmp_path), config=config)
            orch.client = mock_client

            orch.execute_agent = AsyncMock(
                return_value=AgentResult(
                    agent_name="A01",
                    task_type=TaskType.CODING,
                    model_tier=ModelTier.SONNET,
                    success=True,
                    content="ok",
                    structured_data={},
                    timestamp="2025-12-27T00:00:00+00:00",
                )
            )

            decomposition_override = {
                "task": {"text": "Do a big thing"},
                "subtasks": [
                    {
                        "id": "ignored",
                        "title": "Good",
                        "description": "Good",
                        "agent_id": "A01",
                        "inputs": {"project_folder": str(tmp_path)},
                    },
                    {
                        "id": "ignored",
                        "title": "Bad",
                        "description": "Bad",
                        "agent_id": "A999",
                        "inputs": {},
                    },
                ],
            }

            agg = await orch.execute_decomposed_task(
                task_text="Do a big thing",
                context={"project_folder": str(tmp_path)},
                decomposition_override=decomposition_override,
            )

    assert agg["success"] is False
    assert len(agg["runs"]) == 2
    assert agg["runs"][0]["success"] is True
    assert agg["runs"][1]["success"] is False
    assert "Unknown agent_id" in (agg["runs"][1]["error"] or "")
