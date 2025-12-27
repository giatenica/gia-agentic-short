import asyncio

import pytest
from unittest.mock import MagicMock, patch

from src.agents.inter_agent_protocol import (
    build_error_message,
    build_request_message,
    build_response_message,
    validate_agent_message,
)
from src.agents.orchestrator import AgentOrchestrator, OrchestratorConfig, ExecutionMode


def test_agent_message_schema_validation_roundtrip():
    request = build_request_message(
        call_id="call123",
        caller_agent_id="A03",
        target_agent_id="A01",
        reason="Need data analysis",
        context={"project_folder": "/tmp"},
        priority="normal",
        timeout_seconds=60,
        timestamp="2025-12-27T00:00:00+00:00",
    )
    validate_agent_message(request)

    response = build_response_message(
        call_id="call123",
        result={"ok": True},
        execution_time=0.01,
        attempt=1,
        max_attempts=2,
        timestamp="2025-12-27T00:00:01+00:00",
    )
    validate_agent_message(response)

    error = build_error_message(
        call_id="call123",
        error="Call timed out",
        error_code="timeout",
        execution_time=0.02,
        attempt=2,
        max_attempts=2,
        timestamp="2025-12-27T00:00:02+00:00",
    )
    validate_agent_message(error)


@pytest.mark.asyncio
async def test_exchange_agent_message_timeout_returns_structured_error(tmp_path):
    mock_client = MagicMock()

    with patch("src.agents.orchestrator.ClaudeClient", return_value=mock_client):
        with patch("src.agents.orchestrator.CriticalReviewAgent"):
            config = OrchestratorConfig(
                default_mode=ExecutionMode.SINGLE_PASS,
                agent_timeout=10,
                review_timeout=5,
                inter_agent_call_max_attempts=2,
                inter_agent_call_retry_backoff_seconds=0.0,
            )
            orch = AgentOrchestrator(str(tmp_path), config=config)
            orch.client = mock_client

            request = build_request_message(
                call_id="timeout123",
                caller_agent_id="A03",
                target_agent_id="A01",
                reason="Test timeout",
                context={},
                timeout_seconds=60,
            )

            with patch(
                "src.agents.orchestrator.asyncio.wait_for",
                side_effect=asyncio.TimeoutError,
            ):
                msg = await orch.exchange_agent_message(request)

    assert msg["type"] == "error"
    assert msg["error_code"] == "timeout"
    assert msg["attempt"] == 2
    assert msg["max_attempts"] == 2
