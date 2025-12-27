"""Inter-agent request/response protocol.

This module defines a small, structured message protocol used by the orchestrator
for inter-agent calls.

The canonical validation source is the JSON Schema in:
- src/schemas/agent_message.schema.json

This protocol is deliberately minimal. It is intended to support:
- request: caller asks another agent (by registry id) to do a subtask
- response: callee returns a successful result
- error: callee or orchestrator returns a structured failure (timeout, permissions)

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from src.utils.schema_validation import validate_against_schema


_SCHEMA_FILENAME = "agent_message.schema.json"


def _iso_utc_now() -> str:
    """Return the current timestamp as ISO8601 with UTC timezone."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentMessageError:
    """Structured error payload for an inter-agent message."""

    code: str
    message: str


def build_request_message(
    *,
    call_id: str,
    caller_agent_id: str,
    target_agent_id: str,
    reason: str,
    context: Mapping[str, Any],
    priority: str = "normal",
    timeout_seconds: int = 600,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a validated request message dict.

    Args:
        call_id: Unique identifier for the call.
        caller_agent_id: Registry id for the calling agent.
        target_agent_id: Registry id for the callee agent.
        reason: Human readable reason for the request.
        context: JSON-like payload passed to the target agent.
        priority: One of "normal", "high", "critical".
        timeout_seconds: Timeout for the call attempt.
        timestamp: Optional ISO timestamp; when omitted, uses current UTC.

    Returns:
        A dict that conforms to `agent_message.schema.json`.
    """

    msg: Dict[str, Any] = {
        "type": "request",
        "call_id": call_id,
        "timestamp": timestamp or _iso_utc_now(),
        "caller_agent_id": caller_agent_id,
        "target_agent_id": target_agent_id,
        "reason": reason,
        "context": dict(context),
        "priority": priority,
        "timeout_seconds": timeout_seconds,
    }
    validate_agent_message(msg)
    return msg


def build_response_message(
    *,
    call_id: str,
    result: Optional[Mapping[str, Any]],
    execution_time: float,
    attempt: int,
    max_attempts: int,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a validated successful response message dict."""

    msg: Dict[str, Any] = {
        "type": "response",
        "call_id": call_id,
        "timestamp": timestamp or _iso_utc_now(),
        "success": True,
        "result": dict(result) if isinstance(result, Mapping) else None,
        "error": None,
        "error_code": None,
        "execution_time": float(execution_time),
        "attempt": int(attempt),
        "max_attempts": int(max_attempts),
    }
    validate_agent_message(msg)
    return msg


def build_error_message(
    *,
    call_id: str,
    error: str,
    error_code: str,
    execution_time: float,
    attempt: int,
    max_attempts: int,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a validated error message dict."""

    msg: Dict[str, Any] = {
        "type": "error",
        "call_id": call_id,
        "timestamp": timestamp or _iso_utc_now(),
        "success": False,
        "error": error,
        "error_code": error_code,
        "execution_time": float(execution_time),
        "attempt": int(attempt),
        "max_attempts": int(max_attempts),
    }
    validate_agent_message(msg)
    return msg


def validate_agent_message(message: Any) -> None:
    """Validate a message dict against the agent message schema.

    Args:
        message: The message to validate.

    Raises:
        ValueError: When schema validation fails.
    """

    validate_against_schema(message, _SCHEMA_FILENAME)
