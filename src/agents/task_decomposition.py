"""Task decomposition and aggregation helpers.

This module supports splitting a high-level task into multiple subtasks routed to
agents, then aggregating results.

The decomposition output is validated against:
- src/schemas/task_decomposition.schema.json

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from src.llm.claude_client import ClaudeClient, TaskType
from src.utils.schema_validation import validate_against_schema


_DECOMPOSITION_SCHEMA_FILENAME = "task_decomposition.schema.json"


def _short_hash(value: str) -> str:
    """Return a short, stable hash for deterministic IDs."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def deterministic_subtask_id(*, index: int, agent_id: str, title: str) -> str:
    """Create a deterministic subtask id.

    The id depends on index, agent_id, and title to stay stable across runs.
    """

    base = f"{index}:{agent_id}:{title.strip()}"
    return f"ST{index:02d}_{_short_hash(base)}"


def validate_task_decomposition(payload: Any) -> None:
    """Validate task decomposition payload against JSON schema."""
    validate_against_schema(payload, _DECOMPOSITION_SCHEMA_FILENAME)


def normalize_task_decomposition(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a task decomposition payload.

    This enforces deterministic subtask IDs regardless of what the LLM returned.

    Args:
        payload: Candidate decomposition payload.

    Returns:
        A schema-valid decomposition dict with deterministic subtask IDs.

    Raises:
        ValueError: When payload does not validate.
    """

    out: Dict[str, Any] = {
        "task": {"text": str(payload.get("task", {}).get("text") or "").strip()},
        "subtasks": [],
    }

    subtasks = payload.get("subtasks")
    if not isinstance(subtasks, list):
        subtasks = []

    for i, item in enumerate(subtasks, start=1):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "").strip()
        agent_id = str(item.get("agent_id") or "").strip()
        description = str(item.get("description") or "").strip()
        inputs = item.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}

        out["subtasks"].append(
            {
                "id": deterministic_subtask_id(index=i, agent_id=agent_id, title=title),
                "title": title,
                "description": description,
                "agent_id": agent_id,
                "inputs": inputs,
                "depends_on": [str(x) for x in item.get("depends_on", []) if isinstance(x, str) and x.strip()],
                "priority": str(item.get("priority") or "normal"),
            }
        )

    validate_task_decomposition(out)
    return out


def build_decomposition_system_prompt(*, available_agent_ids: List[str]) -> str:
    """Build the system prompt for prompt-driven decomposition."""

    agent_list = ", ".join(available_agent_ids)
    return (
        "You decompose a single high-level task into 2 or more subtasks. "
        "Return ONLY valid JSON that conforms to the expected schema. "
        "\n\n"
        "Schema (informal):\n"
        "{\n"
        "  \"task\": {\"text\": string},\n"
        "  \"subtasks\": [\n"
        "    {\n"
        "      \"id\": string,\n"
        "      \"title\": string,\n"
        "      \"description\": string,\n"
        "      \"agent_id\": string,\n"
        "      \"inputs\": object,\n"
        "      \"depends_on\": [string],\n"
        "      \"priority\": \"normal\"|\"high\"|\"critical\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Valid agent_id values: {agent_list}\n"
    )


async def decompose_task_via_llm(
    *,
    client: ClaudeClient,
    task_text: str,
    available_agent_ids: List[str],
) -> Dict[str, Any]:
    """Decompose a high-level task into subtasks using the LLM.

    Args:
        client: Claude client.
        task_text: High-level task description.
        available_agent_ids: Agent registry ids the LLM may choose from.

    Returns:
        A normalized, schema-valid decomposition dict.

    Raises:
        ValueError: When LLM output is not valid JSON or fails schema validation.
    """

    system = build_decomposition_system_prompt(available_agent_ids=available_agent_ids)
    messages = [
        {
            "role": "user",
            "content": (
                "Decompose this task into subtasks and pick an agent_id for each.\n\n"
                f"Task: {task_text.strip()}\n"
            ),
        }
    ]

    raw = await client.chat_async(
        messages=messages,
        system=system,
        task=TaskType.CODING,
        temperature=0.0,
        cache_system=True,
    )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Decomposition output is not valid JSON: {e}")

    if not isinstance(payload, dict):
        raise ValueError("Decomposition output must be a JSON object")

    return normalize_task_decomposition(payload)


@dataclass(frozen=True)
class SubtaskRunRecord:
    """Represents one subtask execution record."""

    subtask_id: str
    agent_id: str
    success: bool
    error: Optional[str]
    result: Optional[Dict[str, Any]]


def aggregate_subtask_runs(
    *,
    decomposition: Mapping[str, Any],
    runs: List[SubtaskRunRecord],
) -> Dict[str, Any]:
    """Aggregate subtask runs into a single JSON artifact payload."""

    out_runs: List[Dict[str, Any]] = []
    for r in runs:
        out_runs.append(
            {
                "subtask_id": r.subtask_id,
                "agent_id": r.agent_id,
                "success": bool(r.success),
                "error": r.error,
                "result": r.result,
            }
        )

    overall_success = all(r.success for r in runs) if runs else False

    return {
        "schema_version": "1.0",
        "task": dict(decomposition.get("task") or {}),
        "subtasks": list(decomposition.get("subtasks") or []),
        "runs": out_runs,
        "success": bool(overall_success),
    }
