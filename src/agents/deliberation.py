"""Multi-agent deliberation and consensus helpers.

Implements a minimal deliberation loop:
- Collect 2+ agent perspectives
- Detect conflicts
- Produce a consolidated output with rationale
- Escalate via a degraded result when conflicts are unresolved

Artifacts are designed to be reproducible and deterministic when written with
sorted keys.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _normalize_for_compare(text: str) -> str:
    return " ".join((text or "").split()).strip()


def detect_conflict(outputs: List[str]) -> bool:
    """Return True if agent outputs materially differ."""

    normalized = [_normalize_for_compare(x) for x in outputs if _normalize_for_compare(x)]
    return len(set(normalized)) > 1


def _format_perspective(*, agent_id: str, content: str) -> str:
    header = f"Perspective ({agent_id})"
    return f"{header}\n{'=' * len(header)}\n{content.strip()}\n"


@dataclass(frozen=True)
class DeliberationPerspective:
    agent_id: str
    success: bool
    content: str
    error: Optional[str]
    result: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "success": bool(self.success),
            "content": self.content,
            "error": self.error,
            "result": self.result,
        }


def build_consensus(
    *,
    task_text: str,
    agent_ids: List[str],
    perspectives: List[DeliberationPerspective],
) -> Dict[str, Any]:
    """Build consolidated output plus rationale from perspectives.

    This is intentionally minimal. If there is a conflict, we produce a merged
    output and mark the result as degraded.
    """

    successful = [p for p in perspectives if p.success and _normalize_for_compare(p.content)]
    outputs = [p.content for p in successful]

    conflict = detect_conflict(outputs)

    degraded = False
    degradation_reason: Optional[str] = None

    if len(successful) < 2:
        degraded = True
        degradation_reason = "insufficient_successful_agents"

    if conflict:
        degraded = True
        degradation_reason = degradation_reason or "conflicting_outputs"

    if not successful:
        consolidated_output = ""
        rationale = "No successful perspectives were available to consolidate."
    elif not degraded:
        consolidated_output = successful[0].content
        rationale = "All participating agents produced equivalent outputs."
    else:
        consolidated_output = "\n\n".join(
            _format_perspective(agent_id=p.agent_id, content=p.content) for p in successful
        ).strip() + "\n"
        reasons: List[str] = []
        if conflict:
            reasons.append("agent outputs conflict")
        if len(successful) < 2:
            reasons.append("fewer than two successful perspectives")
        rationale = "Degraded consensus: " + "; ".join(reasons) + "."

    return {
        "schema_version": "1.0",
        "task_text": task_text,
        "agent_ids": list(agent_ids),
        "perspectives": [p.to_dict() for p in perspectives],
        "conflict_detected": bool(conflict),
        "degraded": bool(degraded),
        "degradation_reason": degradation_reason,
        "consolidated_output": consolidated_output,
        "rationale": rationale,
    }
