"""Pipeline workflow context.

This module defines a small, serializable state object used by the unified
pipeline runner.

It intentionally stores only stable primitives to support checkpointing and
debugging. Phase implementations remain the source of truth for detailed
artifacts written to the project folder.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkflowContext:
    """Serializable state passed through unified pipeline stages."""

    project_folder: Path
    run_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=_utc_now_iso)

    success: bool = True
    errors: list[str] = field(default_factory=list)

    checkpoints: Dict[str, str] = field(default_factory=dict)
    phase_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def mark_checkpoint(self, name: str) -> None:
        if not name.strip():
            return
        self.checkpoints[name] = _utc_now_iso()

    def record_phase_result(self, phase: str, result: Any) -> None:
        payload: Dict[str, Any]

        if hasattr(result, "to_dict") and callable(getattr(result, "to_dict")):
            payload = result.to_dict()  # type: ignore[assignment]
        elif isinstance(result, dict):
            payload = dict(result)
        else:
            payload = {
                "success": bool(getattr(result, "success", False)),
                "repr": repr(result),
            }

        self.phase_results[phase] = payload

        phase_success = payload.get("success")
        if phase_success is False:
            self.success = False
            phase_errors = payload.get("errors")
            if isinstance(phase_errors, list):
                self.errors.extend([str(e) for e in phase_errors])

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "created_at": self.created_at,
            "project_folder": str(self.project_folder),
            "success": self.success,
            "errors": list(self.errors),
            "checkpoints": dict(self.checkpoints),
            "phase_results": dict(self.phase_results),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "WorkflowContext":
        project_folder = Path(str(payload.get("project_folder", ""))).expanduser().resolve()
        ctx = cls(project_folder=project_folder)

        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id:
            ctx.run_id = run_id

        created_at = payload.get("created_at")
        if isinstance(created_at, str) and created_at:
            ctx.created_at = created_at

        ctx.success = bool(payload.get("success", True))

        errors = payload.get("errors")
        if isinstance(errors, list):
            ctx.errors = [str(e) for e in errors]

        checkpoints = payload.get("checkpoints")
        if isinstance(checkpoints, dict):
            ctx.checkpoints = {str(k): str(v) for k, v in checkpoints.items()}

        phase_results = payload.get("phase_results")
        if isinstance(phase_results, dict):
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in phase_results.items():
                if isinstance(v, dict):
                    out[str(k)] = dict(v)
            ctx.phase_results = out

        return ctx

    def write_json(self, path: Path) -> None:
        import json

        path.write_text(
            json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(cls, path: Path) -> Optional["WorkflowContext"]:
        import json

        if not path.exists() or not path.is_file():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        return cls.from_payload(payload)
