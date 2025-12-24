"""Results section writer.

Sprint 4 (optional) Issue #55 MVP:
- Deterministic, filesystem-first section generation.
- Reads claims/claims.json and outputs/metrics.json.
- Only emits numeric values that come from outputs/metrics.json metric keys.
- Missing metric keys either block or downgrade, depending on config.

This implementation is deliberately conservative:
- It does not call the LLM.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.agents.base import AgentResult, BaseAgent
from src.agents.section_writer import resolve_section_output_path
from src.llm.claude_client import TaskType
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.schema_validation import is_valid_claim_record, is_valid_metric_record
from src.utils.validation import validate_project_folder


OnFailureAction = Literal["block", "downgrade"]


@dataclass(frozen=True)
class ResultsWriterConfig:
    enabled: bool = True
    on_missing_metrics: OnFailureAction = "block"

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "ResultsWriterConfig":
        raw = context.get("results_writer")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", True))
        on_missing = raw.get("on_missing_metrics", "block")
        if on_missing not in ("block", "downgrade"):
            on_missing = "block"

        return cls(enabled=enabled, on_missing_metrics=on_missing)


def _load_json_list(path: Path) -> Tuple[List[Any], Optional[str]]:
    if not path.exists():
        return [], None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        return [], f"{type(e).__name__}"

    if not isinstance(payload, list):
        return [], "not_a_list"

    return payload, None


def _collect_computed_metric_keys(claims: List[Any]) -> Tuple[List[str], List[str]]:
    referenced: set[str] = set()
    invalid: List[str] = []

    for item in claims:
        if not isinstance(item, dict) or not is_valid_claim_record(item):
            invalid.append("invalid_claim_record")
            continue

        if str(item.get("kind")) != "computed":
            continue

        keys = item.get("metric_keys")
        if isinstance(keys, list):
            for k in keys:
                if isinstance(k, str) and k.strip():
                    referenced.add(k.strip())

    return sorted(referenced), invalid


def _index_metrics(metrics: List[Any]) -> Tuple[Dict[str, Dict[str, Any]], int]:
    by_key: Dict[str, Dict[str, Any]] = {}
    invalid = 0

    for item in metrics:
        if not isinstance(item, dict) or not is_valid_metric_record(item):
            invalid += 1
            continue

        k = item.get("metric_key")
        if isinstance(k, str) and k.strip():
            by_key[k.strip()] = item

    return by_key, invalid


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Deterministic and compact across platforms.
        return f"{value:.10g}"
    return str(value)


class ResultsWriterAgent(BaseAgent):
    """Deterministic Results writer constrained by metrics.json."""

    def __init__(self, client=None):
        system_prompt = (
            "You write a Results section for an academic paper. "
            "Only emit numeric results when they are backed by outputs/metrics.json. "
            "If referenced metric keys are missing, block or downgrade based on config."
        )
        super().__init__(
            name="ResultsWriter",
            task_type=TaskType.DOCUMENT_CREATION,
            system_prompt=system_prompt,
            client=client,
        )

    async def execute(self, context: dict) -> AgentResult:
        project_folder = context.get("project_folder")
        if not isinstance(project_folder, str) or not project_folder:
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error="Missing required context: project_folder",
            )

        pf = validate_project_folder(project_folder)
        ensure_project_outputs_layout(pf)

        cfg = ResultsWriterConfig.from_context(context)
        section_id = str(context.get("section_id") or "results").strip() or "results"
        section_title = str(context.get("section_title") or "Results").strip() or "Results"

        output_path, rel = resolve_section_output_path(pf, section_id=section_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cfg.enabled:
            latex = f"\\section{{{section_title}}}\n% Results writer disabled\n"
            output_path.write_text(latex, encoding="utf-8")
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content=latex,
                structured_data={
                    "section_id": section_id,
                    "output_relpath": rel,
                    "metadata": {"enabled": False},
                },
            )

        claims_path = pf / "claims" / "claims.json"
        claims_payload, claims_error = _load_json_list(claims_path)
        referenced_metric_keys, _invalid_claims = _collect_computed_metric_keys(claims_payload)

        metrics_path = pf / "outputs" / "metrics.json"
        metrics_payload, metrics_error = _load_json_list(metrics_path)
        metrics_by_key, invalid_metrics = _index_metrics(metrics_payload)

        missing_metric_keys = sorted([k for k in referenced_metric_keys if k not in metrics_by_key])
        present_metric_keys = sorted([k for k in referenced_metric_keys if k in metrics_by_key])

        if missing_metric_keys and cfg.on_missing_metrics == "block":
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=f"Missing metric keys referenced by computed claims: {missing_metric_keys}",
                structured_data={
                    "section_id": section_id,
                    "output_relpath": rel,
                    "metadata": {
                        "enabled": True,
                        "action": "block",
                        "missing_metric_keys": missing_metric_keys,
                        "present_metric_keys": present_metric_keys,
                        "claims_file_present": claims_path.exists(),
                        "metrics_file_present": metrics_path.exists(),
                        "claims_read_error": claims_error,
                        "metrics_read_error": metrics_error,
                        "metrics_invalid_items": invalid_metrics,
                    },
                },
            )

        chunks: List[str] = [f"\\section{{{section_title}}}"]

        if missing_metric_keys:
            chunks.append("% Results writer downgraded due to missing metrics")

        if present_metric_keys:
            chunks.append("\\begin{itemize}")
            for key in present_metric_keys:
                rec = metrics_by_key.get(key)
                if not isinstance(rec, dict):
                    continue

                name = str(rec.get("name") or key).strip() or key
                value = _format_metric_value(rec.get("value"))
                unit = str(rec.get("unit") or "").strip()

                tail = f" {unit}" if unit else ""
                chunks.append(f"\\item {name}: {value}{tail}.")
            chunks.append("\\end{itemize}")
        else:
            chunks.append("Results are pending metric computation.")

        latex = "\n".join(chunks)
        if not latex.endswith("\n"):
            latex += "\n"

        output_path.write_text(latex, encoding="utf-8")

        metadata = {
            "enabled": True,
            "action": "downgrade" if missing_metric_keys else "pass",
            "missing_metric_keys": missing_metric_keys,
            "present_metric_keys": present_metric_keys,
            "claims_file_present": claims_path.exists(),
            "metrics_file_present": metrics_path.exists(),
            "claims_read_error": claims_error,
            "metrics_read_error": metrics_error,
            "metrics_invalid_items": invalid_metrics,
        }

        return AgentResult(
            agent_name=self.name,
            task_type=self.task_type,
            model_tier=self.model_tier,
            success=True,
            content=latex,
            structured_data={
                "section_id": section_id,
                "output_relpath": rel,
                "metadata": metadata,
            },
        )
