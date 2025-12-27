"""Data analysis execution agent.

Issue #83 scope:
- Deterministically run one or more analysis scripts under analysis/.
- Ensure analysis artifacts are written to the standard outputs layout:
  - outputs/metrics.json (list[MetricRecord])
  - outputs/tables/*.tex
  - outputs/figures/* (optional)
  - outputs/artifacts.json (provenance; written by the analysis runner)

This agent is deliberately conservative:
- It does not call the LLM.
- It runs only scripts located under analysis/.
- It uses the existing analysis runner, which supports a minimal subprocess
  environment to reduce secret leakage.

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
from src.analysis.runner import (
    AnalysisMultiRunResult,
    AnalysisRunResult,
    discover_analysis_scripts,
    run_project_analysis_script,
    run_project_analysis_scripts,
)
from src.llm.claude_client import TaskType
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.schema_validation import is_valid_metric_record
from src.utils.validation import validate_project_folder
from src.claims.generator import generate_claims_from_metrics


OnFailureAction = Literal["block", "downgrade"]


@dataclass(frozen=True)
class AnalysisExecutionConfig:
    enabled: bool = True
    scripts: Optional[List[str]] = None
    timeout_seconds: Optional[int] = None
    sanitize_env: bool = True
    require_figures: bool = False
    on_script_failure: OnFailureAction = "block"
    on_missing_outputs: OnFailureAction = "block"

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "AnalysisExecutionConfig":
        raw = context.get("analysis_execution")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", True))

        scripts = raw.get("scripts")
        if scripts is not None:
            if isinstance(scripts, list) and all(isinstance(s, str) for s in scripts):
                scripts = [s.strip() for s in scripts if s.strip()]
            else:
                scripts = None

        timeout_seconds = raw.get("timeout_seconds")
        if timeout_seconds is not None:
            try:
                timeout_seconds = int(timeout_seconds)
            except Exception:
                timeout_seconds = None

        sanitize_env = bool(raw.get("sanitize_env", True))
        require_figures = bool(raw.get("require_figures", False))

        on_script_failure = raw.get("on_script_failure", "block")
        if on_script_failure not in ("block", "downgrade"):
            on_script_failure = "block"

        on_missing_outputs = raw.get("on_missing_outputs", "block")
        if on_missing_outputs not in ("block", "downgrade"):
            on_missing_outputs = "block"

        return cls(
            enabled=enabled,
            scripts=scripts,
            timeout_seconds=timeout_seconds,
            sanitize_env=sanitize_env,
            require_figures=require_figures,
            on_script_failure=on_script_failure,
            on_missing_outputs=on_missing_outputs,
        )


def _load_metrics(metrics_path: Path) -> Tuple[List[Any], Optional[str]]:
    if not metrics_path.exists():
        return [], None

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return [], type(exc).__name__

    if not isinstance(payload, list):
        return [], "not_a_list"

    return payload, None


def _count_valid_metric_records(items: List[Any]) -> Tuple[int, int]:
    valid = 0
    invalid = 0
    for item in items:
        if not isinstance(item, dict) or not is_valid_metric_record(item):
            invalid += 1
        else:
            valid += 1
    return valid, invalid


def _list_tex_tables(tables_dir: Path) -> List[str]:
    if not tables_dir.exists() or not tables_dir.is_dir():
        return []

    out: List[str] = []
    for p in sorted(tables_dir.glob("*.tex")):
        if p.is_file():
            out.append(p.name)
    return out


def _list_figures(figures_dir: Path) -> List[str]:
    if not figures_dir.exists() or not figures_dir.is_dir():
        return []

    out: List[str] = []
    for p in sorted(figures_dir.iterdir()):
        if p.is_file():
            out.append(p.name)
    return out


def _resolve_scripts(project_folder: Path, scripts: Optional[List[str]]) -> List[str]:
    analysis_dir = project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if scripts:
        return scripts

    return discover_analysis_scripts(project_folder=project_folder)


class DataAnalysisExecutionAgent(BaseAgent):
    """Execute project analysis scripts and validate expected outputs."""

    def __init__(self, client=None):
        super().__init__(
            name="DataAnalysisExecution",
            task_type=TaskType.DATA_ANALYSIS,
            system_prompt=(
                "You run deterministic analysis scripts under analysis/ and validate output artifacts. "
                "You never call the LLM and you avoid leaking secrets into subprocess environments."
            ),
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
        paths = ensure_project_outputs_layout(pf)

        cfg = AnalysisExecutionConfig.from_context(context)
        if not cfg.enabled:
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content="",
                structured_data={"metadata": {"enabled": False}},
            )

        script_paths = _resolve_scripts(pf, cfg.scripts)
        if not script_paths:
            # No scripts configured or discovered.
            if cfg.on_missing_outputs == "block":
                return AgentResult(
                    agent_name=self.name,
                    task_type=self.task_type,
                    model_tier=self.model_tier,
                    success=False,
                    content="",
                    error="No analysis scripts configured or discovered under analysis/",
                )
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content="",
                structured_data={
                    "metadata": {"enabled": True, "action": "downgrade", "reason": "no_scripts"}
                },
            )

        run_results: List[Dict[str, Any]] = []
        created_files_union: set[str] = set()
        last_runner_result: Optional[AnalysisRunResult] = None

        multi_runner_result: Optional[AnalysisMultiRunResult] = None

        if len(script_paths) > 1:
            # Run scripts in one pass and emit a combined artifacts.json that records all runs.
            multi_runner_result = run_project_analysis_scripts(
                project_folder=pf,
                scripts=script_paths,
                timeout_seconds=cfg.timeout_seconds,
                sanitize_env=cfg.sanitize_env,
                stop_on_failure=(cfg.on_script_failure == "block"),
            )
            created_files_union.update(multi_runner_result.created_files)

            for r in multi_runner_result.runs:
                script = str(r.get("script", {}).get("path") or "")
                result = r.get("result", {}) if isinstance(r.get("result"), dict) else {}
                ok = bool(result.get("success"))
                rc_raw = result.get("returncode")
                if isinstance(rc_raw, int):
                    rc = rc_raw
                elif isinstance(rc_raw, str):
                    try:
                        rc = int(rc_raw)
                    except ValueError:
                        rc = -1
                else:
                    rc = -1
                created = r.get("created_files") if isinstance(r.get("created_files"), list) else []
                run_results.append(
                    {
                        "script": script,
                        "success": ok,
                        "returncode": rc,
                        "artifacts_path": multi_runner_result.artifacts_path,
                        "created_files": list(created),
                    }
                )

            if not multi_runner_result.success and cfg.on_script_failure == "block":
                return AgentResult(
                    agent_name=self.name,
                    task_type=self.task_type,
                    model_tier=self.model_tier,
                    success=False,
                    content="",
                    error="One or more analysis scripts failed",
                    structured_data={
                        "runs": run_results,
                        "metadata": {"enabled": True, "action": "block"},
                    },
                )
        else:
            for sp in script_paths:
                rr = run_project_analysis_script(
                    project_folder=pf,
                    script_path=sp,
                    timeout_seconds=cfg.timeout_seconds,
                    sanitize_env=cfg.sanitize_env,
                )
                last_runner_result = rr
                created_files_union.update(rr.created_files)
                run_results.append(
                    {
                        "script": sp,
                        "success": bool(rr.success),
                        "returncode": int(rr.returncode),
                        "artifacts_path": rr.artifacts_path,
                        "created_files": list(rr.created_files),
                    }
                )

                if not rr.success and cfg.on_script_failure == "block":
                    return AgentResult(
                        agent_name=self.name,
                        task_type=self.task_type,
                        model_tier=self.model_tier,
                        success=False,
                        content="",
                        error=f"Analysis script failed: {sp}: returncode={rr.returncode}",
                        structured_data={
                            "runs": run_results,
                            "metadata": {"enabled": True, "action": "block"},
                        },
                    )

        metrics_path = paths.outputs_dir / "metrics.json"
        metrics_payload, metrics_error = _load_metrics(metrics_path)
        valid_metrics, invalid_metrics = _count_valid_metric_records(metrics_payload)

        claims_generation = generate_claims_from_metrics(project_folder=pf)

        tables = _list_tex_tables(paths.outputs_tables_dir)
        figures = _list_figures(paths.outputs_figures_dir)

        missing: List[str] = []
        if valid_metrics < 1:
            missing.append("outputs/metrics.json")
        if not tables:
            missing.append("outputs/tables/*.tex")
        if cfg.require_figures and not figures:
            missing.append("outputs/figures/*")

        if missing and cfg.on_missing_outputs == "block":
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=f"Missing required analysis outputs: {missing}",
                structured_data={
                    "runs": run_results,
                    "metadata": {
                        "enabled": True,
                        "action": "block",
                        "missing_outputs": missing,
                        "metrics_read_error": metrics_error,
                        "valid_metric_records": valid_metrics,
                        "invalid_metric_records": invalid_metrics,
                        "claims_generation": claims_generation,
                        "tables": tables,
                        "figures": figures,
                    },
                },
            )

        action = "pass" if not missing else "downgrade"

        return AgentResult(
            agent_name=self.name,
            task_type=self.task_type,
            model_tier=self.model_tier,
            success=True,
            content="",
            structured_data={
                "runs": run_results,
                "created_files": sorted(created_files_union),
                "artifacts_path": (
                    multi_runner_result.artifacts_path
                    if multi_runner_result is not None
                    else (last_runner_result.artifacts_path if last_runner_result else None)
                ),
                "metadata": {
                    "enabled": True,
                    "action": action,
                    "missing_outputs": missing,
                    "metrics_read_error": metrics_error,
                    "valid_metric_records": valid_metrics,
                    "invalid_metric_records": invalid_metrics,
                    "claims_generation": claims_generation,
                    "tables": tables,
                    "figures": figures,
                },
            },
        )
