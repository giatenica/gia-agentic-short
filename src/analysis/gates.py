"""Analysis gate.

Checks that analysis artifacts exist before drafting numeric results.

Deterministic and filesystem-first:
- validates outputs/metrics.json as a list of MetricRecord
- optionally requires at least one table artifact under outputs/tables/
- optionally requires at least one figure artifact under outputs/figures/

The default policy is permissive when not explicitly enabled.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from loguru import logger

from src.config import INTAKE_SERVER
from src.utils.schema_validation import is_valid_metric_record
from src.tracing import safe_set_current_span_attributes
from src.utils.validation import validate_project_folder


class AnalysisGateError(ValueError):
    """Raised when the analysis gate blocks execution."""


OnFailureAction = Literal["block", "downgrade"]


@dataclass(frozen=True)
class AnalysisGateConfig:
    """Configuration for analysis readiness enforcement."""

    enabled: bool = False
    on_failure: OnFailureAction = "block"

    min_metrics: int = 1
    require_tables: bool = False
    require_figures: bool = False

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "AnalysisGateConfig":
        raw = context.get("analysis_gate")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", False))
        on_failure = raw.get("on_failure", "block")
        if on_failure not in ("block", "downgrade"):
            on_failure = "block"

        def _as_int(val: Any, default: int) -> int:
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        min_metrics = _as_int(raw.get("min_metrics"), cls.min_metrics)
        min_metrics = max(0, min_metrics)
        require_tables = bool(raw.get("require_tables", cls.require_tables))
        require_figures = bool(raw.get("require_figures", cls.require_figures))

        return cls(
            enabled=enabled,
            on_failure=on_failure,
            min_metrics=min_metrics,
            require_tables=require_tables,
            require_figures=require_figures,
        )


def _load_json_list(path: Path) -> Tuple[List[Any], Optional[str]]:
    if not path.exists():
        return [], None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        return [], f"{type(e).__name__}"

    if not isinstance(payload, list):
        return [], "not_a_list"

    return payload, None


def _count_dir_artifacts(dir_path: Path, *, allowed_suffixes: Set[str]) -> Tuple[int, List[str], bool]:
    if not dir_path.exists():
        return 0, [], False

    rels: List[str] = []
    max_files = int(INTAKE_SERVER.MAX_ZIP_FILES)
    for p in dir_path.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in allowed_suffixes:
            continue
        rels.append(str(p.relative_to(dir_path)))
        if len(rels) >= max_files:
            break

    rels.sort()

    return len(rels), rels, True


def check_analysis_gate(
    *,
    project_folder: str | Path,
    config: Optional[AnalysisGateConfig] = None,
) -> Dict[str, Any]:
    """Check that analysis prerequisites are met.

    Expected locations:
    - outputs/metrics.json: list[MetricRecord]
    - outputs/tables/: table artifacts (optional)
    - outputs/figures/: figure artifacts (optional)

    Returns a dict with keys:
    - ok (bool)
    - enabled (bool)
    - action (pass|block|downgrade|disabled)
    - metrics_file_present (bool)
    - metrics_read_error (str|None)
    - metrics_invalid_items (int)
    - metrics_total (int)
    - metrics_valid_items (int)
    - min_metrics (int)
    - require_tables (bool)
    - tables_dir_present (bool)
    - tables_count (int)
    - tables_relpaths (list[str])
    - require_figures (bool)
    - figures_dir_present (bool)
    - figures_count (int)
    - figures_relpaths (list[str])
    """

    cfg = config or AnalysisGateConfig()
    pf = validate_project_folder(project_folder)

    metrics_path = pf / "outputs" / "metrics.json"
    metrics_payload, metrics_error = _load_json_list(metrics_path)
    if metrics_error:
        logger.debug(f"Analysis gate: metrics read error: {metrics_error}")

    metrics_invalid = 0
    metrics_valid = 0
    for item in metrics_payload:
        if not isinstance(item, dict) or not is_valid_metric_record(item):
            metrics_invalid += 1
            continue
        metrics_valid += 1

    tables_dir = pf / "outputs" / "tables"
    figures_dir = pf / "outputs" / "figures"

    tables_count, tables_relpaths, tables_dir_present = _count_dir_artifacts(
        tables_dir,
        allowed_suffixes={".tex", ".csv", ".tsv", ".json"},
    )
    figures_count, figures_relpaths, figures_dir_present = _count_dir_artifacts(
        figures_dir,
        allowed_suffixes={".pdf", ".png", ".jpg", ".jpeg", ".svg"},
    )

    # If the gate is disabled, remain permissive.
    if not cfg.enabled:
        result = {
            "ok": True,
            "enabled": False,
            "action": "disabled",
            "metrics_file_present": metrics_path.exists(),
            "metrics_read_error": metrics_error,
            "metrics_invalid_items": metrics_invalid,
            "metrics_total": len(metrics_payload),
            "metrics_valid_items": metrics_valid,
            "min_metrics": cfg.min_metrics,
            "require_tables": cfg.require_tables,
            "tables_dir_present": tables_dir_present,
            "tables_count": tables_count,
            "tables_relpaths": tables_relpaths,
            "require_figures": cfg.require_figures,
            "figures_dir_present": figures_dir_present,
            "figures_count": figures_count,
            "figures_relpaths": figures_relpaths,
        }

        safe_set_current_span_attributes(
            {
                "gate.name": "analysis",
                "gate.enabled": False,
                "gate.ok": True,
                "gate.action": "disabled",
                "analysis_gate.metrics_file_present": bool(metrics_path.exists()),
                "analysis_gate.metrics_valid_items": int(metrics_valid),
                "analysis_gate.metrics_invalid_items": int(metrics_invalid),
                "analysis_gate.min_metrics": int(cfg.min_metrics),
                "analysis_gate.require_tables": bool(cfg.require_tables),
                "analysis_gate.tables_count": int(tables_count),
                "analysis_gate.require_figures": bool(cfg.require_figures),
                "analysis_gate.figures_count": int(figures_count),
            }
        )

        return result

    has_problem = False

    # Metrics must exist and validate.
    if not metrics_path.exists():
        has_problem = True
    if metrics_error is not None:
        has_problem = True
    if metrics_invalid > 0:
        has_problem = True
    if metrics_valid < cfg.min_metrics:
        has_problem = True

    if cfg.require_tables and tables_count < 1:
        has_problem = True

    if cfg.require_figures and figures_count < 1:
        has_problem = True

    action: Literal["pass", "block", "downgrade"] = "pass"
    ok = True

    if has_problem:
        if cfg.on_failure == "block":
            action = "block"
            ok = False
        else:
            action = "downgrade"

    result = {
        "ok": ok,
        "enabled": True,
        "action": action,
        "metrics_file_present": metrics_path.exists(),
        "metrics_read_error": metrics_error,
        "metrics_invalid_items": metrics_invalid,
        "metrics_total": len(metrics_payload),
        "metrics_valid_items": metrics_valid,
        "min_metrics": cfg.min_metrics,
        "require_tables": cfg.require_tables,
        "tables_dir_present": tables_dir_present,
        "tables_count": tables_count,
        "tables_relpaths": tables_relpaths,
        "require_figures": cfg.require_figures,
        "figures_dir_present": figures_dir_present,
        "figures_count": figures_count,
        "figures_relpaths": figures_relpaths,
    }

    safe_set_current_span_attributes(
        {
            "gate.name": "analysis",
            "gate.enabled": True,
            "gate.ok": bool(ok),
            "gate.action": str(action),
            "analysis_gate.on_failure": str(cfg.on_failure),
            "analysis_gate.metrics_file_present": bool(metrics_path.exists()),
            "analysis_gate.metrics_valid_items": int(metrics_valid),
            "analysis_gate.metrics_invalid_items": int(metrics_invalid),
            "analysis_gate.min_metrics": int(cfg.min_metrics),
            "analysis_gate.require_tables": bool(cfg.require_tables),
            "analysis_gate.tables_count": int(tables_count),
            "analysis_gate.require_figures": bool(cfg.require_figures),
            "analysis_gate.figures_count": int(figures_count),
        }
    )

    return result


def enforce_analysis_gate(
    *,
    project_folder: str | Path,
    config: Optional[AnalysisGateConfig] = None,
) -> Dict[str, Any]:
    """Enforce the analysis gate.

    Raises:
        AnalysisGateError: if the gate is enabled and action=block.
    """

    result = check_analysis_gate(project_folder=project_folder, config=config)
    if result.get("enabled") and result.get("action") == "block":
        raise AnalysisGateError(f"Analysis gate blocked: {result}")
    return result
