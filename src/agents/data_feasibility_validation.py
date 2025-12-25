"""Deterministic data feasibility validation agent.

Issue #84 scope:
- Validate local data feasibility for empirical workflows.
- Produce deterministic, machine-readable outputs:
  - outputs/data_feasibility.json
  - outputs/data_feasibility_report.md

The agent is conservative:
- It does not call the LLM.
- It performs local, deterministic checks over files under data/.
- It supports fail-closed (block) or downgrade behavior.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from loguru import logger

from src.agents.base import AgentResult, BaseAgent
from src.config import INTAKE_SERVER
from src.llm.claude_client import TaskType
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.project_io import load_project_json
from src.utils.validation import validate_project_folder


OnFailureAction = Literal["block", "downgrade"]


def _json_dump_deterministic(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _parse_date_like(value: Any) -> Optional[datetime]:
    """Parse flexible date-like values.

    Supported examples:
    - "2020" (interpreted as 2020-01-01)
    - "2020-12-31"
    - datetime/date
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, (int, float)):
        # Treat as year when it looks like YYYY.
        year = int(value)
        if 1800 <= year <= 2200:
            return datetime(year, 1, 1)
        return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    if len(text) == 4 and text.isdigit():
        year = int(text)
        if 1800 <= year <= 2200:
            return datetime(year, 1, 1)
        return None

    # ISO-ish parsing.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m"):
        try:
            dt = datetime.strptime(text, fmt)
            if fmt in ("%Y-%m", "%Y/%m"):
                return datetime(dt.year, dt.month, 1)
            return dt
        except ValueError:
            continue

    # Last resort: let pandas try in a deterministic way.
    try:
        import pandas as pd

        dt = pd.to_datetime(text, errors="coerce", utc=False)
        if pd.isna(dt):
            return None
        # Convert pandas Timestamp to naive datetime.
        return dt.to_pydatetime()
    except Exception:
        return None


def _infer_date_column(columns: List[str]) -> Optional[str]:
    """Infer a likely date column name from a list of column names."""
    lowered = {c.lower(): c for c in columns}
    for candidate in ("date", "datetime", "timestamp", "time", "dt"):
        if candidate in lowered:
            return lowered[candidate]
    for c in columns:
        cl = c.lower()
        if cl.endswith("_date") or cl.endswith("date"):
            return c
    return None


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


@dataclass(frozen=True)
class DataFeasibilityConfig:
    """Configuration for deterministic data feasibility validation.

    NOTE: `expected_dtypes` is parsed but not enforced yet. It is reserved for
    future type validation, and configuring it currently has no effect.
    """

    enabled: bool = True
    files: Optional[List[str]] = None
    max_rows: int = 200_000
    date_column: Optional[str] = None
    required_columns: Optional[List[str]] = None
    expected_dtypes: Optional[Dict[str, str]] = None
    variables: Optional[List[Dict[str, Any]]] = None
    sample_period: Optional[Dict[str, Any]] = None
    on_failure: OnFailureAction = "block"

    @classmethod
    def from_context(cls, context: Dict[str, Any], project_folder: Path) -> "DataFeasibilityConfig":
        raw: Any = context.get("data_feasibility")

        # Allow configuration via project.json as a fallback.
        if not isinstance(raw, dict):
            project_data = load_project_json(str(project_folder))
            raw = project_data.get("data_feasibility")

        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", True))

        files = raw.get("files")
        if files is not None:
            if isinstance(files, list) and all(isinstance(f, str) for f in files):
                files = [f.strip().lstrip("/") for f in files if f.strip()]
            else:
                files = None

        max_rows = raw.get("max_rows", cls.max_rows)
        try:
            max_rows = int(max_rows)
        except Exception:
            max_rows = cls.max_rows
        if max_rows <= 0:
            max_rows = cls.max_rows

        date_column = raw.get("date_column")
        if date_column is not None and not isinstance(date_column, str):
            date_column = None
        if isinstance(date_column, str):
            date_column = date_column.strip() or None

        required_columns = raw.get("required_columns")
        if required_columns is not None:
            if isinstance(required_columns, list) and all(isinstance(c, str) for c in required_columns):
                required_columns = [c.strip() for c in required_columns if c.strip()]
            else:
                required_columns = None

        expected_dtypes = raw.get("expected_dtypes")
        if expected_dtypes is not None:
            if not isinstance(expected_dtypes, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in expected_dtypes.items()
            ):
                expected_dtypes = None

        variables = raw.get("variables")
        if variables is not None:
            if not isinstance(variables, list) or not all(isinstance(v, dict) for v in variables):
                variables = None

        sample_period = raw.get("sample_period")
        if sample_period is not None and not isinstance(sample_period, dict):
            sample_period = None

        on_failure = raw.get("on_failure", "block")
        if on_failure not in ("block", "downgrade"):
            on_failure = "block"

        return cls(
            enabled=enabled,
            files=files,
            max_rows=max_rows,
            date_column=date_column,
            required_columns=required_columns,
            expected_dtypes=expected_dtypes,
            variables=variables,
            sample_period=sample_period,
            on_failure=on_failure,
        )


def _load_dataframe(path: Path, max_rows: int):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, nrows=max_rows)
        return df, True
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if len(df) > max_rows:
            df = df.head(max_rows)
            return df, False
        return df, True
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=0, nrows=max_rows)
        return df, True
    if suffix == ".dta":
        df = pd.read_stata(path)
        if len(df) > max_rows:
            df = df.head(max_rows)
            return df, False
        return df, True
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            df = pd.DataFrame(payload[:max_rows])
            return df, True
        raise ValueError("Unsupported JSON structure (expected list[object])")

    raise ValueError(f"Unsupported data file type: {suffix}")


def _summarize_dataframe(df, *, date_column: Optional[str]) -> Dict[str, Any]:
    import pandas as pd

    def _timestamp_to_iso_date(ts: Any) -> Optional[str]:
        if ts is None or pd.isna(ts):
            return None
        try:
            return ts.to_pydatetime().date().isoformat()
        except (TypeError, AttributeError, ValueError):
            try:
                ts_naive = ts.tz_localize(None)
                return ts_naive.to_pydatetime().date().isoformat()
            except (TypeError, AttributeError, ValueError):
                return None

    columns = list(df.columns)
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}

    missing_counts = df.isnull().sum()
    missing_values: Dict[str, int] = {
        str(c): int(missing_counts[c]) for c in df.columns if int(missing_counts[c]) > 0
    }

    rows = int(len(df))
    size = int(df.size) if rows > 0 else 0
    missing_pct = float(round(float(missing_counts.sum()) / size * 100, 6)) if size > 0 else 0.0

    detected_date_column = date_column
    if detected_date_column is None:
        detected_date_column = _infer_date_column([str(c) for c in columns])

    date_summary: Optional[Dict[str, Any]] = None
    if detected_date_column and detected_date_column in df.columns:
        series = pd.to_datetime(df[detected_date_column], errors="coerce", utc=False)
        non_null = int(series.notna().sum())
        ratio = float(non_null / len(series)) if len(series) else 0.0

        min_dt = series.min()
        max_dt = series.max()

        if pd.isna(min_dt) or pd.isna(max_dt):
            min_iso = None
            max_iso = None
        else:
            min_iso = _timestamp_to_iso_date(min_dt)
            max_iso = _timestamp_to_iso_date(max_dt)

        year_counts = series.dropna().dt.year.value_counts().to_dict()
        date_summary = {
            "date_column": str(detected_date_column),
            "non_null_ratio": float(round(ratio, 6)),
            "min": min_iso,
            "max": max_iso,
            "counts_by_year": {str(int(k)): int(v) for k, v in sorted(year_counts.items())},
        }

    return {
        "rows_loaded": rows,
        "columns": [str(c) for c in columns],
        "dtypes": {k: v for k, v in dtypes.items()},
        "missing_values": missing_values,
        "missing_pct": missing_pct,
        "date_summary": date_summary,
    }


def _check_required_columns(all_columns: List[str], required: Optional[List[str]]) -> Dict[str, Any]:
    if not required:
        return {"enabled": False, "ok": True, "missing": []}

    universe = set(all_columns)
    missing = sorted(c for c in required if c not in universe)
    return {
        "enabled": True,
        "ok": len(missing) == 0,
        "required": list(required),
        "missing": missing,
    }


def _check_variables(all_columns: List[str], variables: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not variables:
        return {"enabled": False, "ok": True, "variables": []}

    universe = set(all_columns)
    results: List[Dict[str, Any]] = []
    all_ok = True
    for spec in variables:
        name = spec.get("name")
        requires = spec.get("requires")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(requires, list) or not all(isinstance(c, str) for c in requires):
            requires = []
        requires = [c.strip() for c in requires if c.strip()]
        missing = sorted(c for c in requires if c not in universe)
        ok = len(missing) == 0
        all_ok = all_ok and ok
        results.append(
            {
                "name": name.strip(),
                "requires": requires,
                "ok": ok,
                "missing": missing,
            }
        )

    return {"enabled": True, "ok": all_ok, "variables": results}


def _check_sample_period(observed_min: Optional[str], observed_max: Optional[str], sample_period: Optional[Dict[str, Any]]):
    if not sample_period:
        return {"enabled": False, "ok": True}

    required_start = _parse_date_like(sample_period.get("start"))
    required_end = _parse_date_like(sample_period.get("end"))

    obs_min = _parse_date_like(observed_min) if observed_min else None
    obs_max = _parse_date_like(observed_max) if observed_max else None

    issues: List[str] = []
    ok = True

    if required_start and obs_min:
        if obs_min > required_start:
            ok = False
            issues.append("observed_start_after_required_start")
    elif required_start and not obs_min:
        ok = False
        issues.append("missing_observed_start")

    if required_end and obs_max:
        if obs_max < required_end:
            ok = False
            issues.append("observed_end_before_required_end")
    elif required_end and not obs_max:
        ok = False
        issues.append("missing_observed_end")

    return {
        "enabled": True,
        "ok": ok,
        "required_start": required_start.date().isoformat() if required_start else None,
        "required_end": required_end.date().isoformat() if required_end else None,
        "observed_start": observed_min,
        "observed_end": observed_max,
        "issues": issues,
    }


def _render_report(report: Dict[str, Any]) -> str:
    ok = bool(report.get("ok"))
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}

    lines: List[str] = []
    lines.append("# Data Feasibility Report")
    lines.append("")
    lines.append(f"Status: {'PASS' if ok else 'FAIL'}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Files analyzed: {summary.get('files_analyzed', 0)}")
    if summary.get("observed_start") or summary.get("observed_end"):
        lines.append(
            f"- Observed date range: {summary.get('observed_start', 'unknown')} to {summary.get('observed_end', 'unknown')}"
        )
    lines.append("")

    req = checks.get("required_columns")
    if isinstance(req, dict) and req.get("enabled"):
        lines.append("## Required Columns")
        lines.append(f"- OK: {bool(req.get('ok'))}")
        missing = req.get("missing") if isinstance(req.get("missing"), list) else []
        if missing:
            lines.append(f"- Missing: {', '.join(str(x) for x in missing)}")
        lines.append("")

    var = checks.get("variables")
    if isinstance(var, dict) and var.get("enabled"):
        lines.append("## Variable Feasibility")
        lines.append(f"- OK: {bool(var.get('ok'))}")
        vars_list = var.get("variables") if isinstance(var.get("variables"), list) else []
        for item in vars_list:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            item_ok = bool(item.get("ok"))
            missing = item.get("missing") if isinstance(item.get("missing"), list) else []
            if item_ok:
                lines.append(f"- {name}: OK")
            else:
                lines.append(f"- {name}: missing {', '.join(str(x) for x in missing)}")
        lines.append("")

    sp = checks.get("sample_period")
    if isinstance(sp, dict) and sp.get("enabled"):
        lines.append("## Sample Period")
        lines.append(f"- OK: {bool(sp.get('ok'))}")
        lines.append(
            f"- Required: {sp.get('required_start', 'unknown')} to {sp.get('required_end', 'unknown')}"
        )
        lines.append(
            f"- Observed: {sp.get('observed_start', 'unknown')} to {sp.get('observed_end', 'unknown')}"
        )
        issues = sp.get("issues") if isinstance(sp.get("issues"), list) else []
        if issues:
            lines.append(f"- Issues: {', '.join(str(x) for x in issues)}")
        lines.append("")

    files = report.get("files") if isinstance(report.get("files"), list) else []
    lines.append("## Files")
    for f in files:
        if not isinstance(f, dict):
            continue
        path = f.get("path")
        rows = f.get("rows_loaded")
        cols = f.get("columns") if isinstance(f.get("columns"), list) else []
        lines.append(f"- {path}: {rows} rows, {len(cols)} cols")

    lines.append("")
    return "\n".join(lines)


class DataFeasibilityValidationAgent(BaseAgent):
    """Validate dataset schema, variable feasibility, and sample period coverage."""

    def __init__(self, client=None):
        super().__init__(
            name="DataFeasibilityValidation",
            task_type=TaskType.DATA_ANALYSIS,
            system_prompt=(
                "You validate local dataset feasibility deterministically. "
                "You never call the LLM. You emit a JSON summary and a readable report under outputs/."
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
        cfg = DataFeasibilityConfig.from_context(context, pf)

        if not cfg.enabled:
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content="",
                structured_data={"metadata": {"enabled": False}},
            )

        data_dir = pf / "data"
        report_json_path = Path(paths.outputs_dir) / "data_feasibility.json"
        report_md_path = Path(paths.outputs_dir) / "data_feasibility_report.md"

        supported_suffixes = {".csv", ".parquet", ".xlsx", ".xls", ".dta", ".json"}
        files: List[Path] = []
        config_path_errors: List[str] = []

        data_dir_resolved = data_dir.resolve()

        if cfg.files:
            for rel in cfg.files:
                candidate = (data_dir / rel).resolve()
                if not candidate.is_relative_to(data_dir_resolved):
                    config_path_errors.append(f"unsafe_file_path:{rel}")
                    continue
                if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in supported_suffixes:
                    files.append(candidate)
        else:
            if data_dir.exists():
                max_files = int(INTAKE_SERVER.MAX_ZIP_FILES)
                exclude_dirs = {"__pycache__", ".venv", ".git", "node_modules", "temp", "tmp"}
                visited = 0
                for p in data_dir.rglob("*"):
                    if not p.is_file() or p.name.startswith("."):
                        continue
                    if p.suffix.lower() not in supported_suffixes:
                        continue
                    visited += 1
                    if visited > max_files:
                        break
                    try:
                        rel_parts = p.relative_to(data_dir).parts
                    except ValueError:
                        continue
                    if any(part in exclude_dirs for part in rel_parts[:-1]):
                        continue
                    if any(part.startswith(".") for part in rel_parts[:-1]):
                        continue
                    files.append(p)

        files = sorted({f.resolve() for f in files})

        file_summaries: List[Dict[str, Any]] = []
        all_columns: List[str] = []
        observed_start: Optional[str] = None
        observed_end: Optional[str] = None
        errors: List[str] = []

        if config_path_errors:
            errors.extend(sorted(config_path_errors))

        if not data_dir.exists() or not files:
            errors.append("no_data_files")

        for f in files:
            try:
                df, fully_loaded = _load_dataframe(f, cfg.max_rows)
                summary = _summarize_dataframe(df, date_column=cfg.date_column)
                summary["path"] = _safe_relpath(f, data_dir)
                summary["fully_loaded"] = bool(fully_loaded)
                file_summaries.append(summary)
                all_columns.extend(summary.get("columns", []))

                date_summary = summary.get("date_summary")
                if isinstance(date_summary, dict):
                    min_iso = date_summary.get("min")
                    max_iso = date_summary.get("max")
                    if isinstance(min_iso, str):
                        if observed_start is None or min_iso < observed_start:
                            observed_start = min_iso
                    if isinstance(max_iso, str):
                        if observed_end is None or max_iso > observed_end:
                            observed_end = max_iso
            except Exception as exc:
                logger.warning(f"Data feasibility: failed to read {f}: {exc}")
                file_summaries.append(
                    {
                        "path": _safe_relpath(f, data_dir),
                        "error": type(exc).__name__,
                    }
                )
                errors.append(f"file_read_failed:{_safe_relpath(f, data_dir)}")

        # Compute checks.
        all_columns_unique = sorted(set(all_columns))

        required_columns_check = _check_required_columns(all_columns_unique, cfg.required_columns)
        variables_check = _check_variables(all_columns_unique, cfg.variables)
        sample_period_check = _check_sample_period(observed_start, observed_end, cfg.sample_period)

        ok = True
        if errors:
            ok = False
        if required_columns_check.get("enabled") and not required_columns_check.get("ok"):
            ok = False
        if variables_check.get("enabled") and not variables_check.get("ok"):
            ok = False
        if sample_period_check.get("enabled") and not sample_period_check.get("ok"):
            ok = False

        report: Dict[str, Any] = {
            "ok": ok,
            "summary": {
                "files_analyzed": len(file_summaries),
                "observed_start": observed_start,
                "observed_end": observed_end,
                "max_rows": int(cfg.max_rows),
            },
            "checks": {
                "required_columns": required_columns_check,
                "variables": variables_check,
                "sample_period": sample_period_check,
            },
            "files": file_summaries,
            "metadata": {
                "enabled": True,
                "errors": sorted(errors),
            },
        }

        # Always try to write outputs.
        try:
            report_json_path.write_text(_json_dump_deterministic(report), encoding="utf-8")
            report_md_path.write_text(_render_report(report) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to write data feasibility outputs: {exc}")

        structured = {
            "ok": bool(ok),
            "output_json": str(report_json_path.relative_to(pf)).replace("\\", "/"),
            "output_report": str(report_md_path.relative_to(pf)).replace("\\", "/"),
            "metadata": {"enabled": True},
        }

        if ok:
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content="",
                structured_data=structured,
            )

        if cfg.on_failure == "block":
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                structured_data=structured,
                error="Data feasibility validation failed",
            )

        structured["metadata"]["action"] = "downgrade"
        return AgentResult(
            agent_name=self.name,
            task_type=self.task_type,
            model_tier=self.model_tier,
            success=True,
            content="",
            structured_data=structured,
        )
