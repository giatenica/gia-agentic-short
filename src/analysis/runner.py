""" 
Analysis Runner
===============
Execute a Python analysis script from a project's analysis/ folder and record
created files under outputs/artifacts.json.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import INTAKE_SERVER
from src.config import TIMEOUTS
from src.utils.subprocess_env import build_minimal_subprocess_env
from src.utils.subprocess_text import to_text
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.validation import validate_project_folder


@dataclass(frozen=True)
class AnalysisRunResult:
    """Result of executing an analysis script."""

    success: bool
    returncode: int
    stdout: str
    stderr: str
    created_files: List[str]
    artifacts_path: str


@dataclass(frozen=True)
class AnalysisMultiRunResult:
    """Result of executing multiple analysis scripts."""

    success: bool
    runs: List[Dict[str, Any]]
    created_files: List[str]
    artifacts_path: str


def _safe_relpath(child: Path, base: Path) -> str:
    return str(child.relative_to(base)).replace(os.sep, "/")


def _list_project_files(project_folder: Path) -> List[str]:
    files: List[str] = []

    max_files = int(INTAKE_SERVER.MAX_ZIP_FILES)

    exclude_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        ".workflow_cache",
        "sources",
        ".evidence",
        "temp",
        "tmp",
        "node_modules",
        "data",
    }

    for p in project_folder.rglob("*"):
        if not p.is_file():
            continue

        rel_parts = p.relative_to(project_folder).parts
        if any(part in exclude_dirs for part in rel_parts[:-1]):
            continue
        if any(part.startswith(".") for part in rel_parts[:-1]):
            continue

        files.append(_safe_relpath(p, project_folder))
        if len(files) >= max_files:
            break
    files.sort()
    return files


def _script_sha256(script_path: Path) -> str:
    data = script_path.read_bytes()
    return hashlib.sha256(data).hexdigest()


_SCRIPT_PREFIX_RE = re.compile(r"^(?P<prefix>\d{1,4})[-_].+")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _analysis_order_key(relpath: str) -> tuple:
    """Deterministic ordering key for analysis scripts.

    Prefer numeric prefixes like 01_load.py, 02_clean.py. Scripts without prefixes
    sort after prefixed scripts.
    """

    name = Path(relpath).name
    m = _SCRIPT_PREFIX_RE.match(name)
    if m:
        try:
            return (0, int(m.group("prefix")), relpath)
        except Exception:
            return (0, 10**9, relpath)
    return (1, 10**9, relpath)


def discover_analysis_scripts(
    *,
    project_folder: str | Path,
    manifest_relpath: str = "analysis/manifest.json",
) -> List[str]:
    """Discover analysis scripts under <project>/analysis/.

    If analysis/manifest.json exists, its order is respected.
    Otherwise scripts are discovered and ordered deterministically.
    """

    pf = validate_project_folder(project_folder)
    analysis_dir = (pf / "analysis").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (pf / manifest_relpath).resolve()
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid analysis manifest JSON: {type(e).__name__}")

        if isinstance(payload, dict):
            payload = payload.get("scripts")

        if not isinstance(payload, list) or not all(isinstance(s, str) for s in payload):
            raise ValueError("Analysis manifest must be a list of script relpaths or {'scripts': [...]} ")

        scripts: List[str] = []
        for s in payload:
            rel = s.strip().lstrip("/")
            if not rel:
                continue
            sp = (pf / rel).resolve()
            try:
                sp.relative_to(analysis_dir)
            except ValueError:
                raise ValueError(f"Manifest script must be under analysis/: {rel}")
            if sp.suffix != ".py":
                raise ValueError(f"Manifest script must be .py: {rel}")
            if not sp.exists():
                raise FileNotFoundError(f"Manifest script not found: {rel}")
            scripts.append(_safe_relpath(sp, pf))
        return scripts

    max_files = int(INTAKE_SERVER.MAX_ZIP_FILES)
    discovered: List[str] = []
    for p in analysis_dir.rglob("*.py"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if "__pycache__" in p.parts:
            continue
        discovered.append(_safe_relpath(p, pf))
        if len(discovered) >= max_files:
            break

    discovered.sort(key=_analysis_order_key)
    return discovered


def _execute_one_script(
    *,
    project_folder: Path,
    script_abs_path: Path,
    timeout_seconds: int,
    sanitize_env: bool,
) -> tuple[int, str, str, bool]:
    env = build_minimal_subprocess_env(sanitize_env=sanitize_env)

    returncode = -1
    stdout = ""
    stderr = ""
    success = False

    try:
        result = subprocess.run(
            [sys.executable, "-I", "-B", _safe_relpath(script_abs_path, project_folder)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            cwd=str(project_folder),
            env=env,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        returncode = int(result.returncode)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        success = returncode == 0
    except subprocess.TimeoutExpired as e:
        stdout = to_text(e.stdout)
        stderr = to_text(e.stderr)
        if stderr:
            stderr = stderr.rstrip("\n") + "\n"
        stderr += f"Execution timed out after {timeout_seconds} seconds"
        returncode = -1
        success = False
    except Exception as e:
        stdout = ""
        stderr = f"Execution failed: {type(e).__name__}"
        returncode = -1
        success = False

    return returncode, stdout, stderr, success


def run_project_analysis_scripts(
    *,
    project_folder: str | Path,
    scripts: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
    sanitize_env: bool = True,
    stop_on_failure: bool = True,
) -> AnalysisMultiRunResult:
    """Run multiple Python scripts under analysis/ and write a combined artifacts.json."""

    pf = validate_project_folder(project_folder)
    paths = ensure_project_outputs_layout(pf)

    script_relpaths = scripts if scripts is not None else discover_analysis_scripts(project_folder=pf)
    script_relpaths = [s for s in (script_relpaths or []) if isinstance(s, str) and s.strip()]

    analysis_dir = (pf / "analysis").resolve()
    for rel in script_relpaths:
        sp = (pf / rel).resolve()
        try:
            sp.relative_to(analysis_dir)
        except ValueError:
            raise ValueError(f"Analysis script must be under analysis/: {rel}")
        if sp.suffix != ".py":
            raise ValueError(f"Analysis script must be a .py file: {rel}")
        if not sp.exists():
            raise FileNotFoundError(f"Analysis script not found: {rel}")

    before_all = set(_list_project_files(pf))

    timeout = int(timeout_seconds) if timeout_seconds is not None else int(TIMEOUTS.CODE_EXECUTION)

    runs: List[Dict[str, Any]] = []
    overall_success = True

    for rel in script_relpaths:
        sp = (pf / rel).resolve()
        started_at = _now_utc_iso()

        before_files = set(_list_project_files(pf))
        rc, out, err, ok = _execute_one_script(
            project_folder=pf,
            script_abs_path=sp,
            timeout_seconds=timeout,
            sanitize_env=sanitize_env,
        )
        after_files = set(_list_project_files(pf))
        created_files = sorted(after_files - before_files)

        runs.append(
            {
                "started_at": started_at,
                "finished_at": _now_utc_iso(),
                "script": {"path": rel, "sha256": _script_sha256(sp)},
                "result": {
                    "returncode": int(rc),
                    "success": bool(ok),
                    "stdout": out,
                    "stderr": err,
                },
                "created_files": created_files,
            }
        )

        if not ok:
            overall_success = False
            if stop_on_failure:
                break

    artifacts_path = paths.outputs_dir / "artifacts.json"
    rel_artifacts = _safe_relpath(artifacts_path, pf)

    after_all = set(_list_project_files(pf))
    created_all = sorted(after_all - before_all)
    if rel_artifacts not in before_all and rel_artifacts not in created_all:
        created_all = sorted(created_all + [rel_artifacts])

    payload: Dict[str, Any] = {
        "schema_version": "1.1",
        "success": bool(overall_success),
        "runs": runs,
        "created_files": created_all,
    }

    artifacts_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return AnalysisMultiRunResult(
        success=bool(overall_success),
        runs=runs,
        created_files=created_all,
        artifacts_path=rel_artifacts,
    )


def run_project_analysis_script(
    *,
    project_folder: str | Path,
    script_path: str | Path,
    timeout_seconds: Optional[int] = None,
    sanitize_env: bool = True,
) -> AnalysisRunResult:
    """Run a Python script that lives under analysis/ and record outputs.

    Args:
        project_folder: Project folder containing project.json
        script_path: Path to a .py file under <project_folder>/analysis
        timeout_seconds: Optional override for execution timeout
        sanitize_env: When True, avoid inheriting most parent env vars

    Returns:
        AnalysisRunResult
    """

    pf = validate_project_folder(project_folder)
    paths = ensure_project_outputs_layout(pf)

    sp = Path(script_path)
    if not sp.is_absolute():
        sp = (pf / sp).resolve()
    else:
        sp = sp.resolve()

    analysis_dir = (pf / "analysis").resolve()
    try:
        sp.relative_to(analysis_dir)
    except ValueError:
        raise ValueError(f"Analysis script must be under analysis/: {sp}")

    if sp.suffix != ".py":
        raise ValueError(f"Analysis script must be a .py file: {sp}")
    if not sp.exists():
        raise FileNotFoundError(f"Analysis script not found: {sp}")

    before_files = set(_list_project_files(pf))

    timeout = int(timeout_seconds) if timeout_seconds is not None else int(TIMEOUTS.CODE_EXECUTION)
    returncode, stdout, stderr, success = _execute_one_script(
        project_folder=pf,
        script_abs_path=sp,
        timeout_seconds=timeout,
        sanitize_env=sanitize_env,
    )

    after_files = set(_list_project_files(pf))
    created_files = sorted(after_files - before_files)

    artifacts_path = paths.outputs_dir / "artifacts.json"
    rel_artifacts = _safe_relpath(artifacts_path, pf)
    if rel_artifacts not in before_files and rel_artifacts not in created_files:
        created_files = sorted(created_files + [rel_artifacts])

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "script": {
            "path": _safe_relpath(sp, pf),
            "sha256": _script_sha256(sp),
        },
        "result": {
            "returncode": int(returncode),
            "success": bool(success),
            "stdout": stdout,
            "stderr": stderr,
        },
        "created_files": created_files,
    }

    artifacts_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return AnalysisRunResult(
        success=bool(success),
        returncode=int(returncode),
        stdout=stdout,
        stderr=stderr,
        created_files=created_files,
        artifacts_path=rel_artifacts,
    )
