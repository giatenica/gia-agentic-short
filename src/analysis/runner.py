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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import TIMEOUTS
from src.utils.subprocess_env import build_minimal_subprocess_env
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


def _safe_relpath(child: Path, base: Path) -> str:
    return str(child.relative_to(base)).replace(os.sep, "/")


def _list_project_files(project_folder: Path) -> List[str]:
    files: List[str] = []

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
    files.sort()
    return files


def _script_sha256(script_path: Path) -> str:
    data = script_path.read_bytes()
    return hashlib.sha256(data).hexdigest()


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

    env = build_minimal_subprocess_env(sanitize_env=sanitize_env)
    timeout = int(timeout_seconds) if timeout_seconds is not None else int(TIMEOUTS.CODE_EXECUTION)

    returncode = -1
    stdout = ""
    stderr = ""
    success = False

    try:
        result = subprocess.run(
            [sys.executable, "-I", "-B", _safe_relpath(sp, pf)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(pf),
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
        # Keep error deterministic (do not include platform-dependent details).
        stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
        stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
        if stderr:
            stderr = stderr.rstrip("\n") + "\n"
        stderr += f"Execution timed out after {timeout} seconds"
        returncode = -1
        success = False
    except Exception as e:
        # Avoid leaking full exception details; keep a stable marker.
        stdout = ""
        stderr = f"Execution failed: {type(e).__name__}"
        returncode = -1
        success = False

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
