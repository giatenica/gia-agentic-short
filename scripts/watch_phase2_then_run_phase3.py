"""Autonomous watcher for end-to-end workflow.

Waits for the literature workflow (Phase 2) to finish by watching for
`literature_workflow_results.json`. Once present:
- Writes `literature_workflow_issues.json` if missing (non-blocking best effort).
- Starts gap resolution (Phase 3) via `scripts/run_gap_resolution.py`.

Designed to be launched with nohup so it can survive shell restarts.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path


def _log(msg: str) -> None:
    print(msg, flush=True)


def _select_python(repo_root: Path) -> str:
    """Return a Python executable that can import this repo reliably.

    Prefer the dedicated run venv if present, then the default venv, then
    fall back to the current interpreter.
    """

    candidates = [
        repo_root / ".venv_run" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _build_minimal_subprocess_env(parent_env: Mapping[str, str]) -> dict[str, str]:
    """Best-effort minimal environment for subprocesses.

    Keeps basic OS runtime variables and only whitelists AI/workflow keys.
    """

    allow_exact = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "TERM",
        "PWD",
        "PYTHONIOENCODING",
    }

    allow_prefixes = (
        "GIA_",
        "ANTHROPIC_",
        "EDISON_",
        "OPENAI_",
        "AZURE_",
        "HF_",
    )

    env: dict[str, str] = {}
    for key, value in parent_env.items():
        if key in allow_exact or key.startswith(allow_prefixes):
            env[key] = value

    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _write_issues_best_effort(
    *,
    repo_root: Path,
    python_exe: str,
    project_folder: str,
    results_path: Path,
    issues_path: Path,
) -> None:
    """Write issues JSON without ever blocking Phase 3.

    This intentionally does not import any `src.*` modules to keep the watcher
    resilient even under interpreter/env mismatches.
    """

    if issues_path.exists():
        return

    try:
        workflow_results = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as e:
        issues_path.write_text(
            json.dumps(
                {
                    "issues": [
                        {
                            "severity": "warning",
                            "code": "LITERATURE_RESULTS_JSON_INVALID",
                            "message": f"Failed to parse {results_path.name}: {e}",
                        }
                    ]
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return

    helper_code = (
        "import json,sys; "
        "from pathlib import Path; "
        "from src.utils.workflow_issue_tracking import write_workflow_issue_tracking; "
        "project_folder=sys.argv[1]; "
        "results=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')); "
        "write_workflow_issue_tracking(project_folder, results, filename='literature_workflow_issues.json');"
    )

    try:
        completed = subprocess.run(
            [python_exe, "-c", helper_code, project_folder, str(results_path)],
            cwd=str(repo_root),
            env=_build_minimal_subprocess_env(os.environ),
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            capture_output=True,
        )
        if completed.returncode == 0 and issues_path.exists():
            return

        issues_path.write_text(
            json.dumps(
                {
                    "issues": [
                        {
                            "severity": "warning",
                            "code": "ISSUE_TRACKING_FALLBACK",
                            "message": "Failed to write literature_workflow_issues.json via helper; wrote fallback.",
                            "details": {
                                "returncode": completed.returncode,
                                "stdout": completed.stdout[-4000:],
                                "stderr": completed.stderr[-4000:],
                            },
                        }
                    ],
                    "workflow_results_summary": {
                        "keys": sorted(list(workflow_results.keys())),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as e:
        issues_path.write_text(
            json.dumps(
                {
                    "issues": [
                        {
                            "severity": "warning",
                            "code": "ISSUE_TRACKING_EXCEPTION",
                            "message": f"Exception while writing issues: {e}",
                        }
                    ]
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[1]

    if len(argv) != 2:
        _log("Usage: python scripts/watch_phase2_then_run_phase3.py <project_folder>")
        return 2

    project_folder = argv[1]
    project_path = Path(project_folder)
    results_path = project_path / "literature_workflow_results.json"
    issues_path = project_path / "literature_workflow_issues.json"

    python_exe = _select_python(repo_root)

    _log(f"Watcher: project_folder={project_folder}")
    _log(f"Watcher: waiting for {results_path}")

    last_heartbeat = 0.0

    while not results_path.exists():
        now = time.time()
        if now - last_heartbeat >= 60:
            _log(f"Watcher: still waiting for {results_path}")
            last_heartbeat = now
        time.sleep(10)

    _log(f"Watcher: detected {results_path}")

    if not issues_path.exists():
        _write_issues_best_effort(
            repo_root=repo_root,
            python_exe=python_exe,
            project_folder=project_folder,
            results_path=results_path,
            issues_path=issues_path,
        )
        if issues_path.exists():
            _log("Watcher: ensured literature_workflow_issues.json")

    _log("Watcher: starting gap resolution")

    env = _build_minimal_subprocess_env(os.environ)
    cmd = [python_exe, "scripts/run_gap_resolution.py", project_folder]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as e:
        _log(f"Watcher: failed to start gap resolution: {e}")
        return 1

    _log(f"Watcher: gap resolution exited with code {completed.returncode}")
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
