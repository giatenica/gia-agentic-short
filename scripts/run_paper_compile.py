#!/usr/bin/env python3
"""Compile the LaTeX paper for a project folder.

Non-interactive, filesystem-first. Records outcome to:
- paper_compile_results.json
- paper_compile_issues.json

Requires a local TeX install (latexmk preferred).

Exit code behavior:
- Exits 1 only for CLI usage errors.
- Otherwise exits 0 and records failures in the issues file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.llm.claude_client import load_env_file_lenient  # noqa: E402

load_env_file_lenient()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _issue(kind: str, message: str, *, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"kind": kind, "message": message}
    if details:
        out["details"] = details
    return out


def _tail_text(text: str, max_chars: int = 8000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _run(cmd: List[str], *, cwd: Path, timeout_s: int = 900) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "cmd": cmd,
            "cwd": str(cwd),
            "stdout_tail": _tail_text(proc.stdout or ""),
            "stderr_tail": _tail_text(proc.stderr or ""),
        }
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
        stderr = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")
        return {
            "ok": False,
            "returncode": None,
            "cmd": cmd,
            "cwd": str(cwd),
            "timeout": True,
            "stdout_tail": _tail_text(stdout),
            "stderr_tail": _tail_text(stderr),
        }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_paper_compile.py <project_folder>")
        sys.exit(1)

    project_folder = Path(sys.argv[1]).expanduser().resolve()

    results_path = project_folder / "paper_compile_results.json"
    issues_path = project_folder / "paper_compile_issues.json"

    issues: List[Dict[str, Any]] = []

    if not project_folder.exists() or not project_folder.is_dir():
        issues.append(_issue("invalid_project_folder", "Project folder does not exist", details={"path": str(project_folder)}))
        _safe_write_json(issues_path, {"stage": "paper_compile", "generated_at": _utc_now_iso(), "success": False, "issues": issues})
        _safe_write_json(results_path, {"stage": "paper_compile", "generated_at": _utc_now_iso(), "success": False, "result": None})
        print("Project folder invalid. See paper_compile_issues.json", flush=True)
        return

    paper_dir = project_folder / "paper"
    main_tex = paper_dir / "main.tex"

    if not main_tex.exists():
        issues.append(_issue("missing_main_tex", "paper/main.tex does not exist", details={"path": str(main_tex)}))
        _safe_write_json(issues_path, {"stage": "paper_compile", "generated_at": _utc_now_iso(), "project_folder": str(project_folder), "success": False, "issues": issues})
        _safe_write_json(results_path, {"stage": "paper_compile", "generated_at": _utc_now_iso(), "project_folder": str(project_folder), "success": False, "result": None})
        print("paper/main.tex missing. See paper_compile_issues.json", flush=True)
        return

    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")

    build_dir = paper_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    steps: List[Dict[str, Any]] = []

    if latexmk:
        cmd = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-outdir=build",
            "main.tex",
        ]
        steps.append(_run(cmd, cwd=paper_dir))
    else:
        issues.append(_issue("latexmk_missing", "latexmk not found; falling back to pdflatex/bibtex"))
        if not pdflatex:
            issues.append(_issue("pdflatex_missing", "pdflatex not found; cannot compile"))
        else:
            steps.append(_run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex"], cwd=paper_dir))
            if bibtex:
                steps.append(_run([bibtex, "main"], cwd=build_dir))
            else:
                issues.append(_issue("bibtex_missing", "bibtex not found; bibliography may fail"))
            steps.append(_run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex"], cwd=paper_dir))
            steps.append(_run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex"], cwd=paper_dir))

    ok = all(bool(s.get("ok")) for s in steps) if steps else False

    if not ok:
        issues.append(_issue("compile_failed", "LaTeX compilation failed", details={"steps": [{"cmd": s.get("cmd"), "returncode": s.get("returncode"), "timeout": s.get("timeout", False)} for s in steps]}))

    pdf_path = build_dir / "main.pdf"

    _safe_write_json(
        results_path,
        {
            "stage": "paper_compile",
            "generated_at": _utc_now_iso(),
            "project_folder": str(project_folder),
            "success": ok and pdf_path.exists(),
            "result": {
                "pdf_path": str(pdf_path) if pdf_path.exists() else None,
                "build_dir": str(build_dir),
                "steps": steps,
            },
        },
    )

    _safe_write_json(
        issues_path,
        {
            "stage": "paper_compile",
            "generated_at": _utc_now_iso(),
            "project_folder": str(project_folder),
            "success": len(issues) == 0,
            "issues": issues,
        },
    )

    print("\n" + "=" * 60)
    print("PAPER COMPILE COMPLETE")
    print("=" * 60)
    print(f"PDF:    {pdf_path if pdf_path.exists() else 'NOT GENERATED'}")
    print(f"Results:{results_path}")
    print(f"Issues: {issues_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
