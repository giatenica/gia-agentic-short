#!/usr/bin/env python3
"""Assemble generated LaTeX sections into the paper scaffold.

This runner is non-interactive and filesystem-first.

It writes:
- paper_assembly_results.json
- paper_assembly_issues.json

It also writes/updates:
- paper/generated_sections.tex
- paper/main.tex (in a safe, reversible way)

Exit code behavior:
- Exits 1 only for CLI usage errors.
- Otherwise exits 0 and records failures in the issues file.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.llm.claude_client import load_env_file_lenient  # noqa: E402
from src.paper.artifacts_includes import generate_figures_include_tex, generate_tables_include_tex  # noqa: E402

load_env_file_lenient()


_AUTOGEN_INPUT_BEGIN = "% === AUTOGEN: generated sections begin ==="
_AUTOGEN_INPUT_END = "% === AUTOGEN: generated sections end ==="
_AUTOGEN_DISABLE_BEGIN = "% === AUTOGEN: disable template sections begin ==="
_AUTOGEN_DISABLE_END = "% === AUTOGEN: disable template sections end ==="

_AUTOGEN_ARTIFACTS_BEGIN = "% === AUTOGEN: generated artifacts begin ==="
_AUTOGEN_ARTIFACTS_END = "% === AUTOGEN: generated artifacts end ==="


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _issue(kind: str, message: str, *, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"kind": kind, "message": message}
    if details:
        out["details"] = details
    return out


def _discover_section_relpaths(project_folder: Path) -> List[str]:
    sections_dir = project_folder / "outputs" / "sections"
    if not sections_dir.exists() or not sections_dir.is_dir():
        return []

    paths: List[Path] = [p for p in sections_dir.glob("*.tex") if p.is_file() and not p.name.startswith(".")]

    preferred_order = [
        "introduction.tex",
        "related_work.tex",
        "methods.tex",
        "results.tex",
        "discussion.tex",
    ]
    preferred_rank = {name: idx for idx, name in enumerate(preferred_order)}

    def _sort_key(p: Path) -> tuple[int, str]:
        return (preferred_rank.get(p.name, 999), p.name)

    rels: List[str] = []
    for p in sorted(paths, key=_sort_key):
        rels.append(str(p.relative_to(project_folder)))
    return rels


def _write_generated_sections_tex(project_folder: Path, section_relpaths: List[str]) -> Tuple[Path, List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []

    paper_dir = project_folder / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    out_path = paper_dir / "generated_sections.tex"

    if not section_relpaths:
        issues.append(
            _issue(
                "no_sections",
                "No generated section files found under outputs/sections",
                details={"expected_dir": str(project_folder / "outputs" / "sections")},
            )
        )

    lines: List[str] = []
    lines.append("% Generated section includes")
    lines.append("% This file is written by scripts/run_paper_assembly.py")
    lines.append("")

    for rel in section_relpaths:
        # Make it robust to path separators and ensure forward slashes.
        rel_norm = rel.replace("\\", "/")
        # paper/ is one level below project root.
        input_path = "../" + rel_norm
        lines.append(f"% --- {rel_norm} ---")
        lines.append(f"\\input{{{input_path}}}")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path, issues


def _write_generated_tables_figures_tex(project_folder: Path) -> Tuple[Tuple[Path, Path], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []

    paper_dir = project_folder / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    tables_path = paper_dir / "generated_tables.tex"
    figures_path = paper_dir / "generated_figures.tex"

    tables_tex, table_labels = generate_tables_include_tex(project_folder)
    figures_tex, figure_labels = generate_figures_include_tex(project_folder)

    tables_path.write_text(tables_tex, encoding="utf-8")
    figures_path.write_text(figures_tex, encoding="utf-8")

    return (tables_path, figures_path), issues


def _inject_generated_sections_into_main(main_tex: str) -> Tuple[str, bool, Optional[str]]:
    if _AUTOGEN_INPUT_BEGIN in main_tex and _AUTOGEN_DISABLE_BEGIN in main_tex and _AUTOGEN_ARTIFACTS_BEGIN in main_tex:
        return main_tex, False, None

    intro_marker = "%=============================================================================\n% 1. INTRODUCTION"
    refs_marker = "%=============================================================================\n% REFERENCES"

    intro_idx = main_tex.find(intro_marker)
    refs_idx = main_tex.find(refs_marker)

    if intro_idx == -1:
        return main_tex, False, "intro_marker_not_found"
    if refs_idx == -1:
        return main_tex, False, "refs_marker_not_found"
    if refs_idx <= intro_idx:
        return main_tex, False, "marker_order_invalid"

    generated_block = (
        "\n"
        "%=============================================================================\n"
        "% GENERATED SECTIONS (AUTOGEN)\n"
        "%=============================================================================\n"
        f"{_AUTOGEN_INPUT_BEGIN}\n"
        "\\input{generated_sections.tex}\n"
        f"{_AUTOGEN_INPUT_END}\n\n"
    )

    artifacts_block = (
        "%=============================================================================\n"
        "% GENERATED TABLES/FIGURES (AUTOGEN)\n"
        "%=============================================================================\n"
        f"{_AUTOGEN_ARTIFACTS_BEGIN}\n"
        "\\input{generated_tables.tex}\n"
        "\\input{generated_figures.tex}\n"
        f"{_AUTOGEN_ARTIFACTS_END}\n\n"
    )

    # If the sections autogen block already exists (from a prior run) but artifacts
    # were not yet injected, insert artifacts directly after the sections block.
    if _AUTOGEN_INPUT_BEGIN in main_tex and _AUTOGEN_INPUT_END in main_tex and _AUTOGEN_ARTIFACTS_BEGIN not in main_tex:
        insert_at = main_tex.find(_AUTOGEN_INPUT_END) + len(_AUTOGEN_INPUT_END)
        patched = main_tex[:insert_at] + "\n\n" + artifacts_block + main_tex[insert_at:]
        return patched, True, None

    disable_begin = f"{_AUTOGEN_DISABLE_BEGIN}\n\\iffalse\n"
    disable_end = "\\fi\n" + _AUTOGEN_DISABLE_END + "\n"

    before = main_tex[:intro_idx]
    template_sections = main_tex[intro_idx:refs_idx]
    after = main_tex[refs_idx:]

    new_tex = before + generated_block + artifacts_block + disable_begin + template_sections + "\n" + disable_end + after
    return new_tex, True, None


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_paper_assembly.py <project_folder>")
        sys.exit(1)

    project_folder = Path(sys.argv[1]).expanduser().resolve()

    results_path = project_folder / "paper_assembly_results.json"
    issues_path = project_folder / "paper_assembly_issues.json"

    issues: List[Dict[str, Any]] = []

    if not project_folder.exists() or not project_folder.is_dir():
        issues.append(_issue("invalid_project_folder", "Project folder does not exist", details={"path": str(project_folder)}))
        _safe_write_json(
            issues_path,
            {
                "stage": "paper_assembly",
                "generated_at": _utc_now_iso(),
                "success": False,
                "issues": issues,
            },
        )
        _safe_write_json(
            results_path,
            {
                "stage": "paper_assembly",
                "generated_at": _utc_now_iso(),
                "success": False,
                "result": None,
            },
        )
        print("Project folder invalid. See paper_assembly_issues.json", flush=True)
        return

    print(f"Assembling paper for: {project_folder}", flush=True)

    section_relpaths = _discover_section_relpaths(project_folder)
    generated_path, gen_issues = _write_generated_sections_tex(project_folder, section_relpaths)
    issues.extend(gen_issues)

    (generated_tables_path, generated_figures_path), artifacts_issues = _write_generated_tables_figures_tex(project_folder)
    issues.extend(artifacts_issues)

    paper_main_path = project_folder / "paper" / "main.tex"
    if not paper_main_path.exists():
        issues.append(_issue("missing_paper_main", "paper/main.tex does not exist", details={"path": str(paper_main_path)}))
        _safe_write_json(
            results_path,
            {
                "stage": "paper_assembly",
                "generated_at": _utc_now_iso(),
                "project_folder": str(project_folder),
                "success": False,
                "result": {
                    "generated_sections_tex": str(generated_path),
                    "generated_tables_tex": str(generated_tables_path),
                    "generated_figures_tex": str(generated_figures_path),
                    "main_tex_updated": False,
                    "section_relpaths": section_relpaths,
                },
            },
        )
        _safe_write_json(
            issues_path,
            {
                "stage": "paper_assembly",
                "generated_at": _utc_now_iso(),
                "project_folder": str(project_folder),
                "success": False,
                "issues": issues,
            },
        )
        print("paper/main.tex missing. Wrote generated_sections.tex only.", flush=True)
        return

    try:
        main_tex = paper_main_path.read_text(encoding="utf-8")
    except Exception as e:
        issues.append(_issue("read_error", "Failed to read paper/main.tex", details={"error": f"{type(e).__name__}: {e}"}))
        main_tex = ""

    new_tex, changed, error_code = _inject_generated_sections_into_main(main_tex)
    if error_code:
        issues.append(_issue("main_tex_patch_failed", "Could not locate expected markers in paper/main.tex", details={"error": error_code}))

    if changed:
        try:
            paper_main_path.write_text(new_tex, encoding="utf-8")
        except Exception as e:
            issues.append(_issue("write_error", "Failed to write updated paper/main.tex", details={"error": f"{type(e).__name__}: {e}"}))

    _safe_write_json(
        results_path,
        {
            "stage": "paper_assembly",
            "generated_at": _utc_now_iso(),
            "project_folder": str(project_folder),
            "success": len(issues) == 0,
            "result": {
                "generated_sections_tex": str(generated_path),
                "generated_tables_tex": str(generated_tables_path),
                "generated_figures_tex": str(generated_figures_path),
                "main_tex_updated": bool(changed),
                "section_relpaths": section_relpaths,
            },
        },
    )

    _safe_write_json(
        issues_path,
        {
            "stage": "paper_assembly",
            "generated_at": _utc_now_iso(),
            "project_folder": str(project_folder),
            "success": len(issues) == 0,
            "issues": issues,
        },
    )

    print("\n" + "=" * 60)
    print("PAPER ASSEMBLY COMPLETE")
    print("=" * 60)
    print(f"Generated sections include: {generated_path}")
    print(f"Main updated: {paper_main_path}")
    print(f"Results: {results_path}")
    print(f"Issues:  {issues_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
