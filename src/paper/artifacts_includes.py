"""Deterministic LaTeX include generation for analysis artifacts.

Generates LaTeX include files for:
- outputs/tables/*.tex
- outputs/figures/*

These include files are intended to be consumed by the paper assembly step.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import List, Tuple


_FIGURE_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".eps"}


def _sanitize_label_fragment(value: str) -> str:
    # Deterministic, conservative: letters, numbers, underscore only.
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "item"


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:6]


def _labels_for_stems(prefix: str, stems: List[str]) -> List[str]:
    bases = [_sanitize_label_fragment(s) for s in stems]
    counts: dict[str, int] = {}
    for b in bases:
        counts[b] = counts.get(b, 0) + 1

    labels: List[str] = []
    for stem, base in zip(stems, bases, strict=True):
        if counts.get(base, 0) <= 1:
            labels.append(f"{prefix}:{base}")
        else:
            labels.append(f"{prefix}:{base}_{_short_hash(stem)}")
    return labels


def _caption_from_stem(stem: str) -> str:
    # Deterministic caption derived from filename stem.
    return stem.replace("_", " ").replace("-", " ").strip() or stem


def discover_table_tex_paths(project_folder: Path) -> List[Path]:
    """Discover LaTeX table artifact files.

    Args:
        project_folder: Project root folder that contains an `outputs/` directory.

    Returns:
        Sorted list of `.tex` paths under `outputs/tables/`.

        If `outputs/tables/` does not exist, is not a directory, or contains no matching
        files, returns an empty list.
    """
    tables_dir = project_folder / "outputs" / "tables"
    if not tables_dir.exists() or not tables_dir.is_dir():
        return []
    return sorted([p for p in tables_dir.glob("*.tex") if p.is_file() and not p.name.startswith(".")])


def discover_figure_paths(project_folder: Path) -> List[Path]:
    """Discover figure artifact files.

    Args:
        project_folder: Project root folder that contains an `outputs/` directory.

    Returns:
        Sorted list of figure paths under `outputs/figures/`.

        Only files with extensions in `_FIGURE_EXTS` are included. Hidden files are
        skipped. If `outputs/figures/` does not exist or is not a directory, returns
        an empty list.
    """
    figures_dir = project_folder / "outputs" / "figures"
    if not figures_dir.exists() or not figures_dir.is_dir():
        return []
    candidates: List[Path] = []
    for p in figures_dir.iterdir():
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() in _FIGURE_EXTS:
            candidates.append(p)
    return sorted(candidates)


def generate_tables_include_tex(project_folder: Path) -> Tuple[str, List[str]]:
    """Generate a LaTeX include file for all table artifacts.

    This scans `outputs/tables/*.tex` and wraps each table in a `table` environment
    with a deterministic caption and a deterministic, collision-safe label.

    Args:
        project_folder: Project root folder.

    Returns:
        Tuple of:
        - latex: The contents of the generated include file.
        - labels: A list of resolved labels in the same order as discovered artifacts.

        Labels are formatted as `tab:<sanitized_stem>` when unique; if multiple stems
        sanitize to the same base, a short hash suffix is appended:
        `tab:<sanitized_stem>_<sha1_6>`.

        If `outputs/tables/` does not exist or contains no tables, the returned LaTeX
        still contains the header comment block and `labels` is empty.
    """
    paths = discover_table_tex_paths(project_folder)
    lines: List[str] = []
    labels: List[str] = []

    lines.append("% Generated table includes")
    lines.append("% This file is written by scripts/run_paper_assembly.py")
    lines.append("")

    for p in paths:
        stem = p.stem
        caption = _caption_from_stem(stem)
        labels.append(stem)

        rel = p.relative_to(project_folder).as_posix()
        input_path = "../" + rel

        lines.append(f"% --- {rel} ---")
        lines.append("\\begin{table}[!htbp]")
        lines.append("  \\centering")
        lines.append(f"  \\caption{{{caption}}}")
        # label will be filled below once collision-safe labels are computed
        lines.append(f"  \\label{{__LABEL_PLACEHOLDER__}}")
        lines.append(f"  \\input{{{input_path}}}")
        lines.append("\\end{table}")
        lines.append("")

    resolved_labels = _labels_for_stems("tab", labels)
    out = "\n".join(lines)
    for resolved in resolved_labels:
        out = out.replace("\\label{__LABEL_PLACEHOLDER__}", f"\\label{{{resolved}}}", 1)
    return out.rstrip() + "\n", resolved_labels


def generate_figures_include_tex(project_folder: Path) -> Tuple[str, List[str]]:
    """Generate a LaTeX include file for all figure artifacts.

    This scans `outputs/figures/*` and wraps each supported file in a `figure`
    environment with `\\includegraphics`, a deterministic caption, and a
    deterministic, collision-safe label.

    Args:
        project_folder: Project root folder.

    Returns:
        Tuple of:
        - latex: The contents of the generated include file.
        - labels: A list of resolved labels in the same order as discovered artifacts.

        Labels are formatted as `fig:<sanitized_stem>` when unique; if multiple stems
        sanitize to the same base, a short hash suffix is appended:
        `fig:<sanitized_stem>_<sha1_6>`.

        If `outputs/figures/` does not exist or contains no supported figures, the
        returned LaTeX still contains the header comment block and `labels` is empty.
    """
    paths = discover_figure_paths(project_folder)
    lines: List[str] = []
    labels: List[str] = []

    lines.append("% Generated figure includes")
    lines.append("% This file is written by scripts/run_paper_assembly.py")
    lines.append("")

    for p in paths:
        stem = p.stem
        caption = _caption_from_stem(stem)
        labels.append(stem)

        rel = p.relative_to(project_folder).as_posix()
        fig_path = "../" + rel

        lines.append(f"% --- {rel} ---")
        lines.append("\\begin{figure}[!htbp]")
        lines.append("  \\centering")
        lines.append(f"  \\includegraphics[width=\\linewidth]{{{fig_path}}}")
        lines.append(f"  \\caption{{{caption}}}")
        # label will be filled below once collision-safe labels are computed
        lines.append(f"  \\label{{__LABEL_PLACEHOLDER__}}")
        lines.append("\\end{figure}")
        lines.append("")

    resolved_labels = _labels_for_stems("fig", labels)
    out = "\n".join(lines)
    for resolved in resolved_labels:
        out = out.replace("\\label{__LABEL_PLACEHOLDER__}", f"\\label{{{resolved}}}", 1)
    return out.rstrip() + "\n", resolved_labels
