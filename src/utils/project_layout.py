""" 
Project Outputs Layout
======================
Helpers for standardizing the on-disk outputs layout for a project.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.validation import validate_project_folder


@dataclass(frozen=True)
class ProjectOutputsPaths:
    """Resolved outputs layout paths for a project."""

    project_folder: Path
    analysis_dir: Path
    outputs_dir: Path
    outputs_tables_dir: Path
    outputs_figures_dir: Path
    claims_dir: Path


def project_outputs_paths(project_folder: str | Path) -> ProjectOutputsPaths:
    """Return the canonical outputs layout paths for a project folder."""

    pf = validate_project_folder(project_folder)

    analysis_dir = pf / "analysis"
    outputs_dir = pf / "outputs"
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    claims_dir = pf / "claims"

    return ProjectOutputsPaths(
        project_folder=pf,
        analysis_dir=analysis_dir,
        outputs_dir=outputs_dir,
        outputs_tables_dir=tables_dir,
        outputs_figures_dir=figures_dir,
        claims_dir=claims_dir,
    )


def ensure_project_outputs_layout(project_folder: str | Path) -> ProjectOutputsPaths:
    """Ensure the standard outputs layout exists for a project.

    Creates only empty folders:
    - analysis/
    - outputs/tables/
    - outputs/figures/
    - claims/

    The operation is idempotent.
    """

    paths = project_outputs_paths(project_folder)

    paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    paths.outputs_tables_dir.mkdir(parents=True, exist_ok=True)
    paths.outputs_figures_dir.mkdir(parents=True, exist_ok=True)
    paths.claims_dir.mkdir(parents=True, exist_ok=True)

    return paths
