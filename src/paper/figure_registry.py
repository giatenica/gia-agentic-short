"""
Figure Registry
===============
Tracks generated figures and tables for paper assembly, ensuring proper
embedding and reference integrity.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger


@dataclass
class FigureEntry:
    """Registry entry for a figure or table."""
    id: str
    path: str  # Relative path from project folder
    label: str  # LaTeX label (e.g., fig:correlation_matrix)
    caption: str
    artifact_type: str  # "figure" or "table"
    source_gap_id: Optional[str] = None  # Which gap generated this
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    width: Optional[str] = None  # LaTeX width (e.g., "0.8\\textwidth")
    position: str = "htbp"  # LaTeX float position
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "label": self.label,
            "caption": self.caption,
            "artifact_type": self.artifact_type,
            "source_gap_id": self.source_gap_id,
            "created_at": self.created_at,
            "width": self.width,
            "position": self.position,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FigureEntry":
        return cls(
            id=data["id"],
            path=data["path"],
            label=data["label"],
            caption=data["caption"],
            artifact_type=data.get("artifact_type", "figure"),
            source_gap_id=data.get("source_gap_id"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            width=data.get("width"),
            position=data.get("position", "htbp"),
        )
    
    def to_latex(self, relative_to: Optional[Path] = None) -> str:
        """Generate LaTeX code for this figure/table."""
        if self.artifact_type == "table":
            return self._to_latex_table()
        return self._to_latex_figure(relative_to)
    
    def _to_latex_figure(self, relative_to: Optional[Path] = None) -> str:
        """Generate LaTeX figure environment."""
        # Compute path for includegraphics
        if relative_to:
            fig_path = Path(self.path)
            # Remove extension for includegraphics (LaTeX finds it)
            include_path = str(fig_path.with_suffix(""))
        else:
            include_path = str(Path(self.path).with_suffix(""))
        
        width = self.width or "0.8\\textwidth"
        
        lines = [
            f"\\begin{{figure}}[{self.position}]",
            "    \\centering",
            f"    \\includegraphics[width={width}]{{{include_path}}}",
            f"    \\caption{{{self._escape_latex(self.caption)}}}",
            f"    \\label{{{self.label}}}",
            "\\end{figure}",
        ]
        return "\n".join(lines)
    
    def _to_latex_table(self) -> str:
        """Generate LaTeX table include."""
        # For tables, we include the .tex file directly
        lines = [
            f"% Table: {self.id}",
            f"\\input{{{self.path}}}",
        ]
        return "\n".join(lines)
    
    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape special LaTeX characters in caption text."""
        # Map of characters to their LaTeX escape sequences
        replacements = {
            "&": "\\&",
            "%": "\\%",
            "$": "\\$",
            "#": "\\#",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
            "~": "\\textasciitilde{}",
            "^": "\\textasciicircum{}",
        }
        # Match any special character that is not already escaped with a backslash
        pattern = re.compile(r"(?<!\\)[&%$#_{}~^]")

        def _replace(match: re.Match) -> str:
            char = match.group(0)
            return replacements.get(char, char)

        return pattern.sub(_replace, text)
class FigureRegistry:
    """
    Registry for tracking figures and tables generated during analysis.
    
    Features:
    - Register new figures/tables with metadata
    - Check for orphaned references
    - Generate LaTeX include files
    - Persist registry to JSON
    """
    
    def __init__(self, project_folder: Path):
        """
        Initialize figure registry.
        
        Args:
            project_folder: Project folder path
        """
        self.project_folder = Path(project_folder)
        self.entries: Dict[str, FigureEntry] = {}
        self._registry_path = self.project_folder / "outputs" / "figure_registry.json"
        
        # Load existing registry if present
        self._load()
    
    def _load(self) -> None:
        """Load registry from disk if exists."""
        if self._registry_path.exists():
            try:
                data = json.loads(self._registry_path.read_text(encoding="utf-8"))
                for entry_data in data.get("entries", []):
                    entry = FigureEntry.from_dict(entry_data)
                    self.entries[entry.id] = entry
                logger.debug(f"Loaded {len(self.entries)} entries from figure registry")
            except Exception as e:
                logger.warning(f"Failed to load figure registry: {e}")
    
    def save(self) -> None:
        """Persist registry to disk."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [e.to_dict() for e in self.entries.values()],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug(f"Saved {len(self.entries)} entries to figure registry")
    
    def register_figure(
        self,
        id: str,
        path: str,
        caption: str,
        *,
        label: Optional[str] = None,
        source_gap_id: Optional[str] = None,
        width: Optional[str] = None,
        position: str = "htbp",
    ) -> FigureEntry:
        """
        Register a new figure.
        
        Args:
            id: Unique identifier for the figure
            path: Path to figure file (relative to project folder)
            caption: Figure caption
            label: LaTeX label (auto-generated if None)
            source_gap_id: Gap that generated this figure
            width: LaTeX width specification
            position: LaTeX float position
            
        Returns:
            FigureEntry for the registered figure
        """
        if label is None:
            # Generate label from id: CORRELATION_ANALYSIS -> fig:correlation_analysis
            label = "fig:" + id.lower().replace(" ", "_").replace("-", "_")
        
        entry = FigureEntry(
            id=id,
            path=path,
            label=label,
            caption=caption,
            artifact_type="figure",
            source_gap_id=source_gap_id,
            width=width,
            position=position,
        )
        
        self.entries[id] = entry
        logger.info(f"Registered figure: {id} -> {path}")
        return entry
    
    def register_table(
        self,
        id: str,
        path: str,
        caption: str,
        *,
        label: Optional[str] = None,
        source_gap_id: Optional[str] = None,
    ) -> FigureEntry:
        """
        Register a new table.
        
        Args:
            id: Unique identifier for the table
            path: Path to table .tex file (relative to project folder)
            caption: Table caption
            label: LaTeX label (auto-generated if None)
            source_gap_id: Gap that generated this table
            
        Returns:
            FigureEntry for the registered table
        """
        if label is None:
            label = "tab:" + id.lower().replace(" ", "_").replace("-", "_")
        
        entry = FigureEntry(
            id=id,
            path=path,
            label=label,
            caption=caption,
            artifact_type="table",
            source_gap_id=source_gap_id,
        )
        
        self.entries[id] = entry
        logger.info(f"Registered table: {id} -> {path}")
        return entry
    
    def get_entry(self, id: str) -> Optional[FigureEntry]:
        """Get entry by ID."""
        return self.entries.get(id)
    
    def get_by_label(self, label: str) -> Optional[FigureEntry]:
        """Get entry by LaTeX label."""
        for entry in self.entries.values():
            if entry.label == label:
                return entry
        return None
    
    def list_figures(self) -> List[FigureEntry]:
        """List all figure entries."""
        return [e for e in self.entries.values() if e.artifact_type == "figure"]
    
    def list_tables(self) -> List[FigureEntry]:
        """List all table entries."""
        return [e for e in self.entries.values() if e.artifact_type == "table"]
    
    def check_file_exists(self, entry: FigureEntry) -> bool:
        """Check if the file for an entry exists."""
        full_path = self.project_folder / entry.path
        return full_path.exists()
    
    def find_orphaned_entries(self) -> List[FigureEntry]:
        """Find entries whose files no longer exist."""
        orphaned = []
        for entry in self.entries.values():
            if not self.check_file_exists(entry):
                orphaned.append(entry)
        return orphaned
    
    def find_unregistered_files(self) -> List[Path]:
        """Find figure/table files that aren't registered."""
        unregistered = []
        
        # Check figures directory
        figures_dir = self.project_folder / "outputs" / "figures"
        if figures_dir.exists():
            registered_paths = {e.path for e in self.entries.values()}
            for p in figures_dir.glob("*"):
                if p.is_file() and p.suffix.lower() in (".png", ".pdf", ".jpg", ".jpeg", ".svg"):
                    rel_path = str(p.relative_to(self.project_folder))
                    if rel_path not in registered_paths:
                        unregistered.append(p)
        
        # Check tables directory
        tables_dir = self.project_folder / "outputs" / "tables"
        if tables_dir.exists():
            for p in tables_dir.glob("*.tex"):
                rel_path = str(p.relative_to(self.project_folder))
                if rel_path not in registered_paths:
                    unregistered.append(p)
        
        return unregistered
    
    def scan_latex_references(self, tex_content: str) -> Set[str]:
        """
        Scan LaTeX content for figure/table references.
        
        Args:
            tex_content: LaTeX source content
            
        Returns:
            Set of referenced labels
        """
        # Match \ref{fig:...} and \ref{tab:...}
        pattern = r"\\ref\{((?:fig|tab):[^}]+)\}"
        matches = re.findall(pattern, tex_content)
        return set(matches)
    
    def find_undefined_references(self, tex_content: str) -> Set[str]:
        """Find references to figures/tables that aren't registered."""
        referenced = self.scan_latex_references(tex_content)
        registered_labels = {e.label for e in self.entries.values()}
        return referenced - registered_labels
    
    def generate_figures_tex(self) -> str:
        """
        Generate LaTeX file content for all registered figures.
        
        Returns:
            LaTeX content for figures include file
        """
        lines = [
            "% Auto-generated figures include file",
            "% Generated by FigureRegistry",
            "",
        ]
        
        figures = self.list_figures()
        if not figures:
            lines.append("% No figures registered")
            return "\n".join(lines)
        
        for entry in sorted(figures, key=lambda e: e.id):
            if self.check_file_exists(entry):
                lines.append(f"% Figure: {entry.id}")
                lines.append(entry.to_latex())
                lines.append("")
            else:
                lines.append(f"% WARNING: File not found for {entry.id}: {entry.path}")
                lines.append("")
        
        return "\n".join(lines)
    
    def generate_tables_tex(self) -> str:
        """
        Generate LaTeX file content for all registered tables.
        
        Returns:
            LaTeX content for tables include file
        """
        lines = [
            "% Auto-generated tables include file",
            "% Generated by FigureRegistry",
            "",
        ]
        
        tables = self.list_tables()
        if not tables:
            lines.append("% No tables registered")
            return "\n".join(lines)
        
        for entry in sorted(tables, key=lambda e: e.id):
            if self.check_file_exists(entry):
                lines.append(entry.to_latex())
                lines.append("")
            else:
                lines.append(f"% WARNING: File not found for {entry.id}: {entry.path}")
                lines.append("")
        
        return "\n".join(lines)
    
    def write_include_files(self) -> Dict[str, Path]:
        """
        Write figures.tex and tables.tex include files.
        
        Returns:
            Dict with paths to generated files
        """
        paper_dir = self.project_folder / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Write figures.tex
        figures_path = paper_dir / "generated_figures.tex"
        figures_path.write_text(self.generate_figures_tex(), encoding="utf-8")
        paths["figures"] = figures_path
        
        # Write tables.tex
        tables_path = paper_dir / "generated_tables.tex"
        tables_path.write_text(self.generate_tables_tex(), encoding="utf-8")
        paths["tables"] = tables_path
        
        logger.info(f"Wrote figure/table include files to {paper_dir}")
        return paths


def auto_register_from_outputs(project_folder: Path) -> FigureRegistry:
    """
    Auto-register all figures and tables found in outputs directory.
    
    Args:
        project_folder: Project folder path
        
    Returns:
        FigureRegistry with discovered entries
    """
    registry = FigureRegistry(project_folder)
    
    # Register figures
    figures_dir = project_folder / "outputs" / "figures"
    if figures_dir.exists():
        for p in figures_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in (".png", ".pdf", ".jpg", ".jpeg", ".svg"):
                id = p.stem.upper().replace("-", "_")
                rel_path = str(p.relative_to(project_folder))
                
                if id not in registry.entries:
                    # Generate caption from filename
                    caption = p.stem.replace("_", " ").replace("-", " ").title()
                    registry.register_figure(
                        id=id,
                        path=rel_path,
                        caption=caption,
                    )
    
    # Register tables
    tables_dir = project_folder / "outputs" / "tables"
    if tables_dir.exists():
        for p in tables_dir.glob("*.tex"):
            id = p.stem.upper().replace("-", "_")
            rel_path = str(p.relative_to(project_folder))
            
            if id not in registry.entries:
                caption = p.stem.replace("_", " ").replace("-", " ").title()
                registry.register_table(
                    id=id,
                    path=rel_path,
                    caption=caption,
                )
    
    registry.save()
    return registry
