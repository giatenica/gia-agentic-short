""" 
Source Fetcher (Local Ingest MVP)
================================
Discovers and reads local sources inside a project folder.

This is intentionally minimal: it only enumerates files and can load text for
plain-text formats. More advanced parsing (PDF, DOCX, HTML) is out of scope
for the MVP and handled by later steps.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.utils.filesystem import source_id_to_dirname, validate_source_id
from src.utils.validation import validate_project_folder, validate_path


DEFAULT_SEARCH_DIRS = [
    "data/raw data",
    "literature",
    "drafts",
    "paper",
]


DEFAULT_EXCLUDE_DIRS = {
    ".workflow_cache",
    ".evidence",
    "__pycache__",
    "_tmp_extract",
}


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".tex",
    ".bib",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
}


# Binary data formats that can be converted to text summaries
BINARY_DATA_EXTENSIONS = {
    ".parquet",
}


# All supported extensions for evidence extraction
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | BINARY_DATA_EXTENSIONS


def _load_parquet_as_text(path: Path, max_chars: int = 200_000) -> str:
    """Load a parquet file and convert to a text representation.

    Returns a markdown-formatted summary including:
    - Schema (column names and types)
    - Row count
    - Sample rows (head)
    - Basic statistics for numeric columns
    """
    try:
        import pandas as pd
    except ImportError:
        raise ValueError(
            "pandas is required for parquet support; install with: pip install pandas pyarrow"
        )

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise ValueError(f"Failed to read parquet file: {e}")

    lines = []
    lines.append(f"# Parquet Data: {path.name}")
    lines.append("")

    # Schema
    lines.append("## Schema")
    lines.append(f"- **Rows:** {len(df):,}")
    lines.append(f"- **Columns:** {len(df.columns)}")
    lines.append("")
    lines.append("| Column | Type |")
    lines.append("|--------|------|")
    for col in df.columns:
        dtype = str(df[col].dtype)
        lines.append(f"| {col} | {dtype} |")
    lines.append("")

    # Sample data (first 10 rows)
    lines.append("## Sample Data (first 10 rows)")
    lines.append("")
    sample_df = df.head(10)
    # Convert to markdown table
    lines.append("| " + " | ".join(str(c) for c in sample_df.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in sample_df.columns) + " |")
    for _, row in sample_df.iterrows():
        row_vals = []
        for val in row:
            # Truncate long values and escape pipes
            s = str(val)[:50].replace("|", "\\|").replace("\n", " ")
            row_vals.append(s)
        lines.append("| " + " | ".join(row_vals) + " |")
    lines.append("")

    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        lines.append("## Numeric Column Statistics")
        lines.append("")
        stats = df[numeric_cols].describe()
        lines.append(
            "| Statistic | " + " | ".join(str(c) for c in numeric_cols) + " |"
        )
        lines.append("| --- | " + " | ".join("---" for _ in numeric_cols) + " |")
        for stat_name in stats.index:
            vals = [
                f"{stats.loc[stat_name, c]:.4g}"
                if pd.notna(stats.loc[stat_name, c])
                else "N/A"
                for c in numeric_cols
            ]
            lines.append(f"| {stat_name} | " + " | ".join(vals) + " |")
        lines.append("")

    # Date columns info
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if date_cols:
        lines.append("## Date Column Ranges")
        lines.append("")
        for col in date_cols:
            min_date = df[col].min()
            max_date = df[col].max()
            lines.append(f"- **{col}:** {min_date} to {max_date}")
        lines.append("")

    text = "\n".join(lines)
    return text[:max_chars]


@dataclass(frozen=True)
class LocalSource:
    source_id: str
    relative_path: str
    mime_type: str
    size_bytes: int
    sha256: str
    created_at: str


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_metadata(relative_path: Path, size_bytes: int, mtime_seconds: float) -> str:
    """Compute a stable sha256 from file metadata.

    This is a performance-friendly alternative to hashing full file contents.
    """
    h = hashlib.sha256()
    h.update(relative_path.as_posix().encode("utf-8"))
    h.update(b"\0")
    h.update(str(size_bytes).encode("utf-8"))
    h.update(b"\0")
    h.update(str(int(mtime_seconds)).encode("utf-8"))
    return h.hexdigest()


def _is_excluded_path(relative_path: Path) -> bool:
    parts = set(relative_path.parts)
    if parts & DEFAULT_EXCLUDE_DIRS:
        return True
    for part in relative_path.parts:
        if part.startswith(".") and part not in {".", ".."}:
            return True
    return False


class SourceFetcherTool:
    """Local ingest tool for discovering project sources."""

    def __init__(
        self,
        project_folder: str,
        search_dirs: Optional[List[str]] = None,
        max_files: int = 5000,
        hash_contents: bool = True,
        sources_subdir: str = "sources",
    ):
        self.project_folder = validate_project_folder(project_folder)
        self.search_dirs = search_dirs or list(DEFAULT_SEARCH_DIRS)
        self.max_files = max_files
        self.hash_contents = hash_contents
        self.sources_subdir = sources_subdir

    def discover_sources(self) -> List[LocalSource]:
        """Discover files under the project folder.

        Returns:
            List of LocalSource records, sorted by relative path.
        """
        sources: List[LocalSource] = []
        for base in self.search_dirs:
            base_path = self.project_folder / base
            if not base_path.exists() or not base_path.is_dir():
                continue

            for file_path in base_path.rglob("*"):
                if not file_path.is_file():
                    continue

                rel = file_path.relative_to(self.project_folder)
                if _is_excluded_path(rel):
                    continue

                # Enforce safety: ensure path is under project_folder
                try:
                    validate_path(file_path, must_exist=True, must_be_file=True, base_dir=self.project_folder)
                except (ValueError, FileNotFoundError):
                    # Skip symlinks or other paths that resolve outside the project folder.
                    continue

                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                size_bytes = stat.st_size
                if self.hash_contents:
                    try:
                        sha256 = _sha256_file(file_path)
                    except OSError:
                        continue
                else:
                    sha256 = _sha256_metadata(rel, size_bytes=size_bytes, mtime_seconds=stat.st_mtime)

                mime_type, _ = mimetypes.guess_type(str(file_path))
                mime_type = mime_type or "application/octet-stream"

                source_id = f"file:{sha256[:12]}"
                created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                )
                sources.append(
                    LocalSource(
                        source_id=source_id,
                        relative_path=str(rel.as_posix()),
                        mime_type=mime_type,
                        size_bytes=size_bytes,
                        sha256=sha256,
                        created_at=created_at,
                    )
                )

                if len(sources) >= self.max_files:
                    return sorted(sources, key=lambda s: s.relative_path)

        return sorted(sources, key=lambda s: s.relative_path)

    def load_text(self, source: LocalSource, max_chars: int = 200_000) -> str:
        """Load a source file as text.

        Raises:
            ValueError: When file extension is not supported as text.
        """
        path = self.project_folder / source.relative_path
        validate_path(path, must_exist=True, must_be_file=True, base_dir=self.project_folder)

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported text format: {ext}")

        # Handle binary data formats specially
        if ext in BINARY_DATA_EXTENSIONS:
            if ext == ".parquet":
                return _load_parquet_as_text(path, max_chars=max_chars)
            # Future: add handlers for other binary formats here
            raise ValueError(f"No handler for binary format: {ext}")

        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars]

    def ingest_source(self, source: LocalSource) -> dict:
        """Copy the referenced source file into sources/<source_id>/raw/.

        Returns:
            A dict with project-relative paths for downstream steps.
        """
        src_path = self.project_folder / source.relative_path
        validate_path(src_path, must_exist=True, must_be_file=True, base_dir=self.project_folder)

        validate_source_id(source.source_id)

        source_dirname = source_id_to_dirname(source.source_id)
        raw_dir = self.project_folder / self.sources_subdir / source_dirname / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        dest_path = raw_dir / Path(source.relative_path).name
        # Avoid loading large files into memory.
        shutil.copy2(src_path, dest_path)

        return {
            "source_id": source.source_id,
            "source_dir": str((self.project_folder / self.sources_subdir / source_dirname).relative_to(self.project_folder)),
            "raw_path": str(dest_path.relative_to(self.project_folder)),
            "original_relative_path": source.relative_path,
        }


def discover_local_sources(project_folder: str) -> List[LocalSource]:
    """Convenience wrapper for default discovery."""
    return SourceFetcherTool(project_folder).discover_sources()
