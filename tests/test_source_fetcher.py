"""
Tests for Source Fetcher
=======================

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from pathlib import Path

import pytest
from src.evidence.source_fetcher import SourceFetcherTool


@pytest.mark.unit
def test_source_fetcher_discovers_files_in_project(temp_project_folder):
    literature_dir = temp_project_folder / "literature"
    literature_dir.mkdir(exist_ok=True)
    (literature_dir / "notes.md").write_text("hello", encoding="utf-8")

    raw_data = temp_project_folder / "data" / "raw data"
    raw_data.mkdir(parents=True, exist_ok=True)
    (raw_data / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    fetcher = SourceFetcherTool(str(temp_project_folder), max_files=100)
    sources = fetcher.discover_sources()

    rel_paths = {s.relative_path for s in sources}
    assert "literature/notes.md" in rel_paths
    assert "data/raw data/data.csv" in rel_paths


@pytest.mark.unit
def test_source_fetcher_load_text_supported_extensions(temp_project_folder):
    literature_dir = temp_project_folder / "literature"
    literature_dir.mkdir(exist_ok=True)
    (literature_dir / "notes.md").write_text("hello world", encoding="utf-8")

    fetcher = SourceFetcherTool(str(temp_project_folder))
    sources = fetcher.discover_sources()
    md = next(s for s in sources if s.relative_path == "literature/notes.md")
    assert fetcher.load_text(md).startswith("hello")


@pytest.mark.unit
def test_source_fetcher_rejects_unsupported_text_format(temp_project_folder):
    paper_dir = temp_project_folder / "paper"
    paper_dir.mkdir(exist_ok=True)
    (paper_dir / "figure.pdf").write_bytes(b"%PDF-1.7\n")

    fetcher = SourceFetcherTool(str(temp_project_folder))
    sources = fetcher.discover_sources()
    pdf = next(s for s in sources if s.relative_path == "paper/figure.pdf")

    with pytest.raises(ValueError, match="Unsupported text format"):
        fetcher.load_text(pdf)


@pytest.mark.unit
def test_source_fetcher_excludes_default_and_hidden_dirs(temp_project_folder):
    literature_dir = temp_project_folder / "literature"
    literature_dir.mkdir(exist_ok=True)

    # Default-excluded dir name
    pycache_dir = literature_dir / "__pycache__"
    pycache_dir.mkdir(parents=True, exist_ok=True)
    (pycache_dir / "skip.md").write_text("nope", encoding="utf-8")

    # Hidden dir segment
    hidden_dir = literature_dir / ".hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    (hidden_dir / "secret.md").write_text("nope", encoding="utf-8")

    # Control file that should be discovered
    (literature_dir / "keep.md").write_text("ok", encoding="utf-8")

    fetcher = SourceFetcherTool(str(temp_project_folder), max_files=100)
    sources = fetcher.discover_sources()

    rel_paths = {s.relative_path for s in sources}
    assert "literature/keep.md" in rel_paths
    assert "literature/__pycache__/skip.md" not in rel_paths
    assert "literature/.hidden/secret.md" not in rel_paths
