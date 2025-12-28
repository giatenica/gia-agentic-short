"""
Tests for Source Fetcher
=======================

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from pathlib import Path

import pytest
from src.evidence.source_fetcher import LocalSource, SourceFetcherTool


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

    # Other default-excluded dirs (ensure they are ignored even if present under a search dir)
    evidence_dir = literature_dir / ".evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "skip.md").write_text("nope", encoding="utf-8")

    cache_dir = literature_dir / ".workflow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "skip.md").write_text("nope", encoding="utf-8")

    tmp_extract_dir = literature_dir / "_tmp_extract"
    tmp_extract_dir.mkdir(parents=True, exist_ok=True)
    (tmp_extract_dir / "skip.md").write_text("nope", encoding="utf-8")

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
    assert "literature/.evidence/skip.md" not in rel_paths
    assert "literature/.workflow_cache/skip.md" not in rel_paths
    assert "literature/_tmp_extract/skip.md" not in rel_paths
    assert "literature/.hidden/secret.md" not in rel_paths


@pytest.mark.unit
def test_source_fetcher_respects_max_files_and_returns_sorted(temp_project_folder):
    literature_dir = temp_project_folder / "literature"
    literature_dir.mkdir(exist_ok=True)

    # Create more files than the max_files limit
    (literature_dir / "b.md").write_text("b", encoding="utf-8")
    (literature_dir / "a.md").write_text("a", encoding="utf-8")
    (literature_dir / "c.md").write_text("c", encoding="utf-8")

    fetcher = SourceFetcherTool(str(temp_project_folder), max_files=2)
    sources = fetcher.discover_sources()

    assert len(sources) == 2
    rels = [s.relative_path for s in sources]
    assert rels == sorted(rels)


@pytest.mark.unit
def test_source_fetcher_load_text_truncates_to_max_chars(temp_project_folder):
    literature_dir = temp_project_folder / "literature"
    literature_dir.mkdir(exist_ok=True)
    (literature_dir / "notes.txt").write_text("x" * 1000, encoding="utf-8")

    fetcher = SourceFetcherTool(str(temp_project_folder))
    sources = fetcher.discover_sources()
    txt = next(s for s in sources if s.relative_path == "literature/notes.txt")

    text = fetcher.load_text(txt, max_chars=100)
    assert len(text) == 100
    assert text == "x" * 100


@pytest.mark.unit
def test_source_fetcher_ingest_copies_into_sources_raw(temp_project_folder):
    literature_dir = temp_project_folder / "literature"
    literature_dir.mkdir(exist_ok=True)
    original = literature_dir / "notes.md"
    original.write_text("hello world", encoding="utf-8")

    fetcher = SourceFetcherTool(str(temp_project_folder))
    sources = fetcher.discover_sources()
    src = next(s for s in sources if s.relative_path == "literature/notes.md")

    result = fetcher.ingest_source(src)
    raw_path = temp_project_folder / result["raw_path"]
    assert raw_path.exists()
    assert raw_path.read_text(encoding="utf-8") == "hello world"

    # Source folder name should be filesystem-safe
    assert ":" not in Path(result["source_dir"]).name


@pytest.mark.unit
def test_source_fetcher_ingest_rejects_path_outside_project(temp_project_folder):
    fetcher = SourceFetcherTool(str(temp_project_folder))
    evil = LocalSource(
        source_id="file:deadbeef",
        relative_path="../evil.txt",
        mime_type="text/plain",
        size_bytes=1,
        sha256="0" * 64,
        created_at="2025-12-22T00:00:00Z",
    )

    with pytest.raises(ValueError):
        fetcher.ingest_source(evil)


@pytest.mark.unit
def test_source_fetcher_skips_symlink_outside_project(temp_project_folder):
    literature_dir = temp_project_folder / "literature"
    literature_dir.mkdir(exist_ok=True)
    (literature_dir / "keep.md").write_text("ok", encoding="utf-8")

    outside_file = temp_project_folder.parent / "outside.txt"
    outside_file.write_text("outside", encoding="utf-8")

    link_path = literature_dir / "outside_link.txt"
    try:
        link_path.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported in this environment")

    fetcher = SourceFetcherTool(str(temp_project_folder), max_files=100)
    sources = fetcher.discover_sources()
    rel_paths = {s.relative_path for s in sources}

    assert "literature/keep.md" in rel_paths
    assert "literature/outside_link.txt" not in rel_paths


@pytest.mark.unit
def test_source_fetcher_load_parquet_file(temp_project_folder):
    """Test that parquet files can be loaded and converted to text."""
    import pandas as pd

    raw_data = temp_project_folder / "data" / "raw data"
    raw_data.mkdir(parents=True, exist_ok=True)

    # Create a simple parquet file
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "price": [100.0, 101.5, 99.8, 102.3, 103.1],
        "volume": [1000, 1500, 800, 2000, 1200],
        "symbol": ["GOOG", "GOOG", "GOOG", "GOOG", "GOOG"],
    })
    parquet_path = raw_data / "test_data.parquet"
    df.to_parquet(parquet_path)

    fetcher = SourceFetcherTool(str(temp_project_folder))
    sources = fetcher.discover_sources()
    parquet_source = next(s for s in sources if s.relative_path.endswith(".parquet"))

    text = fetcher.load_text(parquet_source)

    # Verify the text contains expected schema information
    assert "# Parquet Data: test_data.parquet" in text
    assert "## Schema" in text
    assert "**Rows:** 5" in text
    assert "**Columns:** 4" in text
    assert "| date |" in text
    assert "| price |" in text
    assert "| volume |" in text
    assert "| symbol |" in text

    # Verify sample data is included
    assert "## Sample Data" in text
    assert "GOOG" in text

    # Verify numeric statistics are included
    assert "## Numeric Column Statistics" in text
    assert "mean" in text
    assert "std" in text


@pytest.mark.unit
def test_source_fetcher_load_parquet_truncates_to_max_chars(temp_project_folder):
    """Test that parquet text output respects max_chars limit."""
    import pandas as pd

    raw_data = temp_project_folder / "data" / "raw data"
    raw_data.mkdir(parents=True, exist_ok=True)

    # Create a larger parquet file
    df = pd.DataFrame({
        "col_" + str(i): range(100) for i in range(20)
    })
    parquet_path = raw_data / "large_data.parquet"
    df.to_parquet(parquet_path)

    fetcher = SourceFetcherTool(str(temp_project_folder))
    sources = fetcher.discover_sources()
    parquet_source = next(s for s in sources if s.relative_path.endswith(".parquet"))

    text = fetcher.load_text(parquet_source, max_chars=500)
    assert len(text) <= 500


@pytest.mark.unit
def test_source_fetcher_parquet_in_supported_extensions():
    """Test that .parquet is in the SUPPORTED_EXTENSIONS set."""
    from src.evidence.source_fetcher import SUPPORTED_EXTENSIONS, BINARY_DATA_EXTENSIONS

    assert ".parquet" in SUPPORTED_EXTENSIONS
    assert ".parquet" in BINARY_DATA_EXTENSIONS

