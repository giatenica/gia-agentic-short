"""
Tests for FigureRegistry utility.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import tempfile
from pathlib import Path

import pytest

from src.paper.figure_registry import (
    FigureEntry,
    FigureRegistry,
    auto_register_from_outputs,
    _generate_caption_from_filename,
)


@pytest.mark.unit
class TestFigureEntry:
    """Tests for FigureEntry dataclass."""

    def test_figure_entry_to_dict(self):
        entry = FigureEntry(
            id="CORRELATION_MATRIX",
            path="outputs/figures/correlation_matrix.png",
            label="fig:correlation_matrix",
            caption="Correlation matrix of key variables",
            artifact_type="figure",
            source_gap_id="CORRELATION_ANALYSIS",
        )
        d = entry.to_dict()
        assert d["id"] == "CORRELATION_MATRIX"
        assert d["path"] == "outputs/figures/correlation_matrix.png"
        assert d["label"] == "fig:correlation_matrix"
        assert d["artifact_type"] == "figure"

    def test_figure_entry_from_dict(self):
        data = {
            "id": "TEST_FIG",
            "path": "outputs/figures/test.png",
            "label": "fig:test_fig",
            "caption": "Test figure",
            "artifact_type": "figure",
            "width": "0.9\\textwidth",
        }
        entry = FigureEntry.from_dict(data)
        assert entry.id == "TEST_FIG"
        assert entry.width == "0.9\\textwidth"

    def test_figure_entry_to_latex(self):
        entry = FigureEntry(
            id="TEST",
            path="outputs/figures/test.png",
            label="fig:test",
            caption="A test figure",
            artifact_type="figure",
        )
        latex = entry.to_latex()
        assert "\\begin{figure}" in latex
        assert "\\includegraphics" in latex
        assert "\\caption{A test figure}" in latex
        assert "\\label{fig:test}" in latex
        assert "\\end{figure}" in latex

    def test_figure_entry_escape_latex_special_chars(self):
        entry = FigureEntry(
            id="TEST",
            path="outputs/figures/test.png",
            label="fig:test",
            caption="Price & Volume (50%)",
            artifact_type="figure",
        )
        latex = entry.to_latex()
        assert "\\&" in latex
        assert "\\%" in latex

    def test_table_entry_to_latex(self):
        entry = FigureEntry(
            id="SUMMARY_STATS",
            path="outputs/tables/summary.tex",
            label="tab:summary_stats",
            caption="Summary statistics",
            artifact_type="table",
        )
        latex = entry.to_latex()
        assert "\\input{" in latex
        assert "Table: SUMMARY_STATS" in latex


@pytest.mark.unit
class TestFigureRegistry:
    """Tests for FigureRegistry class."""

    def test_registry_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FigureRegistry(Path(tmpdir))
            assert len(registry.entries) == 0

    def test_register_figure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FigureRegistry(Path(tmpdir))
            entry = registry.register_figure(
                id="FIG1",
                path="outputs/figures/fig1.png",
                caption="Test figure",
            )
            assert entry.id == "FIG1"
            assert entry.label == "fig:fig1"
            assert "FIG1" in registry.entries

    def test_register_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FigureRegistry(Path(tmpdir))
            entry = registry.register_table(
                id="TAB1",
                path="outputs/tables/tab1.tex",
                caption="Test table",
            )
            assert entry.id == "TAB1"
            assert entry.label == "tab:tab1"
            assert entry.artifact_type == "table"

    def test_get_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FigureRegistry(Path(tmpdir))
            registry.register_figure("FIG1", "path.png", "caption")
            
            entry = registry.get_entry("FIG1")
            assert entry is not None
            assert entry.id == "FIG1"
            
            assert registry.get_entry("NONEXISTENT") is None

    def test_get_by_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FigureRegistry(Path(tmpdir))
            registry.register_figure("FIG1", "path.png", "caption", label="fig:custom")
            
            entry = registry.get_by_label("fig:custom")
            assert entry is not None
            assert entry.id == "FIG1"

    def test_list_figures_and_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FigureRegistry(Path(tmpdir))
            registry.register_figure("FIG1", "f1.png", "Figure 1")
            registry.register_figure("FIG2", "f2.png", "Figure 2")
            registry.register_table("TAB1", "t1.tex", "Table 1")
            
            figures = registry.list_figures()
            tables = registry.list_tables()
            
            assert len(figures) == 2
            assert len(tables) == 1

    def test_save_and_load_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir)
            
            # Create and save
            registry1 = FigureRegistry(pf)
            registry1.register_figure("FIG1", "path.png", "caption")
            registry1.save()
            
            # Load in new instance
            registry2 = FigureRegistry(pf)
            assert "FIG1" in registry2.entries

    def test_check_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir)
            
            # Create a dummy file
            fig_dir = pf / "outputs" / "figures"
            fig_dir.mkdir(parents=True)
            (fig_dir / "exists.png").touch()
            
            registry = FigureRegistry(pf)
            entry_exists = registry.register_figure(
                "EXISTS", "outputs/figures/exists.png", "caption"
            )
            entry_missing = registry.register_figure(
                "MISSING", "outputs/figures/missing.png", "caption"
            )
            
            assert registry.check_file_exists(entry_exists) is True
            assert registry.check_file_exists(entry_missing) is False

    def test_find_orphaned_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir)
            
            # Create only one file
            fig_dir = pf / "outputs" / "figures"
            fig_dir.mkdir(parents=True)
            (fig_dir / "exists.png").touch()
            
            registry = FigureRegistry(pf)
            registry.register_figure("EXISTS", "outputs/figures/exists.png", "cap1")
            registry.register_figure("MISSING", "outputs/figures/missing.png", "cap2")
            
            orphaned = registry.find_orphaned_entries()
            assert len(orphaned) == 1
            assert orphaned[0].id == "MISSING"

    def test_scan_latex_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FigureRegistry(Path(tmpdir))
            
            tex_content = r"""
            As shown in Figure~\ref{fig:correlation} and Table~\ref{tab:summary},
            the results are consistent with \ref{fig:scatter}.
            """
            
            refs = registry.scan_latex_references(tex_content)
            assert "fig:correlation" in refs
            assert "tab:summary" in refs
            assert "fig:scatter" in refs

    def test_find_undefined_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FigureRegistry(Path(tmpdir))
            registry.register_figure("FIG1", "f1.png", "cap", label="fig:defined")
            
            tex_content = r"\ref{fig:defined} and \ref{fig:undefined}"
            
            undefined = registry.find_undefined_references(tex_content)
            assert "fig:undefined" in undefined
            assert "fig:defined" not in undefined

    def test_generate_figures_tex(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir)
            fig_dir = pf / "outputs" / "figures"
            fig_dir.mkdir(parents=True)
            (fig_dir / "test.png").touch()
            
            registry = FigureRegistry(pf)
            registry.register_figure("TEST", "outputs/figures/test.png", "Test caption")
            
            tex = registry.generate_figures_tex()
            assert "\\begin{figure}" in tex
            assert "Test caption" in tex

    def test_write_include_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir)
            
            registry = FigureRegistry(pf)
            registry.register_figure("FIG1", "f.png", "Figure")
            registry.register_table("TAB1", "t.tex", "Table")
            
            paths = registry.write_include_files()
            
            assert paths["figures"].exists()
            assert paths["tables"].exists()


@pytest.mark.unit
class TestAutoRegister:
    """Tests for auto_register_from_outputs function."""

    def test_auto_register_discovers_figures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir)
            
            # Create figure files
            fig_dir = pf / "outputs" / "figures"
            fig_dir.mkdir(parents=True)
            (fig_dir / "correlation_matrix.png").touch()
            (fig_dir / "scatter_plot.pdf").touch()
            
            registry = auto_register_from_outputs(pf)
            
            assert len(registry.entries) == 2
            assert "CORRELATION_MATRIX" in registry.entries
            assert "SCATTER_PLOT" in registry.entries

    def test_auto_register_discovers_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir)
            
            # Create table files
            tab_dir = pf / "outputs" / "tables"
            tab_dir.mkdir(parents=True)
            (tab_dir / "summary_stats.tex").touch()
            
            registry = auto_register_from_outputs(pf)
            
            assert "SUMMARY_STATS" in registry.entries
            assert registry.entries["SUMMARY_STATS"].artifact_type == "table"

    def test_paths_use_forward_slashes(self):
        """Test that all paths stored in registry use forward slashes for cross-platform consistency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir)
            
            # Create test files
            fig_dir = pf / "outputs" / "figures"
            fig_dir.mkdir(parents=True)
            (fig_dir / "test_figure.png").touch()
            
            tab_dir = pf / "outputs" / "tables"
            tab_dir.mkdir(parents=True)
            (tab_dir / "test_table.tex").touch()
            
            # Auto-register files
            registry = auto_register_from_outputs(pf)
            
            # Verify all paths use forward slashes (POSIX style)
            for entry in registry.entries.values():
                assert "\\" not in entry.path, f"Path contains backslash: {entry.path}"
                assert "/" in entry.path, f"Path should contain forward slashes: {entry.path}"
            
            # Verify find_unregistered_files works correctly
            # (it should find nothing since everything is registered)
            unregistered = registry.find_unregistered_files()
            assert len(unregistered) == 0, "All files should be registered"


@pytest.mark.unit
class TestCaptionGeneration:
    """Tests for caption generation helper function."""

    def test_preserves_uppercase_acronyms(self):
        """Test that all-uppercase words (acronyms) are preserved."""
        assert _generate_caption_from_filename("VaR_analysis") == "VaR Analysis"
        assert _generate_caption_from_filename("GDP_growth") == "GDP Growth"
        assert _generate_caption_from_filename("ROI_calculation") == "ROI Calculation"

    def test_capitalizes_lowercase_words(self):
        """Test that lowercase words are capitalized."""
        assert _generate_caption_from_filename("correlation_matrix") == "Correlation Matrix"
        assert _generate_caption_from_filename("scatter_plot") == "Scatter Plot"
        assert _generate_caption_from_filename("time_series") == "Time Series"

    def test_preserves_mixed_case(self):
        """Test that mixed-case words are preserved."""
        assert _generate_caption_from_filename("PyTorch_model") == "PyTorch Model"
        assert _generate_caption_from_filename("TensorFlow_results") == "TensorFlow Results"

    def test_handles_hyphens(self):
        """Test that hyphens are replaced with spaces."""
        assert _generate_caption_from_filename("time-series-plot") == "Time Series Plot"
        assert _generate_caption_from_filename("GDP-analysis") == "GDP Analysis"

    def test_handles_mixed_separators(self):
        """Test files with both underscores and hyphens."""
        assert _generate_caption_from_filename("VaR_time-series") == "VaR Time Series"
        assert _generate_caption_from_filename("GDP-growth_rate") == "GDP Growth Rate"

    def test_handles_single_word(self):
        """Test single-word filenames."""
        assert _generate_caption_from_filename("correlation") == "Correlation"
        assert _generate_caption_from_filename("GDP") == "GDP"

    def test_handles_multiple_acronyms(self):
        """Test filenames with multiple acronyms."""
        assert _generate_caption_from_filename("GDP_vs_CPI") == "GDP Vs CPI"
        assert _generate_caption_from_filename("VaR_and_CVaR") == "VaR And CVaR"

    def test_auto_register_uses_new_caption_logic(self):
        """Test that auto_register uses the new caption generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir)
            
            # Create figure with acronym
            fig_dir = pf / "outputs" / "figures"
            fig_dir.mkdir(parents=True)
            (fig_dir / "VaR_analysis.png").touch()
            
            registry = auto_register_from_outputs(pf)
            
            # Check that caption preserves acronym
            entry = registry.get_entry("VAR_ANALYSIS")
            assert entry is not None
            assert entry.caption == "VaR Analysis"
