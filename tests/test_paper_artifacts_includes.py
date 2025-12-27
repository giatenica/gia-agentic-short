import pytest

from src.paper.artifacts_includes import generate_figures_include_tex, generate_tables_include_tex


@pytest.mark.unit
def test_artifacts_includes_deterministic_and_labels_stable(temp_project_folder):
    (temp_project_folder / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    (temp_project_folder / "outputs" / "figures").mkdir(parents=True, exist_ok=True)

    (temp_project_folder / "outputs" / "tables" / "t1.tex").write_text(
        "\\begin{tabular}{ll}a&b\\\\\\n\\end{tabular}\\n",
        encoding="utf-8",
    )
    (temp_project_folder / "outputs" / "figures" / "fig-1.pdf").write_bytes(b"%PDF-1.4\n")

    tex1, labels1 = generate_tables_include_tex(temp_project_folder)
    tex2, labels2 = generate_tables_include_tex(temp_project_folder)

    assert tex1 == tex2
    assert labels1 == labels2
    assert labels1 == ["tab:t1"]
    assert "\\label{tab:t1}" in tex1

    ftex1, flabels1 = generate_figures_include_tex(temp_project_folder)
    ftex2, flabels2 = generate_figures_include_tex(temp_project_folder)

    assert ftex1 == ftex2
    assert flabels1 == flabels2
    assert flabels1 == ["fig:fig_1"]
    assert "\\label{fig:fig_1}" in ftex1


@pytest.mark.unit
def test_tables_include_dedupes_sanitization_stable(temp_project_folder):
    (temp_project_folder / "outputs" / "tables").mkdir(parents=True, exist_ok=True)

    (temp_project_folder / "outputs" / "tables" / "A B.tex").write_text("x\n", encoding="utf-8")
    (temp_project_folder / "outputs" / "tables" / "A-B.tex").write_text("y\n", encoding="utf-8")

    tex, labels = generate_tables_include_tex(temp_project_folder)
    # Both stems sanitize to the same base; generator must disambiguate deterministically.
    assert len(labels) == 2
    assert labels[0].startswith("tab:A_B_")
    assert labels[1].startswith("tab:A_B_")
    assert labels[0] != labels[1]
    assert f"\\label{{{labels[0]}}}" in tex
    assert f"\\label{{{labels[1]}}}" in tex
