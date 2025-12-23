import pytest

from src.utils.project_layout import ensure_project_outputs_layout, project_outputs_paths


@pytest.mark.unit
def test_project_outputs_paths_are_resolved(temp_project_folder):
    paths = project_outputs_paths(temp_project_folder)

    assert paths.project_folder == temp_project_folder
    assert paths.analysis_dir == temp_project_folder / "analysis"
    assert paths.outputs_dir == temp_project_folder / "outputs"
    assert paths.outputs_tables_dir == temp_project_folder / "outputs" / "tables"
    assert paths.outputs_figures_dir == temp_project_folder / "outputs" / "figures"
    assert paths.claims_dir == temp_project_folder / "claims"


@pytest.mark.unit
def test_ensure_project_outputs_layout_is_idempotent_and_empty(temp_project_folder):
    paths_first = ensure_project_outputs_layout(temp_project_folder)
    paths_second = ensure_project_outputs_layout(temp_project_folder)

    assert paths_first == paths_second

    assert paths_first.analysis_dir.is_dir()
    assert paths_first.outputs_dir.is_dir()
    assert paths_first.outputs_tables_dir.is_dir()
    assert paths_first.outputs_figures_dir.is_dir()
    assert paths_first.claims_dir.is_dir()

    assert list(paths_first.analysis_dir.iterdir()) == []
    assert list(paths_first.outputs_tables_dir.iterdir()) == []
    assert list(paths_first.outputs_figures_dir.iterdir()) == []
    assert list(paths_first.claims_dir.iterdir()) == []
