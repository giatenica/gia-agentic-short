import json

import pytest

from src.analysis.runner import discover_analysis_scripts, run_project_analysis_scripts


@pytest.mark.unit
def test_discover_analysis_scripts_orders_by_numeric_prefix(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    (analysis_dir / "02_second.py").write_text("print('2')\n", encoding="utf-8")
    (analysis_dir / "01_first.py").write_text("print('1')\n", encoding="utf-8")
    (analysis_dir / "z_last.py").write_text("print('z')\n", encoding="utf-8")

    scripts = discover_analysis_scripts(project_folder=temp_project_folder)
    assert scripts == [
        "analysis/01_first.py",
        "analysis/02_second.py",
        "analysis/z_last.py",
    ]


@pytest.mark.unit
def test_discover_analysis_scripts_uses_manifest_order(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    (analysis_dir / "01_first.py").write_text("print('1')\n", encoding="utf-8")
    (analysis_dir / "02_second.py").write_text("print('2')\n", encoding="utf-8")

    manifest = ["analysis/02_second.py", "analysis/01_first.py"]
    (analysis_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    scripts = discover_analysis_scripts(project_folder=temp_project_folder)
    assert scripts == ["analysis/02_second.py", "analysis/01_first.py"]


@pytest.mark.unit
def test_discover_analysis_scripts_uses_manifest_dict_scripts_key(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    (analysis_dir / "01_first.py").write_text("print('1')\n", encoding="utf-8")
    (analysis_dir / "02_second.py").write_text("print('2')\n", encoding="utf-8")

    manifest = {"scripts": ["analysis/02_second.py", "analysis/01_first.py"]}
    (analysis_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    scripts = discover_analysis_scripts(project_folder=temp_project_folder)
    assert scripts == ["analysis/02_second.py", "analysis/01_first.py"]


@pytest.mark.unit
def test_discover_analysis_scripts_rejects_invalid_manifest_json(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "manifest.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid analysis manifest JSON"):
        discover_analysis_scripts(project_folder=temp_project_folder)


@pytest.mark.unit
def test_discover_analysis_scripts_rejects_manifest_script_outside_analysis(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    outside = temp_project_folder / "not_analysis.py"
    outside.write_text("print('nope')\n", encoding="utf-8")

    manifest = ["not_analysis.py"]
    (analysis_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be under analysis/"):
        discover_analysis_scripts(project_folder=temp_project_folder)


@pytest.mark.unit
def test_discover_analysis_scripts_rejects_manifest_missing_script(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = ["analysis/does_not_exist.py"]
    (analysis_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Manifest script not found"):
        discover_analysis_scripts(project_folder=temp_project_folder)


@pytest.mark.unit
def test_multi_runner_writes_combined_artifacts_for_multiple_scripts(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    (analysis_dir / "01_a.py").write_text(
        "from pathlib import Path\n"
        "p = Path('.')\n"
        "(p / 'outputs' / 'tables').mkdir(parents=True, exist_ok=True)\n"
        "(p / 'outputs' / 'tables' / 'a.csv').write_text('a,b\\n1,2\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )

    (analysis_dir / "02_b.py").write_text(
        "from pathlib import Path\n"
        "p = Path('.')\n"
        "(p / 'outputs' / 'tables').mkdir(parents=True, exist_ok=True)\n"
        "(p / 'outputs' / 'tables' / 'b.csv').write_text('c,d\\n3,4\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = run_project_analysis_scripts(project_folder=temp_project_folder)
    assert result.success is True
    assert result.artifacts_path == "outputs/artifacts.json"

    artifacts_path = temp_project_folder / "outputs" / "artifacts.json"
    payload = json.loads(artifacts_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.1"
    assert payload["success"] is True
    assert [r["script"]["path"] for r in payload["runs"]] == ["analysis/01_a.py", "analysis/02_b.py"]

    assert "outputs/tables/a.csv" in payload["created_files"]
    assert "outputs/tables/b.csv" in payload["created_files"]


@pytest.mark.unit
def test_multi_runner_stop_on_failure_true_stops_early(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    (analysis_dir / "01_ok.py").write_text("print('ok')\n", encoding="utf-8")
    (analysis_dir / "02_fail.py").write_text("raise SystemExit(2)\n", encoding="utf-8")
    (analysis_dir / "03_after.py").write_text("print('after')\n", encoding="utf-8")

    result = run_project_analysis_scripts(project_folder=temp_project_folder, stop_on_failure=True)
    assert result.success is False

    payload = json.loads((temp_project_folder / "outputs" / "artifacts.json").read_text(encoding="utf-8"))
    assert [r["script"]["path"] for r in payload["runs"]] == ["analysis/01_ok.py", "analysis/02_fail.py"]


@pytest.mark.unit
def test_multi_runner_stop_on_failure_false_runs_all_scripts(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    (analysis_dir / "01_ok.py").write_text("print('ok')\n", encoding="utf-8")
    (analysis_dir / "02_fail.py").write_text("raise SystemExit(2)\n", encoding="utf-8")
    (analysis_dir / "03_after.py").write_text("print('after')\n", encoding="utf-8")

    result = run_project_analysis_scripts(project_folder=temp_project_folder, stop_on_failure=False)
    assert result.success is False

    payload = json.loads((temp_project_folder / "outputs" / "artifacts.json").read_text(encoding="utf-8"))
    assert [r["script"]["path"] for r in payload["runs"]] == [
        "analysis/01_ok.py",
        "analysis/02_fail.py",
        "analysis/03_after.py",
    ]


@pytest.mark.unit
def test_multi_runner_rejects_no_scripts(temp_project_folder):
    (temp_project_folder / "analysis").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="No analysis scripts provided or discovered"):
        run_project_analysis_scripts(project_folder=temp_project_folder, scripts=[])
