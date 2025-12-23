import hashlib
import json

import pytest

from src.analysis.runner import run_project_analysis_script


@pytest.mark.unit
def test_analysis_runner_executes_script_and_writes_artifacts(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    script_text = (
        "from pathlib import Path\n"
        "p = Path('.')\n"
        "(p / 'outputs' / 'tables').mkdir(parents=True, exist_ok=True)\n"
        "(p / 'outputs' / 'tables' / 'example.csv').write_text('a,b\\n1,2\\n', encoding='utf-8')\n"
        "print('done')\n"
    )
    script_path = analysis_dir / "example_analysis.py"
    script_path.write_text(script_text, encoding="utf-8")

    result = run_project_analysis_script(
        project_folder=temp_project_folder,
        script_path="analysis/example_analysis.py",
    )

    assert result.success is True
    assert result.returncode == 0
    assert result.artifacts_path == "outputs/artifacts.json"

    artifacts_path = temp_project_folder / "outputs" / "artifacts.json"
    assert artifacts_path.is_file()

    payload = json.loads(artifacts_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["script"]["path"] == "analysis/example_analysis.py"
    assert payload["script"]["sha256"] == hashlib.sha256(script_text.encode("utf-8")).hexdigest()
    assert payload["result"]["success"] is True
    assert payload["result"]["returncode"] == 0
    assert payload["result"]["stdout"].strip() == "done"

    assert payload["created_files"] == [
        "outputs/artifacts.json",
        "outputs/tables/example.csv",
    ]


@pytest.mark.unit
def test_analysis_runner_rejects_missing_script(temp_project_folder):
    (temp_project_folder / "analysis").mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="Analysis script not found"):
        run_project_analysis_script(
            project_folder=temp_project_folder,
            script_path="analysis/does_not_exist.py",
        )


@pytest.mark.unit
def test_analysis_runner_rejects_script_outside_analysis(temp_project_folder):
    outside = temp_project_folder / "not_analysis.py"
    outside.write_text("print('nope')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be under analysis/"):
        run_project_analysis_script(
            project_folder=temp_project_folder,
            script_path=outside,
        )


@pytest.mark.unit
def test_analysis_runner_rejects_non_py_script(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    bad = analysis_dir / "not_python.txt"
    bad.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a .py file"):
        run_project_analysis_script(
            project_folder=temp_project_folder,
            script_path="analysis/not_python.txt",
        )


@pytest.mark.unit
def test_analysis_runner_records_nonzero_returncode_and_created_files(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    script_text = (
        "from pathlib import Path\n"
        "p = Path('.')\n"
        "(p / 'outputs' / 'tables').mkdir(parents=True, exist_ok=True)\n"
        "(p / 'outputs' / 'tables' / 'partial.csv').write_text('x,y\\n3,4\\n', encoding='utf-8')\n"
        "raise SystemExit(2)\n"
    )
    (analysis_dir / "fail.py").write_text(script_text, encoding="utf-8")

    result = run_project_analysis_script(
        project_folder=temp_project_folder,
        script_path="analysis/fail.py",
    )

    assert result.success is False
    assert result.returncode == 2

    payload = json.loads((temp_project_folder / "outputs" / "artifacts.json").read_text(encoding="utf-8"))
    assert payload["result"]["success"] is False
    assert payload["result"]["returncode"] == 2
    assert payload["created_files"] == [
        "outputs/artifacts.json",
        "outputs/tables/partial.csv",
    ]


@pytest.mark.unit
def test_analysis_runner_records_timeout(temp_project_folder):
    analysis_dir = temp_project_folder / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    script_text = "import time\ntime.sleep(2)\n"
    (analysis_dir / "sleep.py").write_text(script_text, encoding="utf-8")

    result = run_project_analysis_script(
        project_folder=temp_project_folder,
        script_path="analysis/sleep.py",
        timeout_seconds=1,
    )

    assert result.success is False
    assert result.returncode == -1
    assert "timed out" in result.stderr

    payload = json.loads((temp_project_folder / "outputs" / "artifacts.json").read_text(encoding="utf-8"))
    assert payload["result"]["success"] is False
    assert payload["result"]["returncode"] == -1
    assert "timed out" in payload["result"]["stderr"]
    assert payload["created_files"] == ["outputs/artifacts.json"]
