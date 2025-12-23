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
