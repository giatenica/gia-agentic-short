import json
from pathlib import Path

import pytest

from src.evaluation.suite_runner import EvaluationSuiteConfig, load_test_queries, run_evaluation_suite


@pytest.mark.unit
def test_load_test_queries_parses_builtin_dataset():
    path = Path(__file__).parent.parent / "evaluation" / "test_queries.json"
    queries = load_test_queries(path)

    assert len(queries) > 0
    assert all(q.id and q.title and q.research_question for q in queries)


@pytest.mark.unit
def test_load_test_queries_rejects_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_test_queries(p)


@pytest.mark.unit
def test_load_test_queries_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dupes.json"
    p.write_text(
        json.dumps(
            [
                {"id": "x", "title": "T1", "research_question": "RQ1"},
                {"id": "x", "title": "T2", "research_question": "RQ2"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_test_queries(p)


@pytest.mark.unit
def test_load_test_queries_rejects_empty_or_missing_fields(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_test_queries(p)

    p2 = tmp_path / "missing.json"
    p2.write_text(json.dumps([{"id": "x", "title": "T"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_test_queries(p2)


@pytest.mark.unit
def test_load_test_queries_rejects_unsafe_ids(tmp_path):
    p = tmp_path / "unsafe.json"
    p.write_text(
        json.dumps(
            [
                {"id": "../evil", "title": "T", "research_question": "RQ"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_test_queries(p)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_evaluation_suite_dry_writes_report_and_per_query_projects(tmp_path):
    queries_path = Path(__file__).parent.parent / "evaluation" / "test_queries.json"
    queries = load_test_queries(queries_path)

    cfg = EvaluationSuiteConfig(mode="dry", output_root=tmp_path / "outputs")
    report, report_path = await run_evaluation_suite(queries=queries[:2], config=cfg, run_id="testrun")

    assert report.schema_version == "1.0"
    assert report.mode == "dry"
    assert report.queries_total == 2
    assert report.queries_success == 2
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["queries_total"] == 2
    assert len(payload["results"]) == 2

    r0 = payload["results"][0]
    assert r0["success"] is True
    assert r0["skipped"] is True
    assert "project_folder" in r0
    assert any(p.endswith("project.json") for p in r0["created_files"])
