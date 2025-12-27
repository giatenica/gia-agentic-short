from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline.context import WorkflowContext


@pytest.mark.unit
def test_mark_checkpoint_ignores_blank_name(tmp_path: Path) -> None:
    ctx = WorkflowContext(project_folder=tmp_path)
    ctx.mark_checkpoint(" ")
    ctx.mark_checkpoint("")
    assert ctx.checkpoints == {}


@pytest.mark.unit
def test_mark_checkpoint_sets_timestamp(tmp_path: Path) -> None:
    ctx = WorkflowContext(project_folder=tmp_path)
    ctx.mark_checkpoint("start")
    assert "start" in ctx.checkpoints
    assert isinstance(ctx.checkpoints["start"], str)


@pytest.mark.unit
def test_record_phase_result_accepts_dict_and_marks_failure(tmp_path: Path) -> None:
    ctx = WorkflowContext(project_folder=tmp_path)

    ctx.record_phase_result("phase_1", {"success": False, "errors": ["nope"]})

    assert ctx.success is False
    assert "phase_1" in ctx.phase_results
    assert any("nope" in e for e in ctx.errors)


@pytest.mark.unit
def test_record_phase_result_uses_to_dict(tmp_path: Path) -> None:
    class Result:
        def to_dict(self):
            return {"success": True, "errors": []}

    ctx = WorkflowContext(project_folder=tmp_path)
    ctx.record_phase_result("phase_1", Result())

    assert ctx.success is True
    assert ctx.phase_results["phase_1"]["success"] is True


@pytest.mark.unit
def test_record_phase_result_fallback_repr(tmp_path: Path) -> None:
    class Result:
        success = True

    ctx = WorkflowContext(project_folder=tmp_path)
    ctx.record_phase_result("phase_x", Result())

    assert ctx.phase_results["phase_x"]["success"] is True
    assert "repr" in ctx.phase_results["phase_x"]


@pytest.mark.unit
def test_payload_roundtrip_preserves_fields(tmp_path: Path) -> None:
    ctx = WorkflowContext(project_folder=tmp_path)
    ctx.mark_checkpoint("start")
    ctx.record_phase_result("phase_1", {"success": True, "errors": []})

    payload = ctx.to_payload()
    ctx2 = WorkflowContext.from_payload(payload)

    assert ctx2.run_id == ctx.run_id
    assert ctx2.created_at == ctx.created_at
    assert ctx2.project_folder == tmp_path.resolve()
    assert ctx2.checkpoints.keys() == ctx.checkpoints.keys()
    assert ctx2.phase_results == ctx.phase_results


@pytest.mark.unit
def test_from_payload_filters_non_dict_phase_results(tmp_path: Path) -> None:
    payload = {
        "project_folder": str(tmp_path),
        "phase_results": {"ok": {"success": True}, "bad": "nope"},
    }

    ctx = WorkflowContext.from_payload(payload)
    assert "ok" in ctx.phase_results
    assert "bad" not in ctx.phase_results


@pytest.mark.unit
def test_write_json_and_read_json_roundtrip(tmp_path: Path) -> None:
    ctx = WorkflowContext(project_folder=tmp_path)
    ctx.mark_checkpoint("start")
    ctx.record_phase_result("phase_1", {"success": True, "errors": []})

    out = tmp_path / "ctx.json"
    ctx.write_json(out)

    ctx2 = WorkflowContext.read_json(out)
    assert ctx2 is not None
    assert ctx2.run_id == ctx.run_id
    assert ctx2.project_folder == tmp_path.resolve()


@pytest.mark.unit
def test_read_json_returns_none_on_missing_file(tmp_path: Path) -> None:
    assert WorkflowContext.read_json(tmp_path / "missing.json") is None


@pytest.mark.unit
def test_read_json_returns_none_on_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not-json}", encoding="utf-8")
    assert WorkflowContext.read_json(p) is None


@pytest.mark.unit
def test_read_json_returns_none_on_non_dict_json(tmp_path: Path) -> None:
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert WorkflowContext.read_json(p) is None
