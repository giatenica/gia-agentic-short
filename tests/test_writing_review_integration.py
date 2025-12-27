import json
import os
from unittest.mock import patch

import pytest

from src.agents.writing_review_integration import run_writing_review_stage
from src.agents.base import AgentResult
from src.llm.claude_client import TaskType, ModelTier
from src.citations.registry import make_minimal_citation_record, save_citations


def _write_claims(project_folder, *, metric_key: str):
    claims_dir = project_folder / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "schema_version": "1.0",
            "claim_id": "c1",
            "kind": "computed",
            "statement": "A computed statement",
            "metric_keys": [metric_key],
            "created_at": "2025-01-01T00:00:00Z",
        }
    ]
    (claims_dir / "claims.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_tex(project_folder, relpath: str, tex: str):
    p = project_folder / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tex, encoding="utf-8")


class _StubAgent:
    def __init__(self, execute_fn):
        self._execute_fn = execute_fn

    async def execute(self, context):
        return self._execute_fn(context)


def _stub_writer(project_folder, *, relpath: str):
    def _execute(_ctx):
        p = project_folder / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\\section{Intro}\n", encoding="utf-8")
        return AgentResult(
            agent_name="writer",
            task_type=TaskType.DOCUMENT_CREATION,
            model_tier=ModelTier.SONNET,
            success=True,
            content="",
            structured_data={"output_relpath": relpath},
        )

    return _StubAgent(_execute)


def _stub_referee_success():
    def _execute(_ctx):
        return AgentResult(
            agent_name="referee",
            task_type=TaskType.CLASSIFICATION,
            model_tier=ModelTier.SONNET,
            success=True,
            content="",
            structured_data={"checklist": [], "summary": {}},
        )

    return _StubAgent(_execute)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_writing_stage_blocks_on_citation_gate(temp_project_folder):
    save_citations(
        temp_project_folder,
        [
            make_minimal_citation_record(
                citation_key="Known2020",
                title="Known",
                authors=["A"],
                year=2020,
                status="verified",
            )
        ],
        validate=True,
    )

    # Citation gate scans project docs. Create a doc with an unknown cite.
    _write_tex(temp_project_folder, "drafts/bad.tex", "\\section{X} \\cite{Nope2025}\\n")

    ctx = {
        "project_folder": str(temp_project_folder),
        "writing_review": {
            "enabled": True,
            "writers": [{"agent_id": "A17", "section_id": "intro", "section_title": "Intro"}],
        },
        "citation_gate": {"enabled": True, "on_missing": "block"},
    }

    result = await run_writing_review_stage(ctx)

    assert result.success is False
    assert result.needs_revision is True
    assert result.written_section_relpaths == []
    assert result.gates["citation_gate"]["ok"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_writing_stage_blocks_on_computation_gate(temp_project_folder):
    # Enable computation gate with a computed claim referencing a missing metric.
    _write_claims(temp_project_folder, metric_key="m1")

    ctx = {
        "project_folder": str(temp_project_folder),
        "writing_review": {
            "enabled": True,
            "writers": [{"agent_id": "A17", "section_id": "intro", "section_title": "Intro"}],
        },
        "computation_gate": {"enabled": True, "on_missing_metrics": "block"},
    }

    result = await run_writing_review_stage(ctx)

    assert result.success is False
    assert result.needs_revision is True
    assert result.written_section_relpaths == []
    assert result.gates["computation_gate"]["ok"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_writing_stage_blocks_on_literature_gate(temp_project_folder):
    # Enable literature gate with thresholds that are not met.
    ctx = {
        "project_folder": str(temp_project_folder),
        "writing_review": {
            "enabled": True,
            "writers": [{"agent_id": "A17", "section_id": "intro", "section_title": "Intro"}],
        },
        "literature_gate": {
            "enabled": True,
            "on_failure": "block",
            "min_verified_citations": 1,
            "min_evidence_items_total": 1,
        },
    }

    result = await run_writing_review_stage(ctx)

    assert result.success is False
    assert result.needs_revision is True
    assert result.written_section_relpaths == []
    assert result.gates["literature_gate"]["ok"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_writing_stage_blocks_on_analysis_gate(temp_project_folder):
    # Enable analysis gate requiring metrics; the temp project has no outputs/metrics.json.
    ctx = {
        "project_folder": str(temp_project_folder),
        "writing_review": {
            "enabled": True,
            "writers": [{"agent_id": "A20", "section_id": "results", "section_title": "Results"}],
        },
        "analysis_gate": {
            "enabled": True,
            "on_failure": "block",
            "min_metrics": 1,
        },
    }

    result = await run_writing_review_stage(ctx)

    assert result.success is False
    assert result.needs_revision is True
    assert result.written_section_relpaths == []
    assert result.gates["analysis_gate"]["ok"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_writing_stage_deletes_outputs_on_referee_failure(temp_project_folder):
    save_citations(
        temp_project_folder,
        [
            make_minimal_citation_record(
                citation_key="Known2020",
                title="Known",
                authors=["A"],
                year=2020,
                status="verified",
            )
        ],
        validate=True,
    )

    # Precreate a file with an unknown citation to force referee failure.
    _write_tex(temp_project_folder, "outputs/sections/bad.tex", "\\section{Bad} \\cite{Nope2025}\\n")

    ctx = {
        "project_folder": str(temp_project_folder),
        "writing_review": {
            "enabled": True,
            "writers": [{"agent_id": "A17", "section_id": "intro", "section_title": "Intro"}],
            "review_section_relpaths": ["outputs/sections/bad.tex"],
        },
        "referee_review": {"on_unknown_citation": "block"},
    }

    result = await run_writing_review_stage(ctx)

    assert result.success is False
    assert result.needs_revision is True

    # The writer output should be removed when review fails.
    assert (temp_project_folder / "outputs/sections/intro.tex").exists() is False
    assert result.review is not None
    assert result.review["structured"]["summary"]["unknown_citation_keys"] == ["Nope2025"]


@pytest.mark.unit
@pytest.mark.asyncio
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
async def test_writing_stage_default_gates_warn_do_not_block(temp_project_folder):
    # Create a project text file with an unknown citation; default policy should downgrade, not block.
    _write_tex(temp_project_folder, "drafts/bad.tex", "\\section{X} \\cite{Nope2025}\\n")

    # Create a computed claim referencing a missing metric; default should downgrade, not block.
    _write_claims(temp_project_folder, metric_key="m1")

    def create_agent(agent_id: str):
        if agent_id == "A17":
            return _stub_writer(temp_project_folder, relpath="outputs/sections/intro.tex")
        if agent_id == "A19":
            return _stub_referee_success()
        return None

    ctx = {
        "project_folder": str(temp_project_folder),
        "source_ids": ["s1"],
        "writing_review": {
            "enabled": True,
            "writers": [{"agent_id": "A17", "section_id": "intro", "section_title": "Intro"}],
            "review_agent_id": "A19",
        },
        # No explicit evidence_gate/citation_gate/computation_gate configs.
    }

    with patch("src.agents.writing_review_integration.AgentRegistry.create_agent", side_effect=create_agent):
        result = await run_writing_review_stage(ctx)

    assert result.success is True
    assert result.needs_revision is False

    assert result.gates["citation_gate"]["enabled"] is True
    assert result.gates["citation_gate"]["action"] == "downgrade"
    assert "nope2025" in (result.gates["citation_gate"]["missing_keys"] or [])

    assert result.gates["computation_gate"]["enabled"] is True
    assert result.gates["computation_gate"]["action"] == "downgrade"
    assert "m1" in (result.gates["computation_gate"]["missing_metric_keys"] or [])

    assert result.gates["evidence_gate"]["enabled"] is True
    assert result.gates["evidence_gate"]["action"] == "downgrade"


@pytest.mark.unit
@pytest.mark.asyncio
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
async def test_writing_stage_can_switch_gates_to_blocking_mode(temp_project_folder):
    _write_tex(temp_project_folder, "drafts/bad.tex", "\\section{X} \\cite{Nope2025}\\n")

    ctx = {
        "project_folder": str(temp_project_folder),
        "writing_review": {
            "enabled": True,
            "writers": [{"agent_id": "A17", "section_id": "intro", "section_title": "Intro"}],
        },
        "citation_gate": {"enabled": True, "on_missing": "block"},
    }

    result = await run_writing_review_stage(ctx)

    assert result.success is False
    assert result.needs_revision is True
    assert result.gates["citation_gate"]["ok"] is False
