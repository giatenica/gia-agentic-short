import json

import pytest

from src.agents.writing_review_integration import run_writing_review_stage
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
