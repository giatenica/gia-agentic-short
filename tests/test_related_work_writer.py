import json

import pytest

from src.agents.related_work_writer import RelatedWorkWriterAgent
from src.citations.registry import make_minimal_citation_record, save_citations


def _write_evidence(project_folder, source_id: str, *, excerpt: str = "A quoted fact"):
    sources_dir = project_folder / "sources" / source_id
    sources_dir.mkdir(parents=True, exist_ok=True)

    evidence = [
        {
            "schema_version": "1.0",
            "evidence_id": "ev_1",
            "source_id": source_id,
            "kind": "quote",
            "locator": {"type": "file", "value": "paper.pdf"},
            "excerpt": excerpt,
            "created_at": "2025-01-01T00:00:00Z",
            "parser": {"name": "test"},
        }
    ]

    (sources_dir / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_related_work_writer_cites_canonical_keys(temp_project_folder):
    save_citations(
        temp_project_folder,
        [
            make_minimal_citation_record(
                citation_key="Smith2020",
                title="Test Paper",
                authors=["Smith"],
                year=2020,
                status="verified",
            )
        ],
        validate=True,
    )

    _write_evidence(temp_project_folder, "src_1")

    agent = RelatedWorkWriterAgent()
    result = await agent.execute(
        {
            "project_folder": str(temp_project_folder),
            "section_id": "related_work",
            "section_title": "Related Work",
            "source_citation_map": {"src_1": "Smith2020"},
            "related_work_writer": {"require_verified_citations": True},
        }
    )

    assert result.success is True
    out_rel = result.structured_data.get("output_relpath")
    assert out_rel == "outputs/sections/related_work.tex"

    out_text = (temp_project_folder / out_rel).read_text(encoding="utf-8")
    assert "\\cite{Smith2020}" in out_text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_related_work_writer_downgrades_when_citation_missing(temp_project_folder):
    save_citations(
        temp_project_folder,
        [
            make_minimal_citation_record(
                citation_key="smith2020",
                title="Test Paper",
                authors=["Smith"],
                year=2020,
                status="verified",
            )
        ],
        validate=True,
    )

    _write_evidence(temp_project_folder, "src_1")

    agent = RelatedWorkWriterAgent()
    result = await agent.execute(
        {
            "project_folder": str(temp_project_folder),
            "section_id": "related_work",
            "section_title": "Related Work",
            "related_work_writer": {"on_missing_citation": "downgrade"},
        }
    )

    assert result.success is True
    meta = result.structured_data.get("metadata")
    assert "src_1" in meta.get("missing_citation_sources", [])

    out_rel = result.structured_data.get("output_relpath")
    out_text = (temp_project_folder / out_rel).read_text(encoding="utf-8")
    assert "\\cite{" not in out_text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_related_work_writer_blocks_when_citation_missing(temp_project_folder):
    save_citations(
        temp_project_folder,
        [
            make_minimal_citation_record(
                citation_key="smith2020",
                title="Test Paper",
                authors=["Smith"],
                year=2020,
                status="verified",
            )
        ],
        validate=True,
    )

    _write_evidence(temp_project_folder, "src_1")

    agent = RelatedWorkWriterAgent()
    result = await agent.execute(
        {
            "project_folder": str(temp_project_folder),
            "section_id": "related_work",
            "section_title": "Related Work",
            "related_work_writer": {"on_missing_citation": "block"},
        }
    )

    assert result.success is False
    assert "Missing canonical citation key" in (result.error or "")
