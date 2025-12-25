import json

import pytest

from src.agents.introduction_writer import IntroductionWriterAgent
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
async def test_introduction_writer_writes_deterministic_output(temp_project_folder):
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

    agent = IntroductionWriterAgent()
    ctx = {
        "project_folder": str(temp_project_folder),
        "section_id": "introduction",
        "section_title": "Introduction",
        "source_citation_map": {"src_1": "Smith2020"},
    }

    r1 = await agent.execute(ctx)
    assert r1.success is True
    assert r1.structured_data["output_relpath"] == "outputs/sections/introduction.tex"

    out_path = temp_project_folder / r1.structured_data["output_relpath"]
    content1 = out_path.read_text(encoding="utf-8")

    r2 = await agent.execute(ctx)
    assert r2.success is True
    content2 = out_path.read_text(encoding="utf-8")

    assert content1 == content2
    assert "\\section{Introduction}" in content1
    assert "\\cite{Smith2020}" in content1
    assert content1.endswith("\n")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_introduction_writer_blocks_on_missing_evidence(temp_project_folder):
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

    agent = IntroductionWriterAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "introduction_writer": {"on_missing_evidence": "block"},
        "source_citation_map": {},
    }

    result = await agent.execute(ctx)
    assert result.success is False
    assert (temp_project_folder / "outputs/sections/introduction.tex").exists() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_introduction_writer_blocks_on_missing_citation(temp_project_folder):
    _write_evidence(temp_project_folder, "src_1")

    agent = IntroductionWriterAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "source_citation_map": {"src_1": "Unknown2025"},
        "introduction_writer": {"on_missing_citation": "block"},
    }

    result = await agent.execute(ctx)
    assert result.success is False
    assert "Missing canonical citation key" in (result.error or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_introduction_writer_blocks_on_unverified_citation_when_required(temp_project_folder):
    save_citations(
        temp_project_folder,
        [
            make_minimal_citation_record(
                citation_key="Smith2020",
                title="Test Paper",
                authors=["Smith"],
                year=2020,
                status="unverified",
            )
        ],
        validate=True,
    )
    _write_evidence(temp_project_folder, "src_1")

    agent = IntroductionWriterAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "source_citation_map": {"src_1": "Smith2020"},
        "introduction_writer": {
            "require_verified_citations": True,
            "on_missing_citation": "block",
        },
    }

    result = await agent.execute(ctx)
    assert result.success is False
    assert "Unverified citation key blocked" in (result.error or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_introduction_writer_disabled_writes_section(temp_project_folder):
    agent = IntroductionWriterAgent(client=None)
    ctx = {
        "project_folder": str(temp_project_folder),
        "introduction_writer": {"enabled": False},
    }

    result = await agent.execute(ctx)
    assert result.success is True

    out_path = temp_project_folder / "outputs/sections/introduction.tex"
    assert out_path.exists()
    tex = out_path.read_text(encoding="utf-8")
    assert "\\section{Introduction}" in tex
    assert result.structured_data["metadata"]["enabled"] is False
