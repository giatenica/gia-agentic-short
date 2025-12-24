import json

import pytest

from src.agents.referee_review import RefereeReviewAgent
from src.citations.registry import make_minimal_citation_record, save_citations


def _write_evidence(project_folder, source_id: str, *, n_items: int):
    sources_dir = project_folder / "sources" / source_id
    sources_dir.mkdir(parents=True, exist_ok=True)

    evidence = []
    for i in range(n_items):
        evidence.append(
            {
                "schema_version": "1.0",
                "evidence_id": f"ev_{i}",
                "source_id": source_id,
                "kind": "quote",
                "locator": {"type": "file", "value": "paper.pdf"},
                "excerpt": "A quoted fact",
                "created_at": "2025-01-01T00:00:00Z",
                "parser": {"name": "test"},
            }
        )

    (sources_dir / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_section(project_folder, relpath: str, text: str):
    p = project_folder / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_referee_review_passes_when_citations_known_and_evidence_ok(temp_project_folder):
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

    _write_evidence(temp_project_folder, "src_1", n_items=2)
    _write_section(
        temp_project_folder,
        "outputs/sections/intro.tex",
        "\\section{Intro}\\nText \\cite{Smith2020}\\n",
    )

    agent = RefereeReviewAgent()
    result = await agent.execute(
        {
            "project_folder": str(temp_project_folder),
            "section_relpaths": ["outputs/sections/intro.tex"],
            "source_citation_map": {"src_1": "Smith2020"},
            "referee_review": {"min_evidence_items_per_cited_source": 1, "require_verified_citations": True},
        }
    )

    assert result.success is True
    assert result.structured_data["summary"]["passed"] is True
    checks = {c["check"]: c for c in result.structured_data["checklist"]}
    assert checks["citations_known_keys"]["passed"] is True
    assert checks["evidence_coverage"]["passed"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_referee_review_fails_on_unknown_citation_key(temp_project_folder):
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

    _write_section(
        temp_project_folder,
        "outputs/sections/intro.tex",
        "\\section{Intro}\\nText \\cite{Nope2025}\\n",
    )

    agent = RefereeReviewAgent()
    result = await agent.execute(
        {
            "project_folder": str(temp_project_folder),
            "section_relpaths": ["outputs/sections/intro.tex"],
            "referee_review": {"on_unknown_citation": "block"},
        }
    )

    assert result.success is False
    summary = result.structured_data["summary"]
    assert "Nope2025" in summary["unknown_citation_keys"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_referee_review_fails_on_insufficient_evidence_when_blocking(temp_project_folder):
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

    _write_evidence(temp_project_folder, "src_1", n_items=0)
    _write_section(
        temp_project_folder,
        "outputs/sections/intro.tex",
        "\\section{Intro}\\nText \\cite{Smith2020}\\n",
    )

    agent = RefereeReviewAgent()
    result = await agent.execute(
        {
            "project_folder": str(temp_project_folder),
            "section_relpaths": ["outputs/sections/intro.tex"],
            "source_citation_map": {"src_1": "Smith2020"},
            "referee_review": {
                "min_evidence_items_per_cited_source": 1,
                "on_insufficient_evidence": "block",
            },
        }
    )

    assert result.success is False
    checks = {c["check"]: c for c in result.structured_data["checklist"]}
    assert checks["evidence_coverage"]["severity"] == "error"
