import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.writing_review_integration import run_writing_review_stage
from src.agents.base import AgentResult
from src.llm.claude_client import TaskType, ModelTier


class _StubAgent:
    def __init__(self, execute_fn):
        self._execute_fn = execute_fn

    async def execute(self, context):
        return self._execute_fn(context)


@pytest.mark.unit
@pytest.mark.asyncio
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
async def test_revision_loop_reruns_only_targeted_sections(tmp_path: Path):
    # Minimal project structure expected by the writing stage
    project_folder = tmp_path
    (project_folder / "project.json").write_text("{}\n", encoding="utf-8")
    (project_folder / "outputs").mkdir(parents=True, exist_ok=True)

    # Two writers; each writes its own section.
    writers = [
        {"agent_id": "A17", "section_id": "intro", "section_title": "Introduction"},
        {"agent_id": "A18", "section_id": "methods", "section_title": "Methods"},
    ]

    state = {
        "intro_runs": 0,
        "methods_runs": 0,
        "review_runs": 0,
    }

    intro_rel = "sections/introduction.tex"
    methods_rel = "sections/methods.tex"

    def intro_execute(ctx):
        state["intro_runs"] += 1
        out = project_folder / intro_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r"Intro text\n", encoding="utf-8")
        return AgentResult(
            agent_name="intro",
            task_type=TaskType.DOCUMENT_CREATION,
            model_tier=ModelTier.SONNET,
            success=True,
            content="",
            structured_data={"output_relpath": intro_rel},
        )

    def methods_execute(ctx):
        state["methods_runs"] += 1
        out = project_folder / methods_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        # Put an unknown citation key so the referee can target this section
        out.write_text(r"Methods text\n\cite{BadKey}\n", encoding="utf-8")
        return AgentResult(
            agent_name="methods",
            task_type=TaskType.DOCUMENT_CREATION,
            model_tier=ModelTier.SONNET,
            success=True,
            content="",
            structured_data={"output_relpath": methods_rel},
        )

    def review_execute(ctx):
        state["review_runs"] += 1
        if state["review_runs"] == 1:
            # Fail with unknown citation key; loop should rerun only methods (contains BadKey)
            return AgentResult(
                agent_name="referee",
                task_type=TaskType.CLASSIFICATION,
                model_tier=ModelTier.SONNET,
                success=False,
                content="",
                structured_data={
                    "checklist": [
                        {
                            "check": "citations_known_keys",
                            "pass": False,
                            "details": {"unknown_citation_keys": ["BadKey"]},
                        }
                    ],
                    "summary": {"failed_checks": ["citations_known_keys"]},
                },
            )
        return AgentResult(
            agent_name="referee",
            task_type=TaskType.CLASSIFICATION,
            model_tier=ModelTier.SONNET,
            success=True,
            content="",
            structured_data={"checklist": [], "summary": {}},
        )

    def create_agent(agent_id: str):
        if agent_id == "A17":
            return _StubAgent(intro_execute)
        if agent_id == "A18":
            return _StubAgent(methods_execute)
        if agent_id == "A19":
            return _StubAgent(review_execute)
        return None

    context = {
        "project_folder": str(project_folder),
        "writing_review": {
            "enabled": True,
            "writers": writers,
            "review_agent_id": "A19",
            "max_iterations": 3,
        },
        "source_citation_map": {},
        "referee_review": {},
    }

    with patch("src.agents.writing_review_integration.AgentRegistry.create_agent", side_effect=create_agent):
        result = await run_writing_review_stage(context)

    assert result.success is True
    assert result.needs_revision is False
    # Intro written once; methods written twice due to targeting
    assert state["intro_runs"] == 1
    assert state["methods_runs"] == 2
    assert state["review_runs"] == 2

    hist_path = project_folder / "outputs" / "writing_review_history.json"
    assert hist_path.exists()
    payload = json.loads(hist_path.read_text(encoding="utf-8"))
    assert payload["max_iterations"] == 3
    assert len(payload["iterations"]) == 2
    assert payload["iterations"][0]["targets_next_relpaths"] == [methods_rel]


@pytest.mark.unit
@pytest.mark.asyncio
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
async def test_revision_loop_respects_max_iterations_and_removes_files(tmp_path: Path):
    project_folder = tmp_path
    (project_folder / "project.json").write_text("{}\n", encoding="utf-8")
    (project_folder / "outputs").mkdir(parents=True, exist_ok=True)

    writers = [
        {"agent_id": "A17", "section_id": "intro", "section_title": "Introduction"},
    ]

    intro_rel = "sections/introduction.tex"

    def intro_execute(ctx):
        out = project_folder / intro_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r"Intro text\n", encoding="utf-8")
        return AgentResult(
            agent_name="intro",
            task_type=TaskType.DOCUMENT_CREATION,
            model_tier=ModelTier.SONNET,
            success=True,
            content="",
            structured_data={"output_relpath": intro_rel},
        )

    def review_execute(ctx):
        return AgentResult(
            agent_name="referee",
            task_type=TaskType.CLASSIFICATION,
            model_tier=ModelTier.SONNET,
            success=False,
            content="",
            structured_data={
                "checklist": [
                    {
                        "check": "citations_known_keys",
                        "pass": False,
                        "details": {"unknown_citation_keys": ["BadKey"]},
                    }
                ],
                "summary": {"failed_checks": ["citations_known_keys"]},
            },
        )

    def create_agent(agent_id: str):
        if agent_id == "A17":
            return _StubAgent(intro_execute)
        if agent_id == "A19":
            return _StubAgent(review_execute)
        return None

    context = {
        "project_folder": str(project_folder),
        "writing_review": {
            "enabled": True,
            "writers": writers,
            "review_agent_id": "A19",
            "max_iterations": 2,
        },
        "source_citation_map": {},
        "referee_review": {},
    }

    with patch("src.agents.writing_review_integration.AgentRegistry.create_agent", side_effect=create_agent):
        result = await run_writing_review_stage(context)

    assert result.success is False
    assert result.needs_revision is True

    # On final failure, outputs should be removed
    assert not (project_folder / intro_rel).exists()

    hist_path = project_folder / "outputs" / "writing_review_history.json"
    assert hist_path.exists()
    payload = json.loads(hist_path.read_text(encoding="utf-8"))
    assert payload["max_iterations"] == 2
    assert len(payload["iterations"]) == 2
