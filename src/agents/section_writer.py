"""Section writer agents.

This module defines a minimal, deterministic section-writing interface.

Sprint 4 MVP scope:
- A base section writer agent that writes a LaTeX section to a deterministic
  on-disk location under the project folder.
- A stub implementation that does not call the LLM (used for tests and wiring).

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

from loguru import logger

from src.agents.base import AgentResult, BaseAgent
from src.llm.claude_client import TaskType
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.validation import validate_project_folder


DEFAULT_SECTIONS_RELATIVE_DIR = Path("outputs") / "sections"


def _slugify_section_id(section_id: str) -> str:
    """Convert a user-provided section id into a safe filename stem."""

    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in section_id.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "section"


def resolve_section_output_path(
    project_folder: str | Path,
    *,
    section_id: str,
    relative_dir: Path = DEFAULT_SECTIONS_RELATIVE_DIR,
) -> Tuple[Path, str]:
    """Return (absolute_path, relative_path_str) for a section output file."""

    pf = validate_project_folder(project_folder)
    safe_id = _slugify_section_id(section_id)

    output_dir = pf / relative_dir
    output_path = output_dir / f"{safe_id}.tex"

    rel = str(output_path.relative_to(pf))
    return output_path, rel


class SectionWriterAgent(BaseAgent, ABC):
    """Base class for section writer agents.

    Subclasses implement :meth:`render_section`.
    The base :meth:`execute` writes the content to disk deterministically.
    """

    def __init__(self, name: str, *, system_prompt: str):
        super().__init__(
            name=name,
            task_type=TaskType.DOCUMENT_CREATION,
            system_prompt=system_prompt,
            cache_ttl="ephemeral",
        )

    @abstractmethod
    async def render_section(self, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Return (latex, metadata) for the section."""

    async def execute(self, context: dict) -> AgentResult:
        project_folder = context.get("project_folder")
        if not project_folder:
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error="Missing required input: project_folder",
            )

        pf = validate_project_folder(project_folder)
        ensure_project_outputs_layout(pf)

        section_id = str(context.get("section_id") or "stub").strip() or "stub"
        output_path, rel = resolve_section_output_path(pf, section_id=section_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        latex, metadata = await self.render_section(context)

        if not latex.endswith("\n"):
            latex = latex + "\n"

        output_path.write_text(latex, encoding="utf-8")

        structured: dict[str, Any] = {
            "section_id": section_id,
            "output_relpath": rel,
            "metadata": metadata or {},
        }

        logger.info(f"Wrote section output: {rel}")

        return AgentResult(
            agent_name=self.name,
            task_type=self.task_type,
            model_tier=self.model_tier,
            success=True,
            content=latex,
            structured_data=structured,
        )


class StubSectionWriterAgent(SectionWriterAgent):
    """Minimal section writer that produces deterministic LaTeX output."""

    def __init__(self):
        super().__init__(
            name="SectionWriter",
            system_prompt=(
                "You generate LaTeX section text for an academic paper. "
                "Only use project artifacts provided in the context. "
                "If required evidence or citations are missing, return a minimal, "
                "non-definitive section and include a short TODO list."
            ),
        )

    async def render_section(self, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        section_title = str(context.get("section_title") or "Section").strip() or "Section"

        latex = "\n".join(
            [
                f"\\section{{{section_title}}}",
                "This is a stub section generated for wiring and tests.",
            ]
        )

        metadata = {
            "writer": "stub",
            "section_title": section_title,
        }

        return latex, metadata
