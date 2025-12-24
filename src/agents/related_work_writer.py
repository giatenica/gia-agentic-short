"""Related Work section writer.

Sprint 4 MVP:
- Deterministic, filesystem-first section generation.
- Constrained to canonical citation keys from bibliography/citations.json.
- Evidence-backed by sources/*/evidence.json.

This implementation is deliberately conservative:
- It does not call the LLM.
- It blocks or downgrades when citation linkage is missing, depending on config.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from loguru import logger

from src.agents.base import AgentResult, BaseAgent
from src.agents.section_writer import resolve_section_output_path
from src.citations.registry import load_citations
from src.llm.claude_client import TaskType
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.validation import validate_project_folder


OnFailureAction = Literal["block", "downgrade"]


@dataclass(frozen=True)
class RelatedWorkWriterConfig:
    enabled: bool = True
    on_missing_citation: OnFailureAction = "downgrade"
    on_unverified_citation: OnFailureAction = "downgrade"
    require_verified_citations: bool = False
    max_sources: int = 25
    max_quotes_per_source: int = 2

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "RelatedWorkWriterConfig":
        raw = context.get("related_work_writer")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", True))
        on_missing = raw.get("on_missing_citation", "downgrade")
        on_unverified = raw.get("on_unverified_citation", "downgrade")
        require_verified = bool(raw.get("require_verified_citations", False))

        if on_missing not in ("block", "downgrade"):
            on_missing = "downgrade"
        if on_unverified not in ("block", "downgrade"):
            on_unverified = "downgrade"

        max_sources = int(raw.get("max_sources", 25))
        max_quotes = int(raw.get("max_quotes_per_source", 2))
        if max_sources < 1:
            max_sources = 1
        if max_quotes < 1:
            max_quotes = 1

        return cls(
            enabled=enabled,
            on_missing_citation=on_missing,
            on_unverified_citation=on_unverified,
            require_verified_citations=require_verified,
            max_sources=max_sources,
            max_quotes_per_source=max_quotes,
        )


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _iter_source_evidence_files(project_folder: Path) -> Iterable[Path]:
    sources_dir = project_folder / "sources"
    if not sources_dir.exists() or not sources_dir.is_dir():
        return []

    paths: list[Path] = []
    for p in sources_dir.rglob("evidence.json"):
        if p.is_file():
            paths.append(p)

    return sorted(paths)


def _load_evidence_items(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Evidence file must contain a list: {path}")
    return [p for p in payload if isinstance(p, dict)]


def _citation_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_key: dict[str, Dict[str, Any]] = {}
    for r in records:
        key = str(r.get("citation_key") or "").strip()
        if key:
            by_key[key.lower()] = r
    return by_key


class RelatedWorkWriterAgent(BaseAgent):
    """Deterministic Related Work writer constrained by evidence and citations."""

    def __init__(self, client=None):
        system_prompt = (
            "You write a Related Work section for an academic paper. "
            "Only cite canonical keys and only make claims supported by evidence items. "
            "If required artifacts are missing, you must stop or downgrade based on config."
        )
        super().__init__(
            name="RelatedWorkWriter",
            task_type=TaskType.DOCUMENT_CREATION,
            system_prompt=system_prompt,
            client=client,
        )

    async def execute(self, context: dict) -> AgentResult:
        project_folder = context.get("project_folder")
        if not isinstance(project_folder, str) or not project_folder:
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error="Missing required context: project_folder",
            )

        pf = validate_project_folder(project_folder)
        ensure_project_outputs_layout(pf)

        cfg = RelatedWorkWriterConfig.from_context(context)
        section_id = str(context.get("section_id") or "related_work").strip() or "related_work"
        section_title = str(context.get("section_title") or "Related Work").strip() or "Related Work"

        if not cfg.enabled:
            output_path, rel = resolve_section_output_path(pf, section_id=section_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            latex = f"\\section{{{_latex_escape(section_title)}}}\n% Related Work writer disabled\n"
            output_path.write_text(latex, encoding="utf-8")
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content=latex,
                structured_data={
                    "section_id": section_id,
                    "output_relpath": rel,
                    "metadata": {"enabled": False},
                },
            )

        source_citation_map = context.get("source_citation_map")
        if not isinstance(source_citation_map, dict):
            source_citation_map = {}

        citation_records = load_citations(pf, validate=True)
        by_key = _citation_index(citation_records)

        evidence_files = list(_iter_source_evidence_files(pf))
        if not evidence_files:
            msg = "No sources/*/evidence.json files found"
            if cfg.on_missing_citation == "block":
                return AgentResult(
                    agent_name=self.name,
                    task_type=self.task_type,
                    model_tier=self.model_tier,
                    success=False,
                    content="",
                    error=msg,
                )

        missing_sources: list[str] = []
        unverified_keys: list[str] = []
        used_keys: list[str] = []

        chunks: list[str] = [f"\\section{{{_latex_escape(section_title)}}}"]

        sources_emitted = 0
        for evidence_path in evidence_files:
            if sources_emitted >= cfg.max_sources:
                break

            try:
                items = _load_evidence_items(evidence_path)
            except Exception as e:
                logger.debug(f"Skipping unreadable evidence file: {evidence_path}: {type(e).__name__}: {e}")
                continue

            quote_items = [i for i in items if str(i.get("kind") or "") == "quote"]
            quote_items = sorted(quote_items, key=lambda d: str(d.get("evidence_id") or ""))
            if not quote_items:
                continue

            source_id = str(quote_items[0].get("source_id") or "").strip() or evidence_path.parent.name

            citation_key_raw = source_citation_map.get(source_id)
            citation_key = str(citation_key_raw or "").strip().lower()

            cite_tex = ""
            if citation_key:
                record = by_key.get(citation_key)
                if record is None:
                    missing_sources.append(source_id)
                    citation_key = ""
                else:
                    if cfg.require_verified_citations and str(record.get("status") or "") != "verified":
                        unverified_keys.append(citation_key)
                        if cfg.on_unverified_citation == "block":
                            return AgentResult(
                                agent_name=self.name,
                                task_type=self.task_type,
                                model_tier=self.model_tier,
                                success=False,
                                content="",
                                error=f"Unverified citation key blocked: {citation_key}",
                            )
                        citation_key = ""
                    else:
                        used_keys.append(citation_key)
                        cite_tex = f"\\cite{{{_latex_escape(citation_key)}}}"
            else:
                missing_sources.append(source_id)

            if not cite_tex and cfg.on_missing_citation == "block":
                return AgentResult(
                    agent_name=self.name,
                    task_type=self.task_type,
                    model_tier=self.model_tier,
                    success=False,
                    content="",
                    error=f"Missing canonical citation key for source_id={source_id}",
                )

            heading = _latex_escape(source_id)
            if cite_tex:
                chunks.append(f"\\subsection{{{heading} {cite_tex}}}")
            else:
                chunks.append(f"\\subsection{{{heading}}}")
                chunks.append("\\textit{Note: citation linkage missing; language is non-definitive.}")

            for qi in quote_items[: cfg.max_quotes_per_source]:
                excerpt = _latex_escape(str(qi.get("excerpt") or "").strip())
                locator = qi.get("locator") if isinstance(qi.get("locator"), dict) else {}
                loc_type = _latex_escape(str(locator.get("type") or ""))
                loc_val = _latex_escape(str(locator.get("value") or ""))

                if excerpt:
                    chunks.append("\\begin{quote}")
                    chunks.append(excerpt)
                    chunks.append("\\end{quote}")
                if loc_type or loc_val:
                    chunks.append(f"\\noindent\\textit{{Locator: {loc_type} {loc_val}}}")

            sources_emitted += 1

        if missing_sources and cfg.on_missing_citation == "block":
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=f"Missing citation linkage for sources: {sorted(set(missing_sources))}",
            )

        if sources_emitted == 0:
            chunks.append("\\textit{No quote evidence items were available to generate Related Work.}")

        latex = "\n".join(chunks).rstrip() + "\n"

        output_path, rel = resolve_section_output_path(pf, section_id=section_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex, encoding="utf-8")

        structured: Dict[str, Any] = {
            "section_id": section_id,
            "output_relpath": rel,
            "metadata": {
                "used_citation_keys": sorted(set(used_keys)),
                "missing_citation_sources": sorted(set(missing_sources)),
                "unverified_citation_keys": sorted(set(unverified_keys)),
                "sources_emitted": sources_emitted,
            },
        }

        return AgentResult(
            agent_name=self.name,
            task_type=self.task_type,
            model_tier=self.model_tier,
            success=True,
            content=latex,
            structured_data=structured,
        )
