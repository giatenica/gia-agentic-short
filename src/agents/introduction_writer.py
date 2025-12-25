"""Introduction section writer.

Issue #85 scope:
- Deterministic, filesystem-first section generation.
- Constrained to canonical citation keys from bibliography/citations.json.
- Evidence-backed by sources/*/evidence.json.

This implementation is deliberately conservative:
- It does not call the LLM.
- It blocks or downgrades when required artifacts are missing, depending on config.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from loguru import logger

from src.agents.base import AgentResult, BaseAgent
from src.agents.section_writer import resolve_section_output_path
from src.citations.registry import load_citations
from src.llm.claude_client import TaskType
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.validation import validate_project_folder


OnFailureAction = Literal["block", "downgrade"]


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


@dataclass(frozen=True)
class IntroductionWriterConfig:
    enabled: bool = True
    on_missing_citation: OnFailureAction = "downgrade"
    on_missing_evidence: OnFailureAction = "downgrade"
    require_verified_citations: bool = False
    max_sources: int = 10
    max_quotes_per_source: int = 1

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "IntroductionWriterConfig":
        raw = context.get("introduction_writer")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", True))
        on_missing_citation = raw.get("on_missing_citation", "downgrade")
        on_missing_evidence = raw.get("on_missing_evidence", "downgrade")
        require_verified = bool(raw.get("require_verified_citations", False))

        if on_missing_citation not in ("block", "downgrade"):
            on_missing_citation = "downgrade"
        if on_missing_evidence not in ("block", "downgrade"):
            on_missing_evidence = "downgrade"

        max_sources = int(raw.get("max_sources", 10))
        max_quotes = int(raw.get("max_quotes_per_source", 1))
        if max_sources < 1:
            max_sources = 1
        if max_quotes < 1:
            max_quotes = 1

        return cls(
            enabled=enabled,
            on_missing_citation=on_missing_citation,
            on_missing_evidence=on_missing_evidence,
            require_verified_citations=require_verified,
            max_sources=max_sources,
            max_quotes_per_source=max_quotes,
        )


def _iter_source_evidence_files(project_folder: Path) -> List[Path]:
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


def _load_project_research_question(project_folder: Path) -> Optional[str]:
    path = project_folder / "project.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    rq = obj.get("research_question")
    if isinstance(rq, str) and rq.strip():
        return rq.strip()
    return None


class IntroductionWriterAgent(BaseAgent):
    """Deterministic Introduction writer constrained by evidence and citations."""

    def __init__(self, client=None):
        system_prompt = (
            "You write an Introduction section for an academic paper. "
            "Only cite canonical keys and only make claims supported by evidence items. "
            "If required artifacts are missing, you must stop or downgrade based on config."
        )
        super().__init__(
            name="IntroductionWriter",
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

        cfg = IntroductionWriterConfig.from_context(context)
        section_id = str(context.get("section_id") or "introduction").strip() or "introduction"
        section_title = str(context.get("section_title") or "Introduction").strip() or "Introduction"

        output_path, rel = resolve_section_output_path(pf, section_id=section_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cfg.enabled:
            latex = f"\\section{{{_latex_escape(section_title)}}}\n% Introduction writer disabled\n"
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

        citations = load_citations(pf, validate=True)
        by_key = _citation_index(citations)

        evidence_files = _iter_source_evidence_files(pf)
        if not evidence_files and cfg.on_missing_evidence == "block":
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error="No sources/*/evidence.json files found",
            )

        rq = _load_project_research_question(pf)

        missing_citation_sources: list[str] = []
        unverified_keys: list[str] = []
        used_keys: list[str] = []
        sources_emitted = 0

        chunks: list[str] = [f"\\section{{{_latex_escape(section_title)}}}"]

        if rq:
            chunks.append(f"This paper studies: {_latex_escape(rq)}.")
        else:
            chunks.append("This paper introduces the research question and background.")

        if not evidence_files:
            chunks.append("% Introduction writer downgraded due to missing evidence")
            chunks.append("\\textit{Evidence is not yet available; statements are non-definitive.}")
        else:
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
                citation_key_input = str(citation_key_raw or "").strip()
                citation_key_lookup = citation_key_input.lower()

                cite_tex = ""
                if citation_key_lookup:
                    record = by_key.get(citation_key_lookup)
                    if record is None:
                        missing_citation_sources.append(source_id)
                    else:
                        canonical_key = str(record.get("citation_key") or "").strip()
                        status = str(record.get("status") or "")
                        if cfg.require_verified_citations and status != "verified":
                            unverified_keys.append(canonical_key or citation_key_input)
                            if cfg.on_missing_citation == "block":
                                return AgentResult(
                                    agent_name=self.name,
                                    task_type=self.task_type,
                                    model_tier=self.model_tier,
                                    success=False,
                                    content="",
                                    error=f"Unverified citation key blocked: {canonical_key or citation_key_input}",
                                )
                        else:
                            used_keys.append(canonical_key or citation_key_input)
                            cite_tex = f"\\cite{{{canonical_key or citation_key_input}}}"
                else:
                    missing_citation_sources.append(source_id)

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
                    if excerpt:
                        chunks.append("\\begin{quote}")
                        chunks.append(excerpt)
                        chunks.append("\\end{quote}")

                sources_emitted += 1

        latex = "\n".join(chunks)
        if not latex.endswith("\n"):
            latex += "\n"

        output_path.write_text(latex, encoding="utf-8")

        metadata = {
            "enabled": True,
            "action": "downgrade" if (missing_citation_sources or not evidence_files) else "pass",
            "missing_citation_sources": sorted(set(missing_citation_sources)),
            "unverified_citation_keys": sorted(set(unverified_keys)),
            "used_citation_keys": sorted(set(used_keys)),
            "evidence_file_count": len(evidence_files),
        }

        return AgentResult(
            agent_name=self.name,
            task_type=self.task_type,
            model_tier=self.model_tier,
            success=True,
            content=latex,
            structured_data={
                "section_id": section_id,
                "output_relpath": rel,
                "metadata": metadata,
            },
        )
