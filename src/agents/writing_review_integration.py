"""Writing + review integration helpers.

Sprint 4 Issue #54 MVP:
- Enforce pre-writing gates (when enabled): evidence gate, citation gate, computation gate.
- Run one or more section writers.
- Run referee-style review over generated sections.
- If review fails, stop and return a structured needs-revision payload.

This module is offline-friendly and deterministic for fixed inputs.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.agents.base import AgentResult
from src.agents.registry import AgentRegistry
from src.evidence.gates import EvidenceGateConfig, EvidenceGateError, enforce_evidence_gate
from src.citations.gates import CitationGateConfig, CitationGateError, enforce_citation_gate
from src.claims.gates import ComputationGateConfig, ComputationGateError, enforce_computation_gate
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.validation import validate_project_folder


@dataclass(frozen=True)
class WritingReviewStageResult:
    success: bool
    needs_revision: bool
    written_section_relpaths: List[str]
    gates: Dict[str, Any]
    review: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "needs_revision": self.needs_revision,
            "written_section_relpaths": self.written_section_relpaths,
            "gates": self.gates,
            "review": self.review,
            "error": self.error,
        }


def _remove_written_files(project_folder: Path, relpaths: List[str]) -> None:
    for rel in relpaths:
        try:
            path = project_folder / rel
            if path.exists() and path.is_file():
                path.unlink()
        except Exception as e:
            logger.debug(f"Failed to remove written file {rel}: {type(e).__name__}: {e}")


def _collect_writer_specs(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = context.get("writing_review")
    if not isinstance(raw, dict):
        return []

    writers = raw.get("writers")
    if not isinstance(writers, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in writers:
        if not isinstance(item, dict):
            continue
        agent_id = item.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id.strip():
            continue
        out.append(item)
    return out


def _review_section_relpaths(project_folder: Path, context: Dict[str, Any], written_relpaths: List[str]) -> List[str]:
    raw = context.get("writing_review")
    if isinstance(raw, dict):
        override = raw.get("review_section_relpaths")
        if isinstance(override, list) and all(isinstance(p, str) for p in override):
            return [p for p in override]

    # Default to just-written sections.
    return list(written_relpaths)


async def run_writing_review_stage(context: Dict[str, Any]) -> WritingReviewStageResult:
    """Run gated section writing plus referee review.

    Expects context keys:
    - project_folder (str)
    - writing_review (dict) with:
        - enabled (bool)
        - writers (list[dict]) with at least {agent_id, section_id?, section_title?, ...}
        - review_section_relpaths (optional list[str])
    - source_citation_map (optional dict)

    Gate configs are read from the standard keys used elsewhere:
    - evidence_gate
    - citation_gate
    - computation_gate

    Referee review config is passed via:
    - referee_review
    """

    raw = context.get("writing_review")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return WritingReviewStageResult(
            success=True,
            needs_revision=False,
            written_section_relpaths=[],
            gates={"enabled": False},
            review=None,
            error=None,
        )

    project_folder = context.get("project_folder")
    if not isinstance(project_folder, str) or not project_folder:
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates={"enabled": True},
            review=None,
            error="Missing required context: project_folder",
        )

    pf = validate_project_folder(project_folder)
    ensure_project_outputs_layout(pf)

    gates: Dict[str, Any] = {"enabled": True}

    # Pre-writing gates
    try:
        gate_cfg = EvidenceGateConfig.from_context(context)
        if gate_cfg.require_evidence:
            enforce_evidence_gate(
                project_folder=str(pf),
                source_ids=context.get("source_ids"),
                config=gate_cfg,
            )
        gates["evidence_gate"] = {"ok": True, "enabled": bool(gate_cfg.require_evidence)}
    except EvidenceGateError as e:
        gates["evidence_gate"] = {"ok": False, "enabled": True, "error": str(e)}
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates=gates,
            review=None,
            error="Pre-writing evidence gate blocked",
        )

    try:
        citation_cfg = CitationGateConfig.from_context(context)
        enforce_citation_gate(project_folder=str(pf), config=citation_cfg)
        gates["citation_gate"] = {"ok": True, "enabled": bool(citation_cfg.enabled)}
    except CitationGateError as e:
        gates["citation_gate"] = {"ok": False, "enabled": True, "error": str(e)}
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates=gates,
            review=None,
            error="Pre-writing citation gate blocked",
        )

    try:
        computation_cfg = ComputationGateConfig.from_context(context)
        enforce_computation_gate(project_folder=str(pf), config=computation_cfg)
        gates["computation_gate"] = {"ok": True, "enabled": bool(computation_cfg.enabled)}
    except ComputationGateError as e:
        gates["computation_gate"] = {"ok": False, "enabled": True, "error": str(e)}
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates=gates,
            review=None,
            error="Pre-writing computation gate blocked",
        )

    writers = _collect_writer_specs(context)
    if not writers:
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates=gates,
            review=None,
            error="Writing stage enabled but no writers configured",
        )

    written_relpaths: List[str] = []
    writer_results: List[Dict[str, Any]] = []

    for spec in writers:
        agent_id = str(spec.get("agent_id") or "").strip()
        agent = AgentRegistry.create_agent(agent_id)
        if agent is None:
            _remove_written_files(pf, written_relpaths)
            return WritingReviewStageResult(
                success=False,
                needs_revision=True,
                written_section_relpaths=written_relpaths,
                gates=gates,
                review=None,
                error=f"Failed to instantiate writer agent {agent_id}",
            )

        writer_context = dict(context)
        # Allow per-writer overrides for section id/title and any writer-specific config.
        for key in ("section_id", "section_title", "related_work_writer"):
            if key in spec:
                writer_context[key] = spec[key]

        result: AgentResult = await agent.execute(writer_context)
        writer_results.append({"agent_id": agent_id, "success": result.success, "error": result.error})

        if not result.success:
            _remove_written_files(pf, written_relpaths)
            return WritingReviewStageResult(
                success=False,
                needs_revision=True,
                written_section_relpaths=written_relpaths,
                gates=gates,
                review=None,
                error=f"Writer agent failed: {agent_id}: {result.error}",
            )

        rel = None
        if isinstance(result.structured_data, dict):
            rel = result.structured_data.get("output_relpath")
        if isinstance(rel, str) and rel:
            written_relpaths.append(rel)

    # Referee review
    review_agent_id = str(raw.get("review_agent_id") or "A19").strip() or "A19"
    review_agent = AgentRegistry.create_agent(review_agent_id)
    if review_agent is None:
        _remove_written_files(pf, written_relpaths)
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=written_relpaths,
            gates=gates,
            review=None,
            error=f"Failed to instantiate review agent {review_agent_id}",
        )

    review_relpaths = _review_section_relpaths(pf, context, written_relpaths)
    review_context = {
        "project_folder": str(pf),
        "section_relpaths": review_relpaths,
        "source_citation_map": context.get("source_citation_map") if isinstance(context.get("source_citation_map"), dict) else {},
        "referee_review": context.get("referee_review") if isinstance(context.get("referee_review"), dict) else {},
    }

    review_result: AgentResult = await review_agent.execute(review_context)

    review_payload: Dict[str, Any] = {
        "agent_id": review_agent_id,
        "success": review_result.success,
        "structured": review_result.structured_data if isinstance(review_result.structured_data, dict) else {},
    }

    if not review_result.success:
        _remove_written_files(pf, written_relpaths)
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=written_relpaths,
            gates=gates,
            review=review_payload,
            error="Referee review failed",
        )

    return WritingReviewStageResult(
        success=True,
        needs_revision=False,
        written_section_relpaths=written_relpaths,
        gates=gates,
        review=review_payload,
        error=None,
    )
