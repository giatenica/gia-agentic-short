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
import json
import re
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from src.agents.base import AgentResult
from src.agents.registry import AgentRegistry
from src.evidence.gates import EvidenceGateConfig, EvidenceGateError, enforce_evidence_gate
from src.citations.gates import CitationGateConfig, CitationGateError, enforce_citation_gate
from src.citations.accuracy_gate import (
    CitationAccuracyGateConfig,
    CitationAccuracyGateError,
    enforce_citation_accuracy_gate,
)
from src.claims.gates import ComputationGateConfig, ComputationGateError, enforce_computation_gate
from src.literature.gates import LiteratureGateConfig, LiteratureGateError, enforce_literature_gate
from src.analysis.gates import AnalysisGateConfig, AnalysisGateError, enforce_analysis_gate
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.validation import validate_project_folder
from src.tracing import safe_set_current_span_attributes


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


_CITE_PATTERN = re.compile(r"\\cite[a-zA-Z]*\*?\{([^}]*)\}")


def _parse_citation_keys(tex: str) -> List[str]:
    keys: List[str] = []
    seen: Set[str] = set()
    for m in _CITE_PATTERN.finditer(tex):
        group = m.group(1)
        for raw in group.split(","):
            key = raw.strip()
            if not key:
                continue
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _writing_review_history_path(project_folder: Path) -> Path:
    return project_folder / "outputs" / "writing_review_history.json"


def _write_writing_review_history(project_folder: Path, payload: Dict[str, Any]) -> Path:
    out_path = _writing_review_history_path(project_folder)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _infer_target_relpaths_from_review(
    project_folder: Path,
    *,
    review_relpaths: List[str],
    review_structured: Dict[str, Any],
    source_citation_map: Dict[str, str],
) -> List[str]:
    """Infer which section relpaths should be rewritten based on referee checklist.

    Best-effort: if inference fails, default to rewriting all reviewed sections.
    """

    targets: Set[str] = set()
    checklist = review_structured.get("checklist")
    if not isinstance(checklist, list):
        return list(review_relpaths)

    missing_relpaths: List[str] = []
    unknown_keys: List[str] = []
    unverified_keys: List[str] = []
    cited_sources_below_threshold: List[str] = []

    for item in checklist:
        if not isinstance(item, dict):
            continue
        check = item.get("check")
        details = item.get("details")
        if not isinstance(details, dict):
            details = {}

        if check == "sections_exist":
            raw_missing = details.get("missing_section_relpaths")
            if isinstance(raw_missing, list):
                missing_relpaths = [str(p) for p in raw_missing if isinstance(p, str) and p]
        elif check == "citations_known_keys":
            raw_unknown = details.get("unknown_citation_keys")
            if isinstance(raw_unknown, list):
                unknown_keys = [str(k) for k in raw_unknown if isinstance(k, str) and k]
        elif check == "citations_verified":
            raw_unverified = details.get("unverified_citation_keys")
            if isinstance(raw_unverified, list):
                unverified_keys = [str(k) for k in raw_unverified if isinstance(k, str) and k]
        elif check == "evidence_coverage":
            raw_sources = details.get("sources_below_threshold")
            if isinstance(raw_sources, dict):
                cited_sources_below_threshold = [str(k) for k in raw_sources.keys() if isinstance(k, str) and k]

    for rel in missing_relpaths:
        targets.add(rel)

    keys_of_interest: Set[str] = set()
    keys_of_interest.update(unknown_keys)
    keys_of_interest.update(unverified_keys)

    if cited_sources_below_threshold and source_citation_map:
        for sid in cited_sources_below_threshold:
            key = source_citation_map.get(sid)
            if isinstance(key, str) and key.strip():
                keys_of_interest.add(key.strip())

    if keys_of_interest:
        for rel in review_relpaths:
            try:
                path = project_folder / rel
                if not path.exists() or not path.is_file():
                    continue
                tex = path.read_text(encoding="utf-8")
            except Exception:
                continue
            cited = set(_parse_citation_keys(tex))
            if cited.intersection(keys_of_interest):
                targets.add(rel)

    if not targets:
        return list(review_relpaths)
    return sorted(targets)


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
        safe_set_current_span_attributes(
            {
                "writing_review.enabled": False,
                "writing_review.needs_revision": False,
                "writing_review.writers_ran": 0,
                "writing_review.written_sections_total": 0,
            }
        )
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
        safe_set_current_span_attributes(
            {
                "writing_review.enabled": True,
                "writing_review.needs_revision": True,
                "writing_review.writers_ran": 0,
                "writing_review.written_sections_total": 0,
                "writing_review.error_category": "missing_project_folder",
            }
        )
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

    safe_set_current_span_attributes(
        {
            "writing_review.enabled": True,
            "writing_review.project_folder_name": pf.name,
        }
    )

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
        safe_set_current_span_attributes(
            {
                "writing_review.needs_revision": True,
                "writing_review.writers_ran": 0,
                "writing_review.written_sections_total": 0,
                "writing_review.error_category": "evidence_gate_blocked",
            }
        )
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
        safe_set_current_span_attributes(
            {
                "writing_review.needs_revision": True,
                "writing_review.writers_ran": 0,
                "writing_review.written_sections_total": 0,
                "writing_review.error_category": "citation_gate_blocked",
            }
        )
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates=gates,
            review=None,
            error="Pre-writing citation gate blocked",
        )

    try:
        accuracy_cfg = CitationAccuracyGateConfig.from_context(context)
        accuracy_result = enforce_citation_accuracy_gate(project_folder=str(pf), config=accuracy_cfg)
        gates["citation_accuracy_gate"] = {
            "ok": True,
            "enabled": bool(accuracy_cfg.enabled),
            "action": accuracy_result.get("action"),
            "checked_claims_total": accuracy_result.get("checked_claims_total"),
            "failed_claims_total": accuracy_result.get("failed_claims_total"),
            "skipped_missing_evidence_total": accuracy_result.get("skipped_missing_evidence_total"),
        }
    except CitationAccuracyGateError as e:
        gates["citation_accuracy_gate"] = {"ok": False, "enabled": True, "error": str(e)}
        safe_set_current_span_attributes(
            {
                "writing_review.needs_revision": True,
                "writing_review.writers_ran": 0,
                "writing_review.written_sections_total": 0,
                "writing_review.error_category": "citation_accuracy_gate_blocked",
            }
        )
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates=gates,
            review=None,
            error="Pre-writing citation accuracy gate blocked",
        )

    try:
        computation_cfg = ComputationGateConfig.from_context(context)
        enforce_computation_gate(project_folder=str(pf), config=computation_cfg)
        gates["computation_gate"] = {"ok": True, "enabled": bool(computation_cfg.enabled)}
    except ComputationGateError as e:
        gates["computation_gate"] = {"ok": False, "enabled": True, "error": str(e)}
        safe_set_current_span_attributes(
            {
                "writing_review.needs_revision": True,
                "writing_review.writers_ran": 0,
                "writing_review.written_sections_total": 0,
                "writing_review.error_category": "computation_gate_blocked",
            }
        )
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates=gates,
            review=None,
            error="Pre-writing computation gate blocked",
        )

    try:
        literature_cfg = LiteratureGateConfig.from_context(context)
        enforce_literature_gate(project_folder=str(pf), config=literature_cfg)
        gates["literature_gate"] = {"ok": True, "enabled": bool(literature_cfg.enabled)}
    except LiteratureGateError as e:
        gates["literature_gate"] = {"ok": False, "enabled": True, "error": str(e)}
        safe_set_current_span_attributes(
            {
                "writing_review.needs_revision": True,
                "writing_review.writers_ran": 0,
                "writing_review.written_sections_total": 0,
                "writing_review.error_category": "literature_gate_blocked",
            }
        )
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates=gates,
            review=None,
            error="Pre-writing literature gate blocked",
        )

    try:
        analysis_cfg = AnalysisGateConfig.from_context(context)
        analysis_result = enforce_analysis_gate(project_folder=str(pf), config=analysis_cfg)
        gates["analysis_gate"] = {
            "ok": True,
            "enabled": bool(analysis_cfg.enabled),
            "action": analysis_result.get("action"),
        }
    except AnalysisGateError as e:
        gates["analysis_gate"] = {"ok": False, "enabled": True, "error": str(e)}
        safe_set_current_span_attributes(
            {
                "writing_review.needs_revision": True,
                "writing_review.writers_ran": 0,
                "writing_review.written_sections_total": 0,
                "writing_review.error_category": "analysis_gate_blocked",
            }
        )
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates=gates,
            review=None,
            error="Pre-writing analysis gate blocked",
        )

    writers = _collect_writer_specs(context)
    if not writers:
        safe_set_current_span_attributes(
            {
                "writing_review.needs_revision": True,
                "writing_review.writers_ran": 0,
                "writing_review.writers_configured": 0,
                "writing_review.written_sections_total": 0,
                "writing_review.error_category": "no_writers_configured",
            }
        )
        return WritingReviewStageResult(
            success=False,
            needs_revision=True,
            written_section_relpaths=[],
            gates=gates,
            review=None,
            error="Writing stage enabled but no writers configured",
        )

    writers_configured = len(writers)
    writers_ran = 0

    safe_set_current_span_attributes(
        {
            "writing_review.writers_configured": int(writers_configured),
            "writing_review.review_agent_id": str(raw.get("review_agent_id") or "A19").strip() or "A19",
        }
    )

    max_iterations_raw = raw.get("max_iterations", 3)
    try:
        max_iterations = int(max_iterations_raw)
    except Exception:
        max_iterations = 3
    if max_iterations < 1:
        max_iterations = 1
    if max_iterations > 10:
        max_iterations = 10

    history: List[Dict[str, Any]] = []
    written_relpaths: List[str] = []
    spec_to_relpath: Dict[int, str] = {}
    review_relpaths: Optional[List[str]] = None
    last_review_payload: Optional[Dict[str, Any]] = None

    target_relpaths: Optional[List[str]] = None

    for iteration in range(1, max_iterations + 1):
        # Determine which writers to rerun.
        selected_specs: List[Dict[str, Any]] = []
        if target_relpaths is None:
            selected_specs = list(writers)
        else:
            target_set = set(target_relpaths)
            for idx, spec in enumerate(writers):
                rel = spec_to_relpath.get(idx)
                if isinstance(rel, str) and rel in target_set:
                    selected_specs.append(spec)

        if target_relpaths is not None:
            # Only remove files produced by writers; never delete arbitrary review targets.
            files_to_remove = [p for p in target_relpaths if p in written_relpaths]
            if files_to_remove:
                _remove_written_files(pf, files_to_remove)

        iteration_written: List[str] = []
        iteration_writers_ran: List[str] = []

        for idx, spec in enumerate(writers):
            if spec not in selected_specs:
                continue

            agent_id = str(spec.get("agent_id") or "").strip()
            agent = AgentRegistry.create_agent(agent_id)
            if agent is None:
                _remove_written_files(pf, written_relpaths)
                return WritingReviewStageResult(
                    success=False,
                    needs_revision=True,
                    written_section_relpaths=written_relpaths,
                    gates=gates,
                    review=last_review_payload,
                    error=f"Failed to instantiate writer agent {agent_id}",
                )

            writer_context = dict(context)
            writer_context["revision"] = {
                "iteration": int(iteration),
                "max_iterations": int(max_iterations),
                "target_section_relpaths": list(target_relpaths or []),
                "last_review": last_review_payload,
            }

            # Allow per-writer overrides for section id/title and any writer-specific config.
            for key in (
                "section_id",
                "section_title",
                "source_citation_map",
                "introduction_writer",
                "methods_writer",
                "discussion_writer",
                "related_work_writer",
                "results_writer",
            ):
                if key in spec:
                    writer_context[key] = spec[key]

            result: AgentResult = await agent.execute(writer_context)

            if not result.success:
                _remove_written_files(pf, written_relpaths)
                safe_set_current_span_attributes(
                    {
                        "writing_review.needs_revision": True,
                        "writing_review.writers_ran": int(writers_ran),
                        "writing_review.written_sections_total": int(len(written_relpaths)),
                        "writing_review.error_category": "writer_failed",
                    }
                )
                return WritingReviewStageResult(
                    success=False,
                    needs_revision=True,
                    written_section_relpaths=written_relpaths,
                    gates=gates,
                    review=last_review_payload,
                    error=f"Writer agent failed: {agent_id}: {result.error}",
                )

            rel = None
            if isinstance(result.structured_data, dict):
                rel = result.structured_data.get("output_relpath")
            if isinstance(rel, str) and rel:
                spec_to_relpath[idx] = rel
                if rel not in written_relpaths:
                    written_relpaths.append(rel)
                iteration_written.append(rel)
            iteration_writers_ran.append(agent_id)
            writers_ran += 1

        # Referee review
        review_agent_id = str(raw.get("review_agent_id") or "A19").strip() or "A19"
        review_agent = AgentRegistry.create_agent(review_agent_id)
        if review_agent is None:
            _remove_written_files(pf, written_relpaths)
            safe_set_current_span_attributes(
                {
                    "writing_review.needs_revision": True,
                    "writing_review.writers_ran": int(writers_ran),
                    "writing_review.written_sections_total": int(len(written_relpaths)),
                    "writing_review.review_success": False,
                    "writing_review.error_category": "review_agent_missing",
                }
            )
            return WritingReviewStageResult(
                success=False,
                needs_revision=True,
                written_section_relpaths=written_relpaths,
                gates=gates,
                review=last_review_payload,
                error=f"Failed to instantiate review agent {review_agent_id}",
            )

        if review_relpaths is None:
            review_relpaths = _review_section_relpaths(pf, context, written_relpaths)

        review_context = {
            "project_folder": str(pf),
            "section_relpaths": list(review_relpaths),
            "source_citation_map": context.get("source_citation_map") if isinstance(context.get("source_citation_map"), dict) else {},
            "referee_review": context.get("referee_review") if isinstance(context.get("referee_review"), dict) else {},
        }

        review_result: AgentResult = await review_agent.execute(review_context)

        review_payload: Dict[str, Any] = {
            "agent_id": review_agent_id,
            "success": review_result.success,
            "structured": review_result.structured_data if isinstance(review_result.structured_data, dict) else {},
        }

        history_entry: Dict[str, Any] = {
            "iteration": int(iteration),
            "writers_ran": list(iteration_writers_ran),
            "written_section_relpaths": list(iteration_written),
            "review": {
                "agent_id": review_agent_id,
                "success": bool(review_result.success),
                "summary": (review_payload.get("structured") or {}).get("summary", {}),
            },
        }
        history.append(history_entry)

        last_review_payload = review_payload

        if review_result.success:
            history_payload = {
                "max_iterations": int(max_iterations),
                "review_section_relpaths": list(review_relpaths),
                "iterations": history,
            }
            hist_path = _write_writing_review_history(pf, history_payload)
            review_payload["revision_history_relpath"] = str(hist_path.relative_to(pf))

            safe_set_current_span_attributes(
                {
                    "writing_review.needs_revision": False,
                    "writing_review.writers_ran": int(writers_ran),
                    "writing_review.written_sections_total": int(len(written_relpaths)),
                    "writing_review.review_success": True,
                    "writing_review.review_sections_total": int(len(review_relpaths)),
                    "writing_review.revision_iterations": int(iteration),
                }
            )

            return WritingReviewStageResult(
                success=True,
                needs_revision=False,
                written_section_relpaths=written_relpaths,
                gates=gates,
                review=review_payload,
                error=None,
            )

        # Review failed: infer targets and continue or stop.
        review_structured: Dict[str, Any] = {}
        if isinstance(review_payload.get("structured"), dict):
            review_structured = dict(review_payload.get("structured") or {})
        source_citation_map: Dict[str, str] = {}
        raw_map = context.get("source_citation_map")
        if isinstance(raw_map, dict):
            # Best-effort cast; ignore non-string values.
            source_citation_map = {str(k): str(v) for k, v in raw_map.items() if isinstance(k, str) and isinstance(v, str)}

        inferred_targets = _infer_target_relpaths_from_review(
            pf,
            review_relpaths=list(review_relpaths),
            review_structured=review_structured,
            source_citation_map=source_citation_map,
        )
        target_relpaths = inferred_targets
        history_entry["targets_next_relpaths"] = list(target_relpaths)

        if iteration >= max_iterations:
            # Final failure: remove written outputs to avoid leaving invalid sections.
            _remove_written_files(pf, written_relpaths)

            history_payload = {
                "max_iterations": int(max_iterations),
                "review_section_relpaths": list(review_relpaths),
                "iterations": history,
            }
            hist_path = _write_writing_review_history(pf, history_payload)
            review_payload["revision_history_relpath"] = str(hist_path.relative_to(pf))

            safe_set_current_span_attributes(
                {
                    "writing_review.needs_revision": True,
                    "writing_review.writers_ran": int(writers_ran),
                    "writing_review.written_sections_total": int(len(written_relpaths)),
                    "writing_review.review_success": False,
                    "writing_review.review_sections_total": int(len(review_relpaths)),
                    "writing_review.error_category": "referee_review_failed",
                    "writing_review.revision_iterations": int(iteration),
                }
            )

            return WritingReviewStageResult(
                success=False,
                needs_revision=True,
                written_section_relpaths=written_relpaths,
                gates=gates,
                review=review_payload,
                error="Referee review failed (max iterations reached)",
            )

    # Defensive fallback: if we exit loop unexpectedly.
    _remove_written_files(pf, written_relpaths)
    return WritingReviewStageResult(
        success=False,
        needs_revision=True,
        written_section_relpaths=written_relpaths,
        gates=gates,
        review=last_review_payload,
        error="Writing review loop aborted unexpectedly",
    )
