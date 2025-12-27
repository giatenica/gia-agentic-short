"""Unified full pipeline runner.

This runner chains the existing phase workflows into a single entrypoint.
It is intentionally conservative and filesystem-first.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from src.agents.gap_resolution_workflow import GapResolutionWorkflow
from src.agents.literature_workflow import LiteratureWorkflow
from src.agents.workflow import ResearchWorkflow
from src.agents.writing_review_integration import run_writing_review_stage

from src.pipeline.context import WorkflowContext
from src.claims.generator import generate_claims_from_metrics

from src.pipeline.degradation import (
    make_degradation_event,
    write_degradation_summary,
)


def _default_source_citation_map(project_folder: Path) -> Dict[str, str]:
    return {}


def _default_writing_review_config(project_folder: Path) -> Dict[str, Any]:
    return {
        "enabled": True,
        "review_agent_id": "A19",
        "writers": [
            {
                "agent_id": "A21",
                "section_id": "introduction",
                "section_title": "Introduction",
                "introduction_writer": {
                    "on_missing_citation": "downgrade",
                    "on_unverified_citation": "downgrade",
                    "require_verified_citations": False,
                },
            },
            {
                "agent_id": "A18",
                "section_id": "related_work",
                "section_title": "Related Work",
                "related_work_writer": {
                    "on_missing_citation": "downgrade",
                    "on_unverified_citation": "downgrade",
                    "require_verified_citations": False,
                },
            },
            {
                "agent_id": "A22",
                "section_id": "methods",
                "section_title": "Data and Methodology",
                "methods_writer": {
                    "on_missing_citation": "downgrade",
                    "on_missing_evidence": "downgrade",
                    "on_missing_metrics": "downgrade",
                },
            },
            {
                "agent_id": "A20",
                "section_id": "results",
                "section_title": "Results",
                "results_writer": {
                    "on_missing_metrics": "downgrade",
                },
            },
            {
                "agent_id": "A23",
                "section_id": "discussion",
                "section_title": "Discussion",
                "discussion_writer": {
                    "on_missing_citation": "downgrade",
                    "on_missing_evidence": "downgrade",
                    "on_missing_metrics": "downgrade",
                },
            },
        ],
    }


def _default_referee_review_config(project_folder: Path) -> Dict[str, Any]:
    return {"enabled": True}


def _default_gate_config() -> Dict[str, Dict[str, Any]]:
    """Return default gate configurations.

    By default, gates are enabled in 'warn' mode (downgrade on failure).
    This ensures issues are surfaced without blocking the pipeline.
    """
    return {
        "evidence_gate": {
            "require_evidence": True,
            "min_items_per_source": 1,
        },
        "citation_gate": {
            "enabled": True,
            "on_missing": "downgrade",
            "on_unverified": "downgrade",
        },
        "computation_gate": {
            "enabled": True,
            "on_missing_metrics": "downgrade",
        },
        "claim_evidence_gate": {
            "enabled": True,
            "on_failure": "downgrade",
        },
        "literature_gate": {
            "enabled": True,
            "on_failure": "downgrade",
        },
        "analysis_gate": {
            "enabled": True,
            "on_failure": "downgrade",
        },
        "citation_accuracy_gate": {
            "enabled": True,
            "on_failure": "downgrade",
        },
    }


def _build_writing_context(project_folder: Path, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "project_folder": str(project_folder),
        "source_citation_map": _default_source_citation_map(project_folder),
        "writing_review": _default_writing_review_config(project_folder),
        "referee_review": _default_referee_review_config(project_folder),
    }
    # Apply default gate configs; callers can override via extra.
    ctx.update(_default_gate_config())
    if isinstance(extra, dict) and extra:
        ctx.update(extra)
    return ctx


async def run_full_pipeline(
    project_folder: str,
    *,
    enable_gap_resolution: bool = True,
    enable_writing_review: bool = True,
    workflow_overrides: Optional[Dict[str, Any]] = None,
) -> WorkflowContext:
    """Run Phase 1 -> Phase 2 -> (optional) Phase 3 -> (optional) Phase 4.

    Args:
        project_folder: Project folder path.
        enable_gap_resolution: Run the gap resolution workflow after literature.
        enable_writing_review: Run writing + referee review stage at the end.
        workflow_overrides: Optional dict merged into the literature workflow context
            and the writing-review context.

    Returns:
        WorkflowContext with phase results and checkpoints.
    """

    pf = Path(project_folder).expanduser().resolve()
    context = WorkflowContext(project_folder=pf)

    context.mark_checkpoint("start")

    def _finalize_and_return(ctx: WorkflowContext) -> WorkflowContext:
        try:
            write_degradation_summary(
                project_folder=pf,
                run_id=ctx.run_id,
                degradations=list(ctx.degradations),
                created_at=ctx.created_at,
            )
        except Exception:
            logger.exception(
                "Failed to write degradation summary for project_folder='{}', run_id='{}'",
                pf,
                getattr(ctx, "run_id", None),
            )
        return ctx

    phase1 = ResearchWorkflow()
    phase1_result = await phase1.run(str(pf))
    context.record_phase_result("phase_1", phase1_result)
    context.mark_checkpoint("phase_1_complete")

    if not context.success:
        context.mark_checkpoint("end")
        return _finalize_and_return(context)

    phase2 = LiteratureWorkflow()
    merged_overrides: Dict[str, Any] = dict(workflow_overrides) if isinstance(workflow_overrides, dict) else {}

    # Default to enabling the offline evidence pipeline for the unified runner.
    # Callers can still explicitly disable by passing: {"evidence_pipeline": {"enabled": False}}.
    if "evidence_pipeline" not in merged_overrides:
        merged_overrides["evidence_pipeline"] = {"enabled": True}

    phase2_result = await phase2.run(str(pf), workflow_context=merged_overrides)
    context.record_phase_result("phase_2", phase2_result)

    context.mark_checkpoint("phase_2_complete")

    if not context.success:
        context.mark_checkpoint("end")
        return _finalize_and_return(context)

    if enable_gap_resolution:
        phase3 = GapResolutionWorkflow()
        phase3_result = await phase3.run(str(pf))
        context.record_phase_result("phase_3", phase3_result)
        context.mark_checkpoint("phase_3_complete")

        if not context.success:
            context.mark_checkpoint("end")
            return _finalize_and_return(context)

    if enable_writing_review:
        # Ensure computed claims exist when metrics have been produced by earlier steps.
        # This is filesystem-first and safe to run even when metrics.json is absent.
        try:
            generate_claims_from_metrics(project_folder=pf)
        except Exception as e:
            logger.debug("Claims generation failed in unified pipeline: {}: {}", type(e).__name__, e)
            context.degradations.append(
                make_degradation_event(
                    stage="analysis",
                    reason_code="claims_generation_failed",
                    message=f"Claims generation failed: {type(e).__name__}: {e}",
                    recommended_action="Inspect outputs/metrics.json and outputs/claims.json; rerun analysis if needed.",
                    severity="warning",
                    details={"error_type": type(e).__name__},
                    created_at=context.created_at,
                )
            )

        writing_extra = dict(merged_overrides)
        writing_extra.setdefault("degradations", list(context.degradations))

        writing_context = _build_writing_context(pf, extra=writing_extra)
        writing_result = await run_writing_review_stage(writing_context)
        context.record_phase_result("phase_4_writing_review", writing_result.to_payload())
        context.mark_checkpoint("phase_4_complete")

        if not context.success:
            context.mark_checkpoint("end")
            return _finalize_and_return(context)

    context.mark_checkpoint("end")
    return _finalize_and_return(context)
