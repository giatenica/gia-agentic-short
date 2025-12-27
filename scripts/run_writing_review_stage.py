#!/usr/bin/env python3
"""Run the gated writing + referee review stage for a project folder.

This runner is designed to be non-interactive and filesystem-first.
It writes:
- writing_review_results.json
- writing_review_issues.json

Exit code behavior:
- Exits 1 only for CLI usage errors.
- Otherwise exits 0 and records failures in the issues file.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.llm.claude_client import load_env_file_lenient  # noqa: E402

load_env_file_lenient()

from src.agents.writing_review_integration import run_writing_review_stage  # noqa: E402
from src.claims.generator import generate_claims_from_metrics  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_source_citation_map(project_folder: Path) -> Dict[str, str]:
    # For now, default to empty mapping. Writers will downgrade gracefully when missing.
    return {}


def _default_writing_review_config(project_folder: Path) -> Dict[str, Any]:
    # Keep this permissive by default so it can run even when literature or
    # analysis artifacts are incomplete.
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


def _build_context(project_folder: Path) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "project_folder": str(project_folder),
        "source_citation_map": _default_source_citation_map(project_folder),
        "writing_review": _default_writing_review_config(project_folder),
        "referee_review": {
            "enabled": True,
        },
    }
    # Apply default gate configs so gates are enabled by default in warn mode.
    ctx.update(_default_gate_config())
    return ctx


def _issue(kind: str, message: str, *, details: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "kind": kind,
        "message": message,
    }
    if details:
        out["details"] = details
    return out


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_writing_review_stage.py <project_folder>")
        sys.exit(1)

    project_folder = Path(sys.argv[1]).expanduser().resolve()
    print(f"Starting writing+review stage for: {project_folder}", flush=True)

    results_path = project_folder / "writing_review_results.json"
    issues_path = project_folder / "writing_review_issues.json"

    issues: List[Dict[str, Any]] = []

    if not project_folder.exists() or not project_folder.is_dir():
        issues.append(_issue("invalid_project_folder", "Project folder does not exist", details={"path": str(project_folder)}))
        _safe_write_json(
            issues_path,
            {
                "stage": "writing_review",
                "generated_at": _utc_now_iso(),
                "success": False,
                "issues": issues,
            },
        )
        _safe_write_json(
            results_path,
            {
                "stage": "writing_review",
                "generated_at": _utc_now_iso(),
                "success": False,
                "result": None,
            },
        )
        print("Project folder invalid. See writing_review_issues.json", flush=True)
        return

    context = _build_context(project_folder)

    # Generate claims from metrics before running writing stage.
    # This ensures claims/claims.json is populated for gate validation.
    try:
        claims_result = generate_claims_from_metrics(project_folder=project_folder)
        print(f"Claims generation: {claims_result.get('action', 'unknown')}, claims_written={claims_result.get('claims_written', 0)}", flush=True)
    except Exception as e:
        issues.append(
            _issue(
                "claims_generation_failed",
                f"Claims generation failed: {type(e).__name__}: {e}",
                details={"error_type": type(e).__name__},
            )
        )
        print(f"Warning: Claims generation failed: {e}", flush=True)

    try:
        stage_result = await run_writing_review_stage(context)
        payload = stage_result.to_payload()

        _safe_write_json(
            results_path,
            {
                "stage": "writing_review",
                "generated_at": _utc_now_iso(),
                "project_folder": str(project_folder),
                "success": bool(payload.get("success")),
                "needs_revision": bool(payload.get("needs_revision")),
                "result": payload,
            },
        )

        if not payload.get("success"):
            issues.append(
                _issue(
                    "stage_failed",
                    "Writing review stage returned success=false",
                    details={
                        "error": payload.get("error"),
                        "needs_revision": payload.get("needs_revision"),
                        "written_section_relpaths": payload.get("written_section_relpaths"),
                        "gates": payload.get("gates"),
                    },
                )
            )

        review = payload.get("review")
        if isinstance(review, dict) and not bool(review.get("success", True)):
            issues.append(
                _issue(
                    "referee_review_failed",
                    "Referee review returned success=false",
                    details={"review": review},
                )
            )

    except Exception as e:
        issues.append(_issue("exception", f"{type(e).__name__}: {e}"))
        _safe_write_json(
            results_path,
            {
                "stage": "writing_review",
                "generated_at": _utc_now_iso(),
                "project_folder": str(project_folder),
                "success": False,
                "needs_revision": True,
                "result": None,
                "error": f"{type(e).__name__}: {e}",
            },
        )

    _safe_write_json(
        issues_path,
        {
            "stage": "writing_review",
            "generated_at": _utc_now_iso(),
            "project_folder": str(project_folder),
            "success": len(issues) == 0,
            "issues": issues,
        },
    )

    print("\n" + "=" * 60)
    print("WRITING + REVIEW STAGE COMPLETE")
    print("=" * 60)
    print(f"Results: {results_path}")
    print(f"Issues:  {issues_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
