"""Referee-style adversarial review stage.

Sprint 4 MVP:
- Deterministic, filesystem-first review that can run after section generation.
- Checks:
  - Citation correctness (no unknown citation keys).
  - Evidence coverage thresholds for cited sources (minimum evidence items).

This implementation is deliberately offline:
- It does not call the LLM.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Set

from loguru import logger

from src.agents.base import AgentResult, BaseAgent
from src.agents.section_writer import resolve_section_output_path
from src.citations.registry import load_citations
from src.llm.claude_client import TaskType
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.validation import validate_project_folder


OnFailureAction = Literal["block", "downgrade"]


@dataclass(frozen=True)
class RefereeReviewConfig:
    enabled: bool = True

    # Citation checks
    on_unknown_citation: OnFailureAction = "block"
    require_verified_citations: bool = False
    on_unverified_citation: OnFailureAction = "downgrade"

    # Evidence checks
    min_evidence_items_per_cited_source: int = 1
    on_insufficient_evidence: OnFailureAction = "downgrade"

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "RefereeReviewConfig":
        raw = context.get("referee_review")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", True))

        on_unknown = raw.get("on_unknown_citation", "block")
        on_unverified = raw.get("on_unverified_citation", "downgrade")
        on_evidence = raw.get("on_insufficient_evidence", "downgrade")

        if on_unknown not in ("block", "downgrade"):
            on_unknown = "block"
        if on_unverified not in ("block", "downgrade"):
            on_unverified = "downgrade"
        if on_evidence not in ("block", "downgrade"):
            on_evidence = "downgrade"

        require_verified = bool(raw.get("require_verified_citations", False))

        min_items = int(raw.get("min_evidence_items_per_cited_source", 1))
        if min_items < 0:
            min_items = 0

        return cls(
            enabled=enabled,
            on_unknown_citation=on_unknown,
            require_verified_citations=require_verified,
            on_unverified_citation=on_unverified,
            min_evidence_items_per_cited_source=min_items,
            on_insufficient_evidence=on_evidence,
        )


_CITE_PATTERN = re.compile(r"\\cite[a-zA-Z]*\*?\{([^}]*)\}")


def _iter_section_paths(project_folder: Path, context: Dict[str, Any]) -> List[Path]:
    relpaths = context.get("section_relpaths")
    if isinstance(relpaths, list) and all(isinstance(p, str) for p in relpaths):
        return [project_folder / p for p in relpaths]

    section_ids = context.get("section_ids")
    if isinstance(section_ids, list) and all(isinstance(s, str) for s in section_ids):
        out: List[Path] = []
        for sid in section_ids:
            p, _ = resolve_section_output_path(project_folder, section_id=sid)
            out.append(p)
        return out

    # Default: scan outputs/sections
    sections_dir = project_folder / "outputs" / "sections"
    if not sections_dir.exists() or not sections_dir.is_dir():
        return []

    return sorted([p for p in sections_dir.rglob("*.tex") if p.is_file()])


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


def _citation_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for r in records:
        key = str(r.get("citation_key") or "").strip()
        if key:
            by_key[key] = r
    return by_key


def _iter_source_evidence_files(project_folder: Path) -> Iterable[Path]:
    sources_dir = project_folder / "sources"
    if not sources_dir.exists() or not sources_dir.is_dir():
        return []

    paths: List[Path] = []
    for p in sources_dir.rglob("evidence.json"):
        if p.is_file():
            paths.append(p)

    return sorted(paths)


def _load_evidence_items(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Evidence file must contain a list: {path}")
    return [p for p in payload if isinstance(p, dict)]


def _evidence_count_by_source(project_folder: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for evidence_path in _iter_source_evidence_files(project_folder):
        source_id = evidence_path.parent.name
        try:
            items = _load_evidence_items(evidence_path)
        except Exception as e:
            logger.debug(f"Skipping unreadable evidence file: {evidence_path}: {type(e).__name__}: {e}")
            continue
        counts[source_id] = len(items)
    return counts


class RefereeReviewAgent(BaseAgent):
    """Deterministic referee-style review producing a structured revision checklist."""

    def __init__(self, client=None):
        system_prompt = (
            "You are a referee-style reviewer. "
            "You never invent facts. "
            "You produce a structured checklist of issues and required revisions."
        )
        super().__init__(
            name="RefereeReview",
            task_type=TaskType.DATA_EXTRACTION,
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

        cfg = RefereeReviewConfig.from_context(context)
        if not cfg.enabled:
            checklist = [
                {
                    "check": "review_enabled",
                    "passed": True,
                    "severity": "info",
                    "message": "Referee review disabled by config.",
                }
            ]
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content="Referee review disabled.",
                structured_data={"checklist": checklist, "summary": {"failed": 0, "passed": 1}},
            )

        section_paths = _iter_section_paths(pf, context)
        missing_sections = [str(p.relative_to(pf)) for p in section_paths if not p.exists()]

        citation_records = load_citations(pf, validate=True)
        citations_by_key = _citation_index(citation_records)
        known_keys: Set[str] = set(citations_by_key.keys())

        cited_keys: List[str] = []
        if section_paths:
            for p in section_paths:
                if not p.exists():
                    continue
                tex = p.read_text(encoding="utf-8")
                cited_keys.extend(_parse_citation_keys(tex))

        cited_keys_unique = []
        seen: Set[str] = set()
        for k in cited_keys:
            if k not in seen:
                cited_keys_unique.append(k)
                seen.add(k)

        unknown_keys = sorted([k for k in cited_keys_unique if k not in known_keys])

        unverified_keys: List[str] = []
        if cfg.require_verified_citations:
            for k in cited_keys_unique:
                r = citations_by_key.get(k)
                if r and str(r.get("status") or "") != "verified":
                    unverified_keys.append(k)

        evidence_counts = _evidence_count_by_source(pf)
        source_citation_map = context.get("source_citation_map")
        if not isinstance(source_citation_map, dict):
            source_citation_map = {}

        cited_sources_with_low_evidence: dict[str, int] = {}
        if cfg.min_evidence_items_per_cited_source > 0 and source_citation_map:
            # Only evaluate sources whose mapped citation key is actually cited.
            for source_id, key_raw in source_citation_map.items():
                if not isinstance(source_id, str) or not source_id.strip():
                    continue
                key = str(key_raw or "").strip()
                if not key:
                    continue
                if key not in cited_keys_unique:
                    continue
                count = int(evidence_counts.get(source_id, 0))
                if count < cfg.min_evidence_items_per_cited_source:
                    cited_sources_with_low_evidence[source_id] = count

        checklist: List[dict] = []

        if missing_sections:
            checklist.append(
                {
                    "check": "sections_exist",
                    "passed": False,
                    "severity": "error",
                    "message": "One or more requested section files were missing.",
                    "details": {"missing_section_relpaths": missing_sections},
                }
            )
        else:
            checklist.append(
                {
                    "check": "sections_exist",
                    "passed": True,
                    "severity": "info",
                    "message": "All section files exist.",
                    "details": {"section_count": len(section_paths)},
                }
            )

        if unknown_keys:
            checklist.append(
                {
                    "check": "citations_known_keys",
                    "passed": False,
                    "severity": "error",
                    "message": "Unknown citation keys found in LaTeX.",
                    "details": {"unknown_citation_keys": unknown_keys},
                }
            )
        else:
            checklist.append(
                {
                    "check": "citations_known_keys",
                    "passed": True,
                    "severity": "info",
                    "message": "All citation keys are known.",
                    "details": {"cited_citation_keys": cited_keys_unique},
                }
            )

        if unverified_keys:
            checklist.append(
                {
                    "check": "citations_verified",
                    "passed": False if cfg.on_unverified_citation == "block" else True,
                    "severity": "error" if cfg.on_unverified_citation == "block" else "warning",
                    "message": "One or more citations are not verified.",
                    "details": {"unverified_citation_keys": sorted(unverified_keys)},
                }
            )
        else:
            checklist.append(
                {
                    "check": "citations_verified",
                    "passed": True,
                    "severity": "info",
                    "message": "All cited citations are verified (or verification not required).",
                }
            )

        if cited_sources_with_low_evidence:
            checklist.append(
                {
                    "check": "evidence_coverage",
                    "passed": False if cfg.on_insufficient_evidence == "block" else True,
                    "severity": "error" if cfg.on_insufficient_evidence == "block" else "warning",
                    "message": "Evidence coverage below threshold for one or more cited sources.",
                    "details": {
                        "min_evidence_items_per_cited_source": cfg.min_evidence_items_per_cited_source,
                        "sources_below_threshold": {
                            k: cited_sources_with_low_evidence[k]
                            for k in sorted(cited_sources_with_low_evidence.keys())
                        },
                    },
                }
            )
        else:
            checklist.append(
                {
                    "check": "evidence_coverage",
                    "passed": True,
                    "severity": "info",
                    "message": "Evidence coverage meets threshold (or no source mapping provided).",
                    "details": {
                        "min_evidence_items_per_cited_source": cfg.min_evidence_items_per_cited_source,
                    },
                }
            )

        # Determine overall pass/fail under configured blocking rules.
        failed_checks: List[str] = []
        for item in checklist:
            if item.get("check") == "citations_known_keys" and unknown_keys and cfg.on_unknown_citation == "block":
                failed_checks.append("citations_known_keys")
            if item.get("check") == "sections_exist" and missing_sections:
                failed_checks.append("sections_exist")
            if item.get("check") == "citations_verified" and unverified_keys and cfg.on_unverified_citation == "block":
                failed_checks.append("citations_verified")
            if (
                item.get("check") == "evidence_coverage"
                and cited_sources_with_low_evidence
                and cfg.on_insufficient_evidence == "block"
            ):
                failed_checks.append("evidence_coverage")

        passed = len(failed_checks) == 0

        summary = {
            "passed": passed,
            "failed_checks": sorted(set(failed_checks)),
            "cited_citation_keys": cited_keys_unique,
            "unknown_citation_keys": unknown_keys,
        }

        content_lines = ["Referee review checklist:"]
        for item in checklist:
            status = "PASS" if item.get("passed") else "FAIL"
            content_lines.append(f"- {status}: {item.get('check')} | {item.get('message')}")

        return AgentResult(
            agent_name=self.name,
            task_type=self.task_type,
            model_tier=self.model_tier,
            success=passed,
            content="\n".join(content_lines) + "\n",
            structured_data={"checklist": checklist, "summary": summary},
            error=None if passed else "Referee review found blocking issues",
        )
