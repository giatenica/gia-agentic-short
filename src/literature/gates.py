"""Literature gate.

Checks that the literature layer is sufficiently populated before drafting.

Deterministic and offline:
- counts verified citation records in bibliography/citations.json
- counts evidence items across sources/*/evidence.json
- optional: enforces a minimum evidence item count per source

The default policy is permissive when not explicitly enabled.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from loguru import logger

from src.citations.registry import load_citations
from src.utils.schema_validation import is_valid_citation_record, is_valid_evidence_item
from src.utils.validation import validate_project_folder


class LiteratureGateError(ValueError):
    """Raised when the literature gate blocks execution."""


OnFailureAction = Literal["block", "downgrade"]


@dataclass(frozen=True)
class LiteratureGateConfig:
    """Configuration for literature readiness enforcement."""

    enabled: bool = False
    on_failure: OnFailureAction = "block"

    min_verified_citations: int = 1
    min_evidence_items_total: int = 1

    # If set to a positive integer, every evidence source present must have at least this many items.
    min_evidence_items_per_source: int = 0

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "LiteratureGateConfig":
        raw = context.get("literature_gate")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", False))
        on_failure = raw.get("on_failure", "block")
        if on_failure not in ("block", "downgrade"):
            on_failure = "block"

        def _as_int(val: Any, default: int) -> int:
            try:
                iv = int(val)
            except (TypeError, ValueError):
                return default
            return iv

        min_verified_citations = max(0, _as_int(raw.get("min_verified_citations", cls.min_verified_citations), cls.min_verified_citations))
        min_evidence_items_total = max(0, _as_int(raw.get("min_evidence_items_total", cls.min_evidence_items_total), cls.min_evidence_items_total))
        min_evidence_items_per_source = max(0, _as_int(raw.get("min_evidence_items_per_source", cls.min_evidence_items_per_source), cls.min_evidence_items_per_source))

        return cls(
            enabled=enabled,
            on_failure=on_failure,
            min_verified_citations=min_verified_citations,
            min_evidence_items_total=min_evidence_items_total,
            min_evidence_items_per_source=min_evidence_items_per_source,
        )


def _load_json_list(path: Path) -> Tuple[List[Any], Optional[str]]:
    if not path.exists():
        return [], None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        return [], f"{type(e).__name__}"

    if not isinstance(payload, list):
        return [], "not_a_list"

    return payload, None


def _count_verified_citations(project_folder: Path) -> Tuple[int, int, bool]:
    citations_file_present = (project_folder / "bibliography" / "citations.json").exists()

    invalid = 0
    verified = 0

    try:
        payload = load_citations(project_folder, validate=False)
    except Exception as e:
        logger.debug(f"Literature gate: citations read error: {type(e).__name__}")
        payload = []

    for item in payload:
        if not isinstance(item, dict):
            invalid += 1
            continue

        if not is_valid_citation_record(item):
            invalid += 1
            continue

        if item.get("status") == "verified":
            verified += 1

    return verified, invalid, citations_file_present


def _count_evidence_items(project_folder: Path) -> Tuple[int, Dict[str, int], int, int]:
    """Return (total_valid_items, per_source_valid_counts, invalid_items, evidence_files_scanned)."""

    sources_dir = project_folder / "sources"
    if not sources_dir.exists():
        return 0, {}, 0, 0

    total_valid = 0
    invalid = 0
    evidence_files_scanned = 0
    per_source_counts: Dict[str, int] = {}

    for evidence_path in sorted(sources_dir.glob("*/evidence.json")):
        if not evidence_path.is_file():
            continue

        evidence_files_scanned += 1

        payload, err = _load_json_list(evidence_path)
        if err:
            logger.debug(f"Literature gate: evidence read error: {evidence_path}: {err}")
            continue

        source_id = evidence_path.parent.name
        valid_here = 0

        for item in payload:
            if not isinstance(item, dict):
                invalid += 1
                continue

            if not is_valid_evidence_item(item):
                invalid += 1
                continue

            valid_here += 1

        per_source_counts[source_id] = valid_here
        total_valid += valid_here

    return total_valid, per_source_counts, invalid, evidence_files_scanned


def check_literature_gate(
    *,
    project_folder: str | Path,
    config: Optional[LiteratureGateConfig] = None,
) -> Dict[str, Any]:
    """Check that literature prerequisites are met.

    Expected locations:
    - bibliography/citations.json: list[CitationRecord]
    - sources/*/evidence.json: list[EvidenceItem]

    Returns a dict with keys:
    - ok (bool)
    - enabled (bool)
    - action (pass|block|downgrade|disabled)
    - verified_citations (int)
    - min_verified_citations (int)
    - evidence_items_total (int)
    - min_evidence_items_total (int)
    - sources_below_min (list[str])
    - min_evidence_items_per_source (int)
    - citations_invalid_items (int)
    - evidence_invalid_items (int)
    - evidence_files_scanned (int)
    - citations_file_present (bool)
    """

    cfg = config or LiteratureGateConfig()
    pf = validate_project_folder(project_folder)

    verified_citations, citations_invalid, citations_file_present = _count_verified_citations(pf)

    evidence_items_total, per_source_counts, evidence_invalid, evidence_files_scanned = _count_evidence_items(pf)

    # If the gate is disabled, remain permissive.
    if not cfg.enabled:
        return {
            "ok": True,
            "enabled": False,
            "action": "disabled",
            "verified_citations": verified_citations,
            "min_verified_citations": cfg.min_verified_citations,
            "evidence_items_total": evidence_items_total,
            "min_evidence_items_total": cfg.min_evidence_items_total,
            "sources_below_min": [],
            "min_evidence_items_per_source": cfg.min_evidence_items_per_source,
            "citations_invalid_items": citations_invalid,
            "evidence_invalid_items": evidence_invalid,
            "evidence_files_scanned": evidence_files_scanned,
            "citations_file_present": citations_file_present,
        }

    sources_below_min: list[str] = []
    if cfg.min_evidence_items_per_source > 0:
        sources_below_min = sorted(
            [sid for sid, cnt in per_source_counts.items() if cnt < cfg.min_evidence_items_per_source]
        )

    meets_citations = verified_citations >= cfg.min_verified_citations
    meets_evidence_total = evidence_items_total >= cfg.min_evidence_items_total
    meets_per_source = not sources_below_min

    has_problem = not (meets_citations and meets_evidence_total and meets_per_source)

    action: Literal["pass", "block", "downgrade"] = "pass"
    ok = True

    if has_problem:
        if cfg.on_failure == "block":
            action = "block"
            ok = False
        else:
            action = "downgrade"

    return {
        "ok": ok,
        "enabled": True,
        "action": action,
        "verified_citations": verified_citations,
        "min_verified_citations": cfg.min_verified_citations,
        "evidence_items_total": evidence_items_total,
        "min_evidence_items_total": cfg.min_evidence_items_total,
        "sources_below_min": sources_below_min,
        "min_evidence_items_per_source": cfg.min_evidence_items_per_source,
        "citations_invalid_items": citations_invalid,
        "evidence_invalid_items": evidence_invalid,
        "evidence_files_scanned": evidence_files_scanned,
        "citations_file_present": citations_file_present,
    }


def enforce_literature_gate(
    *,
    project_folder: str | Path,
    config: Optional[LiteratureGateConfig] = None,
) -> Dict[str, Any]:
    """Enforce the literature gate.

    Raises:
        LiteratureGateError: if the gate is enabled and action=block.
    """

    result = check_literature_gate(project_folder=project_folder, config=config)
    if result.get("enabled") and result.get("action") == "block":
        raise LiteratureGateError(f"Literature gate blocked: {result}")
    return result
