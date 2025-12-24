"""Claim to evidence alignment gate.

Checks that source-backed claims include evidence_ids and that those IDs
exist in sources/*/evidence.json, and that citation_keys exist in
bibliography/citations.json.

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
from src.utils.schema_validation import is_valid_claim_record, is_valid_citation_record, is_valid_evidence_item
from src.utils.validation import validate_project_folder


class ClaimEvidenceGateError(ValueError):
    """Raised when the claim-evidence gate blocks execution."""


OnFailureAction = Literal["block", "downgrade"]


@dataclass(frozen=True)
class ClaimEvidenceGateConfig:
    """Configuration for claim to evidence alignment enforcement."""

    enabled: bool = False
    on_failure: OnFailureAction = "block"

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "ClaimEvidenceGateConfig":
        raw = context.get("claim_evidence_gate")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", False))
        on_failure = raw.get("on_failure", "block")
        if on_failure not in ("block", "downgrade"):
            on_failure = "block"

        return cls(enabled=enabled, on_failure=on_failure)


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


def _find_source_backed_refs(claims: List[Any]) -> Tuple[set[str], set[str], List[str], int, int]:
    """Return (evidence_ids, citation_keys, missing_evidence_claim_ids, source_backed_claims_total, invalid_claims)."""

    evidence_ids: set[str] = set()
    citation_keys: set[str] = set()
    missing_evidence_claim_ids: list[str] = []
    source_backed_claims_total = 0
    invalid_claims = 0

    for item in claims:
        if not isinstance(item, dict):
            invalid_claims += 1
            continue

        if not is_valid_claim_record(item):
            invalid_claims += 1
            continue

        if str(item.get("kind")) != "source_backed":
            continue

        source_backed_claims_total += 1

        claim_id = str(item.get("claim_id") or "").strip()

        raw_evidence_ids = item.get("evidence_ids")
        if not isinstance(raw_evidence_ids, list) or not any(
            isinstance(x, str) and x.strip() for x in raw_evidence_ids
        ):
            if claim_id:
                missing_evidence_claim_ids.append(claim_id)
        else:
            for ev_id in raw_evidence_ids:
                if isinstance(ev_id, str) and ev_id.strip():
                    evidence_ids.add(ev_id.strip())

        raw_citation_keys = item.get("citation_keys")
        if isinstance(raw_citation_keys, list):
            for key in raw_citation_keys:
                if isinstance(key, str) and key.strip():
                    citation_keys.add(key.strip())

    missing_evidence_claim_ids = sorted(set(missing_evidence_claim_ids))
    return evidence_ids, citation_keys, missing_evidence_claim_ids, source_backed_claims_total, invalid_claims


def _load_known_citation_keys(project_folder: Path) -> Tuple[set[str], int, bool]:
    invalid = 0
    known: set[str] = set()

    citations_file_present = (project_folder / "bibliography" / "citations.json").exists()

    try:
        citations_payload = load_citations(project_folder, validate=False)
    except Exception as e:
        logger.debug(f"Claim-evidence gate: citations read error: {type(e).__name__}")
        citations_payload = []

    for item in citations_payload:
        if not isinstance(item, dict):
            invalid += 1
            continue

        if not is_valid_citation_record(item):
            invalid += 1
            continue

        key = item.get("citation_key")
        if isinstance(key, str) and key.strip():
            known.add(key.strip())

    return known, invalid, citations_file_present


def _load_known_evidence_ids(project_folder: Path) -> Tuple[set[str], int, int]:
    """Return (known_evidence_ids, invalid_evidence_items, evidence_files_scanned)."""

    sources_dir = project_folder / "sources"
    if not sources_dir.exists():
        return set(), 0, 0

    known: set[str] = set()
    invalid_items = 0
    evidence_files_scanned = 0

    for evidence_path in sorted(sources_dir.glob("*/evidence.json")):
        if not evidence_path.is_file():
            continue

        evidence_files_scanned += 1
        payload, err = _load_json_list(evidence_path)
        if err:
            logger.debug(f"Claim-evidence gate: evidence read error: {evidence_path}: {err}")
            continue

        for item in payload:
            if not isinstance(item, dict):
                invalid_items += 1
                continue

            if not is_valid_evidence_item(item):
                invalid_items += 1
                continue

            ev_id = item.get("evidence_id")
            if isinstance(ev_id, str) and ev_id.strip():
                known.add(ev_id.strip())

    return known, invalid_items, evidence_files_scanned


def check_claim_evidence_gate(
    *,
    project_folder: str | Path,
    config: Optional[ClaimEvidenceGateConfig] = None,
) -> Dict[str, Any]:
    """Check source-backed claims for evidence and citation alignment.

    Expected locations:
    - claims/claims.json: list[ClaimRecord]
    - sources/*/evidence.json: list[EvidenceItem]
    - bibliography/citations.json: list[CitationRecord]

    Returns a dict with keys:
    - ok (bool)
    - enabled (bool)
    - action (pass|block|downgrade|disabled)
    - missing_evidence_claim_ids (list)
    - missing_evidence_ids (list)
    - missing_citation_keys (list)
    - referenced_evidence_ids_total (int)
    - referenced_citation_keys_total (int)
    - source_backed_claims_total (int)
    - claims_invalid_items (int)
    - evidence_invalid_items (int)
    - citations_invalid_items (int)
    - evidence_files_scanned (int)
    - citations_file_present (bool)
    - claims_file_present (bool)
    """

    cfg = config or ClaimEvidenceGateConfig()
    pf = validate_project_folder(project_folder)

    claims_path = pf / "claims" / "claims.json"
    claims_payload, claims_error = _load_json_list(claims_path)
    if claims_error:
        logger.debug(f"Claim-evidence gate: claims read error: {claims_error}")

    referenced_evidence_ids, referenced_citation_keys, missing_evidence_claim_ids, source_backed_claims_total, claims_invalid = (
        _find_source_backed_refs(claims_payload)
    )

    # If the gate is disabled, remain permissive.
    if not cfg.enabled:
        return {
            "ok": True,
            "enabled": False,
            "action": "disabled",
            "missing_evidence_claim_ids": [],
            "missing_evidence_ids": [],
            "missing_citation_keys": [],
            "referenced_evidence_ids_total": len(referenced_evidence_ids),
            "referenced_citation_keys_total": len(referenced_citation_keys),
            "source_backed_claims_total": source_backed_claims_total,
            "claims_invalid_items": claims_invalid,
            "evidence_invalid_items": 0,
            "citations_invalid_items": 0,
            "evidence_files_scanned": 0,
            "citations_file_present": (pf / "bibliography" / "citations.json").exists(),
            "claims_file_present": claims_path.exists(),
        }

    # If there are no source-backed claims, there is nothing to check.
    if source_backed_claims_total == 0:
        return {
            "ok": True,
            "enabled": True,
            "action": "pass",
            "missing_evidence_claim_ids": [],
            "missing_evidence_ids": [],
            "missing_citation_keys": [],
            "referenced_evidence_ids_total": 0,
            "referenced_citation_keys_total": 0,
            "source_backed_claims_total": 0,
            "claims_invalid_items": claims_invalid,
            "evidence_invalid_items": 0,
            "citations_invalid_items": 0,
            "evidence_files_scanned": 0,
            "citations_file_present": (pf / "bibliography" / "citations.json").exists(),
            "claims_file_present": claims_path.exists(),
        }

    known_evidence_ids, evidence_invalid, evidence_files_scanned = _load_known_evidence_ids(pf)
    known_citation_keys, citations_invalid, citations_file_present = _load_known_citation_keys(pf)

    missing_evidence_ids = sorted(referenced_evidence_ids - known_evidence_ids)
    missing_citation_keys = sorted(referenced_citation_keys - known_citation_keys)

    has_problem = bool(missing_evidence_claim_ids or missing_evidence_ids or missing_citation_keys)

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
        "missing_evidence_claim_ids": missing_evidence_claim_ids,
        "missing_evidence_ids": missing_evidence_ids,
        "missing_citation_keys": missing_citation_keys,
        "referenced_evidence_ids_total": len(referenced_evidence_ids),
        "referenced_citation_keys_total": len(referenced_citation_keys),
        "source_backed_claims_total": source_backed_claims_total,
        "claims_invalid_items": claims_invalid,
        "evidence_invalid_items": evidence_invalid,
        "citations_invalid_items": citations_invalid,
        "evidence_files_scanned": evidence_files_scanned,
        "citations_file_present": citations_file_present,
        "claims_file_present": claims_path.exists(),
    }


def enforce_claim_evidence_gate(
    *,
    project_folder: str | Path,
    config: Optional[ClaimEvidenceGateConfig] = None,
) -> Dict[str, Any]:
    """Enforce the claim-evidence gate.

    Raises:
        ClaimEvidenceGateError: if the gate is enabled and action=block.
    """

    result = check_claim_evidence_gate(project_folder=project_folder, config=config)
    if result.get("enabled") and result.get("action") == "block":
        raise ClaimEvidenceGateError(f"Claim-evidence gate blocked: {result}")
    return result
