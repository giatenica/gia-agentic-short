"""Evidence gates.

Gates are small checks that can be used to block synthesis or writing steps when
required evidence is missing.

The default policy is permissive when not explicitly enabled.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from src.evidence.store import EvidenceStore
from src.tracing import safe_set_current_span_attributes


class EvidenceGateError(ValueError):
    """Raised when an evidence gate fails."""


@dataclass(frozen=True)
class EvidenceGateConfig:
    """Configuration for evidence enforcement."""

    require_evidence: bool = False
    min_items_per_source: int = 1

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "EvidenceGateConfig":
        raw = context.get("evidence_gate")
        if not isinstance(raw, dict):
            return cls()

        require_evidence = bool(raw.get("require_evidence", False))
        min_items_per_source = int(raw.get("min_items_per_source", 1))
        if min_items_per_source < 1:
            min_items_per_source = 1

        return cls(require_evidence=require_evidence, min_items_per_source=min_items_per_source)


def _discover_source_ids_from_ledger(store: EvidenceStore) -> List[str]:
    source_ids: set[str] = set()
    for item in store.iter_items(validate=False):
        sid = item.get("source_id")
        if isinstance(sid, str) and sid:
            source_ids.add(sid)
    return sorted(source_ids)


def check_evidence_gate(
    *,
    project_folder: str,
    source_ids: Optional[Iterable[str]] = None,
    config: Optional[EvidenceGateConfig] = None,
) -> Dict[str, Any]:
    """Check whether the project has enough evidence.

    Args:
        project_folder: Project folder path.
        source_ids: Optional iterable of source IDs to check. If omitted, checks all
            sources present under the evidence layout.
        config: EvidenceGateConfig; defaults to require_evidence=False.

    Returns:
        Dict with keys: ok (bool), per_source (dict), total_items (int).

    Notes:
        This function does not raise. Use enforce_evidence_gate to raise.
    """

    cfg = config or EvidenceGateConfig()
    store = EvidenceStore(project_folder)
    if source_ids is None:
        source_ids = _discover_source_ids_from_ledger(store)

    per_source: Dict[str, Dict[str, Any]] = {}
    total = 0

    for source_id in source_ids:
        try:
            items = store.read_evidence_items(source_id, validate=True)
            count = len(items)
            total += count
            per_source[source_id] = {"count": count, "ok": count >= cfg.min_items_per_source}
        except FileNotFoundError:
            per_source[source_id] = {"count": 0, "ok": False, "missing": True}

    all_ok = all(entry.get("ok", False) for entry in per_source.values()) if per_source else False

    sources_total = len(per_source)
    sources_ok = sum(1 for v in per_source.values() if v.get("ok") is True)
    sources_missing = sum(1 for v in per_source.values() if v.get("missing") is True)

    if not cfg.require_evidence:
        result = {"ok": True, "per_source": per_source, "total_items": total, "require_evidence": False}
    else:
        result = {"ok": all_ok, "per_source": per_source, "total_items": total, "require_evidence": True}

    safe_set_current_span_attributes(
        {
            "gate.name": "evidence",
            "gate.enabled": bool(cfg.require_evidence),
            "gate.ok": bool(result.get("ok")),
            "evidence_gate.min_items_per_source": int(cfg.min_items_per_source),
            "evidence_gate.sources_total": sources_total,
            "evidence_gate.sources_ok": sources_ok,
            "evidence_gate.sources_missing": sources_missing,
            "evidence_gate.total_items": int(total),
        }
    )

    return result


def enforce_evidence_gate(
    *,
    project_folder: str,
    source_ids: Optional[Iterable[str]] = None,
    config: Optional[EvidenceGateConfig] = None,
) -> Dict[str, Any]:
    """Enforce the evidence gate.

    Raises:
        EvidenceGateError: if require_evidence is enabled and evidence is missing.
    """

    result = check_evidence_gate(project_folder=project_folder, source_ids=source_ids, config=config)
    if result.get("require_evidence") and not result.get("ok"):
        raise EvidenceGateError(f"Evidence gate failed: {result}")
    return result
