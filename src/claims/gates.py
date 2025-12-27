"""Computation gate.

Checks that computed claims only reference metrics that exist in
outputs/metrics.json.

Default policy:
- Enabled by default in downgrade mode so missing metrics are reported without
    blocking early pipeline runs.
- Users can switch to blocking mode by setting on_missing_metrics to "block".

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

from src.utils.schema_validation import is_valid_claim_record, is_valid_metric_record
from src.tracing import safe_set_current_span_attributes
from src.utils.validation import validate_project_folder


class ComputationGateError(ValueError):
    """Raised when the computation gate blocks execution."""


OnFailureAction = Literal["block", "downgrade"]


@dataclass(frozen=True)
class ComputationGateConfig:
    """Configuration for computed-claim enforcement."""

    enabled: bool = True
    on_missing_metrics: OnFailureAction = "downgrade"

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "ComputationGateConfig":
        raw = context.get("computation_gate")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", False))
        on_missing_metrics = raw.get("on_missing_metrics", "block")
        if on_missing_metrics not in ("block", "downgrade"):
            on_missing_metrics = "block"

        return cls(enabled=enabled, on_missing_metrics=on_missing_metrics)


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


def _find_computed_metric_keys(claims: List[Any]) -> Tuple[set[str], int, int]:
    metric_keys: set[str] = set()
    computed_claims = 0
    invalid_claims = 0

    for item in claims:
        if not isinstance(item, dict):
            invalid_claims += 1
            continue

        if not is_valid_claim_record(item):
            invalid_claims += 1
            continue

        if str(item.get("kind")) != "computed":
            continue

        computed_claims += 1
        keys = item.get("metric_keys")
        if isinstance(keys, list):
            for k in keys:
                if isinstance(k, str) and k.strip():
                    metric_keys.add(k.strip())

    return metric_keys, computed_claims, invalid_claims


def _load_metric_keys(metrics: List[Any]) -> Tuple[set[str], int]:
    metric_keys: set[str] = set()
    invalid_metrics = 0

    for item in metrics:
        if not isinstance(item, dict):
            invalid_metrics += 1
            continue

        if not is_valid_metric_record(item):
            invalid_metrics += 1
            continue

        k = item.get("metric_key")
        if isinstance(k, str) and k.strip():
            metric_keys.add(k.strip())

    return metric_keys, invalid_metrics


def check_computation_gate(
    *,
    project_folder: str | Path,
    config: Optional[ComputationGateConfig] = None,
) -> Dict[str, Any]:
    """Check that computed claims reference existing metrics.

    Expected locations:
    - claims/claims.json: list[ClaimRecord]
    - outputs/metrics.json: list[MetricRecord]

    Returns a dict with keys:
    - ok (bool)
    - enabled (bool)
    - action (pass|block|downgrade|disabled)
    - missing_metric_keys (list)
    - referenced_metric_keys_total (int)
    - computed_claims_total (int)
    - claims_invalid_items (int)
    - metrics_invalid_items (int)
    - metrics_file_present (bool)
    - claims_file_present (bool)
    """

    cfg = config or ComputationGateConfig()
    pf = validate_project_folder(project_folder)

    claims_path = pf / "claims" / "claims.json"
    claims_payload, claims_error = _load_json_list(claims_path)
    if claims_error:
        logger.debug(f"Computation gate: claims read error: {claims_error}")

    referenced_metric_keys, computed_claims_total, claims_invalid = _find_computed_metric_keys(claims_payload)

    metrics_path = pf / "outputs" / "metrics.json"

    # If the gate is disabled, remain permissive.
    if not cfg.enabled:
        result = {
            "ok": True,
            "enabled": False,
            "action": "disabled",
            "missing_metric_keys": [],
            "referenced_metric_keys_total": len(referenced_metric_keys),
            "computed_claims_total": computed_claims_total,
            "claims_invalid_items": claims_invalid,
            "metrics_invalid_items": 0,
            "metrics_file_present": metrics_path.exists(),
            "claims_file_present": claims_path.exists(),
        }

        safe_set_current_span_attributes(
            {
                "gate.name": "computation",
                "gate.enabled": False,
                "gate.ok": True,
                "gate.action": "disabled",
                "computation_gate.referenced_metric_keys_total": int(len(referenced_metric_keys)),
                "computation_gate.computed_claims_total": int(computed_claims_total),
                "computation_gate.missing_metric_keys_total": 0,
                "computation_gate.metrics_file_present": bool(metrics_path.exists()),
                "computation_gate.claims_file_present": bool(claims_path.exists()),
            }
        )

        return result

    # If there are no computed claims, there is nothing to check.
    if not referenced_metric_keys:
        result = {
            "ok": True,
            "enabled": True,
            "action": "pass",
            "missing_metric_keys": [],
            "referenced_metric_keys_total": 0,
            "computed_claims_total": computed_claims_total,
            "claims_invalid_items": claims_invalid,
            "metrics_invalid_items": 0,
            "metrics_file_present": metrics_path.exists(),
            "claims_file_present": claims_path.exists(),
        }

        safe_set_current_span_attributes(
            {
                "gate.name": "computation",
                "gate.enabled": True,
                "gate.ok": True,
                "gate.action": "pass",
                "computation_gate.referenced_metric_keys_total": 0,
                "computation_gate.computed_claims_total": int(computed_claims_total),
                "computation_gate.missing_metric_keys_total": 0,
                "computation_gate.metrics_file_present": bool(metrics_path.exists()),
                "computation_gate.claims_file_present": bool(claims_path.exists()),
            }
        )

        return result

    metrics_payload, metrics_error = _load_json_list(metrics_path)
    if metrics_error:
        logger.debug(f"Computation gate: metrics read error: {metrics_error}")

    known_metric_keys, metrics_invalid = _load_metric_keys(metrics_payload)

    missing = sorted(referenced_metric_keys - known_metric_keys)

    action: Literal["pass", "block", "downgrade"] = "pass"
    ok = True

    if missing:
        if cfg.on_missing_metrics == "block":
            action = "block"
            ok = False
        else:
            action = "downgrade"

    result = {
        "ok": ok,
        "enabled": True,
        "action": action,
        "missing_metric_keys": missing,
        "referenced_metric_keys_total": len(referenced_metric_keys),
        "computed_claims_total": computed_claims_total,
        "claims_invalid_items": claims_invalid,
        "metrics_invalid_items": metrics_invalid,
        "metrics_file_present": metrics_path.exists(),
        "claims_file_present": claims_path.exists(),
    }

    safe_set_current_span_attributes(
        {
            "gate.name": "computation",
            "gate.enabled": True,
            "gate.ok": bool(ok),
            "gate.action": str(action),
            "computation_gate.on_missing_metrics": str(cfg.on_missing_metrics),
            "computation_gate.referenced_metric_keys_total": int(len(referenced_metric_keys)),
            "computation_gate.computed_claims_total": int(computed_claims_total),
            "computation_gate.missing_metric_keys_total": int(len(missing)),
            "computation_gate.metrics_file_present": bool(metrics_path.exists()),
            "computation_gate.claims_file_present": bool(claims_path.exists()),
        }
    )

    return result


def enforce_computation_gate(
    *,
    project_folder: str | Path,
    config: Optional[ComputationGateConfig] = None,
) -> Dict[str, Any]:
    """Enforce the computation gate.

    Raises:
        ComputationGateError: if the gate is enabled and action=block.
    """

    result = check_computation_gate(project_folder=project_folder, config=config)
    if result.get("enabled") and result.get("action") == "block":
        raise ComputationGateError(f"Computation gate blocked: {result}")
    return result
