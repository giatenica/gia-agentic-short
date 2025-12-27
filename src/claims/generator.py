"""Claims generation from computed metrics.

Generates claims/claims.json from outputs/metrics.json so downstream gates and
writers can validate computed statements against canonical metric keys.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.utils.schema_validation import is_valid_metric_record


def _utc_now_iso_z() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _load_json_list(path: Path) -> Tuple[List[Any], Optional[str]]:
    if not path.exists():
        return [], None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        return [], f"{type(e).__name__}"

    if not isinstance(payload, list):
        return [], "not_a_list"

    return payload, None


def _build_claim_from_metric(metric: Dict[str, Any]) -> Dict[str, Any]:
    metric_key = str(metric.get("metric_key") or "").strip()
    name = str(metric.get("name") or metric_key).strip() or metric_key
    value = metric.get("value")
    unit = str(metric.get("unit") or "").strip()

    value_s = _format_metric_value(value)
    tail = f" {unit}" if unit else ""

    statement = f"{name} is {value_s}{tail}."

    created_at = metric.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        created_at = _utc_now_iso_z()

    return {
        "schema_version": "1.0",
        "claim_id": f"computed:{metric_key}",
        "kind": "computed",
        "statement": statement,
        "metric_keys": [metric_key],
        "created_at": created_at,
        "metadata": {
            "metric_key": metric_key,
            "name": name,
            "value": value,
            "unit": unit,
            "description": metric.get("description"),
        },
    }


def generate_claims_from_metrics(
    *,
    project_folder: str | Path,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Generate claims/claims.json from outputs/metrics.json.

    This function is deterministic and filesystem-first.

    Returns a structured summary; it does not raise for missing files.
    """

    pf = Path(project_folder).expanduser().resolve()
    if not pf.exists() or not pf.is_dir():
        raise ValueError(f"Project folder not found: {pf}")

    outputs_dir = pf / "outputs"
    claims_dir = pf / "claims"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    claims_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = outputs_dir / "metrics.json"
    claims_path = claims_dir / "claims.json"

    metrics_payload, metrics_error = _load_json_list(metrics_path)
    if metrics_error:
        logger.debug(f"Claims generation: metrics read error: {metrics_error}")

    valid_metrics: List[Dict[str, Any]] = []
    invalid_metrics = 0

    for item in metrics_payload:
        if not isinstance(item, dict) or not is_valid_metric_record(item):
            invalid_metrics += 1
            continue
        valid_metrics.append(item)

    claims: List[Dict[str, Any]] = []
    for rec in valid_metrics:
        metric_key = rec.get("metric_key")
        if not isinstance(metric_key, str) or not metric_key.strip():
            continue
        claims.append(_build_claim_from_metric(rec))

    claims.sort(key=lambda c: str(c.get("claim_id") or ""))

    if claims_path.exists() and not overwrite:
        return {
            "ok": True,
            "action": "skipped",
            "metrics_file_present": metrics_path.exists(),
            "claims_file_present": True,
            "claims_written": 0,
            "metrics_total": len(metrics_payload),
            "metrics_invalid_items": invalid_metrics,
        }

    claims_path.write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "action": "written",
        "metrics_file_present": metrics_path.exists(),
        "claims_file_present": True,
        "claims_written": len(claims),
        "metrics_total": len(metrics_payload),
        "metrics_invalid_items": invalid_metrics,
    }
