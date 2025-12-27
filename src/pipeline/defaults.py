"""Shared default configuration for pipeline components.

This module centralizes default gate configurations to avoid duplication
across runner.py and standalone scripts.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

from typing import Any, Dict


def default_gate_config() -> Dict[str, Dict[str, Any]]:
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
