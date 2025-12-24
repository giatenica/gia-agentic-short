"""Run the claim to evidence alignment gate on a project folder.

This helper is local/offline and intended for quick validation:
- loads claims/claims.json
- verifies source-backed claims include evidence_ids
- verifies evidence_ids exist in sources/*/evidence.json
- verifies citation_keys exist in bibliography/citations.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the claim to evidence alignment gate")
    parser.add_argument("project_folder", help="Path to a project folder containing project.json")
    parser.add_argument(
        "--enabled",
        action="store_true",
        help="Enable enforcement (default: false). If false, gate is permissive.",
    )
    parser.add_argument(
        "--on-failure",
        choices=["block", "downgrade"],
        default="block",
        help="Behavior when alignment fails (default: block)",
    )

    args = parser.parse_args()

    # Allow running this script directly without requiring installation.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.claims.claim_evidence_gate import (
        ClaimEvidenceGateConfig,
        ClaimEvidenceGateError,
        enforce_claim_evidence_gate,
    )

    cfg = ClaimEvidenceGateConfig(enabled=bool(args.enabled), on_failure=args.on_failure)

    try:
        result = enforce_claim_evidence_gate(project_folder=args.project_folder, config=cfg)
    except ClaimEvidenceGateError as e:
        print(str(e))
        return 2

    print(f"project_folder: {args.project_folder}")
    print(f"enabled: {result.get('enabled')}")
    print(f"action: {result.get('action')}")
    print(f"source_backed_claims_total: {result.get('source_backed_claims_total')}")
    print(f"missing_evidence_claims: {len(result.get('missing_evidence_claim_ids') or [])}")
    print(f"missing_evidence_ids: {len(result.get('missing_evidence_ids') or [])}")
    print(f"missing_citation_keys: {len(result.get('missing_citation_keys') or [])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
