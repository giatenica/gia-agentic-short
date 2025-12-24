"""Run the literature gate on a project folder.

This helper is local/offline and intended for quick validation.

Checks (when enabled):
- minimum number of verified citation records in bibliography/citations.json
- minimum number of evidence items across sources/*/evidence.json
- optional: minimum evidence items per evidence source
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the literature gate")
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
        help="Behavior when thresholds are not met (default: block)",
    )
    parser.add_argument("--min-verified-citations", type=int, default=1)
    parser.add_argument("--min-evidence-items-total", type=int, default=1)
    parser.add_argument("--min-evidence-items-per-source", type=int, default=0)

    args = parser.parse_args()

    # Allow running this script directly without requiring installation.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.literature.gates import LiteratureGateConfig, LiteratureGateError, enforce_literature_gate

    cfg = LiteratureGateConfig(
        enabled=bool(args.enabled),
        on_failure=args.on_failure,
        min_verified_citations=max(0, int(args.min_verified_citations)),
        min_evidence_items_total=max(0, int(args.min_evidence_items_total)),
        min_evidence_items_per_source=max(0, int(args.min_evidence_items_per_source)),
    )

    try:
        result = enforce_literature_gate(project_folder=args.project_folder, config=cfg)
    except LiteratureGateError as e:
        print(str(e))
        return 2

    print(f"project_folder: {args.project_folder}")
    print(f"enabled: {result.get('enabled')}")
    print(f"action: {result.get('action')}")
    print(f"verified_citations: {result.get('verified_citations')}")
    print(f"evidence_items_total: {result.get('evidence_items_total')}")
    print(f"sources_below_min: {len(result.get('sources_below_min') or [])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
