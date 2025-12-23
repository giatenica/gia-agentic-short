"""Run the citation gate on a project folder.

This helper is local/offline and intended for quick validation:
- scans Markdown/LaTeX files for citation keys
- checks them against bibliography/citations.json
- reports pass/block/downgrade
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the citation linter gate")
    parser.add_argument("project_folder", help="Path to a project folder containing project.json")
    parser.add_argument(
        "--enabled",
        action="store_true",
        help="Enable enforcement (default: false). If false, gate is permissive.",
    )
    parser.add_argument(
        "--on-missing",
        choices=["block", "downgrade"],
        default="block",
        help="Behavior when a cited key is missing from citations.json (default: block)",
    )
    parser.add_argument(
        "--on-unverified",
        choices=["block", "downgrade"],
        default="downgrade",
        help="Behavior when a cited key exists but status!=verified (default: downgrade)",
    )
    args = parser.parse_args()

    # Allow running this script directly without requiring installation.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.citations.gates import CitationGateConfig, CitationGateError, enforce_citation_gate

    cfg = CitationGateConfig(enabled=bool(args.enabled), on_missing=args.on_missing, on_unverified=args.on_unverified)

    try:
        result = enforce_citation_gate(project_folder=args.project_folder, config=cfg)
    except CitationGateError as e:
        print(str(e))
        return 2

    print(f"project_folder: {args.project_folder}")
    print(f"enabled: {result.get('enabled')}")
    print(f"action: {result.get('action')}")
    print(f"referenced_keys_total: {result.get('referenced_keys_total')}")
    print(f"missing_keys: {len(result.get('missing_keys') or [])}")
    print(f"unverified_keys: {len(result.get('unverified_keys') or [])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
