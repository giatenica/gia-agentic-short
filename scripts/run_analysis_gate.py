"""Run the analysis gate on a project folder.

This helper is local/offline and intended for quick validation.

Checks (when enabled):
- outputs/metrics.json exists and validates as MetricRecord list
- optional: outputs/tables/ contains at least one table artifact
- optional: outputs/figures/ contains at least one figure artifact
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the analysis gate")
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
        help="Behavior when readiness checks fail (default: block)",
    )
    parser.add_argument("--min-metrics", type=int, default=1)
    parser.add_argument("--require-tables", action="store_true")
    parser.add_argument("--require-figures", action="store_true")

    args = parser.parse_args()

    # Allow running this script directly without requiring installation.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.analysis.gates import AnalysisGateConfig, AnalysisGateError, enforce_analysis_gate

    cfg = AnalysisGateConfig(
        enabled=bool(args.enabled),
        on_failure=args.on_failure,
        min_metrics=max(0, int(args.min_metrics)),
        require_tables=bool(args.require_tables),
        require_figures=bool(args.require_figures),
    )

    try:
        result = enforce_analysis_gate(project_folder=args.project_folder, config=cfg)
    except AnalysisGateError as e:
        print(str(e))
        return 2

    print(f"project_folder: {args.project_folder}")
    print(f"enabled: {result.get('enabled')}")
    print(f"action: {result.get('action')}")
    print(f"metrics_file_present: {result.get('metrics_file_present')}")
    print(f"metrics_valid_items: {result.get('metrics_valid_items')}")
    print(f"tables_count: {result.get('tables_count')}")
    print(f"figures_count: {result.get('figures_count')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
