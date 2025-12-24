#!/usr/bin/env python3
"""Run an evaluation sweep over evaluation/test_queries.json.

This script is designed to be safe by default:
- default mode is dry-run (no LLM calls)
- outputs are materialized under outputs/evaluation_suite/<run_id>/

Use --mode phase1 (or phase1+phase2) to run the live workflows.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument(
        "--queries",
        default=str(Path("evaluation") / "test_queries.json"),
        help="Path to evaluation/test_queries.json",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path("outputs") / "evaluation_suite"),
        help="Where to write suite results",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id (default: UTC timestamp)",
    )
    parser.add_argument(
        "--mode",
        choices=["dry", "phase1", "phase1+phase2"],
        default="dry",
        help="Run mode. dry skips workflows; phase1 runs ResearchWorkflow; phase1+phase2 also runs LiteratureWorkflow.",
    )
    parser.add_argument(
        "--enable-edison",
        action="store_true",
        help="Enable Edison usage for Phase 2 (default: Edison disabled by passing api_key=None)",
    )

    return parser.parse_args()


async def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Only load .env when running live workflows.
    if args.mode != "dry":
        from src.llm.claude_client import load_env_file_lenient

        load_env_file_lenient()

    from src.evaluation.suite_runner import EvaluationSuiteConfig, load_test_queries, run_evaluation_suite

    queries = load_test_queries(args.queries)
    cfg = EvaluationSuiteConfig(
        mode=args.mode,
        output_root=Path(args.output_root),
        disable_edison_by_default=not bool(args.enable_edison),
    )

    report, report_path = await run_evaluation_suite(queries=queries, config=cfg, run_id=args.run_id)

    print(f"mode: {report.mode}")
    print(f"queries_total: {report.queries_total}")
    print(f"queries_success: {report.queries_success}")
    print(f"report_path: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
