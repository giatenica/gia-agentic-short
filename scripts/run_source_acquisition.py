#!/usr/bin/env python3
"""Ingest a project sources list into the evidence layout.

Usage:
  python scripts/run_source_acquisition.py <project_folder> [sources_list_path]

If sources_list_path is omitted, the script tries common defaults:
- sources_list.json
- sources.json
- inputs/sources.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.llm.claude_client import load_env_file_lenient  # noqa: E402

load_env_file_lenient()

from src.evidence.acquisition import find_default_sources_list_path, ingest_sources_list  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_source_acquisition.py <project_folder> [sources_list_path]", flush=True)
        raise SystemExit(1)

    project_folder = sys.argv[1]
    rel = sys.argv[2] if len(sys.argv) > 2 else None

    if not rel:
        rel = find_default_sources_list_path(project_folder)

    if not rel:
        print("No sources list found (expected sources_list.json, sources.json, or inputs/sources.json)", flush=True)
        raise SystemExit(2)

    summary = ingest_sources_list(project_folder=project_folder, sources_list_path=rel)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    raise SystemExit(0 if summary.get("ok") else 3)


if __name__ == "__main__":
    main()
