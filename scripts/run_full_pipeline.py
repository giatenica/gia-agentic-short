#!/usr/bin/env python3
"""Run the unified pipeline for a project folder.

This runner chains:
- Phase 1: ResearchWorkflow
- Phase 2: LiteratureWorkflow
- Phase 3: GapResolutionWorkflow (optional)
- Phase 4: Writing + referee review stage (optional)

It writes:
- full_pipeline_context.json

Exit code behavior:
- Exits 1 only for CLI usage errors.
- Otherwise exits 0 and writes success/errors into the context file.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
root_str = str(ROOT_DIR)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from src.llm.claude_client import load_env_file_lenient  # noqa: E402

load_env_file_lenient()

from src.pipeline.runner import run_full_pipeline  # noqa: E402


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable types gracefully."""
    
    def default(self, obj):
        # Handle pandas Timestamp
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # Handle datetime
        if hasattr(obj, 'strftime'):
            return obj.strftime('%Y-%m-%dT%H:%M:%S')
        # Handle Path objects
        if hasattr(obj, '__fspath__'):
            return str(obj)
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        # Fallback to string representation
        try:
            return str(obj)
        except Exception:
            return f"<non-serializable: {type(obj).__name__}>"


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_full_pipeline.py <project_folder>")
        sys.exit(1)

    project_folder = Path(sys.argv[1]).expanduser().resolve()
    print(f"Starting full pipeline for: {project_folder}", flush=True)

    ctx = await run_full_pipeline(str(project_folder))

    out_path = project_folder / "full_pipeline_context.json"
    out_path.write_text(json.dumps(ctx.to_payload(), indent=2, sort_keys=True, cls=SafeJSONEncoder) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Success: {ctx.success}")
    print(f"Context: {out_path}")
    if ctx.errors:
        print("\nErrors:")
        for e in ctx.errors:
            print(f"  - {e}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
