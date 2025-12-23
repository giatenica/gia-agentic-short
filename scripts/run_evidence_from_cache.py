"""Run the evidence pipeline from cached workflow stage outputs.

This is a local/offline helper that takes an existing project folder containing
`.workflow_cache/<stage>.json`, extracts a text payload, runs:
- MVP parser -> `sources/<source_id>/parsed.json`
- deterministic evidence extraction -> `sources/<source_id>/evidence.json`
- (optional) append to `.evidence/evidence.jsonl` ledger
- evidence gate check

It is designed to be idempotent by default: if per-source artifacts already
exist, it will reuse them unless `--force` is provided.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# Allow running this script directly without requiring installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evidence.extraction import extract_evidence_items
from src.evidence.gates import EvidenceGateConfig, check_evidence_gate
from src.evidence.parser import MVPLineBlockParser
from src.evidence.store import EvidenceStore
from src.utils.validation import validate_project_folder


def _find_first_str_by_key(obj: Any, *, keys: Iterable[str], max_depth: int = 8) -> Optional[str]:
    if max_depth <= 0:
        return None

    if isinstance(obj, dict):
        for key in keys:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val

        for val in obj.values():
            found = _find_first_str_by_key(val, keys=keys, max_depth=max_depth - 1)
            if found:
                return found

    if isinstance(obj, list):
        for val in obj:
            found = _find_first_str_by_key(val, keys=keys, max_depth=max_depth - 1)
            if found:
                return found

    return None


def _created_at_from_payload(payload: dict[str, Any]) -> str:
    raw_ts = payload.get("timestamp")
    if isinstance(raw_ts, str) and raw_ts:
        dt = datetime.fromisoformat(raw_ts)
    else:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.isoformat().replace("+00:00", "Z")


def _ledger_has_source_id(store: EvidenceStore, source_id: str) -> bool:
    for item in store.iter_items(validate=False) or []:
        if item.get("source_id") == source_id:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evidence pipeline from cached stage JSON")
    parser.add_argument("project_folder", help="Path to an existing project folder (must contain project.json)")
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="Process every .workflow_cache/*.json stage file (skips *.lock).",
    )
    parser.add_argument(
        "--stage",
        default="literature_search",
        help="Cache stage name (reads .workflow_cache/<stage>.json). Default: literature_search",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="Evidence source_id to write under sources/. Default: cache:<stage>",
    )
    parser.add_argument("--max-items", type=int, default=25, help="Max evidence items to extract. Default: 25")
    parser.add_argument(
        "--append-ledger",
        action="store_true",
        help="Append extracted items to .evidence/evidence.jsonl if not already present",
    )
    parser.add_argument("--force", action="store_true", help="Re-generate parsed/evidence artifacts")
    parser.add_argument(
        "--require-evidence",
        action="store_true",
        help="Require evidence gate to pass (default: true)",
    )
    parser.add_argument(
        "--no-require-evidence",
        dest="require_evidence",
        action="store_false",
        help="Do not require evidence gate to pass",
    )
    parser.set_defaults(require_evidence=True)
    parser.add_argument(
        "--min-items-per-source",
        type=int,
        default=1,
        help="Evidence gate min items per source. Default: 1",
    )

    args = parser.parse_args()

    project_folder = validate_project_folder(Path(args.project_folder))

    def run_one_stage(stage_name: str) -> dict[str, Any]:
        stage_source_id = args.source_id or f"cache:{stage_name}"
        cache_file = project_folder / ".workflow_cache" / f"{stage_name}.json"
        if not cache_file.exists():
            return {"stage": stage_name, "source_id": stage_source_id, "ok": False, "error": "cache_missing"}

        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        agent_result = payload.get("agent_result") or {}

        text = None
        structured = agent_result.get("structured_data")
        if isinstance(structured, dict):
            text = _find_first_str_by_key(structured, keys=("formatted_answer", "content"))
        if not text:
            text = agent_result.get("content") if isinstance(agent_result.get("content"), str) else None
        if not text or not str(text).strip():
            return {
                "stage": stage_name,
                "source_id": stage_source_id,
                "ok": False,
                "error": "no_text_payload",
            }

        store = EvidenceStore(str(project_folder))
        source_paths = store.source_paths(stage_source_id)

        parsed_exists = source_paths.parsed_path.exists()
        evidence_exists = source_paths.evidence_path.exists()

        if args.force or not (parsed_exists and evidence_exists):
            mvp_parser = MVPLineBlockParser()
            parsed_doc = mvp_parser.parse(text)
            parsed_payload: dict[str, Any] = {
                "parser": {"name": parsed_doc.parser_name, "version": parsed_doc.parser_version},
                "blocks": [
                    {
                        "kind": b.kind,
                        "span": {"start_line": b.span.start_line, "end_line": b.span.end_line},
                        "text": b.text,
                    }
                    for b in parsed_doc.blocks
                ],
            }
            store.write_parsed(stage_source_id, parsed_payload)

            created_at = _created_at_from_payload(payload)
            items = extract_evidence_items(
                parsed=parsed_payload,
                source_id=stage_source_id,
                created_at=created_at,
                max_items=args.max_items,
            )
            store.write_evidence_items(stage_source_id, items)
        else:
            items = store.read_evidence_items(stage_source_id)

        appended = 0
        if args.append_ledger and not _ledger_has_source_id(store, stage_source_id):
            appended = store.append_many(items)

        gate_cfg = EvidenceGateConfig(
            require_evidence=bool(args.require_evidence),
            min_items_per_source=int(args.min_items_per_source),
        )
        gate_result = check_evidence_gate(
            project_folder=str(project_folder),
            source_ids=[stage_source_id],
            config=gate_cfg,
        )

        return {
            "stage": stage_name,
            "source_id": stage_source_id,
            "ok": True,
            "items": len(items),
            "ledger_appended_now": appended,
            "gate_ok": bool(gate_result.get("ok")),
        }

    if args.all_stages:
        cache_dir = project_folder / ".workflow_cache"
        stage_names = sorted(p.stem for p in cache_dir.glob("*.json") if p.is_file())

        results: list[dict[str, Any]] = []
        for stage_name in stage_names:
            try:
                results.append(run_one_stage(stage_name))
            except Exception as e:
                results.append(
                    {
                        "stage": stage_name,
                        "source_id": args.source_id or f"cache:{stage_name}",
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

        print(f"project_folder: {project_folder}")
        print(f"stages_total: {len(results)}")
        if args.append_ledger:
            store = EvidenceStore(str(project_folder))
            print(f"ledger_count_total: {store.count()}")
        print("summary:")
        for r in results:
            if r.get("ok"):
                print(
                    f"- {r['stage']}: items={r.get('items', 0)} gate_ok={r.get('gate_ok')} appended={r.get('ledger_appended_now', 0)}"
                )
            else:
                print(f"- {r['stage']}: ERROR {r.get('error')}")
        return 0

    # Single-stage mode
    stage = args.stage
    result = run_one_stage(stage)
    if not result.get("ok"):
        raise SystemExit(f"Stage {stage} failed: {result.get('error')}")

    store = EvidenceStore(str(project_folder))
    source_paths = store.source_paths(result["source_id"])

    print(f"project_folder: {project_folder}")
    print(f"stage: {result['stage']}")
    print(f"source_id: {result['source_id']}")
    print(f"parsed_path: {source_paths.parsed_path}")
    print(f"evidence_path: {source_paths.evidence_path}")
    print(f"items: {result.get('items', 0)}")
    if args.append_ledger:
        print(f"ledger_appended_now: {result.get('ledger_appended_now', 0)}")
        print(f"ledger_count_total: {store.count()}")
    print(f"gate_ok: {result.get('gate_ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
