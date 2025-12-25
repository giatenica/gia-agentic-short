"""Evaluation suite runner.

This module provides a small, deterministic runner for sweeping the built-in
`evaluation/test_queries.json` dataset.

It supports a safe dry-run mode that does not require external API keys and does
not call any LLM-backed workflows.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from loguru import logger

from src.config import INTAKE_SERVER
from src.utils.project_layout import ensure_project_outputs_layout


RunMode = Literal["dry", "phase1", "phase1+phase2"]


@dataclass(frozen=True)
class EvaluationSuiteConfig:
    """Configuration for an evaluation suite run."""

    mode: RunMode = "dry"
    output_root: Path = Path("outputs") / "evaluation_suite"
    disable_edison_by_default: bool = True

    @property
    def normalized_mode(self) -> RunMode:
        if self.mode == "phase1+phase2":
            return "phase1+phase2"
        if self.mode == "phase1":
            return "phase1"
        return "dry"


@dataclass
class EvaluationQuery:
    """A single evaluation query loaded from evaluation/test_queries.json."""

    id: str
    title: str
    research_question: str

    # Keep any extra fields for project.json generation and future use.
    raw: Dict[str, Any]


@dataclass
class EvaluationQueryResult:
    """Result record for a single evaluation query run."""

    query_id: str
    success: bool
    skipped: bool
    error: Optional[str]

    project_folder: str
    created_files: List[str]

    phase1_success: Optional[bool] = None
    phase2_success: Optional[bool] = None


@dataclass
class EvaluationSuiteReport:
    """Top-level report for a suite run."""

    schema_version: str
    started_at: str
    finished_at: str
    mode: str

    output_root: str
    run_id: str

    queries_total: int
    queries_success: int
    results: List[EvaluationQueryResult]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug(text: str, *, max_len: int = 40) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        return "item"
    return text[:max_len].strip("-")


def load_test_queries(path: str | Path) -> List[EvaluationQuery]:
    """Load evaluation queries.

    Raises ValueError when JSON is invalid or schema is unexpected.
    """

    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to read test queries: {type(e).__name__}")

    if not isinstance(payload, list):
        raise ValueError("test queries payload must be a list")

    out: List[EvaluationQuery] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        qid = item.get("id")
        title = item.get("title")
        rq = item.get("research_question")
        if not isinstance(qid, str) or not qid.strip():
            continue
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(rq, str) or not rq.strip():
            continue
        out.append(EvaluationQuery(id=qid.strip(), title=title.strip(), research_question=rq.strip(), raw=dict(item)))

    if not out:
        raise ValueError("no valid queries loaded")

    # Ensure unique IDs.
    ids = [q.id for q in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query ids")

    # Ensure IDs are safe for filesystem paths.
    for q in out:
        if any(ch in q.id for ch in ("/", "\\")):
            raise ValueError(f"unsafe query id: {q.id}")

    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _list_files(base: Path) -> List[str]:
    files: List[str] = []
    max_files = int(INTAKE_SERVER.MAX_ZIP_FILES)

    exclude_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        ".workflow_cache",
        "node_modules",
        "temp",
        "tmp",
        ".evidence",
    }
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()
        if "/." in f"/{rel}":
            continue
        if any(part in exclude_dirs for part in Path(rel).parts[:-1]):
            continue
        files.append(rel)
        if len(files) >= max_files:
            break
    return sorted(files)


def _materialize_project_folder(
    *,
    base_dir: Path,
    query: EvaluationQuery,
) -> Path:
    """Create an isolated project folder with project.json and standard layout."""

    project_dir = base_dir / "project"
    project_dir.mkdir(parents=True, exist_ok=True)

    project_json = dict(query.raw)
    project_json.setdefault("id", query.id)
    project_json.setdefault("title", query.title)
    project_json.setdefault("research_question", query.research_question)

    _write_json(project_dir / "project.json", project_json)
    ensure_project_outputs_layout(project_dir)

    return project_dir


async def run_evaluation_suite(
    *,
    queries: List[EvaluationQuery],
    config: EvaluationSuiteConfig,
    run_id: Optional[str] = None,
) -> Tuple[EvaluationSuiteReport, Path]:
    """Run the evaluation suite over a set of test queries.

    Behavior by mode:
    - dry: creates per-query isolated project folders and a summary report; no LLM calls
    - phase1: runs Phase 1 workflow for each query
    - phase1+phase2: runs both Phase 1 and Phase 2 workflows for each query

    Side effects:
    - creates directories under Path(config.output_root) / run_id
    - writes report.json into the run directory

    Returns:
        (report, report_path)
    """

    started_at = _utc_now_iso()
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    root = Path(config.output_root).resolve() / rid
    root.mkdir(parents=True, exist_ok=True)

    results: List[EvaluationQueryResult] = []

    ResearchWorkflow = None
    LiteratureWorkflow = None
    EdisonClient = None

    if config.normalized_mode != "dry":
        from src.agents.workflow import ResearchWorkflow as _ResearchWorkflow

        ResearchWorkflow = _ResearchWorkflow

        if config.normalized_mode == "phase1+phase2":
            from src.agents.literature_workflow import LiteratureWorkflow as _LiteratureWorkflow
            from src.llm.edison_client import EdisonClient as _EdisonClient

            LiteratureWorkflow = _LiteratureWorkflow
            EdisonClient = _EdisonClient

    for idx, q in enumerate(queries, start=1):
        query_dir = root / f"{idx:03d}_{_slug(q.id)}_{_slug(q.title)}"
        query_dir.mkdir(parents=True, exist_ok=True)

        project_dir = _materialize_project_folder(base_dir=query_dir, query=q)

        if config.normalized_mode == "dry":
            created = _list_files(project_dir)
            results.append(
                EvaluationQueryResult(
                    query_id=q.id,
                    success=True,
                    skipped=True,
                    error=None,
                    project_folder=str(project_dir),
                    created_files=created,
                    phase1_success=None,
                    phase2_success=None,
                )
            )
            continue

        # Live mode: run workflows.
        phase1_success: Optional[bool] = None
        phase2_success: Optional[bool] = None
        error: Optional[str] = None

        try:
            if ResearchWorkflow is None:
                raise RuntimeError("ResearchWorkflow not available")

            wf1 = ResearchWorkflow()
            wf1_result = await wf1.run(str(project_dir))
            phase1_success = bool(wf1_result.success)

            if config.normalized_mode == "phase1+phase2":
                if LiteratureWorkflow is None or EdisonClient is None:
                    raise RuntimeError("Phase 2 dependencies not available")

                edison_client = None
                if config.disable_edison_by_default:
                    edison_client = EdisonClient(api_key=None)

                wf2 = LiteratureWorkflow(edison_client=edison_client)
                wf2_result = await wf2.run(str(project_dir))
                phase2_success = bool(wf2_result.success)

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.warning(f"Evaluation query failed: {q.id}: {error}")

        created = _list_files(project_dir)

        # Define overall success conservatively.
        ok = bool(error is None and (phase1_success is True) and (phase2_success is not False))

        results.append(
            EvaluationQueryResult(
                query_id=q.id,
                success=ok,
                skipped=False,
                error=error,
                project_folder=str(project_dir),
                created_files=created,
                phase1_success=phase1_success,
                phase2_success=phase2_success,
            )
        )

    finished_at = _utc_now_iso()

    queries_success = sum(1 for r in results if r.success)

    report = EvaluationSuiteReport(
        schema_version="1.0",
        started_at=started_at,
        finished_at=finished_at,
        mode=config.normalized_mode,
        output_root=str(Path(config.output_root).resolve() / rid),
        run_id=rid,
        queries_total=len(results),
        queries_success=queries_success,
        results=results,
    )

    report_path = root / "report.json"
    _write_json(report_path, asdict(report))

    return report, report_path
