"""Discussion / Conclusion section writer.

Issue #85 scope:
- Deterministic, filesystem-first section generation.
- Constrained to canonical citation keys from bibliography/citations.json.
- Source-backed statements should be supported by evidence items in sources/*/evidence.json.
- Numeric claims should only be emitted when backed by outputs/metrics.json.

This implementation is deliberately conservative:
- It does not call the LLM.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.agents.base import AgentResult, BaseAgent
from src.agents.section_writer import resolve_section_output_path
from src.citations.registry import load_citations
from src.llm.claude_client import TaskType
from src.utils.project_layout import ensure_project_outputs_layout
from src.utils.schema_validation import is_valid_claim_record, is_valid_metric_record
from src.utils.validation import validate_project_folder


OnFailureAction = Literal["block", "downgrade"]


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


@dataclass(frozen=True)
class DiscussionWriterConfig:
    enabled: bool = True
    on_missing_citation: OnFailureAction = "downgrade"
    on_missing_evidence: OnFailureAction = "downgrade"
    on_missing_metrics: OnFailureAction = "downgrade"
    max_sources: int = 10

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "DiscussionWriterConfig":
        raw = context.get("discussion_writer")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", True))
        on_missing_citation = raw.get("on_missing_citation", "downgrade")
        on_missing_evidence = raw.get("on_missing_evidence", "downgrade")
        on_missing_metrics = raw.get("on_missing_metrics", "downgrade")
        max_sources = int(raw.get("max_sources", 10))
        if max_sources < 1:
            max_sources = 1

        if on_missing_citation not in ("block", "downgrade"):
            on_missing_citation = "downgrade"
        if on_missing_evidence not in ("block", "downgrade"):
            on_missing_evidence = "downgrade"
        if on_missing_metrics not in ("block", "downgrade"):
            on_missing_metrics = "downgrade"

        return cls(
            enabled=enabled,
            on_missing_citation=on_missing_citation,
            on_missing_evidence=on_missing_evidence,
            on_missing_metrics=on_missing_metrics,
            max_sources=max_sources,
        )


def _load_json_list(path: Path) -> Tuple[List[Any], Optional[str]]:
    if not path.exists():
        return [], None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        return [], f"{type(e).__name__}"

    if not isinstance(payload, list):
        return [], "not_a_list"

    return payload, None


def _collect_computed_metric_keys(claims: List[Any]) -> List[str]:
    keys: set[str] = set()
    for item in claims:
        if not isinstance(item, dict) or not is_valid_claim_record(item):
            continue
        if str(item.get("kind")) != "computed":
            continue
        mks = item.get("metric_keys")
        if isinstance(mks, list):
            for k in mks:
                if isinstance(k, str) and k.strip():
                    keys.add(k.strip())
    return sorted(keys)


def _index_metrics(metrics: List[Any]) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for item in metrics:
        if not isinstance(item, dict) or not is_valid_metric_record(item):
            continue
        k = item.get("metric_key")
        if isinstance(k, str) and k.strip():
            by_key[k.strip()] = item
    return by_key


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _iter_source_ids_with_evidence(project_folder: Path) -> List[str]:
    sources_dir = project_folder / "sources"
    if not sources_dir.exists() or not sources_dir.is_dir():
        return []

    ids: set[str] = set()
    for p in sources_dir.rglob("evidence.json"):
        if p.is_file():
            ids.add(p.parent.name)

    return sorted(ids)


def _citation_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_key: dict[str, Dict[str, Any]] = {}
    for r in records:
        key = str(r.get("citation_key") or "").strip()
        if key:
            by_key[key.lower()] = r
    return by_key


class DiscussionWriterAgent(BaseAgent):
    """Deterministic Discussion / Conclusion writer constrained by artifacts."""

    def __init__(self, client=None):
        system_prompt = (
            "You write a Discussion/Conclusion section for an academic paper. "
            "Only cite canonical keys and only emit numeric values backed by outputs/metrics.json. "
            "If required artifacts are missing, you must stop or downgrade based on config."
        )
        super().__init__(
            name="DiscussionWriter",
            task_type=TaskType.DOCUMENT_CREATION,
            system_prompt=system_prompt,
            client=client,
        )

    async def execute(self, context: dict) -> AgentResult:
        project_folder = context.get("project_folder")
        if not isinstance(project_folder, str) or not project_folder:
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error="Missing required context: project_folder",
            )

        pf = validate_project_folder(project_folder)
        ensure_project_outputs_layout(pf)

        cfg = DiscussionWriterConfig.from_context(context)
        section_id = str(context.get("section_id") or "discussion").strip() or "discussion"
        section_title = str(context.get("section_title") or "Discussion").strip() or "Discussion"

        output_path, rel = resolve_section_output_path(pf, section_id=section_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cfg.enabled:
            latex = f"\\section{{{_latex_escape(section_title)}}}\n% Discussion writer disabled\n"
            output_path.write_text(latex, encoding="utf-8")
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content=latex,
                structured_data={
                    "section_id": section_id,
                    "output_relpath": rel,
                    "metadata": {"enabled": False},
                },
            )

        source_citation_map = context.get("source_citation_map")
        if not isinstance(source_citation_map, dict):
            source_citation_map = {}

        citations = load_citations(pf, validate=True)
        by_key = _citation_index(citations)

        source_ids = _iter_source_ids_with_evidence(pf)
        if not source_ids and cfg.on_missing_evidence == "block":
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error="No sources/*/evidence.json files found",
            )

        claims_path = pf / "claims" / "claims.json"
        claims_payload, claims_error = _load_json_list(claims_path)
        metric_keys = _collect_computed_metric_keys(claims_payload)

        metrics_path = pf / "outputs" / "metrics.json"
        metrics_payload, metrics_error = _load_json_list(metrics_path)
        metrics_by_key = _index_metrics(metrics_payload)

        missing_metric_keys = sorted([k for k in metric_keys if k not in metrics_by_key])
        present_metric_keys = sorted([k for k in metric_keys if k in metrics_by_key])

        if missing_metric_keys and cfg.on_missing_metrics == "block":
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=f"Missing metric keys referenced by computed claims: {missing_metric_keys}",
            )

        missing_citation_sources: list[str] = []
        used_citation_keys: list[str] = []

        chunks: list[str] = [f"\\section{{{_latex_escape(section_title)}}}"]

        if not source_ids:
            chunks.append("% Discussion writer downgraded due to missing evidence")
            chunks.append("\\textit{Evidence is not yet available; conclusions are non-definitive.}")
        else:
            chunks.append("\\subsection{Interpretation in context}")
            chunks.append("\\begin{itemize}")
            for sid in source_ids[: cfg.max_sources]:
                raw = source_citation_map.get(sid)
                citation_key_input = str(raw or "").strip()
                cite_tex = ""
                if citation_key_input:
                    rec = by_key.get(citation_key_input.lower())
                    if isinstance(rec, dict):
                        canonical = str(rec.get("citation_key") or "").strip()
                        if canonical:
                            cite_tex = f"\\cite{{{canonical}}}"
                            used_citation_keys.append(canonical)
                    else:
                        missing_citation_sources.append(sid)
                else:
                    missing_citation_sources.append(sid)

                if not cite_tex and cfg.on_missing_citation == "block":
                    return AgentResult(
                        agent_name=self.name,
                        task_type=self.task_type,
                        model_tier=self.model_tier,
                        success=False,
                        content="",
                        error=f"Missing canonical citation key for source_id={sid}",
                    )

                safe_sid = _latex_escape(sid)
                if cite_tex:
                    chunks.append(f"\\item {safe_sid} {cite_tex}.")
                else:
                    chunks.append(f"\\item {safe_sid}.")
            chunks.append("\\end{itemize}")

        chunks.append("\\subsection{Implications of computed metrics}")
        if missing_metric_keys:
            chunks.append("% Discussion writer downgraded due to missing metrics")

        if present_metric_keys:
            chunks.append("\\begin{itemize}")
            for key in present_metric_keys[:25]:
                rec = metrics_by_key.get(key)
                if not isinstance(rec, dict):
                    continue
                name = str(rec.get("name") or key).strip() or key
                value = _format_metric_value(rec.get("value"))
                unit = str(rec.get("unit") or "").strip()

                safe_name = _latex_escape(name)
                safe_unit = _latex_escape(unit) if unit else ""
                tail = f" {safe_unit}" if safe_unit else ""
                chunks.append(f"\\item {safe_name}: {value}{tail}.")
            chunks.append("\\end{itemize}")
        else:
            chunks.append("Computed metrics are pending.")

        chunks.append("\\subsection{Limitations and next steps}")
        chunks.append("\\begin{itemize}")
        chunks.append("\\item Verify all citations and ensure evidence coverage for all claims.")
        chunks.append("\\item Run the analysis stage to populate missing metrics and artifacts.")
        chunks.append("\\end{itemize}")

        latex = "\n".join(chunks)
        if not latex.endswith("\n"):
            latex += "\n"

        output_path.write_text(latex, encoding="utf-8")

        action = "pass"
        if missing_metric_keys or missing_citation_sources or not source_ids:
            action = "downgrade"

        metadata = {
            "enabled": True,
            "action": action,
            "missing_metric_keys": missing_metric_keys,
            "present_metric_keys": present_metric_keys,
            "missing_citation_sources": sorted(set(missing_citation_sources)),
            "used_citation_keys": sorted(set(used_citation_keys)),
            "claims_read_error": claims_error,
            "metrics_read_error": metrics_error,
        }

        return AgentResult(
            agent_name=self.name,
            task_type=self.task_type,
            model_tier=self.model_tier,
            success=True,
            content=latex,
            structured_data={
                "section_id": section_id,
                "output_relpath": rel,
                "metadata": metadata,
            },
        )
