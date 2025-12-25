"""Citation gates.

Gates are small checks that can be used to block or downgrade writing when
citations are missing or unverified.

The default policy is permissive when not explicitly enabled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple

from loguru import logger

from src.config import INTAKE_SERVER
from src.citations.registry import load_citations
from src.tracing import safe_set_current_span_attributes
from src.utils.validation import validate_project_folder


class CitationGateError(ValueError):
    """Raised when a citation gate blocks execution."""


OnFailureAction = Literal["block", "downgrade"]


@dataclass(frozen=True)
class CitationGateConfig:
    """Configuration for citation enforcement."""

    enabled: bool = False
    on_missing: OnFailureAction = "block"
    on_unverified: OnFailureAction = "downgrade"

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "CitationGateConfig":
        raw = context.get("citation_gate")
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", False))
        on_missing = raw.get("on_missing", "block")
        on_unverified = raw.get("on_unverified", "downgrade")

        if on_missing not in ("block", "downgrade"):
            on_missing = "block"
        if on_unverified not in ("block", "downgrade"):
            on_unverified = "downgrade"

        return cls(enabled=enabled, on_missing=on_missing, on_unverified=on_unverified)


def _iter_project_text_files(project_folder: Path) -> Iterable[Path]:
    exclude_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        ".workflow_cache",
        "sources",
        ".evidence",
        "temp",
        "node_modules",
        "bibliography",
        "literature",
    }

    max_files = int(INTAKE_SERVER.MAX_ZIP_FILES)
    yielded = 0

    for path in project_folder.rglob("*"):
        if not path.is_file():
            continue

        try:
            rel = path.relative_to(project_folder)
        except ValueError:
            continue

        rel_parts = rel.parts

        # Skip hidden and excluded directories.
        if any(part in exclude_dirs for part in rel_parts[:-1]):
            continue
        if any(part.startswith(".") for part in rel_parts[:-1]):
            # Avoid scanning editor and tooling metadata.
            continue

        if path.suffix.lower() in (".md", ".tex"):
            yield path
            yielded += 1
            if yielded >= max_files:
                return


def _extract_citation_keys_from_text(text: str, *, suffix: str) -> Set[str]:
    keys: set[str] = set()

    if suffix.lower() == ".md":
        # Pandoc-style citations: [@key] or [@key1; @key2]
        for match in re.findall(r"\[@?([a-zA-Z0-9_-]+)\]", text):
            keys.add(match.strip().lower())

        # Also capture bare @key in cite spans like @smith2020.
        for match in re.findall(r"(?<![\w-])@([a-zA-Z0-9_-]+)", text):
            keys.add(match.strip().lower())

    if suffix.lower() == ".tex":
        # \cite{key}, \citep{key1,key2}, \citet{key}
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", text):
            for key in group.split(","):
                k = key.strip()
                if k:
                    keys.add(k.lower())

    return keys


def find_referenced_citation_keys(project_folder: str | Path) -> Tuple[Set[str], List[str]]:
    """Return (keys, documents_checked) for citations referenced in project docs."""
    pf = validate_project_folder(project_folder)

    keys: set[str] = set()
    docs: list[str] = []

    for p in _iter_project_text_files(pf):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as e:
            logger.debug(f"Skipping unreadable file in citation scan: {p}: {type(e).__name__}: {e}")
            continue

        rel = str(p.relative_to(pf))
        docs.append(rel)
        keys |= _extract_citation_keys_from_text(text, suffix=p.suffix)

    return keys, sorted(set(docs))


def check_citation_gate(
    *,
    project_folder: str | Path,
    config: Optional[CitationGateConfig] = None,
) -> Dict[str, Any]:
    """Check whether cited keys exist in the registry and are verified.

    Returns a dict with keys:
    - ok (bool)
    - enabled (bool)
    - action (pass|block|downgrade|disabled)
    - missing_keys (list)
    - unverified_keys (list)
    - referenced_keys_total (int)
    - documents_checked (list)
    """

    cfg = config or CitationGateConfig()
    pf = validate_project_folder(project_folder)

    referenced, documents_checked = find_referenced_citation_keys(pf)

    if not cfg.enabled:
        result = {
            "ok": True,
            "enabled": False,
            "action": "disabled",
            "missing_keys": [],
            "unverified_keys": [],
            "referenced_keys_total": len(referenced),
            "documents_checked": documents_checked,
        }

        safe_set_current_span_attributes(
            {
                "gate.name": "citation",
                "gate.enabled": False,
                "gate.ok": True,
                "gate.action": "disabled",
                "citation_gate.referenced_keys_total": int(len(referenced)),
                "citation_gate.documents_checked_total": int(len(documents_checked)),
            }
        )

        return result

    records = load_citations(pf, validate=True)
    by_key: dict[str, Dict[str, Any]] = {}
    for r in records:
        key = str(r.get("citation_key") or "").strip().lower()
        if key:
            by_key[key] = r

    missing = sorted(k for k in referenced if k not in by_key)
    unverified = sorted(
        k for k in referenced if k in by_key and str(by_key[k].get("status") or "").strip() != "verified"
    )

    action: Literal["pass", "block", "downgrade"] = "pass"
    ok = True

    if missing:
        if cfg.on_missing == "block":
            action = "block"
            ok = False
        else:
            action = "downgrade"

    if action != "block" and unverified:
        if cfg.on_unverified == "block":
            action = "block"
            ok = False
        else:
            action = "downgrade"

    result = {
        "ok": ok,
        "enabled": True,
        "action": action,
        "missing_keys": missing,
        "unverified_keys": unverified,
        "referenced_keys_total": len(referenced),
        "documents_checked": documents_checked,
    }

    safe_set_current_span_attributes(
        {
            "gate.name": "citation",
            "gate.enabled": True,
            "gate.ok": bool(ok),
            "gate.action": str(action),
            "citation_gate.on_missing": str(cfg.on_missing),
            "citation_gate.on_unverified": str(cfg.on_unverified),
            "citation_gate.referenced_keys_total": int(len(referenced)),
            "citation_gate.missing_keys_total": int(len(missing)),
            "citation_gate.unverified_keys_total": int(len(unverified)),
            "citation_gate.documents_checked_total": int(len(documents_checked)),
        }
    )

    return result


def enforce_citation_gate(
    *,
    project_folder: str | Path,
    config: Optional[CitationGateConfig] = None,
) -> Dict[str, Any]:
    """Enforce the citation gate.

    Raises:
        CitationGateError: if the gate is enabled and action=block.
    """

    result = check_citation_gate(project_folder=project_folder, config=config)
    if result.get("enabled") and result.get("action") == "block":
        raise CitationGateError(f"Citation gate blocked: {result}")
    return result
