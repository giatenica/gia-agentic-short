""" 
Evidence Store
==============
Append-only JSONL ledger for EvidenceItem records.

Design goals:
- Simple, transparent on-disk format (one JSON object per line)
- File locking to avoid concurrent write corruption
- Schema validation for predictable downstream parsing

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from filelock import FileLock, Timeout
from loguru import logger

from src.config import TIMEOUTS
from src.utils.validation import validate_project_folder
from src.utils.schema_validation import validate_evidence_item
from src.utils.filesystem import source_id_to_dirname


@dataclass(frozen=True)
class EvidenceStorePaths:
    """Resolved paths for an evidence store under a project folder."""

    store_dir: Path
    ledger_path: Path
    lock_path: Path

@dataclass(frozen=True)
class EvidenceProjectPaths:
    """Resolved project-level evidence paths under a project folder."""

    sources_dir: Path
    bibliography_dir: Path


@dataclass(frozen=True)
class EvidenceSourcePaths:
    """Resolved per-source paths under a project folder."""

    source_dir: Path
    raw_dir: Path
    parsed_path: Path
    evidence_path: Path
class EvidenceStore:
    """Append-only EvidenceItem store (JSONL).

    The store is placed under the project folder in `.evidence/evidence.jsonl`.
    """

    def __init__(
        self,
        project_folder: str,
        store_subdir: str = ".evidence",
        ledger_filename: str = "evidence.jsonl",
        lock_timeout_seconds: int = TIMEOUTS.FILE_LOCK,
        sources_subdir: str = "sources",
        bibliography_subdir: str = "bibliography",
    ):
        self.project_folder = validate_project_folder(project_folder)
        self.store_subdir = store_subdir
        self.ledger_filename = ledger_filename
        self.lock_timeout_seconds = lock_timeout_seconds
        self.sources_subdir = sources_subdir
        self.bibliography_subdir = bibliography_subdir

    def project_paths(self) -> EvidenceProjectPaths:
        return EvidenceProjectPaths(
            sources_dir=self.project_folder / self.sources_subdir,
            bibliography_dir=self.project_folder / self.bibliography_subdir,
        )

    def ensure_project_layout(self) -> EvidenceProjectPaths:
        """Ensure project-level evidence directories exist."""
        p = self.project_paths()
        p.sources_dir.mkdir(parents=True, exist_ok=True)
        p.bibliography_dir.mkdir(parents=True, exist_ok=True)
        return p

    def _validate_source_id(self, source_id: str) -> None:
        if not source_id or not isinstance(source_id, str):
            raise ValueError("source_id must be a non-empty string")
        if "/" in source_id or "\\" in source_id:
            raise ValueError("source_id must not contain path separators")
        if ".." in source_id:
            raise ValueError("source_id must not contain '..'")

    def source_paths(self, source_id: str) -> EvidenceSourcePaths:
        """Resolve per-source storage paths for a given source_id."""
        self._validate_source_id(source_id)
        p = self.project_paths()
        source_dir = p.sources_dir / source_id_to_dirname(source_id)
        return EvidenceSourcePaths(
            source_dir=source_dir,
            raw_dir=source_dir / "raw",
            parsed_path=source_dir / "parsed.json",
            evidence_path=source_dir / "evidence.json",
        )

    def ensure_source_layout(self, source_id: str) -> EvidenceSourcePaths:
        """Ensure per-source folders exist (raw dir; parsed/evidence files optional)."""
        self.ensure_project_layout()
        sp = self.source_paths(source_id)
        sp.raw_dir.mkdir(parents=True, exist_ok=True)
        return sp

    def write_parsed(self, source_id: str, parsed: Any) -> EvidenceSourcePaths:
        """Write parsed document representation to sources/<source_id>/parsed.json."""
        sp = self.ensure_source_layout(source_id)
        sp.parsed_path.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return sp

    def read_parsed(self, source_id: str) -> Any:
        """Read parsed document representation from sources/<source_id>/parsed.json.

        The parsed.json file is expected to already exist; a FileNotFoundError
        will be raised if it is missing.
        """
        sp = self.source_paths(source_id)
        with open(sp.parsed_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_evidence_items(self, source_id: str, items: list[Dict[str, Any]]) -> EvidenceSourcePaths:
        """Write evidence items to sources/<source_id>/evidence.json.

        Each item is validated against the EvidenceItem schema.
        """
        for item in items:
            validate_evidence_item(item)

        sp = self.ensure_source_layout(source_id)
        sp.evidence_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return sp

    def read_evidence_items(self, source_id: str, validate: bool = True) -> list[Dict[str, Any]]:
        """Read evidence items from sources/<source_id>/evidence.json.

        The evidence.json file is expected to already exist; a FileNotFoundError
        will be raised if it is missing.
        """
        sp = self.source_paths(source_id)
        with open(sp.evidence_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            raise ValueError(f"Evidence payload must be a list at {sp.evidence_path}")

        if validate:
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError(f"Evidence items must be objects at {sp.evidence_path}")
                validate_evidence_item(item)

        return payload

    def paths(self) -> EvidenceStorePaths:
        store_dir = self.project_folder / self.store_subdir
        ledger_path = store_dir / self.ledger_filename
        lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
        return EvidenceStorePaths(store_dir=store_dir, ledger_path=ledger_path, lock_path=lock_path)

    def ensure_exists(self) -> EvidenceStorePaths:
        p = self.paths()
        p.store_dir.mkdir(parents=True, exist_ok=True)
        return p

    def append(self, item: Dict[str, Any]) -> None:
        """Validate and append a single EvidenceItem record."""
        validate_evidence_item(item)
        p = self.ensure_exists()

        line = json.dumps(item, ensure_ascii=False)

        try:
            with FileLock(p.lock_path, timeout=self.lock_timeout_seconds):
                with open(p.ledger_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.write("\n")
                    f.flush()
        except Timeout as e:
            raise TimeoutError(
                f"Timed out acquiring evidence store lock {p.lock_path} after {self.lock_timeout_seconds}s"
            ) from e
        except OSError as e:
            raise OSError(f"Failed to append evidence item to {p.ledger_path}: {e}")

    def append_many(self, items: Iterable[Dict[str, Any]]) -> int:
        """Append multiple EvidenceItem records in one locked section.

        Returns:
            Number of items written.
        """
        p = self.ensure_exists()
        validated_lines = []
        for item in items:
            validate_evidence_item(item)
            validated_lines.append(json.dumps(item, ensure_ascii=False))

        if not validated_lines:
            return 0

        try:
            with FileLock(p.lock_path, timeout=self.lock_timeout_seconds):
                with open(p.ledger_path, "a", encoding="utf-8") as f:
                    for line in validated_lines:
                        f.write(line)
                        f.write("\n")
                    f.flush()
        except Timeout as e:
            raise TimeoutError(
                f"Timed out acquiring evidence store lock {p.lock_path} after {self.lock_timeout_seconds}s"
            ) from e
        except OSError as e:
            raise OSError(f"Failed to append evidence items to {p.ledger_path}: {e}")

        return len(validated_lines)

    def iter_items(self, validate: bool = True) -> Iterator[Dict[str, Any]]:
        """Iterate EvidenceItem records in insertion order."""
        p = self.paths()
        if not p.ledger_path.exists():
            return

        with open(p.ledger_path, "r", encoding="utf-8") as f:
            for idx, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {p.ledger_path} line {idx}: {e}")

                if not isinstance(obj, dict):
                    raise ValueError(f"Evidence record must be an object at {p.ledger_path} line {idx}")

                if validate:
                    try:
                        validate_evidence_item(obj)
                    except ValueError as e:
                        raise ValueError(f"Invalid EvidenceItem at {p.ledger_path} line {idx}: {e}")

                yield obj

    def load_all(self, validate: bool = True, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        """Load all EvidenceItem records.

        Args:
            validate: Validate schema on read.
            limit: Optional max number of records to read.
        """
        items: list[Dict[str, Any]] = []
        for item in self.iter_items(validate=validate):
            items.append(item)
            if limit is not None and len(items) >= limit:
                break
        return items

    def count(self) -> int:
        """Count records quickly (does not validate schema)."""
        p = self.paths()
        if not p.ledger_path.exists():
            return 0
        count = 0
        with open(p.ledger_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                if raw_line.strip():
                    count += 1
        return count

    def clear(self) -> None:
        """Delete the ledger file (keeps directory)."""
        p = self.paths()
        if p.ledger_path.exists():
            try:
                p.ledger_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete evidence ledger {p.ledger_path}: {e}")
