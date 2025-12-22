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

from src.utils.validation import validate_project_folder
from src.utils.schema_validation import validate_evidence_item


@dataclass(frozen=True)
class EvidenceStorePaths:
    """Resolved paths for an evidence store under a project folder."""

    store_dir: Path
    ledger_path: Path
    lock_path: Path


class EvidenceStore:
    """Append-only EvidenceItem store (JSONL).

    The store is placed under the project folder in `.evidence/evidence.jsonl`.
    """

    def __init__(
        self,
        project_folder: str,
        store_subdir: str = ".evidence",
        ledger_filename: str = "evidence.jsonl",
        lock_timeout_seconds: int = 30,
    ):
        self.project_folder = validate_project_folder(project_folder)
        self.store_subdir = store_subdir
        self.ledger_filename = ledger_filename
        self.lock_timeout_seconds = lock_timeout_seconds

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
