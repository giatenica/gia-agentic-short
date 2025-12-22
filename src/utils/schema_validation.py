""" 
Schema Validation Utilities
===========================
JSON Schema loading and validation helpers.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError


@lru_cache(maxsize=32)
def _load_schema(schema_filename: str) -> Dict[str, Any]:
    """Load a schema JSON file from src/schemas.

    Args:
        schema_filename: File name under src/schemas (for example 'evidence_item.schema.json').

    Raises:
        FileNotFoundError: When schema file is missing.
        ValueError: When schema file is not valid JSON or not a JSON object.
    """
    schemas_dir = Path(__file__).resolve().parent.parent / "schemas"
    schema_path = (schemas_dir / schema_filename).resolve()
    if not schema_path.is_relative_to(schemas_dir.resolve()):
        raise ValueError(f"Schema path escapes schemas directory: {schema_filename}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_filename}")

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load schema {schema_filename}: {e}")

    if not isinstance(schema, dict):
        raise ValueError(f"Schema {schema_filename} must be a JSON object")
    return schema


def validate_against_schema(payload: Any, schema_filename: str) -> None:
    """Validate payload against a JSON Schema.

    Args:
        payload: Any JSON-serializable object.
        schema_filename: File name under src/schemas.

    Raises:
        ValueError: When payload fails validation.
    """
    schema = _load_schema(schema_filename)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if not errors:
        return

    error: ValidationError = errors[0]
    path = "/".join(str(p) for p in error.path)
    prefix = f"Validation failed at '{path}': " if path else "Validation failed: "
    raise ValueError(prefix + error.message)


def validate_evidence_item(item: Dict[str, Any]) -> None:
    """Validate an EvidenceItem dict.

    Uses src/schemas/evidence_item.schema.json.
    """
    validate_against_schema(item, "evidence_item.schema.json")
    _validate_evidence_item_span_ordering(item)


def _validate_evidence_item_span_ordering(item: Dict[str, Any]) -> None:
    """Validate ordering constraints not expressible in JSON Schema.

    Ensures locator.span has non-decreasing end values.
    """
    locator = item.get("locator")
    if not isinstance(locator, dict):
        return

    span = locator.get("span")
    if span is None:
        return
    if not isinstance(span, dict):
        raise ValueError("Validation failed at 'locator/span': span must be an object")

    start_line = span.get("start_line")
    end_line = span.get("end_line")
    if isinstance(start_line, int) and isinstance(end_line, int) and end_line < start_line:
        raise ValueError("Validation failed at 'locator/span': end_line must be >= start_line")

    start_char = span.get("start_char")
    end_char = span.get("end_char")
    if isinstance(start_char, int) and isinstance(end_char, int) and end_char < start_char:
        raise ValueError("Validation failed at 'locator/span': end_char must be >= start_char")


def is_valid_evidence_item(item: Any) -> bool:
    """Return True when item validates as EvidenceItem."""
    if not isinstance(item, dict):
        return False
    try:
        validate_evidence_item(item)
        return True
    except ValueError:
        return False
