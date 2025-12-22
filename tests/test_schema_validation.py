"""
Tests for Schema Validation Utilities
====================================

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import sys
from pathlib import Path
import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.utils.schema_validation import (
    validate_evidence_item,
    is_valid_evidence_item,
)


def _valid_item():
    return {
        "schema_version": "1.0",
        "evidence_id": "ev_001",
        "source_id": "paper:smith2020",
        "kind": "quote",
        "locator": {
            "type": "doi",
            "value": "10.1000/xyz123",
            "span": {"start_line": 10, "end_line": 12},
        },
        "excerpt": "We find a statistically significant effect.",
        "created_at": "2025-12-22T10:11:12Z",
        "parser": {"name": "manual"},
        "metadata": {"note": "unit test"},
    }


@pytest.mark.unit
def test_validate_evidence_item_accepts_valid_payload():
    validate_evidence_item(_valid_item())


@pytest.mark.unit
def test_validate_evidence_item_rejects_missing_required_field():
    item = _valid_item()
    item.pop("excerpt")
    with pytest.raises(ValueError, match="excerpt"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_validate_evidence_item_rejects_additional_properties():
    item = _valid_item()
    item["unexpected"] = "nope"
    with pytest.raises(ValueError, match="Additional properties"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_validate_evidence_item_rejects_invalid_datetime_format():
    item = _valid_item()
    item["created_at"] = "not-a-datetime"
    with pytest.raises(ValueError, match="created_at"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_is_valid_evidence_item_boolean_helper():
    assert is_valid_evidence_item(_valid_item()) is True
    assert is_valid_evidence_item({}) is False
    assert is_valid_evidence_item("not a dict") is False
