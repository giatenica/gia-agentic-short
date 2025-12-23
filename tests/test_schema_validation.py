"""
Tests for Schema Validation Utilities
====================================

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import json
from unittest.mock import patch

import pytest
import src.utils.schema_validation as schema_validation
from src.utils.schema_validation import (
    validate_against_schema,
    validate_evidence_item,
    is_valid_evidence_item,
    validate_metric_record,
    is_valid_metric_record,
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


def _valid_metric_record():
    return {
        "schema_version": "1.0",
        "metric_key": "m:example_metric",
        "name": "Example metric",
        "value": 1.23,
        "created_at": "2025-12-22T10:11:12Z",
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


@pytest.mark.unit
def test_validate_evidence_item_rejects_invalid_kind_enum():
    item = _valid_item()
    item["kind"] = "not-a-kind"
    with pytest.raises(ValueError, match="kind"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_validate_evidence_item_rejects_invalid_locator_type_enum():
    item = _valid_item()
    item["locator"]["type"] = "not-a-locator-type"
    with pytest.raises(ValueError, match="locator"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_validate_evidence_item_rejects_schema_version_mismatch():
    item = _valid_item()
    item["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema_version"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_validate_evidence_item_rejects_empty_required_strings():
    item = _valid_item()
    item["evidence_id"] = ""
    with pytest.raises(ValueError, match="evidence_id"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_validate_evidence_item_rejects_invalid_parser_object():
    item = _valid_item()
    item["parser"] = {"version": "1"}
    with pytest.raises(ValueError, match="parser"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_validate_evidence_item_rejects_invalid_locator_object():
    item = _valid_item()
    item["locator"] = {"type": "doi"}
    with pytest.raises(ValueError, match="locator"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_validate_evidence_item_rejects_span_ordering_end_line_before_start_line():
    item = _valid_item()
    item["locator"]["span"] = {"start_line": 10, "end_line": 9}
    with pytest.raises(ValueError, match="end_line"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_validate_evidence_item_rejects_span_ordering_end_char_before_start_char():
    item = _valid_item()
    item["locator"]["span"] = {"start_char": 10, "end_char": 9}
    with pytest.raises(ValueError, match="end_char"):
        validate_evidence_item(item)


@pytest.mark.unit
def test_validate_evidence_item_accepts_optional_context_field():
    item = _valid_item()
    item["context"] = "This quote appears in the introduction section."
    validate_evidence_item(item)


@pytest.mark.unit
def test_validate_against_schema_accepts_valid_payload():
    validate_against_schema(_valid_item(), "evidence_item.schema.json")


@pytest.mark.unit
def test_validate_against_schema_raises_when_schema_missing():
    with pytest.raises(FileNotFoundError, match="Schema not found"):
        validate_against_schema({"x": 1}, "definitely_missing.schema.json")


@pytest.mark.unit
def test_validate_against_schema_rejects_path_traversal_schema_filename():
    with pytest.raises(ValueError, match="escapes schemas directory"):
        validate_against_schema({"x": 1}, "../evil.schema.json")


@pytest.mark.unit
def test_load_schema_raises_on_invalid_json():
    payload = _valid_item()
    schema_validation._load_schema.cache_clear()
    with patch("src.utils.schema_validation.json.load") as mock_load:
        mock_load.side_effect = json.JSONDecodeError("bad", "{", 1)
        with pytest.raises(ValueError, match="Failed to load schema"):
            validate_against_schema(payload, "evidence_item.schema.json")


@pytest.mark.unit
def test_load_schema_raises_when_schema_not_object():
    payload = _valid_item()
    schema_validation._load_schema.cache_clear()
    with patch("src.utils.schema_validation.json.load") as mock_load:
        mock_load.return_value = []
        with pytest.raises(ValueError, match="must be a JSON object"):
            validate_against_schema(payload, "evidence_item.schema.json")


@pytest.mark.unit
def test_validate_metric_record_accepts_minimal_valid_payload():
    validate_metric_record(_valid_metric_record())


@pytest.mark.unit
def test_validate_metric_record_rejects_missing_required_field():
    record = _valid_metric_record()
    record.pop("metric_key")
    with pytest.raises(ValueError, match="metric_key"):
        validate_metric_record(record)


@pytest.mark.unit
def test_is_valid_metric_record_boolean_helper():
    assert is_valid_metric_record(_valid_metric_record()) is True
    assert is_valid_metric_record({}) is False
    assert is_valid_metric_record("not a dict") is False


@pytest.mark.unit
def test_validate_metric_record_rejects_additional_properties():
    record = _valid_metric_record()
    record["unexpected"] = "nope"
    with pytest.raises(ValueError, match="Additional properties"):
        validate_metric_record(record)


@pytest.mark.unit
def test_validate_metric_record_rejects_invalid_datetime_format():
    record = _valid_metric_record()
    record["created_at"] = "not-a-datetime"
    with pytest.raises(ValueError, match="created_at"):
        validate_metric_record(record)


@pytest.mark.unit
def test_validate_metric_record_rejects_schema_version_mismatch():
    record = _valid_metric_record()
    record["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema_version"):
        validate_metric_record(record)


@pytest.mark.unit
def test_validate_metric_record_rejects_empty_required_strings():
    record = _valid_metric_record()
    record["metric_key"] = ""
    with pytest.raises(ValueError, match="metric_key"):
        validate_metric_record(record)


@pytest.mark.unit
def test_validate_metric_record_accepts_optional_fields():
    record = _valid_metric_record()
    record["unit"] = "bps"
    record["description"] = "Example optional description"
    record["metadata"] = {"note": "unit test"}
    validate_metric_record(record)


@pytest.mark.unit
def test_validate_metric_record_rejects_invalid_value_type():
    record = _valid_metric_record()
    record["value"] = "not-a-number"
    with pytest.raises(ValueError, match="value"):
        validate_metric_record(record)
