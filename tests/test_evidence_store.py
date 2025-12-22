"""
Tests for Evidence Store
=======================

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import json
from pathlib import Path

from filelock import FileLock
import pytest
from src.evidence.store import EvidenceStore


def _valid_item(evidence_id: str = "ev_001"):
    return {
        "schema_version": "1.0",
        "evidence_id": evidence_id,
        "source_id": "paper:smith2020",
        "kind": "quote",
        "locator": {"type": "doi", "value": "10.1000/xyz123"},
        "excerpt": "We find a statistically significant effect.",
        "created_at": "2025-12-22T10:11:12Z",
        "parser": {"name": "manual"},
    }


@pytest.mark.unit
def test_evidence_store_append_and_iter(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    store.append(_valid_item("ev_1"))
    store.append(_valid_item("ev_2"))

    items = list(store.iter_items())
    assert [i["evidence_id"] for i in items] == ["ev_1", "ev_2"]


@pytest.mark.unit
def test_evidence_store_creates_expected_paths(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    p = store.ensure_exists()
    assert p.store_dir.exists()
    assert p.ledger_path.name == "evidence.jsonl"


@pytest.mark.unit
def test_evidence_store_rejects_invalid_item_and_does_not_write(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    bad = _valid_item("ev_bad")
    bad.pop("excerpt")

    with pytest.raises(ValueError):
        store.append(bad)

    assert store.count() == 0


@pytest.mark.unit
def test_evidence_store_iter_raises_on_invalid_json_line(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    p = store.ensure_exists()
    p.ledger_path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        list(store.iter_items())


@pytest.mark.unit
def test_evidence_store_append_many(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    n = store.append_many([_valid_item("ev_1"), _valid_item("ev_2")])
    assert n == 2
    assert store.count() == 2


@pytest.mark.unit
def test_evidence_store_clear_deletes_ledger(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    store.append(_valid_item("ev_1"))
    assert store.count() == 1

    store.clear()
    assert store.count() == 0
    assert store.paths().ledger_path.exists() is False


@pytest.mark.unit
def test_evidence_store_load_all_limit(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    store.append_many([_valid_item("ev_1"), _valid_item("ev_2"), _valid_item("ev_3")])
    items = store.load_all(limit=2)
    assert [i["evidence_id"] for i in items] == ["ev_1", "ev_2"]


@pytest.mark.unit
def test_evidence_store_load_all_validate_false_allows_schema_invalid_object(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    store.append(_valid_item("ev_ok"))

    p = store.ensure_exists()
    bad = _valid_item("ev_bad")
    bad.pop("excerpt")
    with open(p.ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(bad, ensure_ascii=False))
        f.write("\n")

    items = store.load_all(validate=False)
    assert [i["evidence_id"] for i in items] == ["ev_ok", "ev_bad"]

    with pytest.raises(ValueError, match="Invalid EvidenceItem"):
        store.load_all(validate=True)


@pytest.mark.unit
def test_evidence_store_append_times_out_when_lock_held(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder), lock_timeout_seconds=0)
    p = store.ensure_exists()

    with FileLock(p.lock_path, timeout=0):
        with pytest.raises(TimeoutError, match="Timed out acquiring evidence store lock"):
            store.append(_valid_item("ev_1"))
