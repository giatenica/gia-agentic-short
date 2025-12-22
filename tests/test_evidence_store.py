"""
Tests for Evidence Store
=======================

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import sys
from pathlib import Path
import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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
