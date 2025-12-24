import json

import pytest

from src.citations.registry import save_citations
from src.claims.claim_evidence_gate import (
    ClaimEvidenceGateConfig,
    ClaimEvidenceGateError,
    check_claim_evidence_gate,
    enforce_claim_evidence_gate,
)
from src.evidence.store import EvidenceStore


def _write_claims(project_folder, claims):
    (project_folder / "claims").mkdir(exist_ok=True)
    (project_folder / "claims" / "claims.json").write_text(
        json.dumps(claims, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _minimal_evidence_item(*, evidence_id: str, source_id: str):
    return {
        "schema_version": "1.0",
        "evidence_id": evidence_id,
        "source_id": source_id,
        "kind": "quote",
        "locator": {"type": "file", "value": "dummy.txt", "span": {"start_line": 1, "end_line": 1}},
        "excerpt": "A supporting quote.",
        "created_at": "2025-01-01T00:00:00Z",
        "parser": {"name": "mvp"},
    }


@pytest.mark.unit
def test_claim_evidence_gate_passes_when_all_refs_resolve(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    store.write_evidence_items("source1", [_minimal_evidence_item(evidence_id="ev1", source_id="source1")])

    save_citations(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "citation_key": "smith2020",
                "status": "verified",
                "title": "A Paper",
                "authors": ["Smith"],
                "year": 2020,
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
        validate=True,
    )

    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "source_backed",
                "statement": "This is supported.",
                "citation_keys": ["smith2020"],
                "evidence_ids": ["ev1"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    cfg = ClaimEvidenceGateConfig(enabled=True, on_failure="block")
    result = check_claim_evidence_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "pass"
    assert result["missing_evidence_claim_ids"] == []
    assert result["missing_evidence_ids"] == []
    assert result["missing_citation_keys"] == []


@pytest.mark.unit
def test_claim_evidence_gate_blocks_when_evidence_ids_missing_for_source_backed_claim(temp_project_folder):
    save_citations(temp_project_folder, [], validate=False)

    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "source_backed",
                "statement": "Missing evidence ids.",
                "citation_keys": ["smith2020"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    cfg = ClaimEvidenceGateConfig(enabled=True, on_failure="block")
    result = check_claim_evidence_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is False
    assert result["action"] == "block"
    assert result["missing_evidence_claim_ids"] == ["c1"]

    with pytest.raises(ClaimEvidenceGateError):
        enforce_claim_evidence_gate(project_folder=str(temp_project_folder), config=cfg)


@pytest.mark.unit
def test_claim_evidence_gate_blocks_when_evidence_id_unknown(temp_project_folder):
    save_citations(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "citation_key": "smith2020",
                "status": "verified",
                "title": "A Paper",
                "authors": ["Smith"],
                "year": 2020,
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
        validate=True,
    )

    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "source_backed",
                "statement": "Unknown evidence id.",
                "citation_keys": ["smith2020"],
                "evidence_ids": ["missing_ev"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    cfg = ClaimEvidenceGateConfig(enabled=True, on_failure="block")
    result = check_claim_evidence_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is False
    assert result["action"] == "block"
    assert result["missing_evidence_ids"] == ["missing_ev"]


@pytest.mark.unit
def test_claim_evidence_gate_blocks_when_citation_key_missing(temp_project_folder):
    store = EvidenceStore(str(temp_project_folder))
    store.write_evidence_items("source1", [_minimal_evidence_item(evidence_id="ev1", source_id="source1")])

    save_citations(temp_project_folder, [], validate=False)

    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "source_backed",
                "statement": "Unknown citation key.",
                "citation_keys": ["missing2021"],
                "evidence_ids": ["ev1"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    cfg = ClaimEvidenceGateConfig(enabled=True, on_failure="block")
    result = check_claim_evidence_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is False
    assert result["action"] == "block"
    assert result["missing_citation_keys"] == ["missing2021"]


@pytest.mark.unit
def test_claim_evidence_gate_downgrades_when_configured_to_downgrade(temp_project_folder):
    save_citations(temp_project_folder, [], validate=False)

    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "source_backed",
                "statement": "Unknown citation key.",
                "citation_keys": ["missing2021"],
                "evidence_ids": ["ev1"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    cfg = ClaimEvidenceGateConfig(enabled=True, on_failure="downgrade")
    result = check_claim_evidence_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "downgrade"


@pytest.mark.unit
def test_claim_evidence_gate_disabled_is_permissive_even_when_missing(temp_project_folder):
    save_citations(temp_project_folder, [], validate=False)

    _write_claims(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "claim_id": "c1",
                "kind": "source_backed",
                "statement": "Missing evidence ids.",
                "citation_keys": ["smith2020"],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    cfg = ClaimEvidenceGateConfig(enabled=False)
    result = check_claim_evidence_gate(project_folder=str(temp_project_folder), config=cfg)
    assert result["ok"] is True
    assert result["action"] == "disabled"


@pytest.mark.unit
def test_claim_evidence_gate_config_from_context_parses_and_sanitizes_values():
    cfg = ClaimEvidenceGateConfig.from_context({"claim_evidence_gate": {"enabled": True, "on_failure": "downgrade"}})
    assert cfg.enabled is True
    assert cfg.on_failure == "downgrade"

    bad = ClaimEvidenceGateConfig.from_context({"claim_evidence_gate": {"enabled": True, "on_failure": "nope"}})
    assert bad.enabled is True
    assert bad.on_failure == "block"
