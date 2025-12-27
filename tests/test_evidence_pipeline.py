import json
from unittest.mock import patch

import pytest
from pypdf import PdfWriter

from src.evidence.pipeline import EvidencePipelineConfig, run_local_evidence_pipeline
from src.evidence.store import EvidenceStore


@pytest.mark.unit
@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
def test_local_evidence_pipeline_writes_parsed_and_evidence(temp_project_folder):
    literature_dir = temp_project_folder / "literature"
    literature_dir.mkdir(exist_ok=True)
    (literature_dir / "notes.md").write_text(
        "This is a sufficiently long paragraph to extract evidence from. It includes 42%.",
        encoding="utf-8",
    )

    cfg = EvidencePipelineConfig(enabled=True, max_sources=10, ingest_sources=True, append_to_ledger=True)
    summary = run_local_evidence_pipeline(project_folder=str(temp_project_folder), config=cfg)

    assert summary["processed_count"] >= 1
    assert len(summary["source_ids"]) >= 1

    store = EvidenceStore(str(temp_project_folder))
    source_id = summary["source_ids"][0]

    parsed = store.read_parsed(source_id)
    assert isinstance(parsed, dict)
    assert "blocks" in parsed

    evidence = store.read_evidence_items(source_id)
    assert isinstance(evidence, list)
    assert len(evidence) >= 1

    # If append_to_ledger is enabled, the ledger should contain entries.
    assert store.count() >= 1

    # evidence.json should be valid JSON array on disk
    sp = store.source_paths(source_id)
    on_disk = json.loads(sp.evidence_path.read_text(encoding="utf-8"))
    assert isinstance(on_disk, list)


@pytest.mark.unit
@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
def test_local_evidence_pipeline_writes_evidence_coverage_artifact(temp_project_folder):
    literature_dir = temp_project_folder / "literature"
    literature_dir.mkdir(exist_ok=True)
    (literature_dir / "notes.md").write_text(
        "This is a sufficiently long paragraph to extract evidence from. It includes 42%.",
        encoding="utf-8",
    )

    cfg = EvidencePipelineConfig(enabled=True, max_sources=10, ingest_sources=True, append_to_ledger=False)
    run_local_evidence_pipeline(project_folder=str(temp_project_folder), config=cfg)

    coverage_path = temp_project_folder / "outputs" / "evidence_coverage.json"
    assert coverage_path.exists()
    payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert isinstance(payload.get("summary"), dict)


@pytest.mark.unit
@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True)
def test_local_evidence_pipeline_processes_pdf_sources(temp_project_folder):
    literature_dir = temp_project_folder / "literature"
    literature_dir.mkdir(exist_ok=True)

    pdf_path = literature_dir / "paper.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    cfg = EvidencePipelineConfig(enabled=True, max_sources=10, ingest_sources=True, append_to_ledger=False)
    summary = run_local_evidence_pipeline(project_folder=str(temp_project_folder), config=cfg)

    assert summary["processed_count"] >= 1
    assert len(summary["source_ids"]) >= 1

    store = EvidenceStore(str(temp_project_folder))
    pdf_source_id = summary["source_ids"][0]

    parsed = store.read_parsed(pdf_source_id)
    assert isinstance(parsed, dict)
    assert "blocks" in parsed

    evidence = store.read_evidence_items(pdf_source_id)
    assert isinstance(evidence, list)
