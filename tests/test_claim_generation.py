import json

import pytest

from src.claims.generator import generate_claims_from_metrics
from src.utils.schema_validation import validate_claim_record


def _write_metrics(project_folder, metrics):
    outputs_dir = project_folder / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _read_claims(project_folder):
    return json.loads((project_folder / "claims" / "claims.json").read_text(encoding="utf-8"))


@pytest.mark.unit
def test_generate_claims_from_metrics_creates_empty_claims_for_empty_metrics(temp_project_folder):
    _write_metrics(temp_project_folder, [])

    summary = generate_claims_from_metrics(project_folder=temp_project_folder)
    assert summary["ok"] is True
    assert summary["action"] == "written"

    claims = _read_claims(temp_project_folder)
    assert claims == []


@pytest.mark.unit
def test_generate_claims_from_metrics_writes_schema_valid_claims(temp_project_folder):
    _write_metrics(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "metric_key": "m1",
                "name": "Metric 1",
                "value": 1.23,
                "unit": "pct",
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )

    summary = generate_claims_from_metrics(project_folder=temp_project_folder)
    assert summary["ok"] is True
    assert summary["claims_written"] == 1

    claims = _read_claims(temp_project_folder)
    assert isinstance(claims, list) and len(claims) == 1

    claim = claims[0]
    validate_claim_record(claim)
    assert claim["kind"] == "computed"
    assert claim["metric_keys"] == ["m1"]
    assert claim["claim_id"] == "computed:m1"


@pytest.mark.unit
def test_generate_claims_from_metrics_ignores_invalid_metric_records(temp_project_folder):
    _write_metrics(
        temp_project_folder,
        [
            {"not": "a metric"},
            {
                "schema_version": "1.0",
                "metric_key": "m_ok",
                "name": "OK",
                "value": 10,
                "created_at": "2025-01-01T00:00:00Z",
            },
        ],
    )

    summary = generate_claims_from_metrics(project_folder=temp_project_folder)
    assert summary["ok"] is True
    assert summary["claims_written"] == 1

    claims = _read_claims(temp_project_folder)
    assert [c["claim_id"] for c in claims] == ["computed:m_ok"]


@pytest.mark.unit
def test_generate_claims_from_metrics_dedupes_duplicate_metric_keys_last_wins(temp_project_folder):
    _write_metrics(
        temp_project_folder,
        [
            {
                "schema_version": "1.0",
                "metric_key": "m1",
                "name": "Metric 1",
                "value": 1,
                "created_at": "2025-01-01T00:00:00Z",
            },
            {
                "schema_version": "1.0",
                "metric_key": "m1",
                "name": "Metric 1",
                "value": 2,
                "created_at": "2025-01-01T00:00:00Z",
            },
        ],
    )

    summary = generate_claims_from_metrics(project_folder=temp_project_folder)
    assert summary["ok"] is True
    assert summary["claims_written"] == 1
    assert summary["duplicate_metric_keys"] == ["m1"]

    claims = _read_claims(temp_project_folder)
    assert len(claims) == 1
    assert claims[0]["claim_id"] == "computed:m1"
    assert claims[0]["metadata"]["value"] == 2
