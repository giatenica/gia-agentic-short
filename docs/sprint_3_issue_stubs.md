# Sprint 3 Issue Stubs (copy-paste)

Use these as GitHub issues. Each is PR-sized and maps to [docs/sprint_3_checklist.md](sprint_3_checklist.md).

## Issue 1: Project outputs layout helpers
Title: Sprint 3: Standardize project outputs layout (analysis, outputs, claims)

Body:
- Add helpers to create the standard output folders:
  - `analysis/`
  - `outputs/tables/`
  - `outputs/figures/`
  - `claims/`
- Do not create any outputs by default other than empty folders.

Acceptance criteria:
- Layout creation is idempotent.
- Unit tests cover directory creation.

Labels: sprint-3, outputs

---

## Issue 2: MetricRecord schema + validation
Title: Sprint 3: Add MetricRecord schema and validation

Body:
- Add `src/schemas/metric_record.schema.json`.
- Extend `src/utils/schema_validation.py` with `validate_metric_record(...)`.
- Add tests for pass and fail cases.

Acceptance criteria:
- Minimal valid MetricRecord passes.
- Missing required fields fails.

Labels: sprint-3, schemas

---

## Issue 3: ClaimRecord schema + validation
Title: Sprint 3: Add ClaimRecord schema and validation

Body:
- Add `src/schemas/claim_record.schema.json`.
- Extend `src/utils/schema_validation.py` with `validate_claim_record(...)`.
- Define MVP claim types:
  - `computed` claims must reference one or more metric keys.
  - `source_backed` claims must reference citations and evidence ids (can be optional for MVP if evidence ids are not available yet).

Acceptance criteria:
- Minimal valid ClaimRecord passes.
- Computed claims without metric keys fail validation.

Labels: sprint-3, schemas

---

## Issue 4: Analysis runner MVP
Title: Sprint 3: Add analysis runner to execute scripts and capture provenance

Body:
- Add an MVP analysis runner (script-based) under `src/analysis/` or `src/tools/`.
- It should:
  - Run a Python script from `analysis/`.
  - Record run metadata and created files to `outputs/artifacts.json`.
  - Avoid inheriting secrets where practical.

Acceptance criteria:
- Runner can execute a small sample script in tests using the temp project folder.
- Runner writes `outputs/artifacts.json` deterministically for a fixed input.

Labels: sprint-3, analysis

---

## Issue 5: Computation gate
Title: Sprint 3: Add computation gate for missing metrics and claims

Body:
- Add a gate that checks:
  - If any computed claims exist, referenced metric keys exist in `outputs/metrics.json`.
- Gate is off by default.
- Gate action supports block or downgrade.

Acceptance criteria:
- Gate blocks when required metrics are missing.
- Gate passes when metrics are present.
- Unit tests cover both paths.

Labels: sprint-3, gating
