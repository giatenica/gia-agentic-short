# Sprint 1 Checklist: Evidence Ledger (MVP)

Goal: add a minimal evidence ledger that makes source-backed claims auditable. This sprint focuses on filesystem-first storage, simple parsing, evidence item extraction, and gating so downstream synthesis does not create unsupported claims.

## Outcomes
- A project has a stable evidence folder layout.
- A source can be fetched or ingested into the project and parsed into location-indexed text.
- Evidence items can be extracted into a machine-readable schema.
- Downstream stages can enforce gates (stop, downgrade language, or request more sources) when evidence is missing.

## Scope (Sprint 1)
From the roadmap section “Sprint 1: Add the evidence ledger” in [docs/next_steps.md](docs/next_steps.md).

- EvidenceStore layout and utilities
- Minimal SourceFetcherTool (permissible acquisition and snapshots)
- Parsing into location-indexed text (`parsed.json`)
- EvidenceItem schema + minimal extractor
- Workflow gates: require evidence items before evidence-backed synthesis and before “Related Work” writing

Out of scope for Sprint 1
- Full citation verification (Sprint 2)
- Claims registry and computation outputs (Sprint 3)
- Section writers and adversarial review (Sprint 4)

## Definition of Done
- Evidence folder layout is created automatically for new projects.
- At least one parser path exists (HTML or plain text is acceptable as MVP). PDF parsing can be stubbed behind an interface.
- EvidenceItem schema is versioned and validated.
- Any stage that produces source-backed claims fails closed when evidence coverage is below a threshold.
- Unit tests cover the EvidenceStore, parsing output format, evidence extraction schema, and gating behavior.

## PR-sized checklist
Each item below should be a separate PR.

## Current status (as of 2025-12-22)

- PR 1 (EvidenceItem schema + validation): Done; merged via PRs #13 and #15.
  - Implemented: [src/schemas/evidence_item.schema.json](src/schemas/evidence_item.schema.json), [src/utils/schema_validation.py](src/utils/schema_validation.py), unit tests.
  - Note: Schema filename is `evidence_item.schema.json` (not `evidence_item.json`).
- PR 2 (EvidenceStore filesystem API): Partial; core append-only ledger exists and is tested; merged via PRs #14 and #15.
  - Implemented: [src/evidence/store.py](src/evidence/store.py) append-only JSONL ledger under `.evidence/evidence.jsonl`.
  - Missing vs checklist: the per-source layout (`sources/<source_id>/raw/`, `parsed.json`, `evidence.json`) is not implemented yet.
- PR 3 (Source ingestion tool): Partial; local discovery + text loading exists and is tested; merged via PR #15.
  - Implemented: [src/evidence/source_fetcher.py](src/evidence/source_fetcher.py) discovers and reads text files under default project dirs.
  - Missing vs checklist: “ingest/copy into `sources/<source_id>/raw/`” is not implemented yet.
- PR 4 (Parser interface + MVP parser): Done for “location-indexed blocks” (in-memory); merged via PR #16.
  - Implemented: [src/evidence/parser.py](src/evidence/parser.py) outputs blocks with 1-based line spans and has unit tests.
  - Missing vs checklist: writing `sources/<source_id>/parsed.json` is not implemented yet.
- PR 5 (Evidence extraction MVP): Not started.
- PR 6 (Evidence gates in workflows): Not started.

### PR 1: Add EvidenceItem schema and validation
- Add `src/schemas/evidence_item.json` (or equivalent)
- Add `src/utils/schema_validation.py` with `validate_json_schema(data, schema_path)`
- Tests
  - Valid EvidenceItem passes
  - Missing required fields fails
  - Locator variants are validated

### PR 2: EvidenceStore filesystem implementation
- Add `src/evidence/store.py` with an `EvidenceStore` API
- Ensure layout creation:
  - `sources/<source_id>/raw/`
  - `sources/<source_id>/parsed.json`
  - `sources/<source_id>/evidence.json`
  - `bibliography/` folder (empty is ok in Sprint 1)
- Tests
  - Creates folders idempotently
  - Writes and reads parsed and evidence files

### PR 3: Source ingestion tool (minimal)
- Add `src/tools/source_fetcher.py` (or `src/evidence/source_fetcher.py`)
- Support at least one input type:
  - local file copy into `sources/<source_id>/raw/`
  - optional: HTTP download with allowlisted schemes and size limits
- Tests
  - Copies local file
  - Rejects unsupported schemes
  - Enforces size limit (if HTTP path implemented)

### PR 4: Parser interface + MVP parser
- Add `src/evidence/parsers/base.py` with `parse_to_locations(...) -> dict`
- Implement one MVP parser:
  - HTML to heading-based chunks, or
  - Plain text chunker
- Output contract: `parsed.json` contains a list of location-indexed blocks
- Tests
  - Parser output is stable and includes locator fields

### PR 5: Evidence extraction MVP
- Add `src/agents/evidence_extractor.py` or similar
- Input: `parsed.json`
- Output: `evidence.json` as a list of EvidenceItems
- MVP extraction can be heuristic (quote-like sentences, definitions, key findings) but must preserve locators and excerpts
- Tests
  - Produces schema-valid evidence items
  - Deterministic output on a fixed input fixture

### PR 6: Gates in workflows
- Add a gating function that checks evidence coverage
- Enforce in the relevant workflow stage(s):
  - If evidence below threshold: stop and return a structured error, or downgrade language
- Tests
  - Gate blocks when evidence missing
  - Gate passes when evidence present

## Verification plan
- Run unit tests: `pytest -m unit`
- Run full suite: `pytest`
- Manual smoke test
  - Create a sample project
  - Ingest one HTML or text source
  - Parse and extract evidence
  - Confirm a synthesis stage refuses to produce source-backed claims without evidence
