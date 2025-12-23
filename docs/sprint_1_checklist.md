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

## Current status

Sprint 1 is complete. The corresponding GitHub issues are closed:
- #6 EvidenceItem schema + validation
- #7 EvidenceStore filesystem layout
- #8 SourceFetcherTool (local ingest MVP)
- #9 Parser interface + MVP parser
- #10 EvidenceExtractorAgent
- #11 Evidence gates

### PR 1: Add EvidenceItem schema and validation
- Add `src/schemas/evidence_item.schema.json` (or equivalent)
- Add `src/utils/schema_validation.py` with `validate_json_schema(data, schema_path)`
- Tests
  - Valid EvidenceItem passes
  - Missing required fields fails
  - Locator variants are validated

### PR 2: EvidenceStore filesystem implementation
- Add `src/evidence/store.py` with an `EvidenceStore` API
- Layout creation includes:
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

Configuration notes:
- Gate enforcement is off by default.
- To enforce, include in workflow context:
  - `evidence_gate.require_evidence: true`
  - `evidence_gate.min_items_per_source: 1` (default)
## Verification plan
- Run unit tests: `pytest -m unit`
- Run full suite: `pytest`
- Manual smoke test
  - Create a sample project
  - Ingest one HTML or text source
  - Parse and extract evidence
  - Confirm a synthesis stage refuses to produce source-backed claims without evidence
