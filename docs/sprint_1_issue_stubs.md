# Sprint 1 Issue Stubs (copy-paste)

Use these as GitHub issues. Each is PR-sized and maps to [docs/sprint_1_checklist.md](docs/sprint_1_checklist.md).

Created issues:
- https://github.com/giatenica/gia-agentic-short/issues/6
- https://github.com/giatenica/gia-agentic-short/issues/7
- https://github.com/giatenica/gia-agentic-short/issues/8
- https://github.com/giatenica/gia-agentic-short/issues/9
- https://github.com/giatenica/gia-agentic-short/issues/10
- https://github.com/giatenica/gia-agentic-short/issues/11

## Issue 1: EvidenceItem schema + validation utility
Title: Sprint 1: Add EvidenceItem schema and JSON schema validation

Body:
- Add `src/schemas/evidence_item.json` (versioned schema)
- Add `src/utils/schema_validation.py` with `validate_json_schema(data, schema_path)`
- Add unit tests for pass and fail cases

Acceptance criteria:
- EvidenceItem validation fails on missing required fields
- EvidenceItem validation passes on a minimal valid item

Notes:
- Keep schema stable and backward compatible once introduced

Labels: sprint-1, evidence, schemas

---

## Issue 2: EvidenceStore filesystem API
Title: Sprint 1: Implement EvidenceStore filesystem layout and helpers

Body:
- Add `src/evidence/store.py` implementing a filesystem-first EvidenceStore
- Ensure the following paths exist for a project:
  - `sources/<source_id>/raw/`
  - `sources/<source_id>/parsed.json`
  - `sources/<source_id>/evidence.json`
  - `bibliography/`

Acceptance criteria:
- Creating the layout is idempotent
- Store can write and read parsed and evidence payloads
- Unit tests cover layout creation and read and write

Labels: sprint-1, evidence

---

## Issue 3: Minimal source ingestion tool
Title: Sprint 1: Add SourceFetcherTool (local ingest MVP)

Body:
- Add `src/tools/source_fetcher.py` (or `src/evidence/source_fetcher.py`)
- MVP: ingest local files into `sources/<source_id>/raw/` with metadata
- Optional: add HTTP download with allowlisted schemes and size limits

Acceptance criteria:
- Local ingest copies the file into the correct folder
- Tool returns a stable `source_id` and artifact paths
- Unit tests cover ingest success and error cases

Labels: sprint-1, evidence, tools

---

## Issue 4: Parser interface + one MVP parser
Title: Sprint 1: Add parser interface and MVP parser that outputs location-indexed blocks

Body:
- Add `src/evidence/parsers/base.py` with a clear output contract
- Implement one MVP parser:
  - HTML to heading-based blocks, or
  - Plain text chunker
- Write output to `sources/<source_id>/parsed.json`

Acceptance criteria:
- `parsed.json` contains deterministic location-indexed blocks
- Each block contains a locator and text content
- Unit tests for output structure

Labels: sprint-1, evidence, parsing

---

## Issue 5: Evidence extraction MVP
Title: Sprint 1: Add EvidenceExtractorAgent to produce EvidenceItems

Body:
- Add `src/agents/evidence_extractor.py` (or similar)
- Input: parsed blocks from `parsed.json`
- Output: `evidence.json` as a list of EvidenceItems

Acceptance criteria:
- Output EvidenceItems validate against the schema
- Extraction preserves locators and excerpts
- Deterministic output for a fixed input fixture

Labels: sprint-1, evidence, agents

---

## Issue 6: Evidence gates in workflows
Title: Sprint 1: Add evidence gates so synthesis fails closed without evidence

Body:
- Add a gating function that checks evidence presence and minimal coverage
- Apply gating before any stage that produces source-backed claims
- Define behavior when gate fails:
  - stop and return a structured error, or
  - downgrade language and emit a clear warning

Acceptance criteria:
- Gate blocks when evidence is missing
- Gate passes when evidence is present
- Unit tests cover both paths

Labels: sprint-1, workflows, gating
