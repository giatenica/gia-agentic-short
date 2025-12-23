# Sprint 2 Issue Stubs (reference)

Sprint 2 is complete; the issues below are already closed. This file exists to keep a stable copy of the Sprint 2 scope and acceptance criteria.

Closed issues:
- https://github.com/giatenica/gia-agentic-short/issues/27
- https://github.com/giatenica/gia-agentic-short/issues/28
- https://github.com/giatenica/gia-agentic-short/issues/29
- https://github.com/giatenica/gia-agentic-short/issues/30
- https://github.com/giatenica/gia-agentic-short/issues/31

---

## Issue 1: CitationRecord schema + registry
Title: Sprint 2: Add CitationRecord schema + citations registry

Acceptance criteria:
- `CitationRecord` validation exists and fails on missing required fields
- A per-project citations registry exists at `bibliography/citations.json`
- Registry is filesystem-first and schema-validated

---

## Issue 2: Crossref resolver
Title: Sprint 2: Crossref metadata resolver (DOI, title lookup)

Acceptance criteria:
- Crossref can resolve a DOI to a schema-valid CitationRecord
- Resolver uses centralized timeouts and handles common error cases

---

## Issue 3: Bibliography builder
Title: Sprint 2: Bibliography builder (references.bib + citations.json)

Acceptance criteria:
- Deterministic BibTeX written to `bibliography/references.bib`
- Deterministic citations registry written to `bibliography/citations.json`
- DOI-based dedupe is deterministic

---

## Issue 4: Citation linting gate
Title: Sprint 2: Citation linter gate (block or downgrade on unverified citations)

Acceptance criteria:
- Gate is off by default
- Gate can block or downgrade based on missing and unverified citation keys
- Gate scans project writing files (Markdown, LaTeX) for citation keys

---

## Issue 5: Phase 2 integration
Title: Sprint 2: Integrate citation verification into literature workflow outputs

Acceptance criteria:
- Literature synthesis writes canonical bibliography artifacts under `bibliography/`
- If verification is incomplete, outputs avoid definitive citation claims and add a clear disclaimer
- Unit tests mock resolver success and failure paths
