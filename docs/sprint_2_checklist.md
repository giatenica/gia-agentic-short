# Sprint 2 Checklist: Citations and Bibliography Correctness

Goal: add a filesystem-first citation system that produces canonical bibliography artifacts, verifies metadata when possible, and provides gates so downstream writing can block or downgrade language when citations are missing or unverified.

## Outcomes
- A project has a stable bibliography folder layout.
- Citation metadata is stored in a canonical registry (`bibliography/citations.json`) that is schema-validated.
- Canonical BibTeX is produced deterministically (`bibliography/references.bib`).
- Crossref verification can upgrade records to verified metadata.
- A citation linting gate can block or downgrade when citations are missing or unverified.
- The literature workflow Phase 2 writes canonical bibliography artifacts and avoids definitive citation claims when verification is incomplete.

## Scope (Sprint 2)
From the roadmap section “Sprint 2: Citations and bibliography correctness” in [docs/next_steps.md](next_steps.md).

- `CitationRecord` schema and validation
- Citation registry (`bibliography/citations.json`)
- Crossref resolver (DOI and basic bibliographic lookup)
- Deterministic bibliography builder (`bibliography/references.bib`)
- Citation linting gate (opt-in)
- Literature workflow Phase 2 integration (best-effort verification and canonical outputs)

Out of scope for Sprint 2
- Claim to evidence alignment checks (citation accuracy verification against EvidenceItems)
- Computation outputs and metrics registry (Sprint 3)

## Definition of Done
Sprint 2 is complete. The corresponding GitHub issues are closed:
- #27 Add CitationRecord schema + citations registry
- #28 Crossref metadata resolver
- #29 Bibliography builder (references.bib + citations.json)
- #30 Citation linter gate
- #31 Integrate citation verification into literature workflow outputs

## Verification plan
- Run unit tests: `pytest -m unit`
- Run full suite: `pytest`
- Manual smoke test
  - Run literature workflow on a sample project
  - Confirm bibliography artifacts appear under `bibliography/`
  - Confirm `references.bib` at project root stays compatible (if present)
  - Confirm downgrade messaging appears when verification is incomplete
