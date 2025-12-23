# Sprint 3 Checklist: Computation Outputs and Claims Registry (MVP)

Goal: add a strict computation layer that can regenerate quantitative outputs and a minimal claims registry so numeric statements and key findings are traceable to reproducible artifacts.

This sprint is intentionally MVP. It should enable one end-to-end path: run a small analysis script against project data, store outputs in a standard layout, and expose a machine-readable metrics file and claims registry.

## Outcomes
- Standard analysis and outputs layout exists for a project:
  - `analysis/` for scripts
  - `outputs/tables/` for LaTeX tables
  - `outputs/figures/` for figures
  - `outputs/metrics.json` for headline numbers
  - `outputs/artifacts.json` for run metadata and file inventory
  - `claims/claims.json` for claim registry
- A minimal runner can execute an analysis script in a constrained way and capture provenance.
- A minimal schema exists for MetricRecord and ClaimRecord.
- A workflow gate can block or downgrade when computed claims are missing required metrics.

## Scope (Sprint 3)
From the roadmap section “Sprint 3: Computation outputs + claims registry” in [docs/next_steps.md](next_steps.md).

- Minimal metrics schema and validation
- Minimal claims schema and validation
- Standard project output layout helpers
- MVP analysis runner (script-based)
- MVP gate for computed claims and/or missing metrics

Out of scope for Sprint 3
- Full notebook execution support
- Multi-step pipelines and parameter sweeps
- Full section writers that enforce claim usage everywhere (Sprint 4)

## Definition of Done
- A project gets the standard output folders created automatically.
- An analysis runner can execute a script and record provenance in `outputs/artifacts.json`.
- `outputs/metrics.json` validates and can be referenced by writers.
- `claims/claims.json` validates and can reference metrics keys.
- A gate exists and is off by default; it blocks or downgrades when computed claims are unsupported.
- Unit tests cover schemas, layout creation, and gate behavior.

## Verification plan
- Run unit tests: `pytest -m unit`
- Run full suite: `pytest`
- Manual smoke test
  - Create a sample project with data
  - Run the analysis runner
  - Verify outputs are created and validate against schemas
