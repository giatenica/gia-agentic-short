# Next Steps Roadmap (General Capabilities + Architecture)

This roadmap is intentionally general. It describes capabilities that should support many academic writing tasks (empirical papers, theoretical notes, literature reviews, replication studies, methods papers, short articles, and memos), not a single project.

The system can currently plan and scaffold. The missing pieces are the ones that determine whether outputs are defensible: evidence provenance, citation correctness, reproducible computation (when applicable), and section writing that ties claims to verifiable artifacts.

## 0. Guiding principles (acceptance criteria across article types)

- No claim without evidence: every factual statement should link to a source artifact. For text sources, that means a quote or extracted passage with location (page, section, timestamp). For data sources, that means a reproducible transformation and a stored output.
- Prefer reproducibility over fluency: when numbers, tables, figures, or derived facts appear, they must be regenerated from tracked code and inputs.
- Separate retrieval, evidence extraction, synthesis, and writing: avoid “summary of summary” and drifting citations.
- Gate by quality checks: if evidence is thin or citations are unverified, the workflow must stop, downgrade language, or route back to retrieval.
- Keep the architecture modular: different article types should swap in different phases, not fork the whole system.

## 1. Source ingestion and evidence ledger (not only PDFs)

### Why this is blocking
Without a durable evidence layer, the system will sound plausible while being hard to audit. Reviewers tend to punish vagueness: claims like “X finds…” with no traceable support, and empirical claims with no reproducible artifacts.

### Capabilities to add
- Source discovery and resolution to stable identifiers when available (DOI, arXiv ID, Semantic Scholar corpus ID, ISBN, court docket ID, dataset DOI).
- Acquisition of source artifacts where permissible (PDFs, HTML pages, supplementary appendices, datasets, code repositories).
- Parsing into structured text with location preserved (page for PDFs, section or heading for HTML, timestamp markers where relevant).
- Evidence extraction:
	- Quote extraction with location and context windows.
	- Table and figure caption extraction.
	- Key result extraction with exact quoted support.
	- Optional: qualitative memo extraction (definitions, constructs, operationalizations).

### Architectural changes needed
- Introduce a project-level `EvidenceStore` (filesystem first; object store later if desired) with a consistent structure:
	- `sources/<source-id>/raw/` (PDF, HTML snapshot, dataset files, etc)
	- `sources/<source-id>/parsed.json` (location-indexed text)
	- `sources/<source-id>/evidence.json` (quote bank and extracted items)
	- `bibliography/references.bib` (canonical BibTeX where applicable)
- Add a single internal schema for “evidence items” that includes: source id, locator (page or section), excerpt, and extraction timestamp.
- Update synthesis and writing steps to consume evidence items, not ungrounded summaries.

### Agent and tool sketch
- `SourceLocatorAgent`: search and resolve candidate sources to identifiers.
- `SourceFetcherTool`: fetches and snapshots sources where permissible.
- `DocumentParserAgent` (or tool): converts sources into location-indexed text.
- `EvidenceExtractorAgent`: builds evidence items and key-findings lists.
- Refactor synthesis to be evidence-backed by default.

### Acceptance criteria
- For a target set of sources, a high share have parsable text and at least a minimum number of evidence items.
- Output contains no “X finds…” style claims without a corresponding evidence item.

## 2. Citation system and verification layer (metadata correctness + claim checking)

### Why this is blocking
Incorrect citations are an easy reason to reject a manuscript. This is true across disciplines and article types.

### Capabilities to add
- Metadata validation via Crossref (primary), with optional fallback resolvers.
- Version disambiguation:
	- Preprint or working paper versus published version linking.
	- Prefer published versions for citations while tracking alternates.
- Automatic BibTeX generation, plus reference linting.
- Citation accuracy verification:
	- For each key claim attributed to a source, ensure at least one evidence item supports the claim.

### Architectural changes needed
- Canonical bibliography per project (`bibliography/references.bib`) plus a `citations.json` registry tracking canonical keys and verification status.
- A “reference linter” gate before writing begins: block unknown citations or downgrade language.

### Acceptance criteria
- 100% of cited items in the manuscript resolve to validated metadata with a stable identifier.
- No duplicates, no missing year, no missing venue, and page ranges present when available.

## 3. Computation and analysis execution layer (optional but strict when used)

### Why this is blocking
Many manuscripts include quantitative claims. If the system cannot regenerate tables, figures, and numbers on demand, it will eventually contradict itself or drift.

### Capabilities to add
- Safe computation execution:
	- Run code in a constrained environment.
	- Capture dependencies, parameters, and input hashes.
- Output production:
	- Tables and figures suitable for LaTeX.
	- A machine-readable metrics file that text must reference.
- Input QA and feasibility:
	- Schema validation and missingness report.
	- Variable construction feasibility checks.
	- Alignment checks (sample period, definitions, inclusion rules).

### Architectural changes needed
- Standardize project outputs:
	- `analysis/` for scripts and notebooks.
	- `outputs/tables/` for LaTeX tables.
	- `outputs/figures/` for PDFs or PNGs.
	- `outputs/metrics.json` for headline numbers referenced in prose.
- Add a `ClaimsRegistry` that writers must use:
	- quantitative claims map to metrics keys
	- source-backed claims map to evidence item ids

### Acceptance criteria
- Tables and figures can be regenerated from scratch given the project folder.
- All numeric claims in the Results section are pulled from `outputs/metrics.json` or equivalent.

## 4. Configurable writing layer (section writers as plugins)

### Why this is blocking
Scaffolding is not drafting. Writing must be constrained by evidence items, citation registry, and (when relevant) metrics and tables.

### Capabilities to add
- A generic “section writer” interface that can be configured by article type.
- Section-level writing agents that take structured inputs:
	- outline and contribution statement
	- verified citations
	- evidence items
	- claims registry (metrics, tables, figures)
- Output LaTeX sections that:
	- cite using canonical keys
	- reference tables and figures consistently
	- avoid unsupported claims

### Agent sketch
- `SectionWriterAgent` (base interface)
- Common implementations:
	- `IntroductionWriterAgent`
	- `RelatedWorkWriterAgent` (quote-backed)
	- `MethodsWriterAgent`
	- `ResultsWriterAgent` (metrics-backed)
	- `DiscussionWriterAgent`
	- `ConclusionWriterAgent`
- Optional implementations for other article types:
	- `TheoryNoteWriterAgent`
	- `ReplicationWriterAgent`
	- `SurveyWriterAgent`

### Acceptance criteria
- No citation keys appear that are not in the canonical bibliography.
- No results numbers appear that are not in the results registry.
- Literature section contains quote-backed claims for each key cited paper.

## 5. Adversarial review and academic QA (referee simulation + logic and evidence audit)

### Why this is blocking
Experienced reviewers will test identification logic, data construction choices, and whether the paper over-claims. The workflow needs an explicit adversarial review stage that can force revisions.

### Capabilities to add
- `DevilsAdvocateAgent`: challenges contribution, novelty, and plausibility.
- `RefereeSimulatorAgent`: produces a report in referee style with major and minor concerns.
- `EvidenceAuditAgent`: checks that key claims have evidence items and citation keys.
- `MethodAuditAgent`: checks whether the design supports the stated claims (identification, threats, limitations).
- `StatsSanityAgent` (when applicable): clustering choices, multiple testing risk, detectable effect size, placebo and falsification tests.

### Acceptance criteria
- The review stage outputs a structured revision checklist and the workflow must either apply fixes or explicitly record why not.

## 6. Workflow orchestration changes (gates, schemas, and stopping rules)

These changes reduce “looks good but is wrong” outputs.

- Add explicit gates before writing:
	- Literature gate: minimum paper count, verified citations, quote coverage.
	- Analysis gate: tables and figures exist, results registry populated.
	- Consistency gate: cross-file checks pass.
- Add stopping rules:
	- If literature search fails, do not draft literature claims.
	- If citation metadata is unverified, do not format as definitive.
	- If analysis has not run, do not produce numeric conclusions.
- Standardize structured I O between agents (JSON schemas) to reduce drift.
- Make phases configurable by article type, so “literature heavy” tasks and “analysis heavy” tasks can share the same backbone.

## 7. Testing and evaluation additions (to keep quality stable)

- Add unit tests for:
	- bibliography resolution and linting
	- PDF parsing boundaries
	- evidence item schema and quote linking
	- results registry consistency
- Add evaluation sets:
	- citation correctness on a known bibliography
	- claim to quote alignment for a small curated set of papers
	- regression table formatting and number consistency
- Expand tracing to capture:
	- which sources were used
	- which quotes support which claims

## 8. Suggested implementation sequence (minimum viable upgrades)

### Sprint 1: Add the evidence ledger
- Add `EvidenceStore` layout and a minimal `SourceFetcherTool`.
- Add parsing to location-indexed text.
- Require evidence items before synthesis and before “Related Work” writing.

### Sprint 2: Citations and bibliography correctness
- Add DOI and metadata verification and canonical BibTeX.
- Add citation linting gate.

### Sprint 3: Computation outputs + claims registry
- Add analysis runner agent and export of LaTeX tables and figures.
- Add claims registry and enforce writer use.

### Sprint 4: Writing + adversarial review
- Add section writers.
- Add referee and identification audit stage with enforced revision loop.

## How this maps to a project plan (example)

Many projects will still express work as phases (discovery, sources, analysis, writing, polish). The roadmap above defines the shared infrastructure that makes those phases dependable:

- Source phase uses the evidence ledger (Section 1).
- Citation correctness is a gate before serious drafting (Section 2).
- Analysis phase, when needed, produces tables, figures, and metrics that prose must reference (Section 3).
- Writing is a set of pluggable section writers driven by a structured outline and constrained by registries (Section 4).
- Review stages enforce revision loops and prevent unsupported claims from reaching the manuscript (Section 5).

## 9. Architecture spec (schemas and contracts)

This section defines the minimum shared data contracts that all new agents and tools should follow. The goal is interoperability across article types.

### 9.1 Identifiers and storage

- Prefer stable external identifiers when available:
	- DOI, arXiv id, Semantic Scholar corpus id, ISBN.
- Otherwise mint an internal id:
	- `src_<hash>` for sources, `ev_<hash>` for evidence items, `cl_<hash>` for claims, `art_<hash>` for artifacts.
- Store all derived artifacts under the project folder so runs are inspectable and repeatable.

Recommended project layout (minimum):

- `sources/<source_id>/raw/` (downloaded files, snapshots)
- `sources/<source_id>/parsed.json` (location-indexed text)
- `sources/<source_id>/evidence.json` (evidence items)
- `bibliography/references.bib` (canonical BibTeX)
- `bibliography/citations.json` (citation registry)
- `outputs/metrics.json` (numbers referenced by prose)
- `outputs/artifacts.json` (tables, figures, logs)
- `claims/claims.json` (claim registry)

### 9.2 Core schema: `EvidenceItem`

An evidence item is a traceable support unit used to justify a claim.

Minimum fields:

```json
{
	"evidence_id": "ev_...",
	"source_id": "src_...",
	"kind": "quote",
	"locator": {
		"type": "page",
		"value": "12",
		"span": "12-13"
	},
	"excerpt": "...",
	"context": "...",
	"created_at": "2025-12-22T00:00:00Z",
	"parser": {
		"name": "pdf_parser_v1",
		"version": "1.0"
	}
}
```

Notes:

- `kind` can include: `quote`, `table_caption`, `figure_caption`, `definition`, `finding`.
- `locator` must be meaningful for the source type: page for PDFs; heading or section for HTML; timestamp for transcripts.

### 9.3 Core schema: `CitationRecord`

A citation record is the canonical bibliographic entry used by all writers.

```json
{
	"citation_key": "Zingales1994",
	"status": "verified",
	"identifiers": {
		"doi": "10....",
		"arxiv": null
	},
	"title": "...",
	"authors": ["..."],
	"year": 1994,
	"venue": "Journal ...",
	"volume": "12",
	"issue": "3",
	"pages": "123-145",
	"url": "...",
	"version": {
		"type": "published",
		"related_working_paper": "..."
	}
}
```

Rules:

- Writers may only cite keys present in this registry.
- If `status != verified`, prose must downgrade language (no definitive attributions).

### 9.4 Core schema: `ClaimRecord`

A claim record is the unit of meaning that writers are allowed to assert.

```json
{
	"claim_id": "cl_...",
	"type": "source_backed",
	"text": "Zingales (1994) reports ...",
	"support": {
		"citations": ["Zingales1994"],
		"evidence_ids": ["ev_..."]
	},
	"strength": "tentative",
	"created_at": "2025-12-22T00:00:00Z"
}
```

Claim types:

- `source_backed`: must link to at least one evidence item.
- `computed`: must link to a metric key in `outputs/metrics.json`.
- `assumption`: must be labeled as such.
- `design_choice`: must point to the method section inputs.

### 9.5 Core schema: `MetricRecord` (for computed claims)

```json
{
	"metric_key": "main.alpha_baseline",
	"value": 0.0123,
	"units": "USD",
	"description": "Baseline premium estimate",
	"provenance": {
		"run_id": "run_...",
		"script": "analysis/run_main.py",
		"inputs_hash": "sha256:...",
		"created_at": "2025-12-22T00:00:00Z"
	}
}
```

Rule:

- Writers must reference metrics by `metric_key`, not by retyping numbers.

### 9.6 Writer and reviewer contracts

- Writer agents:
	- Input: outline plus registries (`citations.json`, `claims.json`, `metrics.json`).
	- Output: LaTeX sections with citation keys only from the citation registry.
	- Hard constraint: no new facts; only claims in `claims.json`.
- Reviewer agents:
	- Audit that every strong claim maps to evidence, and every number maps to a metric.
	- Output a revision checklist keyed by `claim_id` and `metric_key`.

