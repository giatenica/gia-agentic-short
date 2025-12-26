# GIA Agentic Research Pipeline

[![Python CI](https://github.com/giatenica/gia-agentic-short/actions/workflows/ci.yml/badge.svg)](https://github.com/giatenica/gia-agentic-short/actions/workflows/ci.yml)
[![Security](https://github.com/giatenica/gia-agentic-short/actions/workflows/security.yml/badge.svg)](https://github.com/giatenica/gia-agentic-short/actions/workflows/security.yml)

Fully autonomous academic research pipeline in development.

This repository is building an end to end, agent driven system that goes from project intake to an auditable research output: literature review, structured evidence extraction, optional computation, and paper drafting. The north star is “no claim without traceable support”.

## Author

**Gia Tenica***

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher. For more information see: https://giatenica.com*

## What this is

- A multi agent research pipeline with a clear artifact trail on disk
- A growing set of gates (evidence, citations, analysis) that fail closed or downgrade language
- A schema first approach: JSON schemas are treated as contracts, not suggestions
- A work in progress. Expect changes.

This is not a hosted product. It is a research codebase and a prototype pipeline.

## Architecture at a glance

The pipeline is organized around phases and durable outputs:

- **Intake**: a project folder with `project.json` plus optional data, sources, and notes
- **Workflows**: orchestrated phases that call specialized agents and write Markdown/JSON artifacts
- **Evidence layer (optional)**: offline source ingest and parsing, then schema valid `EvidenceItem` extraction with locators
- **Citations (optional)**: canonical `CitationRecord` registry plus gates and linting
- **Computation (optional)**: analysis scripts produce `MetricRecord` outputs; gates ensure numbers are backed by metrics
- **Writing (optional)**: section writers and referee style review constrained by registries

For roadmap and contracts, see docs/next_steps.md.

### Safety and auditability

- Project folder inputs are validated. Missing or invalid `project.json` should not crash the workflow.
- External dependencies are optional; when they fail, later stages are expected to produce a scaffold output.
- LLM generated code execution runs in a subprocess with isolated Python mode (`-I`) and a minimal environment allowlist. This reduces accidental secret leakage; it is not a full sandbox.

## Agents

The canonical list lives in `src/agents/registry.py`. Current registry IDs:

### Phase 1: Intake and initial analysis

| ID | Agent | Purpose |
|---:|------|---------|
| A01 | DataAnalyst | Analyze project data files and summarize quality and structure |
| A02 | ResearchExplorer | Extract research question, hypotheses, and constraints from the submission |
| A03 | GapAnalyst | Identify missing elements and produce a prioritized gap list |
| A04 | OverviewGenerator | Write `RESEARCH_OVERVIEW.md` |

### Phase 2: Literature and planning

| ID | Agent | Purpose |
|---:|------|---------|
| A05 | HypothesisDeveloper | Turn an overview into testable hypotheses and literature questions |
| A06 | LiteratureSearcher | Search literature (Edison integration when configured) |
| A07 | LiteratureSynthesizer | Produce a literature synthesis and bibliography artifacts |
| A08 | PaperStructurer | Generate LaTeX paper structure |
| A09 | ProjectPlanner | Draft a project plan with milestones and checks |

### Phase 3: Gap resolution

| ID | Agent | Purpose |
|---:|------|---------|
| A10 | GapResolver | Propose code changes or scripts to resolve data or pipeline gaps |
| A11 | OverviewUpdater | Update the overview after gap resolution |

### Quality and tracking

| ID | Agent | Purpose |
|---:|------|---------|
| A12 | CriticalReviewer | Review outputs and surface issues and contradictions |
| A13 | StyleEnforcer | Enforce writing style rules (including banned words list) |
| A14 | ConsistencyChecker | Run cross document consistency checks |
| A15 | ReadinessAssessor | Assess readiness and track timing |

### Evidence and writing (optional)

| ID | Agent | Purpose |
|---:|------|---------|
| A16 | EvidenceExtractor | Extract schema valid evidence items from parsed sources |
| A17 | SectionWriter | Minimal section writer interface (writes LaTeX sections) |
| A18 | RelatedWorkWriter | Write “Related Work” constrained by evidence and citations |
| A19 | RefereeReview | Run deterministic referee style checks over sections |
| A20 | ResultsWriter | Write results constrained by metrics (`outputs/metrics.json`) |
| A21 | IntroductionWriter | Draft an introduction section from registries |
| A22 | MethodsWriter | Draft a methods section from registries |
| A23 | DiscussionWriter | Draft a discussion section from registries |
| A24 | DataAnalysisExecution | Execute project analysis scripts and capture provenance |
| A25 | DataFeasibilityValidation | Check whether the planned analysis is feasible given available data |

## Repository layout

High level structure (see `src/` and `scripts/` for details):

```
gia-agentic-short/
├── src/            # Agents, gates, evidence pipeline, schemas, utilities
├── scripts/        # Local runners for workflows and gates
├── docs/           # Roadmap, checklists, writing style guide
├── tests/          # pytest suite
└── evaluation/     # Evaluation inputs and runners
```

## Contributing

If you want to contribute, please reach out first: me@giatenica.com

This repo is moving quickly and the agent contracts are evolving; coordination up front helps avoid duplicate work.

## Development

Prereqs: Python 3.11+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Unit tests (no external API keys required)
.venv/bin/python -m pytest tests/ -v -m unit
```

## License

Proprietary. All rights reserved.
.venv/bin/python scripts/research_intake_server.py
# Open http://localhost:8080 (or set GIA_INTAKE_PORT)
```

### Project Structure

```
user-input/your-project/
├── project.json                # Project specification
├── analysis/                    # Optional analysis scripts (user-authored)
├── data/
│   └── raw data/              # Data files (parquet, csv)
├── claims/
│   └── claims.json             # Optional ClaimRecord list
├── outputs/
│   ├── artifacts.json          # Analysis provenance (when runner is used)
│   ├── metrics.json            # Optional MetricRecord list
│   ├── tables/
│   └── figures/
│   └── sections/               # Optional LaTeX section outputs (Sprint 4 writers)
├── RESEARCH_OVERVIEW.md        # Generated overview (Phase 1)
├── UPDATED_RESEARCH_OVERVIEW.md # After gap resolution
├── LITERATURE_REVIEW.md        # Literature synthesis (Phase 2)
├── references.bib              # BibTeX bibliography
├── bibliography/               # Canonical bibliography artifacts
│   ├── citations.json
│   └── references.bib
├── paper/
│   └── main.tex               # LaTeX paper structure
├── PROJECT_PLAN.md            # Detailed project plan
└── .workflow_cache/            # Stage caching

# Optional evidence artifacts (when the local evidence pipeline is enabled)
sources/                         # Per-source raw + parsed + extracted evidence
.evidence/evidence.jsonl         # Append-only evidence ledger (JSONL)
```

## Claude Client Features

### Prompt Caching

System prompts can be sent with cache control enabled. Current implementation supports `cache_ttl="ephemeral"`.

```python
from src.llm import get_claude_client

client = get_claude_client()
response = client.chat(
    messages=[{"role": "user", "content": "Analyze..."}],
    system="You are an expert...",  # Cached
    cache_ttl="ephemeral"          # 5-min cache
)
```

### Batch Processing

For non-urgent bulk tasks:

```python
from src.llm import ClaudeClient, BatchRequest

client = ClaudeClient()
requests = [
    BatchRequest(custom_id=f"req_{i}", messages=[...])
    for i in range(100)
]
batch_id = client.create_batch(requests)
```

### Extended Thinking

For complex reasoning:

```python
thinking, response = client.chat_with_thinking(
    messages=[{"role": "user", "content": "Complex question..."}],
    budget_tokens=16000,  # Generous thinking budget
    max_tokens=48000
)
```

## Project Structure

```
gia-agentic-short/
├── src/
│   ├── config.py                # Centralized configuration
│   ├── tracing.py               # OpenTelemetry setup
│   ├── agents/                  # 20 specialized agents (A01-A20)
│   │   ├── base.py              # BaseAgent with best practices
│   │   ├── best_practices.py    # Standards for all agents
│   │   ├── registry.py          # Agent registry and capabilities
│   │   ├── cache.py             # Workflow stage caching
│   │   ├── feedback.py          # Feedback and revision system
│   │   ├── orchestrator.py      # Advanced workflow orchestration
│   │   ├── workflow.py          # Phase 1 research workflow
│   │   ├── gap_resolution_workflow.py  # Gap resolution workflow
│   │   ├── literature_workflow.py  # Phase 2 literature workflow
│   │   ├── data_analyst.py      # A01: Data extraction
│   │   ├── research_explorer.py # A02: Project analysis
│   │   ├── gap_analyst.py       # A03: Gap identification
│   │   ├── overview_generator.py # A04: Overview generation
│   │   ├── hypothesis_developer.py # A05: Hypothesis development
│   │   ├── literature_search.py # A06: Literature search
│   │   ├── literature_synthesis.py # A07: Literature synthesis
│   │   ├── paper_structure.py   # A08: Paper structuring
│   │   ├── project_planner.py   # A09: Project planning
│   │   ├── gap_resolver.py      # A10: Gap resolution
│   │   ├── critical_review.py   # A12: Quality review
│   │   ├── style_enforcer.py    # A13: Style enforcement
│   │   ├── consistency_checker.py # A14: Consistency checking
│   │   ├── readiness_assessor.py # A15: Readiness assessment
│   │   ├── evidence_extractor.py # A16: Evidence extraction
│   │   ├── section_writer.py     # A17: Section writer interface + stub
│   │   ├── related_work_writer.py # A18: Related Work writer
│   │   ├── referee_review.py     # A19: Deterministic referee review
│   │   ├── results_writer.py     # A20: Deterministic Results writer
│   │   └── writing_review_integration.py # Sprint 4 wiring for writing + review stage
│   ├── evidence/                # Evidence pipeline
│   │   ├── extraction.py        # Evidence extraction
│   │   ├── parser.py            # Document parsing
│   │   ├── pipeline.py          # Evidence pipeline
│   │   ├── store.py             # Evidence storage
│   │   └── gates.py             # Evidence gates
│   ├── llm/
│   │   ├── claude_client.py     # Claude API with caching & batching
│   │   └── edison_client.py     # Edison Scientific API client
│   ├── schemas/                 # JSON schemas for validation
│   └── utils/
│       ├── validation.py        # Path and input validation
│       ├── schema_validation.py # JSON schema validation
│       ├── time_tracking.py     # Execution time tracking
│       ├── readiness_scoring.py # Project readiness scoring
│       ├── consistency_validation.py # Cross-document consistency
│       ├── style_validation.py  # Style guide enforcement
│       ├── filesystem.py        # Filesystem helpers
│       └── project_io.py        # Project I/O utilities
├── tests/                       # pytest test suite
├── evaluation/                  # Test queries and metrics
├── user-input/                  # Research projects
├── scripts/
│   ├── run_workflow.py          # Phase 1 workflow runner
│   ├── run_gap_resolution.py    # Gap resolution runner
│   ├── run_literature_workflow.py # Phase 2 workflow runner
│   └── research_intake_server.py # Web intake form server
└── docs/                        # Additional documentation
```

## Building New Agents

Use the best practices module:

```python
from src.agents.base import BaseAgent, AgentResult
from src.agents.best_practices import get_agent_config
from src.llm.claude_client import TaskType

class MyAgent(BaseAgent):
    def __init__(self, client=None):
        # Get configuration with all best practices
        config = get_agent_config(
            agent_name="MyAgent",
            base_prompt="You are an expert at...",
            task_type=TaskType.DATA_ANALYSIS,  # Routes to Sonnet
            add_date=True,      # Include current date
            add_web_awareness=True,  # Include web search awareness
        )
        
        super().__init__(
            name=config["name"],
            task_type=config["task_type"],
            system_prompt=config["system_prompt"],
            client=client,
            cache_ttl="ephemeral",  # 5-min cache
        )
    
    async def execute(self, context: dict) -> AgentResult:
        # Agent implementation
        response, tokens = await self._call_claude(prompt)
        return self._build_result(
            success=True,
            content=response,
            tokens_used=tokens,
        )
```

## Testing

```bash
# Run all unit tests
.venv/bin/python -m pytest tests/ -v -m unit

# Run specific test file
.venv/bin/python -m pytest tests/test_agents.py -v

# Run with coverage
.venv/bin/python -m pytest tests/ -v --cov=src
```

## Cost Management

| Strategy | When to Use |
|----------|-------------|
| Prompt Caching | Repeated system prompts and stable instructions |
| Batch Processing | Non-urgent bulk tasks (10+ requests) |
| Model Selection | Use Haiku for simple tasks; use Opus for complex reasoning |

## Reliability and Performance

### API Resilience

- **Retry Logic**: Exponential backoff for transient failures (Claude and Edison)
- **Timeouts**: Centralized timeout configuration in `src/config.py`
- **Rate Limiting**: Automatic retry on rate limits with backoff
- **Thread-Safe**: Token usage tracking is thread-safe for concurrent operations

### Cache System

- **Thread Safety**: File locking via `filelock` prevents corruption
- **Optimized I/O**: Combined validation and loading in single read
- **Hash Validation**: SHA256 input hashing detects stale cache entries
- **24-Hour TTL**: Configurable expiration with stage-aware dependencies
- **Lock Cleanup**: Orphaned lock files are automatically cleaned up

### Tracing

- **OpenTelemetry**: Full distributed tracing support
- **Graceful Shutdown**: Tracer provider cleanup via atexit handler
- **HTTP Instrumentation**: Automatic tracing of API calls

### Security

- **Path Validation**: All file paths validated against project boundaries
- **Zip Extraction**: Secure extraction with path traversal prevention
- **Code Execution**: Subprocess isolation with minimal environment
- **Input Sanitization**: All user inputs validated before processing
- **API Key Protection**: Keys loaded from environment, never logged

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ANTHROPIC_API_KEY not set` | Add key to `.env` file or environment |
| `Rate limit exceeded` | Retry logic handles automatically; wait if persistent |
| `Cache validation failed` | Delete `.workflow_cache/` folder and re-run |
| `Edison API timeout` | Check network; API has retry logic with exponential backoff |
| `Path validation error` | Ensure project folder exists and has correct structure |

### Debug Mode

Enable detailed logging:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Enable tracing
export ENABLE_TRACING=true
export OTLP_ENDPOINT=http://localhost:4318/v1/traces
```

### Reset Workflow State

```bash
# Clear cached workflow stages
rm -rf user-input/your-project/.workflow_cache/

# Regenerate all outputs
.venv/bin/python scripts/run_workflow.py user-input/your-project
```

## Contributing

### Development Setup

```bash
# Clone and install
git clone https://github.com/giatenica/gia-agentic-short.git
cd gia-agentic-short
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Adding New Agents

1. Create agent file in `src/agents/`
2. Inherit from `BaseAgent`
3. Register in `src/agents/registry.py`
4. Add tests in `tests/test_*.py`
5. Update workflow if needed

### Code Standards

- Type hints on all function signatures
- Docstrings for public methods
- No bare `except:` clauses (use specific exceptions)
- Follow existing naming conventions
- Run tests before committing: `.venv/bin/python -m pytest tests/ -v -m unit`

## License

Proprietary. All rights reserved.
