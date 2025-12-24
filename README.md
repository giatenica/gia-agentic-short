# GIA Agentic Research Pipeline

[![Python CI](https://github.com/giatenica/gia-agentic-short/actions/workflows/ci.yml/badge.svg)](https://github.com/giatenica/gia-agentic-short/actions/workflows/ci.yml)
[![Security](https://github.com/giatenica/gia-agentic-short/actions/workflows/security.yml/badge.svg)](https://github.com/giatenica/gia-agentic-short/actions/workflows/security.yml)

Autonomous academic research system (current module: quantitative finance).

## Author

**Gia Tenica***

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher. For more information see: https://giatenica.com*

## Overview

This project implements an agentic research pipeline using the Claude 4.5 model family. The system automates academic research workflows from project intake through literature review and paper structuring, with support for:

- Multi-agent workflows with specialized agents per task
- Automated data analysis and gap resolution
- Edison Scientific API integration for literature search
- Research overview generation and synthesis
- Optional local evidence pipeline (offline source ingest, parsing, and schema-valid evidence extraction)
- LaTeX paper structure generation
- Prompt caching and batch processing support
- OpenTelemetry tracing for debugging

### Safety and Reliability Notes

- Workflows validate the project folder and handle missing or invalid `project.json` without crashing.
- Edison is treated as an optional external dependency; when it is unavailable, the workflow records a failure for that stage and downstream synthesis produces a scaffold output.
- LLM-generated code execution (gap resolution) runs in a subprocess with a minimal environment and isolated Python mode (`-I`). It is designed to reduce accidental secret leakage; it is not a full sandbox.

## Architecture

### Model Selection

| Task Type | Model | Use Case |
|-----------|-------|----------|
| Complex Reasoning | Claude Opus 4.5 | Research synthesis, hypothesis development, project planning |
| Coding/Agents | Claude Sonnet 4.5 | Code generation, literature synthesis, paper structure |
| High-Volume | Claude Haiku 4.5 | Classification, summarization, data extraction |

### Agent Framework

All agents inherit from `BaseAgent` which provides:
- **Current date awareness**: Models know today's date for temporal reasoning
- **Web search awareness**: Models flag when they need current information
- **Optimal model selection**: Task-based automatic model routing
- **Prompt caching**: Uses Anthropic prompt caching controls when enabled
- **Critical rules**: No hallucination, no banned words

### Phase 1 Agents (Initial Analysis)

| Agent | ID | Task Type | Model | Purpose |
|-------|-----|-----------|-------|---------|
| DataAnalyst | A01 | Data Extraction | Haiku | Analyze datasets and generate statistics |
| ResearchExplorer | A02 | Data Analysis | Sonnet | Analyze what the user has provided |
| GapAnalyst | A03 | Complex Reasoning | Opus | Identify missing elements for research |
| OverviewGenerator | A04 | Document Creation | Sonnet | Generate research overview documents |

### Phase 2 Agents (Literature and Planning)

| Agent | ID | Task Type | Model | Purpose |
|-------|-----|-----------|-------|---------|
| HypothesisDeveloper | A05 | Complex Reasoning | Opus | Formulate testable hypotheses |
| LiteratureSearcher | A06 | Data Analysis | Sonnet | Search literature via Edison API |
| LiteratureSynthesizer | A07 | Document Creation | Sonnet | Synthesize literature and create .bib |
| PaperStructurer | A08 | Document Creation | Sonnet | Create LaTeX paper structure |
| ProjectPlanner | A09 | Complex Reasoning | Opus | Create detailed project plan |

### Gap Resolution Agents

| Agent | ID | Task Type | Model | Purpose |
|-------|-----|-----------|-------|---------|
| GapResolver | A10 | Coding | Sonnet | Generate code to resolve data gaps |
| OverviewUpdater | A11 | Complex Reasoning | Opus | Synthesize findings into updated overview |

### Quality Assurance Agents

| Agent | ID | Task Type | Model | Purpose |
|-------|-----|-----------|-------|---------|
| CriticalReviewer | A12 | Complex Reasoning | Opus | Review outputs for quality with extended thinking |
| StyleEnforcer | A13 | Data Extraction | Haiku | Check banned words and style compliance |
| ConsistencyChecker | A14 | Data Analysis | Sonnet | Verify cross-document consistency |
| ReadinessAssessor | A15 | Data Extraction | Haiku | Assess project readiness and track time |

### Evidence Pipeline Agent

| Agent | ID | Task Type | Model | Purpose |
|-------|-----|-----------|-------|---------|
| EvidenceExtractor | A16 | Data Extraction | Haiku | Extract schema-valid evidence items from parsed sources |

### Writing and Review Agents (Optional)

These agents are designed to be deterministic and filesystem-first when possible.

| Agent | ID | Task Type | Model | Purpose |
|-------|-----|-----------|-------|---------|
| SectionWriter | A17 | Document Creation | Sonnet | Minimal section writer interface (stub) writing LaTeX under outputs/sections/ |
| RelatedWorkWriter | A18 | Document Creation | Sonnet | Deterministic Related Work writer constrained by evidence and canonical citations |
| RefereeReview | A19 | Data Extraction | Haiku | Deterministic referee-style checks over generated LaTeX sections |
| ResultsWriter | A20 | Document Creation | Sonnet | Deterministic Results writer; only emits numbers backed by outputs/metrics.json |

### Agent Registry Summary

- Phase 1 (Initial Analysis): A01–A04
- Phase 2 (Literature and Planning): A05–A09
- Phase 3 (Gap Resolution): A10–A11
- Quality Assurance: A12–A14
- Project Tracking: A15
- Evidence Pipeline: A16
- Writing and Review (optional): A17–A20

## Setup

### Prerequisites

- Python 3.11+
- Virtual environment

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

| Package | Purpose |
|---------|---------|
| anthropic | Claude API client |
| edison-client | Edison Scientific literature search |
| httpx | Async HTTP with timeout configuration |
| tenacity | Retry logic with exponential backoff |
| filelock | Thread-safe file operations |
| opentelemetry-* | Distributed tracing |
| loguru | Structured logging |
| pandas | Data analysis utilities |
| pyarrow | Parquet support for pandas |

### Environment Configuration

Create a `.env` file:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_key

# Optional - Edison Scientific (for literature search)
EDISON_API_KEY=your_edison_key

# Optional - Financial Data APIs
NASDAQ_DATA_LINK_API_KEY=your_key
ALPHAVANTAGE_API_KEY=your_key
FRED_API_KEY=your_key

# Optional - Tracing
ENABLE_TRACING=false
OTLP_ENDPOINT=http://localhost:4318/v1/traces
```

Note: If `EDISON_API_KEY` is not set, the literature workflow will skip Edison calls. The literature synthesis stage will produce a scaffold output so the workflow can still complete.

Note: Library imports do not automatically read `.env`. The CLI scripts in `scripts/` call `load_env_file_lenient()` on startup. If you create a `ClaudeClient` directly, either export the needed environment variables, call `load_env_file_lenient()` yourself, or set `GIA_LOAD_ENV_FILE=1`.

## Usage

### Run Research Workflow

```bash
# Phase 1: Initial analysis workflow
.venv/bin/python scripts/run_workflow.py user-input/your-project

# Phase 1.5: Gap resolution workflow
.venv/bin/python scripts/run_gap_resolution.py user-input/your-project

# Phase 2: Literature and planning workflow (requires Phase 1)
.venv/bin/python scripts/run_literature_workflow.py user-input/your-project
```

### Evidence Outputs (Optional)

There are two ways to generate local evidence artifacts:

1) Offline from cached stage files (no model calls):

```bash
.venv/bin/python scripts/run_evidence_from_cache.py user-input/your-project --all-stages --append-ledger
```

This reads `.workflow_cache/*.json`, writes per-stage evidence under `sources/`, optionally appends to `.evidence/evidence.jsonl`, then runs the evidence gate.

2) As a hook during orchestrator execution (off by default):

The orchestrator can write `sources/cache_<stage>/parsed.json` and `sources/cache_<stage>/evidence.json` after each stage result and run `check_evidence_gate(...)`.
Enable it by passing `OrchestratorConfig(enable_evidence_hook=True, ...)` when constructing `AgentOrchestrator`.

Notes:
- The hook is best-effort: failures are logged and do not fail the workflow.
- Ledger append is disabled by default; set `evidence_hook_append_ledger=True` if you want `.evidence/evidence.jsonl` populated.

### Citation Gate (Optional)

The citation gate can lint Markdown/LaTeX citations in a project folder against `bibliography/citations.json`.

Run it locally:

```bash
.venv/bin/python scripts/run_citation_gate.py user-input/your-project --enabled
```

Behavior:
- Missing cited keys default to `block`
- Unverified citations default to `downgrade`

### Analysis Runner (Optional)

The analysis runner executes a Python script under `analysis/` and writes a deterministic provenance record to `outputs/artifacts.json`.

Notes:
- Execution uses isolated Python mode (`-I`) and a minimal environment allowlist to reduce accidental secret leakage.
- The artifacts file includes the script path, SHA-256, subprocess return code, and a list of created files.

See: `src/analysis/runner.py` (`run_project_analysis_script`).

### Computation Gate (Optional)

The computation gate validates that computed claims only reference metrics that exist in `outputs/metrics.json`.

- Claims live at `claims/claims.json` as `ClaimRecord` objects.
- Metrics live at `outputs/metrics.json` as `MetricRecord` objects.

The gate is off by default and supports `block` or `downgrade` behavior when referenced metric keys are missing.

See: `src/claims/gates.py` (`check_computation_gate`, `enforce_computation_gate`).

### Start Intake Server

```bash
# Start web server for project submission
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
