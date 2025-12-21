# GIA Agentic Research Pipeline

[![Python CI](https://github.com/giatenica/gia-agentic-short/actions/workflows/ci.yml/badge.svg)](https://github.com/giatenica/gia-agentic-short/actions/workflows/ci.yml)
[![Security](https://github.com/giatenica/gia-agentic-short/actions/workflows/security.yml/badge.svg)](https://github.com/giatenica/gia-agentic-short/actions/workflows/security.yml)

Autonomous AI-powered academic research system for academic research (current module: quantitative finance).

## Author

**Gia Tenica***

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher. For more information see: https://giatenica.com

## Overview

This project implements an agentic research pipeline using the Claude 4.5 model family. The system automates academic research workflows from project intake to gap resolution, with support for:

- Multi-agent workflows with specialized agents per task
- Automated data analysis and gap resolution
- Research overview generation and synthesis
- Prompt caching and batch processing for cost efficiency
- OpenTelemetry tracing for debugging

## Architecture

### Model Selection

| Task Type | Model | Use Case |
|-----------|-------|----------|
| Complex Reasoning | Claude Opus 4.5 | Research synthesis, scientific analysis, academic writing |
| Coding/Agents | Claude Sonnet 4.5 | Code generation, data analysis, agent workflows |
| High-Volume | Claude Haiku 4.5 | Classification, summarization, extraction |

### Agent Framework

All agents inherit from `BaseAgent` which provides:
- **Current date awareness**: Models know today's date for temporal reasoning
- **Web search awareness**: Models flag when they need current information
- **Optimal model selection**: Task-based automatic model routing
- **Prompt caching**: 1-hour TTL by default (90% cost savings on cache hits)
- **Critical rules**: No hallucination, no banned words

### Available Agents

| Agent | Task Type | Model | Purpose |
|-------|-----------|-------|---------|
| ResearchExplorer | Multi-step Research | Opus | Explore research questions and data |
| DataAnalyst | Data Analysis | Sonnet | Analyze datasets and generate statistics |
| GapAnalyst | Data Analysis | Sonnet | Identify gaps in research readiness |
| OverviewGenerator | Document Creation | Sonnet | Generate research overview documents |
| GapResolver | Coding | Sonnet | Generate code to resolve data gaps |
| OverviewUpdater | Complex Reasoning | Opus | Synthesize findings into updated overview |

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

### Environment Configuration

Create a `.env` file:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_key

# Optional - Financial Data APIs
NASDAQ_DATA_LINK_API_KEY=your_key
ALPHAVANTAGE_API_KEY=your_key
FRED_API_KEY=your_key
```

## Usage

### Run Research Workflow

```bash
# Full workflow for a project
.venv/bin/python run_workflow.py user-input/your-project

# Gap resolution workflow
.venv/bin/python run_gap_resolution.py user-input/your-project
```

### Project Structure

```
user-input/your-project/
├── research_request.json       # Project specification
├── data/
│   └── raw data/              # Data files (parquet, csv)
├── RESEARCH_OVERVIEW.md        # Generated overview
├── UPDATED_RESEARCH_OVERVIEW.md # After gap resolution
└── .workflow_cache/            # Stage caching
```

## Claude Client Features

### Prompt Caching (90% Cost Savings)

System prompts are cached by default. Best practices:
- Use 1-hour cache for stable agent prompts
- Use 5-min cache for dynamic content
- Place static content at prompt beginning
- Minimum 1024 tokens to cache (4096 for Opus/Haiku)

```python
from src.llm import get_claude_client

client = get_claude_client()
response = client.chat(
    messages=[{"role": "user", "content": "Analyze..."}],
    system="You are an expert...",  # Cached
    cache_ttl="ephemeral_1h"        # 1-hour cache
)
```

### Batch Processing (50% Cost Savings)

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
│   ├── agents/
│   │   ├── base.py              # BaseAgent with best practices
│   │   ├── best_practices.py    # Standards for all agents
│   │   ├── workflow.py          # Main research workflow
│   │   ├── gap_resolution_workflow.py
│   │   ├── research_explorer.py
│   │   ├── data_analyst.py
│   │   ├── gap_analyst.py
│   │   ├── gap_resolver.py
│   │   ├── overview_generator.py
│   │   └── cache.py             # Workflow caching
│   ├── llm/
│   │   └── claude_client.py     # Claude API with caching & batching
│   └── tracing.py               # OpenTelemetry setup
├── tests/                       # pytest test suite
├── evaluation/                  # Test queries and metrics
├── user-input/                  # Research projects
├── run_workflow.py              # Workflow runner
└── run_gap_resolution.py        # Gap resolution runner
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
            cache_ttl="ephemeral_1h",  # 1-hour cache
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
```

## Cost Optimization

| Strategy | Savings | When to Use |
|----------|---------|-------------|
| Prompt Caching | 90% on cache hits | Always enable for system prompts |
| Batch Processing | 50% | Non-urgent bulk tasks (10+ requests) |
| Model Selection | Variable | Use Haiku for simple tasks, Opus only for complex |
| 1-Hour Cache | Free refresh | Agentic workflows, stable prompts |

## License

Proprietary. All rights reserved.
