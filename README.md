# GIA Agentic Research Pipeline

Autonomous AI-powered academic research system for quantitative finance.

## Author

**Gia Tenica***

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher. For more information see: https://giatenica.com

## Overview

This project implements an agentic research pipeline using Claude Opus 4.5 as the primary model. The system supports:

- Automated literature review and synthesis
- Quantitative financial analysis
- Academic paper generation
- Multi-agent workflows with tracing

## Setup

### Prerequisites

- Python 3.11+
- Virtual environment (included)

### Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file with your API keys:

```env
# Claude API (Primary)
ANTHROPIC_API_KEY=your_key_here

# Financial Data APIs
NASDAQ_DATA_LINK_API_KEY=your_key_here
ALPHAVANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here

# Optional
GITHUB_TOKEN=your_github_token
EDISON_API_KEY=your_key_here
```

## API Integrations

| API | Purpose | Status |
|-----|---------|--------|
| Anthropic (Claude) | Primary LLM | Active |
| Alpha Vantage | Stock quotes and fundamentals | Active |
| FRED | Economic data | Active |
| Yahoo Finance | Market data | Active |
| Nasdaq Data Link | Premium financial data | Requires subscription |

## Project Structure

```
gia-agentic-short/
├── src/
│   ├── llm/
│   │   ├── __init__.py
│   │   └── claude_client.py    # Claude API with batch & caching
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_api_keys.py        # API validation tests
├── .env                        # API keys (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

## Claude Client Features

The `ClaudeClient` class provides:

### Prompt Caching
Reuse expensive system prompts across requests to reduce costs:

```python
from src.llm import get_claude_client

client = get_claude_client()
response = client.chat(
    messages=[{"role": "user", "content": "Analyze this data..."}],
    system="You are a quantitative finance expert...",  # Cached
    cache_system=True
)
```

### Batch Processing
Submit up to 10,000 requests asynchronously:

```python
from src.llm import ClaudeClient, BatchRequest

client = ClaudeClient()

requests = [
    BatchRequest(
        custom_id=f"req_{i}",
        messages=[{"role": "user", "content": f"Analyze stock {ticker}"}]
    )
    for i, ticker in enumerate(["AAPL", "MSFT", "GOOGL"])
]

batch_id = client.create_batch(requests)
```

### Extended Thinking
Enable complex reasoning for difficult problems:

```python
thinking, response = client.chat_with_thinking(
    messages=[{"role": "user", "content": "Complex analysis question..."}],
    budget_tokens=10000
)
```

## Testing

Run API validation tests:

```bash
.venv/bin/python tests/test_api_keys.py
```

## License

Proprietary. All rights reserved.
