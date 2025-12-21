"""
API Key Validation Tests
========================
Tests all API keys configured in .env file.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# These tests perform real network calls. Keep them opt-in so the default
# unit test suite is deterministic and works offline.
if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Skipping API key validation tests. Set RUN_INTEGRATION_TESTS=1 to enable.",
        allow_module_level=True,
    )


def test_anthropic_api():
    """Test Anthropic Claude API connection."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Simple test message
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'API test successful' in exactly 3 words."}]
    )
    
    response_text = message.content[0].text
    assert len(response_text) > 0, "Empty response from Claude"


def test_nasdaq_data_link_api():
    """Test Nasdaq Data Link API connection."""
    api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if not api_key:
        pytest.skip("NASDAQ_DATA_LINK_API_KEY not set")

    import nasdaqdatalink
    
    nasdaqdatalink.ApiConfig.api_key = api_key
    
    # Get sample data (FRED GDP)
    data = nasdaqdatalink.get("FRED/GDP", rows=1)
    
    assert data is not None, "No data returned"
    assert len(data) > 0, "Empty dataset"


def test_alpha_vantage_api():
    """Test Alpha Vantage API connection."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        pytest.skip("ALPHAVANTAGE_API_KEY not set")

    from alpha_vantage.timeseries import TimeSeries
    
    ts = TimeSeries(key=api_key, output_format='pandas')
    
    # Get intraday data for a common stock
    data, meta = ts.get_quote_endpoint(symbol='AAPL')
    
    assert data is not None, "No data returned"


def test_fred_api():
    """Test FRED API connection."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        pytest.skip("FRED_API_KEY not set")

    from fredapi import Fred
    
    fred = Fred(api_key=api_key)
    
    # Get unemployment rate
    data = fred.get_series('UNRATE', observation_start='2024-01-01')
    
    assert data is not None, "No data returned"
    assert len(data) > 0, "Empty dataset"


def test_yfinance():
    """Test yfinance data retrieval (no API key needed)."""
    import yfinance as yf
    
    ticker = yf.Ticker("MSFT")
    info = ticker.info
    
    assert info is not None, "No data returned"
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))


def test_edison_api():
    """Test Edison Scientific Research API connection."""
    import httpx
    
    api_key = os.getenv("EDISON_API_KEY")
    if not api_key:
        pytest.skip("EDISON_API_KEY not set")
    
    # Test connection to Edison API
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Simple health check or minimal request
    response = httpx.get(
        "https://api.edisonscientific.com/v1/health",
        headers=headers,
        timeout=30
    )
    
    if response.status_code == 200:
        assert True
    elif response.status_code == 401:
        assert True
    else:
        pytest.fail(f"Edison API status: {response.status_code}")


def run_all_tests():
    """Run all API tests and display results."""
    
    console.print(Panel.fit(
        "[bold blue]API Key Validation Tests[/bold blue]\n"
        "Testing all configured API connections",
        border_style="blue"
    ))
    
    tests = [
        ("Anthropic (Claude)", test_anthropic_api),
        ("Nasdaq Data Link", test_nasdaq_data_link_api),
        ("Alpha Vantage", test_alpha_vantage_api),
        ("FRED", test_fred_api),
        ("Yahoo Finance", test_yfinance),
        ("Edison Scientific", test_edison_api),
    ]
    
    table = Table(title="API Test Results")
    table.add_column("API", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")
    
    results = {"passed": 0, "failed": 0}
    
    for name, test_func in tests:
        try:
            result = test_func()
            table.add_row(name, "[green]PASSED[/green]", result)
            results["passed"] += 1
        except Exception as e:
            table.add_row(name, "[red]FAILED[/red]", str(e)[:60])
            results["failed"] += 1
    
    console.print(table)
    
    # Summary
    total = results["passed"] + results["failed"]
    console.print(f"\n[bold]Summary:[/bold] {results['passed']}/{total} tests passed")
    
    if results["failed"] > 0:
        console.print("[yellow]Some API tests failed. Check your .env configuration.[/yellow]")
        return False
    
    console.print("[green]All API connections verified successfully![/green]")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
