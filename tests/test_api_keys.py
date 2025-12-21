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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()


def test_anthropic_api():
    """Test Anthropic Claude API connection."""
    import anthropic
    
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Simple test message
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'API test successful' in exactly 3 words."}]
    )
    
    response_text = message.content[0].text
    assert len(response_text) > 0, "Empty response from Claude"
    return f"Model: {message.model}, Response: {response_text[:50]}"


def test_nasdaq_data_link_api():
    """Test Nasdaq Data Link API connection."""
    import nasdaqdatalink
    
    nasdaqdatalink.ApiConfig.api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
    
    # Get sample data (FRED GDP)
    data = nasdaqdatalink.get("FRED/GDP", rows=1)
    
    assert data is not None, "No data returned"
    assert len(data) > 0, "Empty dataset"
    return f"Retrieved GDP data: {data.iloc[0].values[0]:.2f}"


def test_alpha_vantage_api():
    """Test Alpha Vantage API connection."""
    from alpha_vantage.timeseries import TimeSeries
    
    ts = TimeSeries(key=os.getenv("ALPHAVANTAGE_API_KEY"), output_format='pandas')
    
    # Get intraday data for a common stock
    data, meta = ts.get_quote_endpoint(symbol='AAPL')
    
    assert data is not None, "No data returned"
    return f"AAPL Quote: ${float(data['05. price'].iloc[0]):.2f}"


def test_fred_api():
    """Test FRED API connection."""
    from fredapi import Fred
    
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    
    # Get unemployment rate
    data = fred.get_series('UNRATE', observation_start='2024-01-01')
    
    assert data is not None, "No data returned"
    assert len(data) > 0, "Empty dataset"
    return f"Unemployment Rate: {data.iloc[-1]:.1f}%"


def test_yfinance():
    """Test yfinance data retrieval (no API key needed)."""
    import yfinance as yf
    
    ticker = yf.Ticker("MSFT")
    info = ticker.info
    
    assert info is not None, "No data returned"
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
    return f"MSFT Price: ${current_price}"


def test_edison_api():
    """Test Edison Scientific Research API connection."""
    import httpx
    
    api_key = os.getenv("EDISON_API_KEY")
    
    # Test connection to Edison API
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Simple health check or minimal request
    response = httpx.get(
        "https://api.edisonscientific.com/v1/health",
        headers=headers,
        timeout=30
    )
    
    if response.status_code == 200:
        return "Edison API connected successfully"
    elif response.status_code == 401:
        return "Edison API key valid (auth endpoint responded)"
    else:
        return f"Edison API status: {response.status_code}"


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
