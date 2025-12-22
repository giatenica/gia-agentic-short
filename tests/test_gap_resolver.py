"""
Tests for Gap Resolver Agent
============================
Unit tests for the gap resolution functionality.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agents.gap_resolver import (
    GapResolverAgent,
    OverviewUpdaterAgent,
    CodeExecutor,
    CodeExecutionResult,
    GapResolution,
)
from src.agents.gap_resolution_workflow import (
    GapResolutionWorkflow,
    GapResolutionWorkflowResult,
)
from src.agents.base import AgentResult
from src.llm.claude_client import TaskType, ModelTier


class TestCodeExecutionResult:
    """Tests for CodeExecutionResult dataclass."""
    
    def test_successful_result(self):
        """Test creating a successful execution result."""
        result = CodeExecutionResult(
            success=True,
            code="print('hello')",
            stdout="hello\n",
            stderr="",
            execution_time=0.5,
        )
        assert result.success is True
        assert result.stdout == "hello\n"
        assert result.error is None
    
    def test_failed_result(self):
        """Test creating a failed execution result."""
        result = CodeExecutionResult(
            success=False,
            code="raise Exception('test')",
            stdout="",
            stderr="Exception: test",
            execution_time=0.1,
            error="Exception: test",
        )
        assert result.success is False
        assert result.error == "Exception: test"
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = CodeExecutionResult(
            success=True,
            code="x = 1",
            stdout="",
            stderr="",
            execution_time=0.1,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["code"] == "x = 1"
        assert "execution_time" in d


class TestGapResolution:
    """Tests for GapResolution dataclass."""
    
    def test_basic_resolution(self):
        """Test creating a basic gap resolution."""
        resolution = GapResolution(
            gap_id="C1",
            gap_type="critical",
            gap_description="Symbol verification needed",
            resolution_approach="Check unique symbols in data",
        )
        assert resolution.gap_id == "C1"
        assert resolution.resolved is False
    
    def test_resolved_gap(self):
        """Test a resolved gap."""
        exec_result = CodeExecutionResult(
            success=True,
            code="print('test')",
            stdout="GOOGL, GOOG found",
            stderr="",
            execution_time=1.0,
        )
        resolution = GapResolution(
            gap_id="C1",
            gap_type="critical",
            gap_description="Symbol verification",
            resolution_approach="Check symbols",
            code_generated="print('test')",
            execution_result=exec_result,
            findings="Both symbols present",
            resolved=True,
        )
        assert resolution.resolved is True
        assert resolution.execution_result.success is True
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        resolution = GapResolution(
            gap_id="C1",
            gap_type="critical",
            gap_description="Test gap",
            resolution_approach="Test approach",
        )
        d = resolution.to_dict()
        assert d["gap_id"] == "C1"
        assert d["gap_type"] == "critical"
        assert d["resolved"] is False


class TestCodeExecutor:
    """Tests for CodeExecutor class."""
    
    @pytest.fixture
    def executor(self):
        """Create a code executor for testing."""
        return CodeExecutor(timeout=10)
    
    def test_simple_code_execution(self, executor):
        """Test executing simple Python code."""
        result = executor.execute("print('hello world')")
        assert result.success is True
        assert "hello world" in result.stdout
    
    def test_code_with_computation(self, executor):
        """Test executing code that computes a value."""
        code = """
import math
result = math.sqrt(16)
print(f'Result: {result}')
"""
        result = executor.execute(code)
        assert result.success is True
        assert "4.0" in result.stdout
    
    def test_code_with_error(self, executor):
        """Test executing code that raises an error."""
        code = "raise ValueError('test error')"
        result = executor.execute(code)
        assert result.success is False
        assert "ValueError" in result.stderr or "ValueError" in str(result.error)
    
    def test_code_with_syntax_error(self, executor):
        """Test executing code with syntax error."""
        code = "def broken(:"
        result = executor.execute(code)
        assert result.success is False
    
    def test_timeout_handling(self):
        """Test that code execution respects timeout."""
        executor = CodeExecutor(timeout=1)
        code = """
import time
time.sleep(10)
print('done')
"""
        result = executor.execute(code)
        assert result.success is False
        assert "timed out" in str(result.error).lower()
    
    def test_working_directory(self, executor):
        """Test executing code in specific directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = """
import os
print(os.getcwd())
"""
            result = executor.execute(code, working_dir=tmpdir)
            assert result.success is True
            assert tmpdir in result.stdout or os.path.basename(tmpdir) in result.stdout

    def test_does_not_inherit_api_keys(self, executor, monkeypatch):
        """Test that executed code does not see parent API key environment variables."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "should_not_leak")
        monkeypatch.setenv("SOME_SERVICE_API_KEY", "should_not_leak")
        monkeypatch.setenv("GITHUB_TOKEN", "should_not_leak")

        code = """
import os
print(os.getenv('ANTHROPIC_API_KEY'))
print(os.getenv('SOME_SERVICE_API_KEY'))
print(os.getenv('GITHUB_TOKEN'))
"""
        result = executor.execute(code)
        assert result.success is True
        # If not present, os.getenv prints 'None'
        assert "should_not_leak" not in result.stdout


class TestGapResolverAgent:
    """Tests for GapResolverAgent class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Claude client."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('src.llm.claude_client.anthropic.Anthropic'):
                with patch('src.llm.claude_client.anthropic.AsyncAnthropic'):
                    from src.llm.claude_client import ClaudeClient
                    client = ClaudeClient()
                    return client
    
    @pytest.mark.unit
    def test_agent_initialization(self, mock_client):
        """Test that GapResolverAgent initializes correctly."""
        agent = GapResolverAgent(client=mock_client)
        assert agent.name == "GapResolver"
        assert agent.task_type == TaskType.CODING
        assert agent.executor is not None
    
    @pytest.mark.unit
    def test_extract_data_paths(self, mock_client):
        """Test data path extraction from project folder."""
        agent = GapResolverAgent(client=mock_client)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data structure
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "test.parquet").touch()
            (data_dir / "test.csv").touch()
            
            paths = agent._extract_data_paths(tmpdir, {})
            
            assert len(paths["parquet_files"]) == 1
            assert len(paths["csv_files"]) == 1
            assert paths["project_folder"] == tmpdir
    
    @pytest.mark.unit
    def test_extract_code(self, mock_client):
        """Test code extraction from Claude response."""
        agent = GapResolverAgent(client=mock_client)
        
        response = """Here's the code to analyze the data:
        
```python
import pandas as pd
df = pd.read_parquet('test.parquet')
print(df.head())
```

This will load and display the data."""
        
        code = agent._extract_code(response)
        assert code is not None
        assert "import pandas" in code
        assert "pd.read_parquet" in code
    
    @pytest.mark.unit
    def test_extract_code_no_block(self, mock_client):
        """Test code extraction when no code block present."""
        agent = GapResolverAgent(client=mock_client)
        
        response = "Just some text without code"
        code = agent._extract_code(response)
        assert code is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_resolve_gap_retries_when_execution_fails(self, mock_client):
        """Gap resolution retries code generation when execution fails."""
        agent = GapResolverAgent(client=mock_client, max_code_attempts=2)

        gap = {
            "id": "C1",
            "type": "critical",
            "description": "Test gap",
            "code_approach": "Print hello",
        }
        data_paths = {"all_files": [], "parquet_files": [], "csv_files": [], "project_folder": "/tmp"}

        # First attempt code fails, second attempt code succeeds, then interpretation.
        agent._call_claude = AsyncMock(
            side_effect=[
                ("""```python\nraise ValueError('boom')\n```""", 10),
                ("""```python\nprint('ok')\n```""", 10),
                ("STATUS: RESOLVED\nFINDINGS: ok\n", 5),
            ]
        )

        fail_exec = CodeExecutionResult(
            success=False,
            code="raise ValueError('boom')",
            stdout="",
            stderr="ValueError: boom",
            execution_time=0.01,
            error="ValueError: boom",
        )
        ok_exec = CodeExecutionResult(
            success=True,
            code="print('ok')",
            stdout="ok\n",
            stderr="",
            execution_time=0.01,
            error=None,
        )
        agent.executor.execute = MagicMock(side_effect=[fail_exec, ok_exec])

        resolution, tokens = await agent._resolve_gap(gap, data_paths, project_folder="/tmp")

        assert tokens > 0
        assert resolution.execution_result is not None
        assert resolution.execution_result.success is True
        assert resolution.resolved is True
        assert len(resolution.execution_attempts) == 2
        assert resolution.execution_attempts[0]["execution_success"] is False
        assert resolution.execution_attempts[1]["execution_success"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_resolve_gap_stops_after_max_attempts(self, mock_client):
        """Gap resolution returns unresolved when all execution attempts fail."""
        agent = GapResolverAgent(client=mock_client, max_code_attempts=2)

        gap = {
            "id": "C1",
            "type": "critical",
            "description": "Test gap",
            "code_approach": "Raise",
        }
        data_paths = {"all_files": [], "parquet_files": [], "csv_files": [], "project_folder": "/tmp"}

        agent._call_claude = AsyncMock(
            side_effect=[
                ("""```python\nraise ValueError('boom')\n```""", 10),
                ("""```python\nraise ValueError('boom2')\n```""", 10),
            ]
        )

        fail_exec_1 = CodeExecutionResult(
            success=False,
            code="raise ValueError('boom')",
            stdout="",
            stderr="ValueError: boom",
            execution_time=0.01,
            error="ValueError: boom",
        )
        fail_exec_2 = CodeExecutionResult(
            success=False,
            code="raise ValueError('boom2')",
            stdout="",
            stderr="ValueError: boom2",
            execution_time=0.01,
            error="ValueError: boom2",
        )
        agent.executor.execute = MagicMock(side_effect=[fail_exec_1, fail_exec_2])

        resolution, _ = await agent._resolve_gap(gap, data_paths, project_folder="/tmp")

        assert resolution.execution_result is not None
        assert resolution.execution_result.success is False
        assert resolution.resolved is False
        assert len(resolution.execution_attempts) == 2
        assert "failed after" in resolution.findings.lower()

    @pytest.mark.unit
    def test_extract_status_from_interpretation(self, mock_client):
        """Test STATUS parsing from interpretation text."""
        agent = GapResolverAgent(client=mock_client)

        assert agent._extract_status_from_interpretation("STATUS: RESOLVED") == "RESOLVED"
        assert agent._extract_status_from_interpretation("status: partially resolved") == "PARTIALLY_RESOLVED"
        assert agent._extract_status_from_interpretation("  STATUS: unresolved  ") == "UNRESOLVED"
        assert agent._extract_status_from_interpretation("FINDINGS: none") is None
        assert agent._extract_status_from_interpretation("STATUS: PENDING") is None


class TestOverviewUpdaterAgent:
    """Tests for OverviewUpdaterAgent class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Claude client."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('src.llm.claude_client.anthropic.Anthropic'):
                with patch('src.llm.claude_client.anthropic.AsyncAnthropic'):
                    from src.llm.claude_client import ClaudeClient
                    client = ClaudeClient()
                    return client
    
    @pytest.mark.unit
    def test_agent_initialization(self, mock_client):
        """Test that OverviewUpdaterAgent initializes correctly."""
        agent = OverviewUpdaterAgent(client=mock_client)
        assert agent.name == "OverviewUpdater"
        assert agent.task_type == TaskType.COMPLEX_REASONING
    
    @pytest.mark.unit
    def test_build_resolution_summary(self, mock_client):
        """Test building resolution summary from resolutions."""
        agent = OverviewUpdaterAgent(client=mock_client)
        
        resolutions = [
            {
                "gap_id": "C1",
                "gap_type": "critical",
                "gap_description": "Symbol verification",
                "resolution_approach": "Check unique values",
                "resolved": True,
                "findings": "Both GOOGL and GOOG found",
                "execution_result": {
                    "success": True,
                    "execution_time": 1.5,
                }
            }
        ]
        
        summary = agent._build_resolution_summary(resolutions)
        assert "Gap C1" in summary
        assert "RESOLVED" in summary
        assert "Symbol verification" in summary
    
    @pytest.mark.unit
    def test_extract_markdown_with_frontmatter(self, mock_client):
        """Test markdown extraction when response has frontmatter."""
        agent = OverviewUpdaterAgent(client=mock_client)
        
        response = """---
project_id: test
---
# Research Overview
Content here"""
        
        markdown = agent._extract_markdown(response)
        assert markdown.startswith("---")
        assert "# Research Overview" in markdown
    
    @pytest.mark.unit
    def test_extract_markdown_with_block(self, mock_client):
        """Test markdown extraction from code block."""
        agent = OverviewUpdaterAgent(client=mock_client)
        
        response = """Here's the updated overview:

```markdown
---
project_id: test
---
# Overview
```

That's the update."""
        
        markdown = agent._extract_markdown(response)
        assert "project_id: test" in markdown


class TestGapResolutionWorkflowResult:
    """Tests for GapResolutionWorkflowResult dataclass."""
    
    @pytest.mark.unit
    def test_basic_result(self):
        """Test creating a basic workflow result."""
        result = GapResolutionWorkflowResult(
            success=True,
            project_id="test123",
            project_folder="/path/to/project",
            original_overview_path="/path/to/RESEARCH_OVERVIEW.md",
        )
        assert result.success is True
        assert result.gaps_resolved == 0
        assert result.gaps_total == 0
    
    @pytest.mark.unit
    def test_result_with_resolutions(self):
        """Test result with gap resolutions."""
        result = GapResolutionWorkflowResult(
            success=True,
            project_id="test123",
            project_folder="/path/to/project",
            original_overview_path="/path/to/RESEARCH_OVERVIEW.md",
            gaps_resolved=2,
            gaps_total=3,
            code_executions=[
                {"gap_id": "C1", "success": True, "resolved": True},
                {"gap_id": "C2", "success": True, "resolved": True},
                {"gap_id": "C3", "success": True, "resolved": False},
            ],
        )
        assert result.gaps_resolved == 2
        assert len(result.code_executions) == 3
    
    @pytest.mark.unit
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = GapResolutionWorkflowResult(
            success=True,
            project_id="test123",
            project_folder="/path/to/project",
            original_overview_path="/path/to/RESEARCH_OVERVIEW.md",
            total_tokens=1000,
            total_time=60.5,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["project_id"] == "test123"
        assert d["total_tokens"] == 1000


class TestGapResolutionWorkflow:
    """Tests for GapResolutionWorkflow class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Claude client."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('src.llm.claude_client.anthropic.Anthropic'):
                with patch('src.llm.claude_client.anthropic.AsyncAnthropic'):
                    from src.llm.claude_client import ClaudeClient
                    client = ClaudeClient()
                    return client
    
    @pytest.mark.unit
    def test_workflow_initialization(self, mock_client):
        """Test workflow initializes correctly."""
        workflow = GapResolutionWorkflow(client=mock_client)
        assert workflow.gap_resolver is not None
        assert workflow.overview_updater is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_workflow_missing_overview(self, mock_client):
        """Test workflow fails gracefully when overview missing."""
        workflow = GapResolutionWorkflow(client=mock_client, use_cache=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project.json but no overview
            project_json = Path(tmpdir) / "project.json"
            project_json.write_text('{"id": "test"}')
            
            result = await workflow.run(tmpdir)
            
            assert result.success is False
            assert "RESEARCH_OVERVIEW.md not found" in result.errors[0]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_workflow_missing_project_json(self, mock_client):
        """Test workflow fails when project.json missing."""
        workflow = GapResolutionWorkflow(client=mock_client, use_cache=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create overview but no project.json
            overview = Path(tmpdir) / "RESEARCH_OVERVIEW.md"
            overview.write_text("# Test Overview")
            
            result = await workflow.run(tmpdir)
            
            assert result.success is False
            assert "project.json not found" in result.errors[0]
    
    @pytest.mark.unit
    def test_result_from_cache(self, mock_client):
        """Test reconstructing AgentResult from cache."""
        workflow = GapResolutionWorkflow(client=mock_client)
        
        cached_data = {
            "agent_name": "GapResolver",
            "task_type": "coding",
            "model_tier": "sonnet",
            "success": True,
            "content": "Test content",
            "structured_data": {"test": "data"},
            "error": None,
            "tokens_used": 100,
            "execution_time": 5.0,
            "timestamp": "2025-01-01T00:00:00",
        }
        
        result = workflow._result_from_cache(cached_data)
        
        assert result.agent_name == "GapResolver"
        assert result.success is True
        assert result.tokens_used == 100
