"""
Workflow Unit Tests
===================
Tests for the research workflow orchestrator.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.workflow import ResearchWorkflow, WorkflowResult
from src.agents.base import AgentResult
from src.llm.claude_client import TaskType, ModelTier


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""
    
    @pytest.mark.unit
    def test_workflow_result_creation(self):
        """WorkflowResult should be created with required fields."""
        result = WorkflowResult(
            success=True,
            project_id="test_001",
            project_folder="/path/to/project",
        )
        
        assert result.success is True
        assert result.project_id == "test_001"
        assert result.total_tokens == 0
        assert result.errors == []
    
    @pytest.mark.unit
    def test_workflow_result_to_dict(self):
        """to_dict should serialize correctly."""
        result = WorkflowResult(
            success=True,
            project_id="test_001",
            project_folder="/path/to/project",
            total_tokens=5000,
            total_time=45.5,
        )
        
        d = result.to_dict()
        
        assert d["success"] is True
        assert d["project_id"] == "test_001"
        assert d["total_tokens"] == 5000
        assert d["total_time"] == 45.5
        assert "agents" in d
    
    @pytest.mark.unit
    def test_workflow_result_with_errors(self):
        """WorkflowResult should track errors."""
        result = WorkflowResult(
            success=False,
            project_id="test_001",
            project_folder="/path/to/project",
            errors=["Agent 1 failed", "Agent 2 timeout"],
        )
        
        assert result.success is False
        assert len(result.errors) == 2


class TestResearchWorkflow:
    """Tests for ResearchWorkflow class."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    @patch('src.agents.workflow.init_tracing')
    @patch('src.agents.workflow.get_tracer')
    def test_workflow_initialization(self, mock_get_tracer, mock_init_tracing, mock_async_anthropic, mock_anthropic):
        """Workflow should initialize with all agents."""
        mock_get_tracer.return_value = MagicMock()
        
        workflow = ResearchWorkflow()
        
        assert workflow.client is not None
        assert workflow.data_analyst is not None
        assert workflow.research_explorer is not None
        assert workflow.gap_analyst is not None
        assert workflow.overview_generator is not None
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    @patch('src.agents.workflow.init_tracing')
    @patch('src.agents.workflow.get_tracer')
    async def test_workflow_missing_project_folder(
        self, mock_get_tracer, mock_init_tracing, mock_async_anthropic, mock_anthropic
    ):
        """Workflow should fail gracefully with missing folder."""
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        mock_get_tracer.return_value = mock_tracer
        
        workflow = ResearchWorkflow()
        result = await workflow.run("/nonexistent/path/project")
        
        assert result.success is False
        assert len(result.errors) > 0
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    @patch('src.agents.workflow.init_tracing')
    @patch('src.agents.workflow.get_tracer')
    async def test_workflow_missing_project_json(
        self, mock_get_tracer, mock_init_tracing, mock_async_anthropic, mock_anthropic, tmp_path
    ):
        """Workflow should fail when project.json is missing."""
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        mock_get_tracer.return_value = mock_tracer
        
        # Create folder without project.json
        project_folder = tmp_path / "empty_project"
        project_folder.mkdir()
        
        workflow = ResearchWorkflow()
        result = await workflow.run(str(project_folder))
        
        assert result.success is False

    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    @patch('src.agents.workflow.init_tracing')
    @patch('src.agents.workflow.get_tracer')
    def test_save_workflow_results_handles_non_json_values(
        self, mock_get_tracer, mock_init_tracing, mock_async_anthropic, mock_anthropic, tmp_path
    ):
        """Saving workflow_results.json should not crash on non-JSON types in structured_data."""
        mock_get_tracer.return_value = MagicMock()
        workflow = ResearchWorkflow()

        # datetime is not JSON serializable by default and mimics the real-world issue
        # where agent outputs may contain pandas.Timestamp.
        agent_result = AgentResult(
            agent_name="TestAgent",
            task_type=TaskType.DATA_ANALYSIS,
            model_tier=ModelTier.SONNET,
            success=True,
            content="ok",
            structured_data={"generated_at": datetime.now()},
        )

        result = WorkflowResult(
            success=True,
            project_id="test_001",
            project_folder=str(tmp_path),
            data_analysis=agent_result,
        )

        workflow._save_workflow_results(tmp_path, result)
        saved = (tmp_path / "workflow_results.json").read_text()

        parsed = json.loads(saved)
        assert parsed["project_id"] == "test_001"
        assert parsed["agents"]["data_analysis"]["structured_data"]["generated_at"]


class TestWorkflowIntegration:
    """Integration-style tests with mocked agents."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    @patch('src.agents.workflow.init_tracing')
    @patch('src.agents.workflow.get_tracer')
    def test_workflow_with_shared_client(
        self, mock_get_tracer, mock_init_tracing, mock_async_anthropic, mock_anthropic
    ):
        """All agents should share the same client."""
        mock_get_tracer.return_value = MagicMock()
        
        workflow = ResearchWorkflow()
        
        # All agents should use the same client instance
        assert workflow.data_analyst.client is workflow.client
        assert workflow.research_explorer.client is workflow.client
        assert workflow.gap_analyst.client is workflow.client
        assert workflow.overview_generator.client is workflow.client
