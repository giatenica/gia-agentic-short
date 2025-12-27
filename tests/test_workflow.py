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


class TestLiteratureWorkflowEvidenceIntegration:
    """Tests for evidence pipeline integration in literature workflow."""

    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}, clear=True)
    def test_evidence_pipeline_runs_on_acquired_sources(self, temp_project_folder):
        """Verify evidence pipeline processes acquired sources in sources/ directory."""
        from src.evidence.pipeline import run_evidence_pipeline_for_acquired_sources, EvidencePipelineConfig
        from src.utils.filesystem import source_id_to_dirname
        
        # Create a sources directory with a downloaded source (simulating Step 3.5)
        # Note: source_id_to_dirname converts : to _ for filesystem safety
        source_id = "arxiv:2301.12345"
        dirname = source_id_to_dirname(source_id)  # arxiv_2301.12345
        sources_dir = temp_project_folder / "sources" / dirname
        raw_dir = sources_dir / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "source.txt").write_text(
            "This is evidence text extracted from a paper. It contains statistical claims at 95% confidence.",
            encoding="utf-8",
        )
        
        # Run evidence pipeline on acquired sources (use dirname as source_id since that's how it's stored)
        cfg = EvidencePipelineConfig(enabled=True, max_sources=10, ingest_sources=False)
        result = run_evidence_pipeline_for_acquired_sources(
            project_folder=str(temp_project_folder),
            config=cfg,
            source_ids=[dirname],  # Use dirname since that's what discover_acquired_sources returns
        )
        
        assert result["processed_count"] >= 1
        assert dirname in result["source_ids"]
        
        # Verify evidence was extracted
        evidence_path = sources_dir / "evidence.json"
        assert evidence_path.exists()
        
        import json
        evidence = json.loads(evidence_path.read_text())
        assert isinstance(evidence, list)
        assert len(evidence) >= 1
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}, clear=True)
    def test_discover_acquired_sources_finds_sources_with_raw_files(self, temp_project_folder):
        """Verify discover_acquired_sources finds sources in sources/ directory."""
        from src.evidence.pipeline import discover_acquired_sources
        from src.utils.filesystem import source_id_to_dirname
        
        # Create sources with raw files (use dirnames since that's how they're stored)
        for sid in ["arxiv:123", "pdf:abc456"]:
            dirname = source_id_to_dirname(sid)
            raw_dir = temp_project_folder / "sources" / dirname / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "source.txt").write_text("content", encoding="utf-8")
        
        # Create a source without raw files (should be skipped)
        empty_source = temp_project_folder / "sources" / "empty_source"
        empty_source.mkdir(parents=True)
        
        source_ids = discover_acquired_sources(str(temp_project_folder))
        
        assert len(source_ids) == 2
        assert source_id_to_dirname("arxiv:123") in source_ids
        assert source_id_to_dirname("pdf:abc456") in source_ids
        assert "empty_source" not in source_ids
