"""
Tests for Agent Orchestrator
============================
Tests the orchestrator's execution, review, and iteration capabilities.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from src.agents.orchestrator import (
    AgentOrchestrator,
    OrchestratorConfig,
    ExecutionMode,
    ExecutionState,
    create_orchestrator,
)
from src.agents.feedback import (
    ConvergenceCriteria,
    QualityScore,
    FeedbackResponse,
    AgentCallRequest,
)
from src.agents.base import AgentResult
from src.llm.claude_client import TaskType, ModelTier


@pytest.fixture
def temp_project_folder():
    """Create a temporary project folder for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_client():
    """Create a properly mocked ClaudeClient."""
    client = MagicMock()
    client.get_model_for_task.return_value = ModelTier.OPUS
    return client


@pytest.fixture
def orchestrator(temp_project_folder, mock_client):
    """Create an orchestrator with mocked client."""
    with patch('src.agents.orchestrator.ClaudeClient', return_value=mock_client):
        with patch('src.agents.critical_review.CriticalReviewAgent'):
            config = OrchestratorConfig(
                default_mode=ExecutionMode.SINGLE_PASS,
                agent_timeout=10,
                review_timeout=5,
            )
            orch = AgentOrchestrator(temp_project_folder, config=config)
            orch.client = mock_client
            return orch


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        
        assert config.default_mode == ExecutionMode.WITH_REVIEW
        assert config.enable_inter_agent_calls is True
        assert config.max_call_depth == 2
        assert config.auto_review is True
        assert config.cache_versions is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = OrchestratorConfig(
            default_mode=ExecutionMode.ITERATIVE,
            max_call_depth=3,
            convergence=ConvergenceCriteria(
                max_iterations=5,
                quality_threshold=0.9,
            ),
        )
        
        assert config.default_mode == ExecutionMode.ITERATIVE
        assert config.max_call_depth == 3
        assert config.convergence.max_iterations == 5


class TestExecutionState:
    """Tests for ExecutionState tracking."""
    
    def test_execution_state_creation(self):
        """Test creating execution state."""
        state = ExecutionState(
            agent_id="A05",
            agent_name="HypothesisDeveloper",
        )
        
        assert state.iteration == 0
        assert state.converged is False
        assert len(state.quality_scores) == 0
    
    def test_execution_state_to_dict(self):
        """Test state serialization."""
        state = ExecutionState(
            agent_id="A05",
            agent_name="HypothesisDeveloper",
            iteration=2,
            quality_scores=[0.5, 0.7],
            converged=True,
            convergence_reason="Quality threshold met",
        )
        
        d = state.to_dict()
        
        assert d["agent_id"] == "A05"
        assert d["iteration"] == 2
        assert d["converged"] is True


class TestOrchestratorInitialization:
    """Tests for orchestrator initialization."""
    
    def test_orchestrator_init(self, temp_project_folder, mock_client):
        """Test orchestrator initialization."""
        with patch('src.agents.orchestrator.ClaudeClient', return_value=mock_client):
            with patch('src.agents.critical_review.CriticalReviewAgent'):
                orch = AgentOrchestrator(temp_project_folder)
                
                assert orch.project_folder == temp_project_folder
                assert orch.cache is not None
    
    def test_orchestrator_with_custom_config(self, temp_project_folder, mock_client):
        """Test orchestrator with custom config."""
        with patch('src.agents.orchestrator.ClaudeClient', return_value=mock_client):
            with patch('src.agents.critical_review.CriticalReviewAgent'):
                config = OrchestratorConfig(
                    default_mode=ExecutionMode.ITERATIVE,
                    max_call_depth=5,
                )
                orch = AgentOrchestrator(temp_project_folder, config=config)
                
                assert orch.config.default_mode == ExecutionMode.ITERATIVE
                assert orch.config.max_call_depth == 5


class TestPermissionChecking:
    """Tests for permission checking."""
    
    def test_check_permission_allowed(self, orchestrator):
        """Test that allowed calls pass permission check."""
        # A03 can call A01
        assert orchestrator._check_permission("A03", "A01") is True
    
    def test_check_permission_denied(self, orchestrator):
        """Test that disallowed calls fail permission check."""
        # A01 cannot call A03
        assert orchestrator._check_permission("A01", "A03") is False
    
    def test_check_permission_disabled(self, temp_project_folder, mock_client):
        """Test that all calls fail when inter-agent calls disabled."""
        with patch('src.agents.orchestrator.ClaudeClient', return_value=mock_client):
            with patch('src.agents.critical_review.CriticalReviewAgent'):
                config = OrchestratorConfig(enable_inter_agent_calls=False)
                orch = AgentOrchestrator(temp_project_folder, config=config)
                
                # Even allowed calls should fail
                assert orch._check_permission("A03", "A01") is False


class TestCallDepthTracking:
    """Tests for call depth tracking."""
    
    def test_check_call_depth_ok(self, orchestrator):
        """Test depth check passes within limit."""
        orchestrator._call_stack = ["A03"]
        assert orchestrator._check_call_depth() is True
    
    def test_check_call_depth_exceeded(self, orchestrator):
        """Test depth check fails when exceeded."""
        orchestrator._call_stack = ["A03", "A01"]  # Depth 2, max is 2
        assert orchestrator._check_call_depth() is False


class TestAgentExecution:
    """Tests for agent execution."""
    
    @pytest.mark.asyncio
    async def test_execute_unknown_agent(self, orchestrator):
        """Test executing unknown agent returns error."""
        result = await orchestrator.execute_agent("A99", {})
        
        assert result.success is False
        assert "not found" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_agent_with_cache_hit(self, orchestrator):
        """Test that cached results are returned."""
        # Mock cache to return a valid result
        cached_result = {
            "agent_name": "HypothesisDeveloper",
            "task_type": "complex_reasoning",
            "model_tier": "opus",
            "success": True,
            "content": "Cached hypothesis",
            "structured_data": {},
            "tokens_used": 100,
            "execution_time": 1.0,
        }
        
        orchestrator.cache.get_if_valid = MagicMock(
            return_value=(True, cached_result)
        )
        
        result = await orchestrator.execute_agent("A05", {"project_folder": "/test"})
        
        assert result.success is True
        assert result.content == "Cached hypothesis"


class TestInterAgentCalls:
    """Tests for inter-agent call handling."""
    
    @pytest.mark.asyncio
    async def test_handle_call_permission_denied(self, orchestrator):
        """Test call handling with permission denied."""
        request = AgentCallRequest(
            call_id="test123",
            caller_agent_id="A01",  # A01 can't call anyone
            target_agent_id="A03",
            reason="Test call",
        )
        
        response = await orchestrator.handle_inter_agent_call(request)
        
        assert response.success is False
        assert "Permission denied" in response.error
    
    @pytest.mark.asyncio
    async def test_handle_call_depth_exceeded(self, orchestrator):
        """Test call handling when depth exceeded."""
        # Fill call stack to max
        orchestrator._call_stack = ["A09", "A03"]
        
        request = AgentCallRequest(
            call_id="test123",
            caller_agent_id="A03",
            target_agent_id="A01",
            reason="Test call",
        )
        
        response = await orchestrator.handle_inter_agent_call(request)
        
        assert response.success is False
        assert "depth" in response.error.lower()


class TestExecutionSummary:
    """Tests for execution summary."""
    
    def test_get_execution_summary_empty(self, orchestrator):
        """Test summary with no executions."""
        summary = orchestrator.get_execution_summary()
        assert summary == {}
    
    def test_get_execution_summary_with_state(self, orchestrator):
        """Test summary with execution state."""
        orchestrator.execution_states["A05"] = ExecutionState(
            agent_id="A05",
            agent_name="HypothesisDeveloper",
            iteration=2,
            converged=True,
        )
        
        summary = orchestrator.get_execution_summary()
        
        assert "A05" in summary
        assert summary["A05"]["iteration"] == 2
    
    def test_get_agent_summary(self, orchestrator):
        """Test getting summary for specific agent."""
        orchestrator.execution_states["A05"] = ExecutionState(
            agent_id="A05",
            agent_name="HypothesisDeveloper",
        )
        
        summary = orchestrator.get_agent_summary("A05")
        assert summary is not None
        assert summary["agent_id"] == "A05"
        
        # Non-existent agent
        assert orchestrator.get_agent_summary("A99") is None


class TestFactoryFunction:
    """Tests for orchestrator factory function."""
    
    def test_create_orchestrator(self, temp_project_folder, mock_client):
        """Test factory function creates orchestrator."""
        with patch('src.agents.orchestrator.ClaudeClient', return_value=mock_client):
            with patch('src.agents.critical_review.CriticalReviewAgent'):
                orch = create_orchestrator(
                    temp_project_folder,
                    mode=ExecutionMode.ITERATIVE,
                    max_iterations=5,
                    quality_threshold=0.9,
                )
                
                assert orch.config.default_mode == ExecutionMode.ITERATIVE
                assert orch.config.convergence.max_iterations == 5
                assert orch.config.convergence.quality_threshold == 0.9


class TestExecutionModes:
    """Tests for different execution modes."""
    
    def test_execution_mode_values(self):
        """Test execution mode enum values."""
        assert ExecutionMode.SINGLE_PASS.value == "single_pass"
        assert ExecutionMode.WITH_REVIEW.value == "with_review"
        assert ExecutionMode.ITERATIVE.value == "iterative"


class TestIterativeRevisionLoop:
    """Tests for iterative execution correctness."""

    @pytest.mark.asyncio
    async def test_execute_iterative_does_not_reexecute_on_revision(self, orchestrator):
        """Revision loop should review revised content, not re-run the agent."""
        # Spec lookup
        with patch("src.agents.orchestrator.AgentRegistry.get") as mock_get:
            spec = MagicMock()
            spec.name = "DummyAgent"
            spec.supports_revision = True
            mock_get.return_value = spec

            # Avoid cache interactions affecting flow
            orchestrator.cache.save_version = MagicMock()
            orchestrator.cache.get_best_version = MagicMock(return_value=None)

            # Mock agent instance with revise
            dummy_agent = MagicMock()
            dummy_agent.self_critique = AsyncMock(return_value={"scores": {"overall": 0.0}})
            dummy_agent.revise = AsyncMock(
                return_value=AgentResult(
                    agent_name="DummyAgent",
                    task_type=TaskType.CODING,
                    model_tier=ModelTier.SONNET,
                    success=True,
                    content="revised",
                )
            )
            orchestrator._get_agent_instance = MagicMock(return_value=dummy_agent)

            initial_result = AgentResult(
                agent_name="DummyAgent",
                task_type=TaskType.CODING,
                model_tier=ModelTier.SONNET,
                success=True,
                content="initial",
            )
            initial_feedback = FeedbackResponse(
                request_id="r1",
                reviewer_agent_id="A12",
                quality_score=QualityScore(overall=0.1),
                issues=[],
                summary="needs work",
                revision_required=True,
            )

            orchestrator.execute_with_review = AsyncMock(return_value=(initial_result, initial_feedback))

            # After revision, review indicates no further revision.
            orchestrator.review_result = AsyncMock(
                return_value=FeedbackResponse(
                    request_id="r2",
                    reviewer_agent_id="A12",
                    quality_score=QualityScore(overall=0.9),
                    issues=[],
                    summary="ok",
                    revision_required=False,
                )
            )

            # Keep threshold high so self-critique does not short-circuit review.
            orchestrator.config.review_threshold = 0.99

            result = await orchestrator.execute_iterative(
                agent_id="A01",
                context={"project_data": {"id": "test"}},
                content_type="general",
                max_iterations=2,
                convergence=ConvergenceCriteria(max_iterations=2, quality_threshold=0.8),
            )

            assert result.success is True
            assert result.content == "revised"
            # execute_with_review should only be used for the initial execution.
            assert orchestrator.execute_with_review.await_count == 1
            assert dummy_agent.revise.await_count == 1


class TestReviewExistingResult:
    """Tests for reviewing pre-existing results."""

    @pytest.mark.asyncio
    async def test_review_existing_result_skips_on_failed_result(self, orchestrator):
        failed = AgentResult(
            agent_name="DummyAgent",
            task_type=TaskType.CODING,
            model_tier=ModelTier.SONNET,
            success=False,
            content="",
            error="boom",
        )

        feedback = await orchestrator._review_existing_result(
            agent_id="A01",
            result=failed,
            content_type="general",
        )

        assert feedback.revision_required is True
        assert feedback.quality_score.overall == 0.0

    @pytest.mark.asyncio
    async def test_review_existing_result_self_critique_shortcuts(self, orchestrator):
        orchestrator.config.review_threshold = 0.5
        orchestrator.config.auto_review = True

        ok = AgentResult(
            agent_name="DummyAgent",
            task_type=TaskType.CODING,
            model_tier=ModelTier.SONNET,
            success=True,
            content="hello",
        )

        dummy_agent = MagicMock()
        dummy_agent.self_critique = AsyncMock(return_value={"scores": {"overall": 0.9}, "summary": "great"})
        orchestrator._get_agent_instance = MagicMock(return_value=dummy_agent)

        orchestrator.review_result = AsyncMock()

        feedback = await orchestrator._review_existing_result(
            agent_id="A01",
            result=ok,
            content_type="general",
        )

        assert feedback.request_id == "self_critique"
        assert feedback.revision_required is False
        assert orchestrator.review_result.await_count == 0

    @pytest.mark.asyncio
    async def test_review_existing_result_falls_back_when_agent_missing(self, orchestrator):
        orchestrator.config.auto_review = True
        orchestrator._get_agent_instance = MagicMock(return_value=None)

        ok = AgentResult(
            agent_name="DummyAgent",
            task_type=TaskType.CODING,
            model_tier=ModelTier.SONNET,
            success=True,
            content="hello",
        )

        orchestrator.review_result = AsyncMock(
            return_value=FeedbackResponse(
                request_id="r1",
                reviewer_agent_id="A12",
                quality_score=QualityScore(overall=0.3),
                issues=[],
                summary="reviewed",
                revision_required=True,
            )
        )

        feedback = await orchestrator._review_existing_result(
            agent_id="A01",
            result=ok,
            content_type="general",
        )

        assert feedback.request_id == "r1"
        assert orchestrator.review_result.await_count == 1
