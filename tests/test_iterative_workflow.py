"""
Integration Tests for Iterative Refinement Workflow
====================================================
Tests the complete iterative refinement pipeline including:
- Agent registry discovery
- Feedback protocol flow
- Version tracking
- Orchestrator coordination
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os
import json

from src.agents.registry import (
    AGENT_REGISTRY,
    AgentRegistry,
    AgentCapability,
    ModelTier as RegistryModelTier,
)
from src.agents.feedback import (
    FeedbackRequest,
    FeedbackResponse,
    QualityScore,
    Issue,
    Severity,
    IssueCategory,
    ConvergenceCriteria,
    RevisionTrigger,
)
from src.agents.base import AgentResult
from src.agents.cache import WorkflowCache
from src.agents.orchestrator import (
    AgentOrchestrator,
    OrchestratorConfig,
    ExecutionMode,
    ExecutionState,
)
from src.llm.claude_client import ModelTier, TaskType


class TestIterativeRefinementWorkflow:
    """Integration tests for full iterative workflow."""
    
    @pytest.fixture
    def project_folder(self):
        """Create a temporary project folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create basic project structure
            os.makedirs(os.path.join(tmpdir, "drafts"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "literature"), exist_ok=True)
            
            # Create project.json
            project = {
                "project_id": "test_project",
                "title": "Test Research Project",
                "research_question": "What factors affect market liquidity?",
            }
            with open(os.path.join(tmpdir, "project.json"), "w") as f:
                json.dump(project, f)
            
            yield tmpdir
    
    def test_agent_registry_integration(self):
        """Test that all agents are properly registered with capabilities."""
        # Verify all agents exist (using dict)
        assert len(AGENT_REGISTRY) == 25  # A01-A25
        
        # Test capability-based discovery via AgentRegistry class
        hypothesis_agents = AgentRegistry.get_by_capability(
            AgentCapability.HYPOTHESIS_DEVELOPMENT
        )
        assert len(hypothesis_agents) >= 1
        assert any(a.id == "A05" for a in hypothesis_agents)
        
        # Test permission checking
        assert AgentRegistry.can_call("A03", "A01")  # LitSearch can call ResearchExplorer
        assert not AgentRegistry.can_call("A01", "A03")  # But not vice versa
    
    def test_feedback_protocol_flow(self):
        """Test complete feedback request/response cycle."""
        # Create quality score (using actual fields)
        score = QualityScore(
            overall=0.75,
            accuracy=0.8,
            clarity=0.7,
            completeness=0.75,
            methodology=0.8,
        )
        
        # Create issues
        issues = [
            Issue(
                severity=Severity.MAJOR,
                category=IssueCategory.CLARITY,
                description="Hypothesis statement is vague",
                location="hypothesis",
                suggestion="Add specific measurable predictions",
            ),
            Issue(
                severity=Severity.MINOR,
                category=IssueCategory.COMPLETENESS,
                description="Missing alternative hypothesis",
            ),
        ]
        
        # Create feedback response
        response = FeedbackResponse(
            request_id="test123",
            reviewer_agent_id="A12",
            quality_score=score,
            issues=issues,
            revision_required=True,
            revision_priority=["Add measurable predictions", "Include alternative hypothesis"],
            summary="Hypothesis needs refinement for specificity",
        )
        
        # Check has_blocking_issues
        assert response.has_blocking_issues is False  # No CRITICAL issues
        assert response.revision_required is True
        
        # Serialize and deserialize
        d = response.to_dict()
        restored = FeedbackResponse.from_dict(d)
        assert restored.quality_score.overall == 0.75
        assert len(restored.issues) == 2
    
    def test_convergence_criteria(self):
        """Test convergence detection logic."""
        criteria = ConvergenceCriteria(
            quality_threshold=0.8,
            max_iterations=3,
            min_improvement=0.05,
        )
        
        # Should continue - below threshold
        result1 = criteria.should_stop(
            current_score=0.6,
            previous_score=None,
            iteration=1,
            critical_count=0,
            major_count=1,
        )
        assert result1[0] is False
        
        # Should stop - above threshold
        result2 = criteria.should_stop(
            current_score=0.85,
            previous_score=0.6,
            iteration=2,
            critical_count=0,
            major_count=0,
        )
        assert result2[0] is True
        assert "threshold" in result2[1].lower()
        
        # Should stop - max iterations
        result3 = criteria.should_stop(
            current_score=0.7,
            previous_score=0.65,
            iteration=3,
            critical_count=0,
            major_count=0,
        )
        assert result3[0] is True
        assert "Maximum" in result3[1]
    
    def test_cache_version_tracking(self, project_folder):
        """Test version tracking in cache."""
        cache = WorkflowCache(project_folder)
        
        # Save multiple versions
        result1 = {
            "content": "Version 1 hypothesis",
            "agent_name": "HypothesisDeveloper",
            "success": True,
        }
        result2 = {
            "content": "Version 2 hypothesis - improved",
            "agent_name": "HypothesisDeveloper",
            "success": True,
        }
        result3 = {
            "content": "Version 3 hypothesis - final",
            "agent_name": "HypothesisDeveloper",
            "success": True,
        }
        
        context = {"project_folder": project_folder}
        
        cache.save_version("hypothesis", result1, context, version=0, quality_score=0.5)
        cache.save_version("hypothesis", result2, context, version=1, quality_score=0.75)
        cache.save_version("hypothesis", result3, context, version=2, quality_score=0.9)
        
        # Get all versions
        versions = cache.get_all_versions("hypothesis")
        assert len(versions) == 3
        
        # Get latest version
        latest = cache.get_latest_version("hypothesis")
        assert latest is not None
        
        # Get best version (returns tuple of version_number, result)
        best_result = cache.get_best_version("hypothesis")
        assert best_result is not None
        version_num, result = best_result
        # The version with quality_score=0.9 is version 2
        assert version_num == 2
    
    def test_execution_state_tracking(self, project_folder):
        """Test execution state tracking across iterations."""
        state = ExecutionState(
            agent_id="A05",
            agent_name="HypothesisDeveloper",
        )
        
        # Track iterations
        state.iteration = 1
        state.quality_scores.append(0.5)
        
        state.iteration = 2
        state.quality_scores.append(0.75)
        
        state.iteration = 3
        state.quality_scores.append(0.9)
        state.converged = True
        state.convergence_reason = "Quality threshold met"
        
        # Verify tracking
        d = state.to_dict()
        assert d["iteration"] == 3
        assert d["quality_scores"] == [0.5, 0.75, 0.9]
        assert d["converged"] is True
    
    @pytest.mark.asyncio
    async def test_orchestrator_permission_enforcement(self, project_folder):
        """Test that orchestrator enforces permission matrix."""
        mock_client = MagicMock()
        mock_client.get_model_for_task.return_value = ModelTier.OPUS
        
        with patch('src.agents.orchestrator.ClaudeClient', return_value=mock_client):
            with patch('src.agents.critical_review.CriticalReviewAgent'):
                config = OrchestratorConfig(
                    enable_inter_agent_calls=True,
                    max_call_depth=2,
                )
                orch = AgentOrchestrator(project_folder, config=config)
                
                # Allowed call
                assert orch._check_permission("A03", "A01") is True
                
                # Disallowed call
                assert orch._check_permission("A01", "A03") is False
    
    @pytest.mark.asyncio
    async def test_orchestrator_call_depth_limiting(self, project_folder):
        """Test that orchestrator limits call depth."""
        mock_client = MagicMock()
        mock_client.get_model_for_task.return_value = ModelTier.OPUS
        
        with patch('src.agents.orchestrator.ClaudeClient', return_value=mock_client):
            with patch('src.agents.critical_review.CriticalReviewAgent'):
                config = OrchestratorConfig(max_call_depth=2)
                orch = AgentOrchestrator(project_folder, config=config)
                
                # Within limit
                orch._call_stack = ["A05"]
                assert orch._check_call_depth() is True
                
                # At limit
                orch._call_stack = ["A05", "A03"]
                assert orch._check_call_depth() is False
    
    def test_agent_result_versioning(self):
        """Test AgentResult supports iteration tracking."""
        result = AgentResult(
            agent_name="HypothesisDeveloper",
            task_type=TaskType.COMPLEX_REASONING,
            model_tier=ModelTier.OPUS,
            success=True,
            content="Initial hypothesis",
            structured_data={"hypothesis": "H1"},
            tokens_used=500,
            execution_time=2.5,
            iteration=1,
            quality_scores={"overall": 0.6},
        )
        
        # Create revised version using actual API
        revised = result.with_revision(
            new_content="Improved hypothesis with specific predictions",
            feedback="Add measurable predictions",
            new_quality_scores={"overall": 0.8},
        )
        
        assert revised.iteration == 2
        assert revised.quality_scores["overall"] == 0.8
        assert len(revised.previous_versions) == 1
        assert "Initial hypothesis" in revised.previous_versions
    
    def test_full_workflow_simulation(self, project_folder):
        """Simulate a complete iterative refinement workflow."""
        cache = WorkflowCache(project_folder)
        context = {"project_folder": project_folder}
        
        # Iteration 1: Initial generation
        result_v1 = {
            "agent_name": "HypothesisDeveloper",
            "task_type": "complex_reasoning",
            "model_tier": "opus",
            "success": True,
            "content": "Market liquidity is affected by trading volume",
            "structured_data": {"hypothesis": "H1: Volume affects liquidity"},
            "tokens_used": 500,
            "execution_time": 2.0,
            "iteration": 0,
        }
        cache.save_version("hypothesis_develop", result_v1, context, version=0, quality_score=0.5)
        
        # Review feedback (simulated)
        feedback_v1 = FeedbackResponse(
            request_id="review_v1",
            reviewer_agent_id="A12",
            quality_score=QualityScore(overall=0.5),
            issues=[
                Issue(
                    severity=Severity.MAJOR,
                    category=IssueCategory.METHODOLOGY,
                    description="Hypothesis is trivial and lacks novelty",
                    suggestion="Add specific mechanisms and testable predictions",
                ),
            ],
            revision_required=True,
        )
        
        # Check if revision needed
        assert feedback_v1.revision_required is True
        
        # Iteration 2: Revised generation
        result_v2 = {
            "agent_name": "HypothesisDeveloper",
            "task_type": "complex_reasoning",
            "model_tier": "opus",
            "success": True,
            "content": "Trading volume affects market liquidity through information asymmetry reduction",
            "structured_data": {
                "hypothesis": "H1: Volume reduces info asymmetry, improving liquidity",
                "predictions": ["P1: Higher volume correlates with lower spreads"],
            },
            "tokens_used": 600,
            "execution_time": 2.5,
            "iteration": 1,
        }
        cache.save_version("hypothesis_develop", result_v2, context, version=1, quality_score=0.75)
        
        # Check convergence
        criteria = ConvergenceCriteria(quality_threshold=0.8)
        should_stop, _ = criteria.should_stop(
            current_score=0.75,
            previous_score=0.5,
            iteration=2,
            critical_count=0,
            major_count=0,
        )
        assert should_stop is False  # Below threshold
        
        # Iteration 3: Final refinement
        result_v3 = {
            "agent_name": "HypothesisDeveloper",
            "task_type": "complex_reasoning",
            "model_tier": "opus",
            "success": True,
            "content": "Trading volume reduces information asymmetry, which decreases adverse selection costs and improves market liquidity",
            "structured_data": {
                "hypothesis": "H1: Volume -> Info asymmetry reduction -> Liquidity improvement",
                "predictions": [
                    "P1: Higher volume correlates with lower bid-ask spreads",
                    "P2: Effect is stronger in periods of high uncertainty",
                ],
                "mechanism": "Information asymmetry channel",
            },
            "tokens_used": 700,
            "execution_time": 3.0,
            "iteration": 2,
        }
        cache.save_version("hypothesis_develop", result_v3, context, version=2, quality_score=0.9)
        
        # Final convergence check - note: iteration param is 1-indexed for "reached max"
        should_stop, reason = criteria.should_stop(
            current_score=0.9,
            previous_score=0.75,
            iteration=2,  # Not yet at max
            critical_count=0,
            major_count=0,
        )
        assert should_stop is True
        assert "threshold" in reason.lower()
        
        # Verify version history
        all_versions = cache.get_all_versions("hypothesis_develop")
        assert len(all_versions) == 3
        
        best_result = cache.get_best_version("hypothesis_develop")
        assert best_result is not None
        version_num, _ = best_result
        assert version_num == 2  # Version 2 has highest score (0.9)


class TestAgentIntegration:
    """Tests for specific agent integrations."""
    
    def test_registry_model_assignment(self):
        """Test that agents have appropriate model tiers assigned."""
        # Complex reasoning agents should use Opus
        hypothesis_dev = AgentRegistry.get("A05")
        assert hypothesis_dev is not None
        assert hypothesis_dev.model_tier == RegistryModelTier.OPUS
        
        # Review agent should use Opus
        critical_review = AgentRegistry.get("A12")
        assert critical_review is not None
        assert critical_review.model_tier == RegistryModelTier.OPUS
        
        # Data analyst uses Haiku for fast tasks
        data_analyst = AgentRegistry.get("A01")
        assert data_analyst is not None
        assert data_analyst.model_tier == RegistryModelTier.HAIKU
    
    def test_registry_permission_matrix_completeness(self):
        """Test that permission matrix is complete and sensible."""
        matrix = AgentRegistry.get_permissions_matrix()
        
        # All registered agents should be in matrix
        for agent_id in AGENT_REGISTRY.keys():
            assert agent_id in matrix, f"{agent_id} not in permissions matrix"
        
        # Critical review agent (A12) should be callable by most agents
        for agent_id, can_call in matrix.items():
            if agent_id != "A12":  # A12 can't call itself
                # Most Phase 2+ agents should be able to call A12
                spec = AgentRegistry.get(agent_id)
                if spec and "A12" in spec.can_call:
                    assert "A12" in can_call, f"{agent_id} should be able to call A12"
    
    def test_capability_coverage(self):
        """Test that all key capabilities are covered by agents."""
        # Essential capabilities for research workflow
        essential_capabilities = [
            AgentCapability.RESEARCH_EXPLORATION,
            AgentCapability.HYPOTHESIS_DEVELOPMENT,
            AgentCapability.LITERATURE_SEARCH,
            AgentCapability.LITERATURE_SYNTHESIS,
            AgentCapability.DATA_ANALYSIS,
            AgentCapability.CRITICAL_REVIEW,
        ]
        
        for cap in essential_capabilities:
            agents = AgentRegistry.get_by_capability(cap)
            assert len(agents) >= 1, f"No agents with capability {cap}"
