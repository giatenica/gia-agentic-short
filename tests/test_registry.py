"""
Tests for Agent Registry
========================
Tests the agent registry, permissions, and discovery functionality.
"""

import pytest
from src.agents.registry import (
    AgentRegistry,
    AgentSpec,
    AgentCapability,
    ModelTier,
    AgentInputSchema,
    AgentOutputSchema,
    get_agent,
    can_agent_call,
    AGENT_REGISTRY,
)


class TestAgentRegistry:
    """Tests for AgentRegistry class."""
    
    def test_registry_has_all_agents(self):
        """Verify all expected agents are registered."""
        expected_ids = [
            "A01", "A02", "A03", "A04",  # Phase 1
            "A05", "A06", "A07", "A08", "A09",  # Phase 2
            "A10", "A11",  # Phase 3
            "A12", "A13", "A14",  # Quality assurance
            "A15",  # Tracking
            "A16",  # Evidence pipeline
            "A17",  # Writing
            "A18",  # Writing
            "A19",  # Review
            "A20",  # Writing
            "A21",  # Writing
            "A22",  # Writing
            "A23",  # Writing
        ]
        
        for agent_id in expected_ids:
            spec = AgentRegistry.get(agent_id)
            assert spec is not None, f"Agent {agent_id} not found in registry"
            assert spec.id == agent_id
    
    def test_get_by_id(self):
        """Test getting agent by ID."""
        spec = AgentRegistry.get("A05")
        assert spec is not None
        assert spec.name == "HypothesisDeveloper"
        assert spec.model_tier == ModelTier.OPUS
    
    def test_get_by_name(self):
        """Test getting agent by name."""
        spec = AgentRegistry.get_by_name("HypothesisDeveloper")
        assert spec is not None
        assert spec.id == "A05"
        
        # Also test by class name
        spec2 = AgentRegistry.get_by_name("HypothesisDevelopmentAgent")
        assert spec2 is not None
        assert spec2.id == "A05"
    
    def test_get_nonexistent_agent(self):
        """Test getting non-existent agent returns None."""
        assert AgentRegistry.get("A99") is None
        assert AgentRegistry.get_by_name("NonExistentAgent") is None
    
    def test_get_by_capability(self):
        """Test finding agents by capability."""
        # Find agents with GAP_ANALYSIS capability
        agents = AgentRegistry.get_by_capability(AgentCapability.GAP_ANALYSIS)
        agent_ids = [a.id for a in agents]
        
        assert "A03" in agent_ids  # GapAnalyst
        assert "A10" in agent_ids  # GapResolver
    
    def test_get_by_model_tier(self):
        """Test finding agents by model tier."""
        opus_agents = AgentRegistry.get_by_model_tier(ModelTier.OPUS)
        opus_ids = [a.id for a in opus_agents]
        
        # A03, A05, A09, A11, A12 should use Opus
        assert "A03" in opus_ids
        assert "A05" in opus_ids
        assert "A09" in opus_ids
        assert "A12" in opus_ids
        
        # A01 should NOT be in Opus (uses Haiku)
        assert "A01" not in opus_ids
    
    def test_list_all(self):
        """Test listing all agents."""
        agents = AgentRegistry.list_all()
        assert len(agents) == 23  # A01-A23
    
    def test_list_ids(self):
        """Test listing all agent IDs."""
        ids = AgentRegistry.list_ids()
        assert len(ids) == 23  # A01-A23
        assert "A01" in ids
        assert "A12" in ids
        assert "A13" in ids  # StyleEnforcer
        assert "A15" in ids  # ReadinessAssessor
        assert "A16" in ids  # EvidenceExtractor
        assert "A17" in ids  # SectionWriter
        assert "A18" in ids  # RelatedWorkWriter
        assert "A19" in ids  # RefereeReview
        assert "A20" in ids  # ResultsWriter
        assert "A21" in ids  # IntroductionWriter
        assert "A22" in ids  # MethodsWriter
        assert "A23" in ids  # DiscussionWriter


class TestAgentPermissions:
    """Tests for inter-agent call permissions."""
    
    def test_can_call_allowed(self):
        """Test that allowed calls are permitted."""
        # A03 (GapAnalyst) can call A01 (DataAnalyst) and A02 (ResearchExplorer)
        assert AgentRegistry.can_call("A03", "A01")
        assert AgentRegistry.can_call("A03", "A02")
    
    def test_can_call_denied(self):
        """Test that disallowed calls are denied."""
        # A01 (DataAnalyst) cannot call anyone
        assert not AgentRegistry.can_call("A01", "A02")
        assert not AgentRegistry.can_call("A01", "A03")
        
        # A06 (LiteratureSearcher) cannot call anyone
        assert not AgentRegistry.can_call("A06", "A05")
    
    def test_can_call_nonexistent(self):
        """Test calling with non-existent agent."""
        assert not AgentRegistry.can_call("A99", "A01")
        assert not AgentRegistry.can_call("A01", "A99")
    
    def test_get_callable_agents(self):
        """Test getting list of callable agents."""
        # A03 can call A01 and A02
        callable_agents = AgentRegistry.get_callable_agents("A03")
        callable_ids = [a.id for a in callable_agents]
        
        assert "A01" in callable_ids
        assert "A02" in callable_ids
        assert len(callable_ids) == 2
    
    def test_permissions_matrix(self):
        """Test getting full permissions matrix."""
        matrix = AgentRegistry.get_permissions_matrix()
        
        assert "A03" in matrix
        assert "A01" in matrix["A03"]
        assert "A02" in matrix["A03"]
        
        # A01 should have empty permissions
        assert matrix["A01"] == []


class TestAgentSpec:
    """Tests for individual agent specifications."""
    
    def test_data_analyst_spec(self):
        """Test DataAnalyst specification."""
        spec = AgentRegistry.get("A01")
        
        assert spec.name == "DataAnalyst"
        assert spec.class_name == "DataAnalystAgent"
        assert spec.module_path == "src.agents.data_analyst"
        assert spec.model_tier == ModelTier.HAIKU
        assert AgentCapability.DATA_ANALYSIS in spec.capabilities
        assert "project_folder" in spec.input_schema.required
        assert spec.supports_revision is True
        assert spec.uses_extended_thinking is False
    
    def test_hypothesis_developer_spec(self):
        """Test HypothesisDeveloper specification."""
        spec = AgentRegistry.get("A05")
        
        assert spec.name == "HypothesisDeveloper"
        assert spec.model_tier == ModelTier.OPUS
        assert AgentCapability.HYPOTHESIS_DEVELOPMENT in spec.capabilities
        assert spec.uses_extended_thinking is True
    
    def test_critical_reviewer_spec(self):
        """Test CriticalReviewer specification."""
        spec = AgentRegistry.get("A12")
        
        assert spec.name == "CriticalReviewer"
        assert spec.model_tier == ModelTier.OPUS
        assert AgentCapability.CRITICAL_REVIEW in spec.capabilities
        assert AgentCapability.CONSISTENCY_CHECK in spec.capabilities
        assert spec.supports_revision is False  # Reviewer doesn't revise itself
        assert spec.can_call == ["A14"]  # Can request cross-document validation


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_agent_function(self):
        """Test get_agent convenience function."""
        spec = get_agent("A05")
        assert spec is not None
        assert spec.id == "A05"
    
    def test_can_agent_call_function(self):
        """Test can_agent_call convenience function."""
        assert can_agent_call("A03", "A01")
        assert not can_agent_call("A01", "A03")


class TestAgentSummary:
    """Tests for registry summary generation."""
    
    def test_summary_generation(self):
        """Test that summary is generated correctly."""
        summary = AgentRegistry.summary()
        
        # Check that it contains expected sections
        assert "Phase 1 - Initial Analysis" in summary
        assert "Phase 2 - Literature" in summary
        assert "Quality Assurance" in summary
        
        # Check that it contains agent info
        assert "HypothesisDeveloper" in summary
        assert "CriticalReviewer" in summary
