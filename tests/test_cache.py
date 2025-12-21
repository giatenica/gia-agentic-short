"""
Tests for Workflow Stage Cache
==============================
"""

import os
import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.agents.cache import WorkflowCache, CacheEntry


@pytest.fixture
def temp_project():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create project.json
        project_data = {
            "id": "test123",
            "title": "Test Project",
            "research_question": "What is the answer?"
        }
        with open(Path(tmpdir) / "project.json", "w") as f:
            json.dump(project_data, f)
        yield tmpdir


@pytest.fixture
def cache(temp_project):
    """Create a cache instance for testing."""
    return WorkflowCache(temp_project)


@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return {
        "project_folder": "/test/project",
        "project_data": {
            "id": "test123",
            "title": "Test Project",
            "research_question": "What is the answer?"
        }
    }


@pytest.fixture
def sample_agent_result():
    """Sample agent result for testing."""
    return {
        "agent_name": "DataAnalyst",
        "task_type": "data_extraction",
        "model_tier": "haiku",
        "success": True,
        "content": "Test analysis content",
        "structured_data": {"key": "value"},
        "error": None,
        "tokens_used": 100,
        "execution_time": 1.5,
        "timestamp": datetime.now().isoformat(),
    }


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = CacheEntry(
            stage_name="data_analyst",
            timestamp="2025-01-01T12:00:00",
            project_id="test123",
            agent_result={"success": True},
            input_hash="abc123",
        )
        
        result = entry.to_dict()
        
        assert result["stage_name"] == "data_analyst"
        assert result["timestamp"] == "2025-01-01T12:00:00"
        assert result["project_id"] == "test123"
        assert result["agent_result"]["success"] is True
        assert result["input_hash"] == "abc123"
    
    def test_from_dict(self):
        """Test reconstruction from dictionary."""
        data = {
            "stage_name": "research_explorer",
            "timestamp": "2025-01-01T12:00:00",
            "project_id": "proj456",
            "agent_result": {"success": False, "error": "test error"},
            "input_hash": "def456",
        }
        
        entry = CacheEntry.from_dict(data)
        
        assert entry.stage_name == "research_explorer"
        assert entry.project_id == "proj456"
        assert entry.agent_result["error"] == "test error"


class TestWorkflowCache:
    """Tests for WorkflowCache class."""
    
    @pytest.mark.unit
    def test_cache_directory_created(self, temp_project):
        """Test that cache directory is created on init."""
        cache = WorkflowCache(temp_project)
        cache_dir = Path(temp_project) / ".workflow_cache"
        assert cache_dir.exists()
        assert cache_dir.is_dir()
    
    @pytest.mark.unit
    def test_save_and_load(self, cache, sample_context, sample_agent_result):
        """Test saving and loading cache entries."""
        cache.save("data_analyst", sample_agent_result, sample_context, "test123")
        
        loaded = cache.load("data_analyst")
        
        assert loaded is not None
        assert loaded["agent_name"] == "DataAnalyst"
        assert loaded["success"] is True
        assert loaded["tokens_used"] == 100
    
    @pytest.mark.unit
    def test_has_valid_cache_fresh(self, cache, sample_context, sample_agent_result):
        """Test has_valid_cache returns True for fresh cache."""
        cache.save("data_analyst", sample_agent_result, sample_context, "test123")
        
        assert cache.has_valid_cache("data_analyst", sample_context)
    
    @pytest.mark.unit
    def test_has_valid_cache_no_cache(self, cache, sample_context):
        """Test has_valid_cache returns False when no cache exists."""
        assert not cache.has_valid_cache("data_analyst", sample_context)
    
    @pytest.mark.unit
    def test_has_valid_cache_input_changed(self, cache, sample_context, sample_agent_result):
        """Test has_valid_cache returns False when inputs change."""
        cache.save("data_analyst", sample_agent_result, sample_context, "test123")
        
        # Modify context
        modified_context = sample_context.copy()
        modified_context["project_data"] = {
            "id": "different123",
            "title": "Different Project",
            "research_question": "Different question"
        }
        
        assert not cache.has_valid_cache("data_analyst", modified_context)
    
    @pytest.mark.unit
    def test_has_valid_cache_failed_result(self, cache, sample_context, sample_agent_result):
        """Test has_valid_cache returns False for failed results."""
        failed_result = sample_agent_result.copy()
        failed_result["success"] = False
        failed_result["error"] = "Something went wrong"
        
        cache.save("data_analyst", failed_result, sample_context, "test123")
        
        assert not cache.has_valid_cache("data_analyst", sample_context)
    
    @pytest.mark.unit
    def test_clear_specific_stage(self, cache, sample_context, sample_agent_result):
        """Test clearing a specific stage's cache."""
        cache.save("data_analyst", sample_agent_result, sample_context, "test123")
        cache.save("research_explorer", sample_agent_result, sample_context, "test123")
        
        cache.clear("data_analyst")
        
        assert cache.load("data_analyst") is None
        assert cache.load("research_explorer") is not None
    
    @pytest.mark.unit
    def test_clear_all(self, cache, sample_context, sample_agent_result):
        """Test clearing all cache."""
        cache.save("data_analyst", sample_agent_result, sample_context, "test123")
        cache.save("research_explorer", sample_agent_result, sample_context, "test123")
        
        cache.clear()
        
        assert cache.load("data_analyst") is None
        assert cache.load("research_explorer") is None
    
    @pytest.mark.unit
    def test_clear_from_stage(self, cache, sample_context, sample_agent_result):
        """Test clearing cache from a specific stage onwards."""
        cache.save("data_analyst", sample_agent_result, sample_context, "test123")
        cache.save("research_explorer", sample_agent_result, sample_context, "test123")
        cache.save("gap_analyst", sample_agent_result, sample_context, "test123")
        
        cache.clear_from_stage("research_explorer")
        
        # data_analyst should remain
        assert cache.load("data_analyst") is not None
        # research_explorer and later should be cleared
        assert cache.load("research_explorer") is None
        assert cache.load("gap_analyst") is None
    
    @pytest.mark.unit
    def test_get_status(self, cache, sample_context, sample_agent_result):
        """Test getting cache status for all stages."""
        cache.save("data_analyst", sample_agent_result, sample_context, "test123")
        
        status = cache.get_status()
        
        assert status["data_analyst"]["cached"] is True
        assert status["data_analyst"]["success"] is True
        assert status["research_explorer"]["cached"] is False
        assert status["gap_analyst"]["cached"] is False
        assert status["overview_generator"]["cached"] is False
    
    @pytest.mark.unit
    def test_input_hash_consistency(self, cache, sample_context):
        """Test that same inputs produce same hash."""
        hash1 = cache._compute_input_hash(sample_context, "data_analyst")
        hash2 = cache._compute_input_hash(sample_context, "data_analyst")
        
        assert hash1 == hash2
    
    @pytest.mark.unit
    def test_input_hash_different_stages(self, cache, sample_context, sample_agent_result):
        """Test that different stages consider different inputs."""
        # Add data_analysis to context (relevant for research_explorer)
        context_with_data = sample_context.copy()
        context_with_data["data_analysis"] = sample_agent_result
        
        hash_stage1 = cache._compute_input_hash(sample_context, "data_analyst")
        hash_stage2 = cache._compute_input_hash(context_with_data, "research_explorer")
        
        # Different stages with different inputs should have different hashes
        assert hash_stage1 != hash_stage2
    
    @pytest.mark.unit
    def test_load_nonexistent(self, cache):
        """Test loading non-existent cache returns None."""
        result = cache.load("nonexistent_stage")
        assert result is None
    
    @pytest.mark.unit
    def test_get_if_valid_returns_tuple(self, cache, sample_context, sample_agent_result):
        """Test get_if_valid returns tuple of (is_valid, data)."""
        cache.save("data_analyst", sample_agent_result, sample_context, "test123")
        
        is_valid, data = cache.get_if_valid("data_analyst", sample_context)
        
        assert is_valid is True
        assert data is not None
        assert data["agent_name"] == "DataAnalyst"
        assert data["success"] is True
    
    @pytest.mark.unit
    def test_get_if_valid_no_cache(self, cache, sample_context):
        """Test get_if_valid returns (False, None) when no cache exists."""
        is_valid, data = cache.get_if_valid("data_analyst", sample_context)
        
        assert is_valid is False
        assert data is None
    
    @pytest.mark.unit
    def test_get_if_valid_input_changed(self, cache, sample_context, sample_agent_result):
        """Test get_if_valid returns (False, None) when inputs change."""
        cache.save("data_analyst", sample_agent_result, sample_context, "test123")
        
        # Modify context
        modified_context = sample_context.copy()
        modified_context["project_data"] = {
            "id": "different123",
            "title": "Different Project",
            "research_question": "Different question"
        }
        
        is_valid, data = cache.get_if_valid("data_analyst", modified_context)
        
        assert is_valid is False
        assert data is None
    
    @pytest.mark.unit
    def test_get_if_valid_failed_result(self, cache, sample_context, sample_agent_result):
        """Test get_if_valid returns (False, None) for failed results."""
        failed_result = sample_agent_result.copy()
        failed_result["success"] = False
        failed_result["error"] = "Something went wrong"
        
        cache.save("data_analyst", failed_result, sample_context, "test123")
        
        is_valid, data = cache.get_if_valid("data_analyst", sample_context)
        
        assert is_valid is False
        assert data is None
