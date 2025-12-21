"""
Workflow Stage Cache
====================
Caches agent outputs between workflow stages to enable resumable workflows.
If a workflow fails at a later stage, previous stage results can be reused.

Cache Structure:
    {project_folder}/.workflow_cache/
        ├── data_analyst.json
        ├── research_explorer.json
        ├── gap_analyst.json
        └── overview_generator.json

Each cache file contains:
    - stage_name: Name of the workflow stage
    - timestamp: When the result was generated
    - project_id: Project identifier
    - agent_result: Full AgentResult as dict
    - input_hash: Hash of inputs (to detect if inputs changed)

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from filelock import FileLock
from loguru import logger


@dataclass
class CacheEntry:
    """A single cached stage result."""
    stage_name: str
    timestamp: str
    project_id: str
    agent_result: dict
    input_hash: str
    
    def to_dict(self) -> dict:
        return {
            "stage_name": self.stage_name,
            "timestamp": self.timestamp,
            "project_id": self.project_id,
            "agent_result": self.agent_result,
            "input_hash": self.input_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(
            stage_name=data["stage_name"],
            timestamp=data["timestamp"],
            project_id=data["project_id"],
            agent_result=data["agent_result"],
            input_hash=data["input_hash"],
        )


class WorkflowCache:
    """
    Manages caching of workflow stage results.
    
    Usage:
        cache = WorkflowCache(project_folder)
        
        # Check if stage is cached
        if cache.has_valid_cache("data_analyst", context):
            result = cache.load("data_analyst")
        else:
            result = await agent.execute(context)
            cache.save("data_analyst", result, context)
    """
    
    CACHE_DIR = ".workflow_cache"
    
    # Stage names in order - Phase 1: Initial Analysis
    STAGES = [
        "data_analyst",
        "research_explorer", 
        "gap_analyst",
        "overview_generator",
    ]
    
    # Stage names - Phase 2: Literature and Planning
    LITERATURE_STAGES = [
        "hypothesis_developer",
        "literature_search",
        "literature_synthesis",
        "paper_structure",
        "project_planner",
    ]
    
    # All stages combined
    ALL_STAGES = STAGES + LITERATURE_STAGES
    
    def __init__(self, project_folder: str, max_age_hours: int = 24):
        """
        Initialize cache for a project.
        
        Args:
            project_folder: Path to the project folder
            max_age_hours: Maximum age of cache entries before they're invalidated
        """
        self.project_folder = Path(project_folder)
        self.cache_dir = self.project_folder / self.CACHE_DIR
        self.max_age_hours = max_age_hours
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Workflow cache initialized at {self.cache_dir}")
    
    def _get_cache_path(self, stage_name: str) -> Path:
        """Get path to cache file for a stage."""
        return self.cache_dir / f"{stage_name}.json"
    
    def _compute_input_hash(self, context: dict, stage_name: str) -> str:
        """
        Compute hash of relevant inputs for a stage.
        
        This helps detect if inputs changed since caching.
        """
        # Select relevant context keys based on stage
        relevant_keys = {
            # Phase 1: Initial Analysis
            "data_analyst": ["project_folder", "project_data"],
            "research_explorer": ["project_folder", "project_data", "data_analysis"],
            "gap_analyst": ["project_folder", "project_data", "data_analysis", "research_analysis"],
            "overview_generator": ["project_folder", "project_data", "data_analysis", "research_analysis", "gap_analysis"],
            # Phase 2: Literature and Planning
            "hypothesis_developer": ["project_folder", "project_data", "research_overview"],
            "literature_search": ["project_folder", "hypothesis_result"],
            "literature_synthesis": ["project_folder", "hypothesis_result", "literature_result"],
            "paper_structure": ["project_folder", "project_data", "hypothesis_result", "literature_result"],
            "project_planner": ["project_folder", "project_data", "hypothesis_result", "literature_result", "paper_structure"],
        }
        
        keys = relevant_keys.get(stage_name, list(context.keys()))
        
        # Build hashable representation
        hash_data = {}
        for key in keys:
            if key in context:
                value = context[key]
                # For project_data, include key fields
                if key == "project_data" and isinstance(value, dict):
                    hash_data[key] = {
                        "id": value.get("id"),
                        "title": value.get("title"),
                        "research_question": value.get("research_question"),
                    }
                # For agent results, include success and timestamp
                elif isinstance(value, dict) and "agent_name" in value:
                    hash_data[key] = {
                        "success": value.get("success"),
                        "timestamp": value.get("timestamp"),
                    }
                else:
                    hash_data[key] = str(value)[:500]  # Truncate long strings
        
        # Compute hash
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]
    
    def _is_cache_fresh(self, timestamp: str) -> bool:
        """Check if cache entry is still fresh based on max age."""
        try:
            cache_time = datetime.fromisoformat(timestamp)
            age = datetime.now() - cache_time
            return age.total_seconds() < (self.max_age_hours * 3600)
        except (ValueError, TypeError):
            return False
    
    def get_if_valid(self, stage_name: str, context: dict) -> Tuple[bool, Optional[dict]]:
        """
        Check validity and return data in a single file read operation.
        
        This is more efficient than calling has_valid_cache() then load()
        as it avoids reading the file twice.
        
        Args:
            stage_name: Name of the workflow stage
            context: Current workflow context for hash validation
            
        Returns:
            Tuple of (is_valid, agent_result_dict or None)
        """
        cache_path = self._get_cache_path(stage_name)
        
        if not cache_path.exists():
            logger.debug(f"No cache found for {stage_name}")
            return False, None
        
        try:
            with open(cache_path) as f:
                data = json.load(f)
            
            entry = CacheEntry.from_dict(data)
            
            # Check freshness
            if not self._is_cache_fresh(entry.timestamp):
                logger.debug(f"Cache expired for {stage_name}")
                return False, None
            
            # Check input hash
            current_hash = self._compute_input_hash(context, stage_name)
            if entry.input_hash != current_hash:
                logger.debug(f"Cache invalidated for {stage_name} (inputs changed)")
                return False, None
            
            # Check if agent result indicates success
            if not entry.agent_result.get("success", False):
                logger.debug(f"Cache skipped for {stage_name} (previous run failed)")
                return False, None
            
            logger.info(f"Valid cache found for {stage_name} (from {entry.timestamp})")
            return True, entry.agent_result
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid cache file for {stage_name}: {e}")
            return False, None
    
    def has_valid_cache(self, stage_name: str, context: dict) -> bool:
        """
        Check if a valid cache exists for a stage.
        
        Note: For better performance, consider using get_if_valid() which
        combines the check and load in a single file read.
        
        Cache is valid if:
        - Cache file exists
        - Cache is not expired
        - Input hash matches (inputs haven't changed)
        """
        is_valid, _ = self.get_if_valid(stage_name, context)
        return is_valid
    
    def load(self, stage_name: str) -> Optional[dict]:
        """
        Load cached result for a stage.
        
        Note: This does NOT validate the cache. Use get_if_valid() for
        combined validation and loading in a single file read.
        
        Returns:
            Agent result dict if cache exists, None otherwise
        """
        cache_path = self._get_cache_path(stage_name)
        
        try:
            with open(cache_path) as f:
                data = json.load(f)
            
            entry = CacheEntry.from_dict(data)
            logger.info(f"Loaded cached result for {stage_name}")
            return entry.agent_result
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache for {stage_name}: {e}")
            return None
    
    def save(self, stage_name: str, agent_result: dict, context: dict, project_id: str = "unknown"):
        """
        Save agent result to cache with file locking for thread safety.
        
        Args:
            stage_name: Name of the workflow stage
            agent_result: AgentResult.to_dict() output
            context: Current workflow context
            project_id: Project identifier
        """
        cache_path = self._get_cache_path(stage_name)
        lock_path = cache_path.with_suffix('.json.lock')
        
        entry = CacheEntry(
            stage_name=stage_name,
            timestamp=datetime.now().isoformat(),
            project_id=project_id,
            agent_result=agent_result,
            input_hash=self._compute_input_hash(context, stage_name),
        )
        
        try:
            # Use file lock to prevent race conditions
            with FileLock(lock_path, timeout=30):
                with open(cache_path, "w") as f:
                    json.dump(entry.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Cached result for {stage_name}")
            
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save cache for {stage_name}: {e}")
    
    def clear(self, stage_name: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            stage_name: Specific stage to clear, or None to clear all
        """
        if stage_name:
            cache_path = self._get_cache_path(stage_name)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache for {stage_name}")
        else:
            # Clear all stages (Phase 1 and Phase 2)
            for stage in self.ALL_STAGES:
                cache_path = self._get_cache_path(stage)
                if cache_path.exists():
                    cache_path.unlink()
            logger.info("Cleared all workflow cache")
    
    def clear_from_stage(self, stage_name: str):
        """
        Clear cache from a specific stage onwards.
        
        Useful when you want to re-run from a certain point.
        Works with both Phase 1 and Phase 2 stages.
        """
        # Try Phase 1 stages first
        if stage_name in self.STAGES:
            stage_index = self.STAGES.index(stage_name)
            for stage in self.STAGES[stage_index:]:
                self.clear(stage)
            logger.info(f"Cleared cache from {stage_name} onwards")
        # Try Phase 2 stages
        elif stage_name in self.LITERATURE_STAGES:
            stage_index = self.LITERATURE_STAGES.index(stage_name)
            for stage in self.LITERATURE_STAGES[stage_index:]:
                self.clear(stage)
            logger.info(f"Cleared cache from {stage_name} onwards")
        else:
            logger.warning(f"Unknown stage: {stage_name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get cache status for all stages (Phase 1 and Phase 2)."""
        status = {}
        for stage in self.ALL_STAGES:
            cache_path = self._get_cache_path(stage)
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        data = json.load(f)
                    status[stage] = {
                        "cached": True,
                        "timestamp": data.get("timestamp"),
                        "success": data.get("agent_result", {}).get("success", False),
                    }
                except (json.JSONDecodeError, KeyError):
                    status[stage] = {"cached": False, "error": "invalid cache"}
            else:
                status[stage] = {"cached": False}
        return status
