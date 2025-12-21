"""
Workflow Stage Cache
====================
Caches agent outputs between workflow stages to enable resumable workflows.
If a workflow fails at a later stage, previous stage results can be reused.

Extended with version tracking for iterative refinement:
- Multiple versions per stage (for revision history)
- Version metadata with quality scores
- Version comparison and rollback

Cache Structure:
    {project_folder}/.workflow_cache/
        ├── data_analyst.json           # Latest version
        ├── data_analyst_v1.json        # Version 1
        ├── data_analyst_v2.json        # Version 2 (revision)
        ├── research_explorer.json
        └── ...

Each cache file contains:
    - stage_name: Name of the workflow stage
    - timestamp: When the result was generated
    - project_id: Project identifier
    - agent_result: Full AgentResult as dict
    - input_hash: Hash of inputs (to detect if inputs changed)
    - version: Version number (0 for initial, 1+ for revisions)

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from filelock import FileLock
from loguru import logger


@dataclass
class CacheEntry:
    """A single cached stage result with version tracking."""
    stage_name: str
    timestamp: str
    project_id: str
    agent_result: dict
    input_hash: str
    version: int = 0                     # 0 = initial, 1+ = revisions
    quality_score: Optional[float] = None  # Overall quality if assessed
    feedback_summary: Optional[str] = None  # What prompted this version
    
    def to_dict(self) -> dict:
        return {
            "stage_name": self.stage_name,
            "timestamp": self.timestamp,
            "project_id": self.project_id,
            "agent_result": self.agent_result,
            "input_hash": self.input_hash,
            "version": self.version,
            "quality_score": self.quality_score,
            "feedback_summary": self.feedback_summary,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(
            stage_name=data["stage_name"],
            timestamp=data["timestamp"],
            project_id=data["project_id"],
            agent_result=data["agent_result"],
            input_hash=data["input_hash"],
            version=data.get("version", 0),
            quality_score=data.get("quality_score"),
            feedback_summary=data.get("feedback_summary"),
        )


@dataclass
class VersionInfo:
    """Summary information about a cached version."""
    version: int
    timestamp: str
    success: bool
    quality_score: Optional[float]
    feedback_summary: Optional[str]
    file_path: str


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
                        "version": data.get("version", 0),
                        "quality_score": data.get("quality_score"),
                    }
                except (json.JSONDecodeError, KeyError):
                    status[stage] = {"cached": False, "error": "invalid cache"}
            else:
                status[stage] = {"cached": False}
        return status
    
    # ========== Version Management Methods ==========
    
    def _get_version_path(self, stage_name: str, version: int) -> Path:
        """Get path to a specific version file."""
        return self.cache_dir / f"{stage_name}_v{version}.json"
    
    def save_version(
        self,
        stage_name: str,
        agent_result: dict,
        context: dict,
        project_id: str = "unknown",
        version: int = 0,
        quality_score: Optional[float] = None,
        feedback_summary: Optional[str] = None,
    ):
        """
        Save a specific version of agent result.
        
        This saves both to the main cache file (latest) and to a
        versioned file for history tracking.
        
        Args:
            stage_name: Name of the workflow stage
            agent_result: AgentResult.to_dict() output
            context: Current workflow context
            project_id: Project identifier
            version: Version number (0 = initial, 1+ = revisions)
            quality_score: Overall quality score if assessed
            feedback_summary: What feedback prompted this version
        """
        # Save to version-specific file
        version_path = self._get_version_path(stage_name, version)
        lock_path = version_path.with_suffix('.json.lock')
        
        entry = CacheEntry(
            stage_name=stage_name,
            timestamp=datetime.now().isoformat(),
            project_id=project_id,
            agent_result=agent_result,
            input_hash=self._compute_input_hash(context, stage_name),
            version=version,
            quality_score=quality_score,
            feedback_summary=feedback_summary,
        )
        
        try:
            with FileLock(lock_path, timeout=30):
                with open(version_path, "w") as f:
                    json.dump(entry.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Saved version {version} for {stage_name}")
            
            # Also save as latest (main cache file)
            self.save(stage_name, agent_result, context, project_id)
            
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save version {version} for {stage_name}: {e}")
    
    def get_version(self, stage_name: str, version: int) -> Optional[dict]:
        """
        Load a specific version of cached result.
        
        Args:
            stage_name: Name of the workflow stage
            version: Version number to load
            
        Returns:
            Agent result dict if version exists, None otherwise
        """
        version_path = self._get_version_path(stage_name, version)
        
        try:
            with open(version_path) as f:
                data = json.load(f)
            
            entry = CacheEntry.from_dict(data)
            logger.info(f"Loaded version {version} for {stage_name}")
            return entry.agent_result
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Version {version} not found for {stage_name}: {e}")
            return None
    
    def get_all_versions(self, stage_name: str) -> List[VersionInfo]:
        """
        Get information about all versions of a stage.
        
        Returns:
            List of VersionInfo sorted by version number
        """
        versions = []
        
        # Check for version files (v0, v1, v2, etc.)
        for version in range(100):  # Reasonable upper limit
            version_path = self._get_version_path(stage_name, version)
            if version_path.exists():
                try:
                    with open(version_path) as f:
                        data = json.load(f)
                    
                    versions.append(VersionInfo(
                        version=data.get("version", version),
                        timestamp=data.get("timestamp", ""),
                        success=data.get("agent_result", {}).get("success", False),
                        quality_score=data.get("quality_score"),
                        feedback_summary=data.get("feedback_summary"),
                        file_path=str(version_path),
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue
            else:
                # Stop if we hit a gap in versions
                if version > 0:
                    break
        
        return sorted(versions, key=lambda v: v.version)
    
    def get_latest_version(self, stage_name: str) -> Optional[int]:
        """
        Get the latest version number for a stage.
        
        Returns:
            Latest version number, or None if no versions exist
        """
        versions = self.get_all_versions(stage_name)
        if versions:
            return versions[-1].version
        return None
    
    def get_best_version(self, stage_name: str) -> Optional[Tuple[int, dict]]:
        """
        Get the version with the highest quality score.
        
        Returns:
            Tuple of (version_number, agent_result) or None if no versions
        """
        versions = self.get_all_versions(stage_name)
        if not versions:
            return None
        
        # Filter to versions with quality scores
        scored = [v for v in versions if v.quality_score is not None]
        
        if scored:
            # Return highest scored version
            best = max(scored, key=lambda v: v.quality_score)
            result = self.get_version(stage_name, best.version)
            if result:
                return best.version, result
        
        # Fall back to latest version
        latest = versions[-1]
        result = self.get_version(stage_name, latest.version)
        if result:
            return latest.version, result
        
        return None
    
    def clear_versions(self, stage_name: str, keep_latest: bool = True):
        """
        Clear version history for a stage.
        
        Args:
            stage_name: Name of the workflow stage
            keep_latest: If True, keeps the latest version
        """
        versions = self.get_all_versions(stage_name)
        
        for v in versions:
            if keep_latest and v == versions[-1]:
                continue
            
            version_path = Path(v.file_path)
            if version_path.exists():
                version_path.unlink()
                logger.debug(f"Deleted version {v.version} for {stage_name}")
        
        logger.info(f"Cleared version history for {stage_name}")
    
    def get_version_diff_summary(
        self,
        stage_name: str,
        version_a: int,
        version_b: int,
    ) -> Optional[dict]:
        """
        Get a summary of differences between two versions.
        
        Args:
            stage_name: Name of the workflow stage
            version_a: First version number
            version_b: Second version number
            
        Returns:
            Dict with difference summary, or None if versions not found
        """
        result_a = self.get_version(stage_name, version_a)
        result_b = self.get_version(stage_name, version_b)
        
        if not result_a or not result_b:
            return None
        
        content_a = result_a.get("content", "")
        content_b = result_b.get("content", "")
        
        # Simple length comparison
        len_a = len(content_a)
        len_b = len(content_b)
        
        # Quality score comparison
        versions = {v.version: v for v in self.get_all_versions(stage_name)}
        score_a = versions.get(version_a, VersionInfo(0, "", False, None, None, "")).quality_score
        score_b = versions.get(version_b, VersionInfo(0, "", False, None, None, "")).quality_score
        
        return {
            "version_a": version_a,
            "version_b": version_b,
            "length_change": len_b - len_a,
            "length_change_percent": ((len_b - len_a) / max(len_a, 1)) * 100,
            "quality_score_a": score_a,
            "quality_score_b": score_b,
            "quality_improvement": (score_b - score_a) if (score_a and score_b) else None,
        }
