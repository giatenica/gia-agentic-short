"""
Extended Research Workflow with Gap Resolution
===============================================
Extends the base workflow with gap resolution and overview updating capabilities.

This workflow is designed to be run after the initial overview is generated,
taking the RESEARCH_OVERVIEW.md and resolving actionable gaps through
code execution, then producing an updated overview.

Workflow:
1. Load existing RESEARCH_OVERVIEW.md
2. GapResolverAgent - Identify and resolve data-related gaps via code
3. OverviewUpdaterAgent - Create UPDATED_RESEARCH_OVERVIEW.md

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from src.llm.claude_client import ClaudeClient
from src.agents.gap_resolver import GapResolverAgent, OverviewUpdaterAgent
from src.agents.base import AgentResult
from src.agents.cache import WorkflowCache
from src.agents.consistency_checker import ConsistencyCheckerAgent
from src.agents.readiness_assessor import ReadinessAssessorAgent
from src.utils.validation import validate_project_folder
from src.utils.workflow_issue_tracking import write_workflow_issue_tracking
from src.tracing import init_tracing, get_tracer
from src.config import GAP_RESOLUTION
from loguru import logger


@dataclass
class GapResolutionWorkflowResult:
    """Result from the gap resolution workflow execution."""
    success: bool
    project_id: str
    project_folder: str
    original_overview_path: str
    gap_resolution: Optional[AgentResult] = None
    updated_overview: Optional[AgentResult] = None
    updated_overview_path: Optional[str] = None
    consistency_check: Optional[AgentResult] = None  # Cross-document consistency validation
    readiness_assessment: Optional[AgentResult] = None  # Project readiness assessment
    gaps_resolved: int = 0
    gaps_total: int = 0
    iterations_run: int = 1  # Number of iterations executed
    total_tokens: int = 0
    total_time: float = 0.0
    errors: list = field(default_factory=list)
    code_executions: list = field(default_factory=list)
    lenient_success: bool = False  # True if success due to lenient mode
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "project_id": self.project_id,
            "project_folder": self.project_folder,
            "original_overview_path": self.original_overview_path,
            "updated_overview_path": self.updated_overview_path,
            "gaps_resolved": self.gaps_resolved,
            "gaps_total": self.gaps_total,
            "iterations_run": self.iterations_run,
            "lenient_success": self.lenient_success,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "errors": self.errors,
            "code_executions": self.code_executions,
            "agents": {
                "gap_resolution": self.gap_resolution.to_dict() if self.gap_resolution else None,
                "updated_overview": self.updated_overview.to_dict() if self.updated_overview else None,
                "consistency_check": self.consistency_check.to_dict() if self.consistency_check else None,
                "readiness_assessment": self.readiness_assessment.to_dict() if self.readiness_assessment else None,
            }
        }


class GapResolutionWorkflow:
    """
    Workflow that resolves research gaps and updates the overview.
    
    This is designed to be run after the initial ResearchWorkflow completes,
    taking the generated RESEARCH_OVERVIEW.md as input.
    
    Sequence:
    1. Load RESEARCH_OVERVIEW.md and project.json
    2. Run GapResolverAgent to identify and resolve data gaps
    3. Run OverviewUpdaterAgent to create updated overview
    4. Save UPDATED_RESEARCH_OVERVIEW.md
    """
    
    def __init__(
        self,
        client: Optional[ClaudeClient] = None,
        use_cache: bool = True,
        cache_max_age_hours: int = 24,
        code_execution_timeout: int = None,
        max_iterations: int = None,
        lenient_mode: bool = None,
        min_resolved_ratio: float = None,
    ):
        """
        Initialize gap resolution workflow.
        
        Args:
            client: Optional shared ClaudeClient
            use_cache: Whether to use stage caching
            cache_max_age_hours: Maximum cache age
            code_execution_timeout: Timeout for code execution in seconds (default from config)
            max_iterations: Maximum workflow iterations for retrying unresolved gaps (default from config)
            lenient_mode: If True, workflow succeeds if min_resolved_ratio is met (default from config)
            min_resolved_ratio: Minimum ratio of gaps that must be resolved for lenient success (default from config)
        """
        self.client = client or ClaudeClient()
        self.use_cache = use_cache
        self.cache_max_age_hours = cache_max_age_hours
        
        # Load from config with optional overrides
        self.code_execution_timeout = code_execution_timeout if code_execution_timeout is not None else GAP_RESOLUTION.EXECUTION_TIMEOUT
        self.max_iterations = max_iterations if max_iterations is not None else GAP_RESOLUTION.MAX_ITERATIONS
        self.lenient_mode = lenient_mode if lenient_mode is not None else GAP_RESOLUTION.LENIENT_MODE
        self.min_resolved_ratio = min_resolved_ratio if min_resolved_ratio is not None else GAP_RESOLUTION.MIN_RESOLVED_RATIO
        
        # Initialize tracing
        init_tracing()
        self.tracer = get_tracer("gap-resolution-workflow")
        
        # Initialize agents
        self.gap_resolver = GapResolverAgent(
            client=self.client,
            execution_timeout=self.code_execution_timeout,
        )
        self.overview_updater = OverviewUpdaterAgent(client=self.client)
        self.consistency_checker = ConsistencyCheckerAgent(client=self.client)
        self.readiness_assessor = ReadinessAssessorAgent(client=self.client)
        
        logger.info(
            f"Gap resolution workflow initialized with 4 agents "
            f"(cache={'enabled' if use_cache else 'disabled'}, "
            f"max_iterations={self.max_iterations}, lenient_mode={self.lenient_mode})"
        )
    
    async def run(self, project_folder: str) -> GapResolutionWorkflowResult:
        """
        Execute the gap resolution workflow.
        
        Args:
            project_folder: Path to project folder with RESEARCH_OVERVIEW.md
            
        Returns:
            GapResolutionWorkflowResult with all findings
        """
        import time
        start_time = time.time()
        
        with self.tracer.start_as_current_span("gap_resolution_workflow") as workflow_span:
            workflow_span.set_attribute("project_folder", project_folder)
            
            try:
                project_path = validate_project_folder(project_folder)
            except Exception as e:
                workflow_span.set_attribute("error", str(e))
                return GapResolutionWorkflowResult(
                    success=False,
                    project_id="unknown",
                    project_folder=project_folder,
                    original_overview_path=str(Path(project_folder) / "RESEARCH_OVERVIEW.md"),
                    errors=[str(e)],
                )

            overview_path = project_path / "RESEARCH_OVERVIEW.md"
            project_json_path = project_path / "project.json"
            
            # Validate inputs exist
            if not overview_path.exists():
                return GapResolutionWorkflowResult(
                    success=False,
                    project_id="unknown",
                    project_folder=project_folder,
                    original_overview_path=str(overview_path),
                    errors=["RESEARCH_OVERVIEW.md not found. Run initial workflow first."],
                )
            
            # Load data
            try:
                research_overview = overview_path.read_text(encoding="utf-8")
            except OSError as e:
                workflow_span.set_attribute("error", f"Failed to read RESEARCH_OVERVIEW.md: {e}")
                return GapResolutionWorkflowResult(
                    success=False,
                    project_id="unknown",
                    project_folder=project_folder,
                    original_overview_path=str(overview_path),
                    errors=[f"Failed to read RESEARCH_OVERVIEW.md: {e}"],
                )

            try:
                with open(project_json_path, "r", encoding="utf-8") as f:
                    project_data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                workflow_span.set_attribute("error", f"Failed to read project.json: {e}")
                return GapResolutionWorkflowResult(
                    success=False,
                    project_id="unknown",
                    project_folder=project_folder,
                    original_overview_path=str(overview_path),
                    errors=[f"Failed to read project.json: {e}"],
                )
            
            project_id = project_data.get("id", "unknown")
            workflow_span.set_attribute("project_id", project_id)
            logger.info(f"Starting gap resolution workflow for project {project_id}")
            
            # Initialize cache
            cache = None
            if self.use_cache:
                cache = WorkflowCache(project_folder, max_age_hours=self.cache_max_age_hours)
            
            result = GapResolutionWorkflowResult(
                success=True,
                project_id=project_id,
                project_folder=project_folder,
                original_overview_path=str(overview_path),
            )
            
            # Build context
            context = {
                "project_folder": project_folder,
                "project_data": project_data,
                "research_overview": research_overview,
            }
            
            # Step 1: Gap Resolution with iterations
            unresolved_gap_ids = set()  # Track gaps that failed to resolve
            all_code_executions = []
            
            for iteration in range(1, self.max_iterations + 1):
                result.iterations_run = iteration
                iter_label = f"[Iter {iteration}/{self.max_iterations}]"
                logger.info(f"{iter_label} Step 1: Running Gap Resolver...")
                
                with self.tracer.start_as_current_span(f"gap_resolver_iter_{iteration}") as span:
                    span.set_attribute("agent", "GapResolver")
                    span.set_attribute("model_tier", "sonnet")
                    span.set_attribute("iteration", iteration)
                    
                    try:
                        # For subsequent iterations, skip cache and focus on unresolved gaps
                        use_cache_this_iter = cache and iteration == 1
                        
                        if use_cache_this_iter and cache.has_valid_cache("gap_resolver", context):
                            cached_data = cache.load("gap_resolver")
                            gap_result = self._result_from_cache(cached_data)
                            span.set_attribute("cached", True)
                            logger.info(f"{iter_label} Using cached Gap Resolver result")
                        else:
                            # Add unresolved gaps to context for retry iterations
                            if iteration > 1 and unresolved_gap_ids:
                                context["retry_gap_ids"] = list(unresolved_gap_ids)
                                context["iteration"] = iteration
                                logger.info(f"{iter_label} Retrying {len(unresolved_gap_ids)} unresolved gaps")
                            
                            gap_result = await self.gap_resolver.execute(context)
                            
                            # Only cache first iteration results
                            if cache and gap_result.success and iteration == 1:
                                cache.save("gap_resolver", gap_result.to_dict(), context, project_id)
                            span.set_attribute("cached", False)
                        
                        result.gap_resolution = gap_result
                        result.total_tokens += gap_result.tokens_used
                        
                        # Extract gap statistics
                        structured = gap_result.structured_data or {}
                        result.gaps_resolved = structured.get("resolved_count", 0)
                        result.gaps_total = structured.get("total_gaps", 0)
                        
                        # Track code executions across iterations
                        iter_executions = [
                            {
                                "gap_id": r.get("gap_id"),
                                "success": r.get("execution_result", {}).get("success", False),
                                "resolved": r.get("resolved", False),
                                "iteration": iteration,
                            }
                            for r in structured.get("resolutions", [])
                        ]
                        all_code_executions.extend(iter_executions)
                        result.code_executions = all_code_executions
                        
                        # Update unresolved gap tracking
                        unresolved_gap_ids = {
                            r.get("gap_id")
                            for r in structured.get("resolutions", [])
                            if not r.get("resolved", False)
                        }
                        
                        context["gap_resolutions"] = structured
                        span.set_attribute("tokens_used", gap_result.tokens_used)
                        span.set_attribute("success", gap_result.success)
                        span.set_attribute("gaps_resolved", result.gaps_resolved)
                        span.set_attribute("gaps_total", result.gaps_total)
                        
                        if not gap_result.success:
                            result.errors.append(f"Gap resolution failed (iter {iteration}): {gap_result.error}")
                            span.set_attribute("error", gap_result.error)
                            
                    except Exception as e:
                        logger.error(f"{iter_label} Gap resolution error: {e}")
                        result.errors.append(f"Gap resolution error (iter {iteration}): {str(e)}")
                        span.set_attribute("error", str(e))
                
                # Check if all gaps resolved; no need for more iterations
                if result.gaps_total > 0 and result.gaps_resolved >= result.gaps_total:
                    logger.info(f"{iter_label} All gaps resolved, skipping remaining iterations")
                    break
                
                # Check if we've hit minimum threshold in lenient mode
                if self.lenient_mode and result.gaps_total > 0:
                    resolved_ratio = result.gaps_resolved / result.gaps_total
                    if resolved_ratio >= self.min_resolved_ratio:
                        logger.info(
                            f"{iter_label} Met threshold ({resolved_ratio:.1%} >= {self.min_resolved_ratio:.1%}), "
                            f"continuing to try remaining gaps..."
                        )
                
                # Log progress between iterations
                if iteration < self.max_iterations and unresolved_gap_ids:
                    logger.info(
                        f"{iter_label} Completed with {result.gaps_resolved}/{result.gaps_total} resolved, "
                        f"{len(unresolved_gap_ids)} remaining"
                    )
            
            # Step 2: Overview Update (only if gaps were processed)
            if result.gaps_total > 0:
                logger.info("Step 2/2: Running Overview Updater...")
                with self.tracer.start_as_current_span("overview_updater") as span:
                    span.set_attribute("agent", "OverviewUpdater")
                    span.set_attribute("model_tier", "opus")
                    try:
                        # Check cache first
                        if cache and cache.has_valid_cache("overview_updater", context):
                            cached_data = cache.load("overview_updater")
                            update_result = self._result_from_cache(cached_data)
                            span.set_attribute("cached", True)
                            logger.info("Step 2/2: Using cached Overview Updater result")
                        else:
                            update_result = await self.overview_updater.execute(context)
                            if cache and update_result.success:
                                cache.save("overview_updater", update_result.to_dict(), context, project_id)
                            span.set_attribute("cached", False)
                        
                        result.updated_overview = update_result
                        result.total_tokens += update_result.tokens_used
                        span.set_attribute("tokens_used", update_result.tokens_used)
                        span.set_attribute("success", update_result.success)
                        
                        if update_result.success:
                            # Save updated overview
                            updated_path = self._save_updated_overview(project_path, update_result)
                            result.updated_overview_path = str(updated_path)
                            span.set_attribute("updated_overview_path", str(updated_path))
                            logger.info(f"Updated overview saved to: {updated_path}")
                        else:
                            result.errors.append(f"Overview update failed: {update_result.error}")
                            span.set_attribute("error", update_result.error)
                            
                    except Exception as e:
                        logger.error(f"Overview update error: {e}")
                        result.errors.append(f"Overview update error: {str(e)}")
                        span.set_attribute("error", str(e))
            else:
                logger.info("Step 2/4: Skipping Overview Updater (no gaps to process)")
            
            # Step 3: Cross-Document Consistency Check (non-blocking)
            logger.info("Step 3/4: Running Consistency Check...")
            with self.tracer.start_as_current_span("consistency_check") as span:
                span.set_attribute("agent", "ConsistencyChecker")
                span.set_attribute("model_tier", "sonnet")
                try:
                    consistency_result = await self.consistency_checker.check_consistency(project_folder)
                    result.consistency_check = consistency_result
                    result.total_tokens += consistency_result.tokens_used
                    span.set_attribute("tokens_used", consistency_result.tokens_used)
                    span.set_attribute("success", consistency_result.success)
                    
                    # Log consistency issues but don't fail workflow
                    if consistency_result.structured_data:
                        critical_count = consistency_result.structured_data.get("critical_count", 0)
                        high_count = consistency_result.structured_data.get("high_count", 0)
                        score = consistency_result.structured_data.get("score", 1.0)
                        span.set_attribute("critical_issues", critical_count)
                        span.set_attribute("high_issues", high_count)
                        span.set_attribute("consistency_score", score)
                        
                        if critical_count > 0:
                            logger.warning(f"Consistency check found {critical_count} critical issues")
                        elif high_count > 0:
                            logger.warning(f"Consistency check found {high_count} high-severity issues")
                        else:
                            logger.info(f"Consistency check passed (score: {score:.2f})")
                except Exception as e:
                    logger.error(f"Consistency check error: {e}")
                    span.set_attribute("error", str(e))
                    # Don't add to errors - consistency check is non-blocking
            
            # Step 4: Project Readiness Assessment (non-blocking)
            logger.info("Step 4/4: Running Readiness Assessment...")
            with self.tracer.start_as_current_span("readiness_assessment") as span:
                span.set_attribute("agent", "ReadinessAssessor")
                span.set_attribute("model_tier", "haiku")
                try:
                    assessment_result = await self.readiness_assessor.assess_project(project_folder)
                    result.readiness_assessment = assessment_result
                    result.total_tokens += assessment_result.tokens_used
                    span.set_attribute("tokens_used", assessment_result.tokens_used)
                    span.set_attribute("success", assessment_result.success)
                    
                    if assessment_result.structured_data:
                        overall_score = assessment_result.structured_data.get("overall_score", 0)
                        automation_gaps = assessment_result.structured_data.get("automation_gaps", [])
                        span.set_attribute("overall_score", overall_score)
                        span.set_attribute("automation_gap_count", len(automation_gaps))
                        
                        if automation_gaps:
                            logger.info(f"Readiness assessment: {overall_score:.0%} complete, {len(automation_gaps)} automation gaps")
                        else:
                            logger.info(f"Readiness assessment: {overall_score:.0%} complete, fully automated")
                except Exception as e:
                    logger.error(f"Readiness assessment error: {e}")
                    span.set_attribute("error", str(e))
                    # Don't add to errors - readiness assessment is non-blocking
            
            # Save workflow results
            self._save_workflow_results(project_path, result)

            # Persist non-fatal issues for later fixes (non-blocking)
            try:
                write_workflow_issue_tracking(
                    project_folder,
                    result.to_dict(),
                    filename="gap_resolution_issues.json",
                )
            except Exception as e:
                logger.warning(f"Failed to write gap resolution workflow issue tracking: {e}")
            
            result.total_time = time.time() - start_time
            
            # Determine success based on mode and resolution ratio
            if len(result.errors) == 0:
                # No errors; full success
                result.success = True
            elif self.lenient_mode and result.gaps_total > 0:
                # Lenient mode: succeed if we resolved at least min_resolved_ratio
                resolved_ratio = result.gaps_resolved / result.gaps_total
                if resolved_ratio >= self.min_resolved_ratio:
                    result.success = True
                    result.lenient_success = True
                    logger.info(
                        f"Lenient success: {result.gaps_resolved}/{result.gaps_total} gaps resolved "
                        f"({resolved_ratio:.1%} >= {self.min_resolved_ratio:.1%} threshold)"
                    )
                else:
                    result.success = False
                    logger.warning(
                        f"Below threshold: {result.gaps_resolved}/{result.gaps_total} gaps resolved "
                        f"({resolved_ratio:.1%} < {self.min_resolved_ratio:.1%} threshold)"
                    )
            else:
                result.success = False
            
            workflow_span.set_attribute("total_tokens", result.total_tokens)
            workflow_span.set_attribute("total_time", result.total_time)
            workflow_span.set_attribute("success", result.success)
            workflow_span.set_attribute("lenient_success", result.lenient_success)
            
            logger.info(
                f"Gap resolution workflow completed in {result.total_time:.2f}s, "
                f"{result.total_tokens} tokens, "
                f"{result.gaps_resolved}/{result.gaps_total} gaps resolved"
                f"{' (lenient)' if result.lenient_success else ''}"
            )
            
            return result
    
    def _result_from_cache(self, cached_data: dict) -> AgentResult:
        """Reconstruct AgentResult from cached dictionary."""
        from src.llm.claude_client import TaskType, ModelTier
        
        return AgentResult(
            agent_name=cached_data.get("agent_name", "unknown"),
            task_type=TaskType(cached_data.get("task_type", "coding")),
            model_tier=ModelTier(cached_data.get("model_tier", "sonnet")),
            success=cached_data.get("success", False),
            content=cached_data.get("content", ""),
            structured_data=cached_data.get("structured_data", {}),
            error=cached_data.get("error"),
            tokens_used=cached_data.get("tokens_used", 0),
            execution_time=cached_data.get("execution_time", 0.0),
            timestamp=cached_data.get("timestamp", ""),
        )
    
    def _save_updated_overview(self, project_path: Path, update_result: AgentResult) -> Path:
        """Save the updated overview to the project folder."""
        updated_path = project_path / "UPDATED_RESEARCH_OVERVIEW.md"
        
        with open(updated_path, "w") as f:
            f.write(update_result.content)
        
        return updated_path
    
    def _save_workflow_results(self, project_path: Path, result: GapResolutionWorkflowResult):
        """Save workflow results as JSON."""
        results_path = project_path / "gap_resolution_results.json"
        
        with open(results_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.debug(f"Gap resolution results saved to: {results_path}")


async def run_gap_resolution_workflow(project_folder: str) -> GapResolutionWorkflowResult:
    """
    Convenience function to run the gap resolution workflow.
    
    Args:
        project_folder: Path to the project folder
        
    Returns:
        GapResolutionWorkflowResult with all findings
    """
    workflow = GapResolutionWorkflow()
    return await workflow.run(project_folder)


def run_gap_resolution_sync(project_folder: str) -> GapResolutionWorkflowResult:
    """Synchronous wrapper for running the gap resolution workflow."""
    return asyncio.run(run_gap_resolution_workflow(project_folder))
