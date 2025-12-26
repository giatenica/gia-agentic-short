"""
Research Workflow Orchestrator
==============================
Chains agents together to analyze research projects and generate
comprehensive overviews ready for literature review.

Workflow:
1. DataAnalystAgent (Haiku) - Examine uploaded data files
2. ResearchExplorerAgent (Sonnet) - Analyze what user provided
3. GapAnalysisAgent (Opus) - Identify missing elements
4. OverviewGeneratorAgent (Sonnet) - Create comprehensive overview

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from src.llm.claude_client import ClaudeClient
from src.agents.data_analyst import DataAnalystAgent
from src.agents.research_explorer import ResearchExplorerAgent
from src.agents.gap_analyst import GapAnalysisAgent
from src.agents.overview_generator import OverviewGeneratorAgent
from src.agents.base import AgentResult
from src.agents.cache import WorkflowCache
from src.agents.consistency_checker import ConsistencyCheckerAgent
from src.agents.readiness_assessor import ReadinessAssessorAgent
from src.utils.validation import validate_project_folder
from src.utils.workflow_issue_tracking import write_workflow_issue_tracking
from src.tracing import init_tracing, get_tracer
from loguru import logger


def _json_default(value):
    """Best-effort JSON serializer for nested workflow results.

    Some agent structured outputs can contain non-JSON-serializable objects
    (for example, pandas.Timestamp). We convert these to stable primitives.
    """
    if value is None:
        return None
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return sorted(list(value))

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    return str(value)


@dataclass
class WorkflowResult:
    """Result from the complete workflow execution."""
    success: bool
    project_id: str
    project_folder: str
    data_analysis: Optional[AgentResult] = None
    research_analysis: Optional[AgentResult] = None
    gap_analysis: Optional[AgentResult] = None
    overview: Optional[AgentResult] = None
    overview_path: Optional[str] = None
    consistency_check: Optional[AgentResult] = None  # Cross-document consistency validation
    readiness_assessment: Optional[AgentResult] = None  # Time tracking and readiness scoring
    total_tokens: int = 0
    total_time: float = 0.0
    errors: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "project_id": self.project_id,
            "project_folder": self.project_folder,
            "overview_path": self.overview_path,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "errors": self.errors,
            "agents": {
                "data_analysis": self.data_analysis.to_dict() if self.data_analysis else None,
                "research_analysis": self.research_analysis.to_dict() if self.research_analysis else None,
                "gap_analysis": self.gap_analysis.to_dict() if self.gap_analysis else None,
                "overview": self.overview.to_dict() if self.overview else None,
                "consistency_check": self.consistency_check.to_dict() if self.consistency_check else None,
                "readiness_assessment": self.readiness_assessment.to_dict() if self.readiness_assessment else None,
            }
        }


class ResearchWorkflow:
    """
    Orchestrates the multi-agent workflow for research project analysis.
    
    Sequence:
    1. Load project data from project.json
    2. Run DataAnalystAgent on uploaded data files
    3. Run ResearchExplorerAgent on project submission
    4. Run GapAnalysisAgent with combined context
    5. Run OverviewGeneratorAgent to create final overview
    6. Save results to project folder
    """
    
    def __init__(self, client: Optional[ClaudeClient] = None, use_cache: bool = True, cache_max_age_hours: int = 24):
        """
        Initialize workflow with shared Claude client.
        
        Args:
            client: Optional shared ClaudeClient (creates new if not provided)
            use_cache: Whether to use stage caching (default: True)
            cache_max_age_hours: Maximum age of cache entries in hours (default: 24)
        """
        self.client = client or ClaudeClient()
        self.use_cache = use_cache
        self.cache_max_age_hours = cache_max_age_hours
        
        # Initialize tracing
        init_tracing()
        self.tracer = get_tracer("research-workflow")
        
        # Initialize agents with shared client
        self.data_analyst = DataAnalystAgent(client=self.client)
        self.research_explorer = ResearchExplorerAgent(client=self.client)
        self.gap_analyst = GapAnalysisAgent(client=self.client)
        self.overview_generator = OverviewGeneratorAgent(client=self.client)
        self.consistency_checker = ConsistencyCheckerAgent(client=self.client)
        self.readiness_assessor = ReadinessAssessorAgent(client=self.client)
        
        logger.info(f"Research workflow initialized with 6 agents (cache={'enabled' if use_cache else 'disabled'})")
    
    async def run(self, project_folder: str) -> WorkflowResult:
        """
        Execute the complete research analysis workflow.
        
        Args:
            project_folder: Path to the project folder containing project.json
            
        Returns:
            WorkflowResult with all agent results and generated overview
        """
        import time
        start_time = time.time()
        
        with self.tracer.start_as_current_span("research_workflow") as workflow_span:
            workflow_span.set_attribute("project_folder", project_folder)

            try:
                project_path = validate_project_folder(project_folder)
            except Exception as e:
                workflow_span.set_attribute("error", str(e))
                return WorkflowResult(
                    success=False,
                    project_id="unknown",
                    project_folder=project_folder,
                    errors=[str(e)],
                )

            project_json_path = project_path / "project.json"
            try:
                with open(project_json_path, "r", encoding="utf-8") as f:
                    project_data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                workflow_span.set_attribute("error", f"Failed to read project.json: {e}")
                return WorkflowResult(
                    success=False,
                    project_id="unknown",
                    project_folder=project_folder,
                    errors=[f"Failed to read project.json: {e}"],
                )
            
            project_id = project_data.get("id", "unknown")
            workflow_span.set_attribute("project_id", project_id)
            logger.info(f"Starting workflow for project {project_id}")
            
            # Initialize cache if enabled
            cache = None
            if self.use_cache:
                cache = WorkflowCache(project_folder, max_age_hours=self.cache_max_age_hours)
                cache_status = cache.get_status()
                cached_stages = [s for s, st in cache_status.items() if st.get("cached") and st.get("success")]
                if cached_stages:
                    logger.info(f"Cache status: {len(cached_stages)} stages cached: {cached_stages}")
            
            result = WorkflowResult(
                success=True,
                project_id=project_id,
                project_folder=project_folder,
            )
            
            # Build context that gets enriched by each agent
            context = {
                "project_folder": project_folder,
                "project_data": project_data,
            }
            
            # Step 1: Data Analysis
            logger.info("Step 1/4: Running Data Analyst...")
            with self.tracer.start_as_current_span("data_analyst") as span:
                span.set_attribute("agent", "DataAnalyst")
                span.set_attribute("model_tier", "haiku")
                try:
                    # Check cache first
                    if cache and cache.has_valid_cache("data_analyst", context):
                        cached_data = cache.load("data_analyst")
                        data_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 1/4: Using cached Data Analyst result")
                    else:
                        data_result = await self.data_analyst.execute(context)
                        if cache and data_result.success:
                            cache.save("data_analyst", data_result.to_dict(), context, project_id)
                        span.set_attribute("cached", False)
                    
                    result.data_analysis = data_result
                    result.total_tokens += data_result.tokens_used
                    context["data_analysis"] = data_result.to_dict()
                    span.set_attribute("tokens_used", data_result.tokens_used)
                    span.set_attribute("success", data_result.success)
                    
                    if not data_result.success:
                        result.errors.append(f"Data analysis failed: {data_result.error}")
                        span.set_attribute("error", data_result.error)
                except Exception as e:
                    logger.error(f"Data analysis error: {e}")
                    result.errors.append(f"Data analysis error: {str(e)}")
                    span.set_attribute("error", str(e))
            
            # Step 2: Research Exploration
            logger.info("Step 2/4: Running Research Explorer...")
            with self.tracer.start_as_current_span("research_explorer") as span:
                span.set_attribute("agent", "ResearchExplorer")
                span.set_attribute("model_tier", "sonnet")
                try:
                    # Check cache first
                    if cache and cache.has_valid_cache("research_explorer", context):
                        cached_data = cache.load("research_explorer")
                        research_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 2/4: Using cached Research Explorer result")
                    else:
                        research_result = await self.research_explorer.execute(context)
                        if cache and research_result.success:
                            cache.save("research_explorer", research_result.to_dict(), context, project_id)
                        span.set_attribute("cached", False)
                    
                    result.research_analysis = research_result
                    result.total_tokens += research_result.tokens_used
                    context["research_analysis"] = research_result.to_dict()
                    span.set_attribute("tokens_used", research_result.tokens_used)
                    span.set_attribute("success", research_result.success)
                    
                    if not research_result.success:
                        result.errors.append(f"Research analysis failed: {research_result.error}")
                        span.set_attribute("error", research_result.error)
                except Exception as e:
                    logger.error(f"Research exploration error: {e}")
                    result.errors.append(f"Research exploration error: {str(e)}")
                    span.set_attribute("error", str(e))
            
            # Step 3: Gap Analysis
            logger.info("Step 3/4: Running Gap Analyst...")
            with self.tracer.start_as_current_span("gap_analyst") as span:
                span.set_attribute("agent", "GapAnalyst")
                span.set_attribute("model_tier", "opus")
                try:
                    # Check cache first
                    if cache and cache.has_valid_cache("gap_analyst", context):
                        cached_data = cache.load("gap_analyst")
                        gap_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 3/4: Using cached Gap Analyst result")
                    else:
                        gap_result = await self.gap_analyst.execute(context)
                        if cache and gap_result.success:
                            cache.save("gap_analyst", gap_result.to_dict(), context, project_id)
                        span.set_attribute("cached", False)
                    
                    result.gap_analysis = gap_result
                    result.total_tokens += gap_result.tokens_used
                    context["gap_analysis"] = gap_result.to_dict()
                    span.set_attribute("tokens_used", gap_result.tokens_used)
                    span.set_attribute("success", gap_result.success)
                    
                    if not gap_result.success:
                        result.errors.append(f"Gap analysis failed: {gap_result.error}")
                        span.set_attribute("error", gap_result.error)
                except Exception as e:
                    logger.error(f"Gap analysis error: {e}")
                    result.errors.append(f"Gap analysis error: {str(e)}")
                    span.set_attribute("error", str(e))
            
            # Step 4: Overview Generation
            logger.info("Step 4/4: Running Overview Generator...")
            with self.tracer.start_as_current_span("overview_generator") as span:
                span.set_attribute("agent", "OverviewGenerator")
                span.set_attribute("model_tier", "sonnet")
                try:
                    # Check cache first
                    if cache and cache.has_valid_cache("overview_generator", context):
                        cached_data = cache.load("overview_generator")
                        overview_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 4/4: Using cached Overview Generator result")
                    else:
                        overview_result = await self.overview_generator.execute(context)
                        if cache and overview_result.success:
                            cache.save("overview_generator", overview_result.to_dict(), context, project_id)
                        span.set_attribute("cached", False)
                    
                    result.overview = overview_result
                    result.total_tokens += overview_result.tokens_used
                    span.set_attribute("tokens_used", overview_result.tokens_used)
                    span.set_attribute("success", overview_result.success)
                    
                    if overview_result.success:
                        # Save overview to project folder
                        overview_path = self._save_overview(project_path, overview_result)
                        result.overview_path = str(overview_path)
                        span.set_attribute("overview_path", str(overview_path))
                        logger.info(f"Overview saved to: {overview_path}")
                    else:
                        result.errors.append(f"Overview generation failed: {overview_result.error}")
                        span.set_attribute("error", overview_result.error)
                except Exception as e:
                    logger.error(f"Overview generation error: {e}")
                    result.errors.append(f"Overview generation error: {str(e)}")
                    span.set_attribute("error", str(e))
            
            # Step 5: Cross-Document Consistency Check (non-blocking)
            logger.info("Step 5/6: Running Consistency Check...")
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
            
            # Step 6: Readiness Assessment (time tracking and readiness scoring)
            logger.info("Step 6/6: Running Readiness Assessment...")
            with self.tracer.start_as_current_span("readiness_assessment") as span:
                span.set_attribute("agent", "ReadinessAssessor")
                span.set_attribute("model_tier", "haiku")
                try:
                    # Pass current workflow results for time tracking
                    assessment_result = await self.readiness_assessor.assess_project(
                        project_folder,
                        workflow_results=result.to_dict(),
                    )
                    result.readiness_assessment = assessment_result
                    result.total_tokens += assessment_result.tokens_used
                    span.set_attribute("tokens_used", assessment_result.tokens_used)
                    span.set_attribute("success", assessment_result.success)
                    
                    if assessment_result.structured_data:
                        readiness = (assessment_result.structured_data.get("readiness") or {})
                        completion = readiness.get("overall_completion", 0)
                        automation_coverage = assessment_result.structured_data.get("automation_coverage", 0)
                        span.set_attribute("overall_completion", completion)
                        span.set_attribute("automation_coverage", automation_coverage)
                        logger.info(f"Readiness: {completion:.1f}% complete, {automation_coverage:.1f}% automated")
                except Exception as e:
                    logger.error(f"Readiness assessment error: {e}")
                    span.set_attribute("error", str(e))
                    # Don't add to errors - assessment is non-blocking
            
            # Save workflow results
            self._save_workflow_results(project_path, result)

            # Persist non-fatal issues for later fixes (non-blocking)
            try:
                write_workflow_issue_tracking(project_folder, result.to_dict())
            except Exception as e:
                logger.warning(f"Failed to write workflow issue tracking: {e}")
            
            result.total_time = time.time() - start_time
            result.success = len(result.errors) == 0
            
            workflow_span.set_attribute("total_tokens", result.total_tokens)
            workflow_span.set_attribute("total_time", result.total_time)
            workflow_span.set_attribute("success", result.success)
            workflow_span.set_attribute("error_count", len(result.errors))
            
            logger.info(
                f"Workflow completed in {result.total_time:.2f}s, "
                f"{result.total_tokens} tokens, "
                f"{len(result.errors)} errors"
            )
            
            return result
    
    def _result_from_cache(self, cached_data: dict) -> AgentResult:
        """Reconstruct AgentResult from cached dictionary.
        
        Note: cache.load() returns the agent_result dict directly (not wrapped),
        so cached_data IS the agent_result, not {agent_result: {...}}.
        """
        from src.llm.claude_client import TaskType, ModelTier
        
        # cached_data IS the agent_result (cache.load returns entry.agent_result directly)
        agent_result = cached_data
        
        # Parse task_type and model_tier from cached values
        task_type_str = agent_result.get("task_type", "data_analysis")
        model_tier_str = agent_result.get("model_tier", "sonnet")
        
        # Convert strings to enums with fallbacks
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.DATA_ANALYSIS
        
        try:
            model_tier = ModelTier(model_tier_str)
        except ValueError:
            model_tier = ModelTier.SONNET
        
        return AgentResult(
            agent_name=agent_result.get("agent_name", "unknown"),
            task_type=task_type,
            model_tier=model_tier,
            success=agent_result.get("success", False),
            content=agent_result.get("content", ""),
            structured_data=agent_result.get("structured_data", {}),
            error=agent_result.get("error"),
            tokens_used=agent_result.get("tokens_used", 0),
            execution_time=agent_result.get("execution_time", 0.0),
            timestamp=agent_result.get("timestamp", ""),
        )
    
    def _save_overview(self, project_path: Path, overview_result: AgentResult) -> Path:
        """Save the generated overview to the project folder."""
        overview_path = project_path / "RESEARCH_OVERVIEW.md"
        
        # Generate markdown with metadata
        markdown = self.overview_generator.generate_markdown_overview(overview_result)
        
        with open(overview_path, "w") as f:
            f.write(markdown)
        
        return overview_path
    
    def _save_workflow_results(self, project_path: Path, result: WorkflowResult):
        """Save complete workflow results as JSON."""
        results_path = project_path / "workflow_results.json"
        
        with open(results_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=_json_default)
        
        logger.debug(f"Workflow results saved to: {results_path}")


async def run_workflow_for_project(project_folder: str) -> WorkflowResult:
    """
    Convenience function to run the workflow for a project folder.
    
    Args:
        project_folder: Path to the project folder
        
    Returns:
        WorkflowResult with all findings
    """
    workflow = ResearchWorkflow()
    return await workflow.run(project_folder)


def run_workflow_sync(project_folder: str) -> WorkflowResult:
    """
    Synchronous wrapper for running the workflow.
    
    Useful for calling from non-async code like the intake server.
    """
    return asyncio.run(run_workflow_for_project(project_folder))

