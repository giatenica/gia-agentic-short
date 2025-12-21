"""
Literature Workflow Orchestrator
================================
Chains the literature phase agents together to develop hypotheses,
search literature, and create paper structure.

Workflow:
1. HypothesisDevelopmentAgent (Opus) - Formulate testable hypothesis
2. LiteratureSearchAgent (Sonnet) - Search via Edison API
3. LiteratureSynthesisAgent (Sonnet) - Process and synthesize results
4. PaperStructureAgent (Sonnet) - Create LaTeX paper structure
5. ProjectPlannerAgent (Opus) - Create detailed project plan

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.llm.claude_client import ClaudeClient
from src.llm.edison_client import EdisonClient
from src.agents.hypothesis_developer import HypothesisDevelopmentAgent
from src.agents.literature_search import LiteratureSearchAgent
from src.agents.literature_synthesis import LiteratureSynthesisAgent
from src.agents.paper_structure import PaperStructureAgent
from src.agents.project_planner import ProjectPlannerAgent
from src.agents.base import AgentResult
from src.agents.cache import WorkflowCache
from src.tracing import init_tracing, get_tracer
from loguru import logger


@dataclass
class LiteratureWorkflowResult:
    """Result from the literature workflow execution."""
    success: bool
    project_id: str
    project_folder: str
    hypothesis_result: Optional[AgentResult] = None
    literature_search_result: Optional[AgentResult] = None
    literature_synthesis_result: Optional[AgentResult] = None
    paper_structure_result: Optional[AgentResult] = None
    project_plan_result: Optional[AgentResult] = None
    total_tokens: int = 0
    total_time: float = 0.0
    errors: list = field(default_factory=list)
    files_created: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "project_id": self.project_id,
            "project_folder": self.project_folder,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "errors": self.errors,
            "files_created": self.files_created,
            "agents": {
                "hypothesis": self.hypothesis_result.to_dict() if self.hypothesis_result else None,
                "literature_search": self.literature_search_result.to_dict() if self.literature_search_result else None,
                "literature_synthesis": self.literature_synthesis_result.to_dict() if self.literature_synthesis_result else None,
                "paper_structure": self.paper_structure_result.to_dict() if self.paper_structure_result else None,
                "project_plan": self.project_plan_result.to_dict() if self.project_plan_result else None,
            }
        }


class LiteratureWorkflow:
    """
    Orchestrates the literature phase workflow for research projects.
    
    Sequence:
    1. Load research overview and project data
    2. Run HypothesisDevelopmentAgent to formulate hypothesis
    3. Run LiteratureSearchAgent to search via Edison API
    4. Run LiteratureSynthesisAgent to process results
    5. Run PaperStructureAgent to create LaTeX structure
    6. Run ProjectPlannerAgent to create project plan
    7. Save all results to project folder
    
    Prerequisites:
    - Research overview (RESEARCH_OVERVIEW.md) must exist
    - Edison API key must be configured
    """
    
    def __init__(
        self,
        client: Optional[ClaudeClient] = None,
        edison_client: Optional[EdisonClient] = None,
        use_cache: bool = True,
        cache_max_age_hours: int = 24,
        edison_timeout: int = 1200,
    ):
        """
        Initialize workflow with clients.
        
        Args:
            client: Optional shared ClaudeClient (creates new if not provided)
            edison_client: Optional EdisonClient (creates new if not provided)
            use_cache: Whether to use stage caching (default: True)
            cache_max_age_hours: Maximum age of cache entries in hours (default: 24)
            edison_timeout: Timeout for Edison API in seconds (default: 20 minutes)
        """
        self.client = client or ClaudeClient()
        self.edison_client = edison_client or EdisonClient()
        self.use_cache = use_cache
        self.cache_max_age_hours = cache_max_age_hours
        self.edison_timeout = edison_timeout
        
        # Initialize tracing
        init_tracing()
        self.tracer = get_tracer("literature-workflow")
        
        # Initialize agents with shared clients
        self.hypothesis_developer = HypothesisDevelopmentAgent(client=self.client)
        self.literature_searcher = LiteratureSearchAgent(
            client=self.client,
            edison_client=self.edison_client,
            search_timeout=edison_timeout,
        )
        self.literature_synthesizer = LiteratureSynthesisAgent(client=self.client)
        self.paper_structurer = PaperStructureAgent(client=self.client)
        self.project_planner = ProjectPlannerAgent(client=self.client)
        
        logger.info(f"Literature workflow initialized with 5 agents (cache={'enabled' if use_cache else 'disabled'})")
    
    async def run(self, project_folder: str) -> LiteratureWorkflowResult:
        """
        Execute the complete literature phase workflow.
        
        Args:
            project_folder: Path to the project folder containing RESEARCH_OVERVIEW.md
            
        Returns:
            LiteratureWorkflowResult with all agent results
        """
        import time
        start_time = time.time()
        
        with self.tracer.start_as_current_span("literature_workflow") as workflow_span:
            workflow_span.set_attribute("project_folder", project_folder)
            
            project_path = Path(project_folder)
            
            # Load prerequisites
            project_json_path = project_path / "project.json"
            overview_path = project_path / "RESEARCH_OVERVIEW.md"
            
            # Load project data
            project_data = {}
            if project_json_path.exists():
                with open(project_json_path) as f:
                    project_data = json.load(f)
            
            project_id = project_data.get("id", "unknown")
            workflow_span.set_attribute("project_id", project_id)
            
            # Load research overview
            research_overview = ""
            if overview_path.exists():
                research_overview = overview_path.read_text()
            else:
                workflow_span.set_attribute("error", "RESEARCH_OVERVIEW.md not found")
                return LiteratureWorkflowResult(
                    success=False,
                    project_id=project_id,
                    project_folder=project_folder,
                    errors=["RESEARCH_OVERVIEW.md not found - run initial workflow first"],
                )
            
            logger.info(f"Starting literature workflow for project {project_id}")
            
            # Initialize cache if enabled
            cache = None
            if self.use_cache:
                cache = WorkflowCache(project_folder, max_age_hours=self.cache_max_age_hours)
            
            result = LiteratureWorkflowResult(
                success=True,
                project_id=project_id,
                project_folder=project_folder,
            )
            
            # Build context that gets enriched by each agent
            context = {
                "project_folder": project_folder,
                "project_data": project_data,
                "research_overview": research_overview,
            }
            
            # Step 1: Hypothesis Development
            logger.info("Step 1/5: Developing hypothesis...")
            with self.tracer.start_as_current_span("hypothesis_developer") as span:
                span.set_attribute("agent", "HypothesisDeveloper")
                span.set_attribute("model_tier", "opus")
                try:
                    if cache and cache.has_valid_cache("hypothesis_developer", context):
                        cached_data = cache.load("hypothesis_developer")
                        hyp_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 1/5: Using cached hypothesis result")
                    else:
                        hyp_result = await self.hypothesis_developer.execute(context)
                        if cache:
                            cache.save("hypothesis_developer", hyp_result.to_dict(), context, project_id)
                        span.set_attribute("cached", False)
                    
                    result.hypothesis_result = hyp_result
                    result.total_tokens += hyp_result.tokens_used
                    context["hypothesis_result"] = hyp_result.to_dict()
                    span.set_attribute("tokens_used", hyp_result.tokens_used)
                    span.set_attribute("success", hyp_result.success)
                    
                    if not hyp_result.success:
                        result.errors.append(f"Hypothesis development failed: {hyp_result.error}")
                        span.set_attribute("error", hyp_result.error)
                except Exception as e:
                    logger.error(f"Hypothesis development error: {e}")
                    result.errors.append(f"Hypothesis development error: {str(e)}")
                    span.set_attribute("error", str(e))
            
            # Step 2: Literature Search
            logger.info("Step 2/5: Searching literature via Edison API...")
            with self.tracer.start_as_current_span("literature_search") as span:
                span.set_attribute("agent", "LiteratureSearcher")
                span.set_attribute("model_tier", "sonnet")
                try:
                    if cache and cache.has_valid_cache("literature_search", context):
                        cached_data = cache.load("literature_search")
                        lit_search_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 2/5: Using cached literature search result")
                    else:
                        lit_search_result = await self.literature_searcher.execute(context)
                        if cache:
                            cache.save("literature_search", lit_search_result.to_dict(), context, project_id)
                        span.set_attribute("cached", False)
                    
                    result.literature_search_result = lit_search_result
                    result.total_tokens += lit_search_result.tokens_used
                    context["literature_result"] = lit_search_result.to_dict()
                    span.set_attribute("tokens_used", lit_search_result.tokens_used)
                    span.set_attribute("success", lit_search_result.success)
                    
                    if not lit_search_result.success:
                        result.errors.append(f"Literature search failed: {lit_search_result.error}")
                        span.set_attribute("error", lit_search_result.error)
                except Exception as e:
                    logger.error(f"Literature search error: {e}")
                    result.errors.append(f"Literature search error: {str(e)}")
                    span.set_attribute("error", str(e))
            
            # Step 3: Literature Synthesis
            logger.info("Step 3/5: Synthesizing literature...")
            with self.tracer.start_as_current_span("literature_synthesis") as span:
                span.set_attribute("agent", "LiteratureSynthesizer")
                span.set_attribute("model_tier", "sonnet")
                try:
                    if cache and cache.has_valid_cache("literature_synthesis", context):
                        cached_data = cache.load("literature_synthesis")
                        lit_synth_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 3/5: Using cached synthesis result")
                    else:
                        lit_synth_result = await self.literature_synthesizer.execute(context)
                        if cache:
                            cache.save("literature_synthesis", lit_synth_result.to_dict(), context, project_id)
                        span.set_attribute("cached", False)
                    
                    result.literature_synthesis_result = lit_synth_result
                    result.total_tokens += lit_synth_result.tokens_used
                    context["literature_synthesis"] = lit_synth_result.to_dict()
                    span.set_attribute("tokens_used", lit_synth_result.tokens_used)
                    span.set_attribute("success", lit_synth_result.success)
                    
                    # Track files created
                    if lit_synth_result.success and lit_synth_result.structured_data:
                        files = lit_synth_result.structured_data.get("files_saved", {})
                        result.files_created.update(files)
                    
                    if not lit_synth_result.success:
                        result.errors.append(f"Literature synthesis failed: {lit_synth_result.error}")
                        span.set_attribute("error", lit_synth_result.error)
                except Exception as e:
                    logger.error(f"Literature synthesis error: {e}")
                    result.errors.append(f"Literature synthesis error: {str(e)}")
                    span.set_attribute("error", str(e))
            
            # Step 4: Paper Structure
            logger.info("Step 4/5: Creating paper structure...")
            with self.tracer.start_as_current_span("paper_structure") as span:
                span.set_attribute("agent", "PaperStructurer")
                span.set_attribute("model_tier", "sonnet")
                try:
                    if cache and cache.has_valid_cache("paper_structure", context):
                        cached_data = cache.load("paper_structure")
                        paper_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 4/5: Using cached paper structure")
                    else:
                        paper_result = await self.paper_structurer.execute(context)
                        if cache:
                            cache.save("paper_structure", paper_result.to_dict(), context, project_id)
                        span.set_attribute("cached", False)
                    
                    result.paper_structure_result = paper_result
                    result.total_tokens += paper_result.tokens_used
                    context["paper_structure"] = paper_result.to_dict()
                    span.set_attribute("tokens_used", paper_result.tokens_used)
                    span.set_attribute("success", paper_result.success)
                    
                    # Track files created
                    if paper_result.success and paper_result.structured_data:
                        files = paper_result.structured_data.get("files_saved", {})
                        result.files_created.update(files)
                    
                    if not paper_result.success:
                        result.errors.append(f"Paper structure failed: {paper_result.error}")
                        span.set_attribute("error", paper_result.error)
                except Exception as e:
                    logger.error(f"Paper structure error: {e}")
                    result.errors.append(f"Paper structure error: {str(e)}")
                    span.set_attribute("error", str(e))
            
            # Step 5: Project Planning
            logger.info("Step 5/5: Creating project plan...")
            with self.tracer.start_as_current_span("project_planner") as span:
                span.set_attribute("agent", "ProjectPlanner")
                span.set_attribute("model_tier", "opus")
                try:
                    if cache and cache.has_valid_cache("project_planner", context):
                        cached_data = cache.load("project_planner")
                        plan_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 5/5: Using cached project plan")
                    else:
                        plan_result = await self.project_planner.execute(context)
                        if cache:
                            cache.save("project_planner", plan_result.to_dict(), context, project_id)
                        span.set_attribute("cached", False)
                    
                    result.project_plan_result = plan_result
                    result.total_tokens += plan_result.tokens_used
                    span.set_attribute("tokens_used", plan_result.tokens_used)
                    span.set_attribute("success", plan_result.success)
                    
                    # Track files created
                    if plan_result.success and plan_result.structured_data:
                        files = plan_result.structured_data.get("files_saved", {})
                        result.files_created.update(files)
                    
                    if not plan_result.success:
                        result.errors.append(f"Project planning failed: {plan_result.error}")
                        span.set_attribute("error", plan_result.error)
                except Exception as e:
                    logger.error(f"Project planning error: {e}")
                    result.errors.append(f"Project planning error: {str(e)}")
                    span.set_attribute("error", str(e))
            
            # Finalize
            result.total_time = time.time() - start_time
            
            # Determine overall success
            result.success = (
                result.hypothesis_result and result.hypothesis_result.success and
                result.literature_synthesis_result and result.literature_synthesis_result.success and
                result.paper_structure_result and result.paper_structure_result.success and
                result.project_plan_result and result.project_plan_result.success
            )
            
            # Save workflow results
            results_path = project_path / "literature_workflow_results.json"
            with open(results_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            workflow_span.set_attribute("success", result.success)
            workflow_span.set_attribute("total_tokens", result.total_tokens)
            workflow_span.set_attribute("total_time", result.total_time)
            
            logger.info(
                f"Literature workflow completed: success={result.success}, "
                f"tokens={result.total_tokens}, time={result.total_time:.1f}s"
            )
            
            return result
    
    def _result_from_cache(self, cached_data: dict) -> AgentResult:
        """Convert cached dictionary back to AgentResult."""
        agent_result = cached_data.get("agent_result", {})
        return AgentResult(
            agent_name=agent_result.get("agent_name", "unknown"),
            success=agent_result.get("success", False),
            content=agent_result.get("content", ""),
            tokens_used=agent_result.get("tokens_used", 0),
            execution_time=agent_result.get("execution_time", 0),
            error=agent_result.get("error"),
            structured_data=agent_result.get("structured_data"),
        )


async def run_literature_workflow(project_folder: str) -> LiteratureWorkflowResult:
    """
    Convenience function to run the literature workflow.
    
    Args:
        project_folder: Path to the project folder
        
    Returns:
        LiteratureWorkflowResult
    """
    workflow = LiteratureWorkflow()
    return await workflow.run(project_folder)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.literature_workflow <project_folder>")
        sys.exit(1)
    
    project_folder = sys.argv[1]
    result = asyncio.run(run_literature_workflow(project_folder))
    
    print(f"\nWorkflow completed: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total time: {result.total_time:.1f}s")
    print(f"Files created: {len(result.files_created)}")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
