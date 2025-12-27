"""
Literature Workflow Orchestrator
================================
Chains the literature phase agents together to develop hypotheses,
search literature, and create paper structure.

Reviewer note on the Edison stage:
- The Edison API call (Step 2) is an external literature retrieval + synthesis step.
    It returns a narrative response plus structured citation metadata.
- Those outputs are persisted in the workflow results and then consumed by
    `LiteratureSynthesisAgent`, which writes project artifacts like `LITERATURE_REVIEW.md`,
    `references.bib`, and `citations_data.json`.
    This makes the external call's contribution explicit and repeatable.

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

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from src.llm.claude_client import ClaudeClient
from src.llm.edison_client import EdisonClient
from src.agents.hypothesis_developer import HypothesisDevelopmentAgent
from src.agents.literature_search import LiteratureSearchAgent
from src.agents.literature_synthesis import LiteratureSynthesisAgent
from src.agents.paper_structure import PaperStructureAgent
from src.agents.project_planner import ProjectPlannerAgent
from src.agents.base import AgentResult
from src.agents.cache import WorkflowCache
from src.agents.consistency_checker import ConsistencyCheckerAgent
from src.agents.readiness_assessor import ReadinessAssessorAgent
from src.agents.data_analysis_execution import DataAnalysisExecutionAgent, AnalysisExecutionConfig
from src.utils.validation import validate_project_folder
from src.utils.workflow_issue_tracking import write_workflow_issue_tracking
from src.evidence.gates import EvidenceGateConfig, EvidenceGateError, enforce_evidence_gate
from src.evidence.pipeline import EvidencePipelineConfig, run_local_evidence_pipeline, run_evidence_pipeline_for_acquired_sources
from src.evidence.acquisition import acquire_sources_from_citations, SourceAcquisitionConfig
from src.agents.writing_review_integration import run_writing_review_stage
from src.citations.source_map import build_source_citation_map, write_source_citation_map
from src.tracing import init_tracing, get_tracer
from loguru import logger

from src.pipeline.degradation import make_degradation_event


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
    consistency_check_result: Optional[AgentResult] = None  # Cross-document consistency validation
    readiness_assessment: Optional[AgentResult] = None  # Project readiness assessment
    analysis_execution_result: Optional[AgentResult] = None  # Analysis scripts execution result
    writing_review: Optional[dict] = None  # Structured needs-revision payload (when enabled)
    total_tokens: int = 0
    total_time: float = 0.0
    errors: list = field(default_factory=list)
    files_created: dict = field(default_factory=dict)
    evidence_pipeline_result: Optional[dict] = None
    degradations: list[Dict[str, Any]] = field(default_factory=list)

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
            "writing_review": self.writing_review,
            "evidence_pipeline_result": self.evidence_pipeline_result,
            "degradations": self.degradations,
            "agents": {
                "hypothesis": self.hypothesis_result.to_dict() if self.hypothesis_result else None,
                "literature_search": self.literature_search_result.to_dict() if self.literature_search_result else None,
                "literature_synthesis": self.literature_synthesis_result.to_dict() if self.literature_synthesis_result else None,
                "paper_structure": self.paper_structure_result.to_dict() if self.paper_structure_result else None,
                "project_plan": self.project_plan_result.to_dict() if self.project_plan_result else None,
                "consistency_check": self.consistency_check_result.to_dict() if self.consistency_check_result else None,
                "readiness_assessment": self.readiness_assessment.to_dict() if self.readiness_assessment else None,
                "analysis_execution": self.analysis_execution_result.to_dict() if self.analysis_execution_result else None,
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
        - Edison API key should be configured for real literature retrieval; if it is
            unavailable, downstream stages can generate scaffold outputs.
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
        self.consistency_checker = ConsistencyCheckerAgent(client=self.client)
        self.readiness_assessor = ReadinessAssessorAgent(client=self.client)
        self.analysis_executor = DataAnalysisExecutionAgent(client=self.client)

        logger.info(f"Literature workflow initialized with 8 agents (cache={'enabled' if use_cache else 'disabled'})")

    async def run(self, project_folder: str, workflow_context: Optional[Dict[str, Any]] = None) -> LiteratureWorkflowResult:
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

            try:
                project_path = validate_project_folder(project_folder)
            except Exception as e:
                workflow_span.set_attribute("error", str(e))
                return LiteratureWorkflowResult(
                    success=False,
                    project_id="unknown",
                    project_folder=project_folder,
                    errors=[str(e)],
                )
            
            # Load prerequisites
            project_json_path = project_path / "project.json"
            overview_path = project_path / "RESEARCH_OVERVIEW.md"
            
            # Load project data
            try:
                with open(project_json_path, "r", encoding="utf-8") as f:
                    project_data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                workflow_span.set_attribute("error", f"Failed to read project.json: {e}")
                return LiteratureWorkflowResult(
                    success=False,
                    project_id="unknown",
                    project_folder=project_folder,
                    errors=[f"Failed to read project.json: {e}"],
                )
            
            project_id = project_data.get("id", "unknown")
            workflow_span.set_attribute("project_id", project_id)
            
            # Load research overview
            research_overview = ""
            if overview_path.exists():
                try:
                    research_overview = overview_path.read_text(encoding="utf-8")
                except OSError as e:
                    workflow_span.set_attribute("error", f"Failed to read RESEARCH_OVERVIEW.md: {e}")
                    return LiteratureWorkflowResult(
                        success=False,
                        project_id=project_id,
                        project_folder=project_folder,
                        errors=[f"Failed to read RESEARCH_OVERVIEW.md: {e}"],
                    )
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
            context: Dict[str, Any] = {
                "project_folder": project_folder,
                "project_data": project_data,
                "research_overview": research_overview,
            }

            if isinstance(workflow_context, dict) and workflow_context:
                # Allow callers to pass in additional configuration.
                # Explicit keys in workflow_context override existing context values, including core keys.
                context.update(workflow_context)

            # Default to running the local evidence pipeline unless callers explicitly
            # disable it. This keeps the workflow offline-friendly while ensuring
            # deterministic section writers have a chance to find evidence artifacts.
            if "evidence_pipeline" not in context:
                context["evidence_pipeline"] = {"enabled": True}

            # Optional Step 0: Local evidence pipeline (discover -> ingest -> parse -> write parsed.json -> extract evidence)
            pipeline_cfg = EvidencePipelineConfig.from_context(context)
            if pipeline_cfg.enabled:
                logger.info("Step 0/5: Running local evidence pipeline...")
                try:
                    pipeline_result = run_local_evidence_pipeline(project_folder=project_folder, config=pipeline_cfg)
                    context["source_ids"] = pipeline_result.get("source_ids", [])
                    context["evidence_pipeline_result"] = pipeline_result
                    result.evidence_pipeline_result = pipeline_result

                    errors = pipeline_result.get("errors")
                    if isinstance(errors, list) and errors:
                        result.degradations.append(
                            make_degradation_event(
                                stage="evidence",
                                reason_code="evidence_pipeline_partial_failure",
                                message="Local evidence pipeline encountered errors.",
                                recommended_action="Inspect outputs/evidence_coverage.json and per-source artifacts; rerun with fewer sources or fix inputs.",
                                details={
                                    "errors": [str(e) for e in errors][:20],
                                    "discovered_count": pipeline_result.get("discovered_count"),
                                    "processed_count": pipeline_result.get("processed_count"),
                                },
                            )
                        )
                except Exception as e:
                    # Best-effort: keep the literature workflow running even if local evidence fails.
                    msg = f"Local evidence pipeline error: {e}"
                    logger.warning(msg)
                    result.errors.append(msg)
                    result.degradations.append(
                        make_degradation_event(
                            stage="evidence",
                            reason_code="evidence_pipeline_failed",
                            message=msg,
                            recommended_action="Ensure sources are readable and PDFs can be parsed; rerun the evidence pipeline.",
                            details={"error_type": type(e).__name__},
                        )
                    )
            else:
                result.degradations.append(
                    make_degradation_event(
                        stage="evidence",
                        reason_code="evidence_pipeline_disabled",
                        message="Local evidence pipeline was disabled for this run.",
                        recommended_action="Enable evidence_pipeline in workflow_context to generate evidence artifacts.",
                        severity="info",
                    )
                )
            
            # Step 1: Hypothesis Development
            logger.info("Step 1/5: Developing hypothesis...")
            with self.tracer.start_as_current_span("hypothesis_developer") as span:
                span.set_attribute("agent", "HypothesisDeveloper")
                span.set_attribute("model_tier", "opus")
                try:
                    is_cached, cached_data = (False, None)
                    if cache:
                        is_cached, cached_data = cache.get_if_valid("hypothesis_developer", context)
                    
                    if is_cached and cached_data:
                        hyp_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 1/5: Using cached hypothesis result")
                    else:
                        hyp_result = await self.hypothesis_developer.execute(context)
                        if cache and hyp_result.success:
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
                    # Set empty hypothesis_result so subsequent steps don't fail
                    context["hypothesis_result"] = {}
            
            # Step 2: Literature Search
            logger.info("Step 2/5: Searching literature via Edison API...")
            with self.tracer.start_as_current_span("literature_search") as span:
                span.set_attribute("agent", "LiteratureSearcher")
                span.set_attribute("model_tier", "sonnet")
                try:
                    is_cached, cached_data = (False, None)
                    if cache:
                        is_cached, cached_data = cache.get_if_valid("literature_search", context)
                    
                    if is_cached and cached_data:
                        lit_search_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 2/5: Using cached literature search result")
                    else:
                        logger.info("Step 2/5: Executing Edison literature search...")
                        lit_search_result = await self.literature_searcher.execute(context)
                        logger.info(f"Step 2/5: Edison search returned, success={lit_search_result.success}")
                        if cache and lit_search_result.success:
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

                    try:
                        payload = lit_search_result.to_dict()
                        structured = payload.get("structured_data") if isinstance(payload, dict) else None
                        fb = structured.get("fallback_metadata") if isinstance(structured, dict) else None
                        if isinstance(fb, dict) and fb.get("degraded") is True:
                            result.degradations.append(
                                make_degradation_event(
                                    stage="literature",
                                    reason_code="literature_search_degraded",
                                    message="Literature search fallback chain completed with degraded results.",
                                    recommended_action="Configure Edison API access or provide a manual sources list.",
                                    details={"used_provider": fb.get("used_provider"), "attempts": fb.get("attempts")},
                                )
                            )
                    except Exception as e:
                        logger.debug(
                            "Failed to extract fallback literature metadata from search result: {}: {}",
                            type(e).__name__,
                            e,
                        )
                except Exception as e:
                    import traceback
                    logger.error(f"Literature search error: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    result.errors.append(f"Literature search error: {str(e)}")
                    span.set_attribute("error", str(e))
                    # Set empty result so subsequent steps don't fail
                    context["literature_result"] = {}
            
            # Step 3: Literature Synthesis
            logger.info("Step 3/5: Synthesizing literature...")
            with self.tracer.start_as_current_span("literature_synthesis") as span:
                span.set_attribute("agent", "LiteratureSynthesizer")
                span.set_attribute("model_tier", "sonnet")
                try:
                    gate_cfg = EvidenceGateConfig.from_context(context)
                    if gate_cfg.require_evidence:
                        try:
                            enforce_evidence_gate(
                                project_folder=context.get("project_folder", ""),
                                source_ids=context.get("source_ids"),
                                config=gate_cfg,
                            )
                        except EvidenceGateError as e:
                            result.errors.append(str(e))
                            span.set_attribute("error", str(e))
                            return LiteratureWorkflowResult(
                                success=False,
                                project_id=project_id,
                                project_folder=project_folder,
                                hypothesis_result=result.hypothesis_result,
                                literature_search_result=result.literature_search_result,
                                literature_synthesis_result=None,
                                paper_structure_result=None,
                                project_plan_result=None,
                                consistency_check_result=None,
                                readiness_assessment=None,
                                total_tokens=result.total_tokens,
                                total_time=result.total_time,
                                errors=result.errors,
                                files_created=result.files_created,
                                evidence_pipeline_result=result.evidence_pipeline_result,
                                degradations=result.degradations,
                            )

                    is_cached, cached_data = (False, None)
                    if cache:
                        is_cached, cached_data = cache.get_if_valid("literature_synthesis", context)
                    
                    if is_cached and cached_data:
                        lit_synth_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 3/5: Using cached synthesis result")
                    else:
                        logger.info("Step 3/5: Executing literature synthesis...")
                        lit_synth_result = await self.literature_synthesizer.execute(context)
                        logger.info(f"Step 3/5: Synthesis returned, success={lit_synth_result.success}")
                        if cache and lit_synth_result.success:
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
                    import traceback
                    logger.error(f"Literature synthesis error: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    result.errors.append(f"Literature synthesis error: {str(e)}")
                    span.set_attribute("error", str(e))
                    # Set empty result so subsequent steps don't fail
                    context["literature_synthesis"] = {}

            # Step 3.5 (optional): Source Acquisition - download PDFs/HTMLs from citations
            # This populates sources/<source_id>/ directories for the evidence pipeline
            source_acquisition_cfg = context.get("source_acquisition")
            if isinstance(source_acquisition_cfg, dict) and source_acquisition_cfg.get("enabled", False):
                logger.info("Step 3.5: Running source acquisition from citations...")
                with self.tracer.start_as_current_span("source_acquisition") as span:
                    span.set_attribute("enabled", True)
                    try:
                        # Get citations from literature search results
                        lit_search_data = context.get("literature_search", {})
                        citations_data = lit_search_data.get("citations_data", [])
                        
                        if citations_data:
                            cfg = SourceAcquisitionConfig.from_context(context)
                            acq_result = acquire_sources_from_citations(
                                project_folder=project_folder,
                                citations_data=citations_data,
                                config=cfg,
                            )
                            context["source_acquisition_result"] = acq_result
                            span.set_attribute("sources_acquired", len(acq_result.get("created_source_ids", [])))
                            span.set_attribute("errors_count", len(acq_result.get("errors", [])))
                            
                            if not acq_result.get("ok", False):
                                msg = f"Source acquisition had errors: {acq_result.get('errors', [])}"
                                logger.warning(msg)
                                # Non-fatal - we continue with whatever sources we got
                            else:
                                logger.info(f"Source acquisition completed: {len(acq_result.get('created_source_ids', []))} sources acquired")
                        else:
                            logger.info("Step 3.5: No citations available for source acquisition")
                            span.set_attribute("skipped", True)
                            span.set_attribute("reason", "no_citations")
                            context["source_acquisition_result"] = {
                                "ok": True,
                                "skipped": True,
                                "reason": "no_citations_data",
                            }
                    except Exception as e:
                        import traceback
                        logger.error(f"Source acquisition error: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        span.set_attribute("error", str(e))
                        # Non-fatal error - continue without source acquisition
                        context["source_acquisition_result"] = {
                            "ok": False,
                            "error": str(e),
                        }

            # Step 3.6: Run evidence pipeline on acquired sources
            # This extracts evidence items from the PDFs/HTML downloaded in Step 3.5.
            # Note: Step 0 (evidence pipeline at workflow start) handles pre-existing sources.
            # This step processes newly acquired sources in sources/<source_id>/raw/.
            acq_result = context.get("source_acquisition_result", {})
            acquired_ids = acq_result.get("created_source_ids", [])
            if acquired_ids and pipeline_cfg.enabled:
                logger.info(f"Step 3.6: Running evidence pipeline on {len(acquired_ids)} acquired sources...")
                with self.tracer.start_as_current_span("evidence_pipeline_acquired") as span:
                    span.set_attribute("source_count", len(acquired_ids))
                    try:
                        # Process acquired sources specifically
                        acquired_pipeline_result = run_evidence_pipeline_for_acquired_sources(
                            project_folder=project_folder,
                            config=pipeline_cfg,
                            source_ids=acquired_ids,
                        )
                        # Merge with any previous evidence pipeline result
                        prev_source_ids = context.get("source_ids", [])
                        new_source_ids = acquired_pipeline_result.get("source_ids", [])
                        context["source_ids"] = list(set(prev_source_ids + new_source_ids))
                        context["evidence_pipeline_acquired_result"] = acquired_pipeline_result
                        
                        span.set_attribute("processed_count", acquired_pipeline_result.get("processed_count", 0))
                        span.set_attribute("evidence_items_total", sum(
                            ps.get("evidence_items_count", 0) 
                            for ps in acquired_pipeline_result.get("per_source", [])
                        ))
                        
                        errors = acquired_pipeline_result.get("errors")
                        if isinstance(errors, list) and errors:
                            result.degradations.append(
                                make_degradation_event(
                                    stage="evidence",
                                    reason_code="acquired_evidence_partial_failure",
                                    message="Evidence pipeline on acquired sources encountered errors.",
                                    recommended_action="Inspect outputs/evidence_coverage.json; some acquired sources may have parsing issues.",
                                    details={"errors": [str(e) for e in errors][:10]},
                                )
                            )
                            span.set_attribute("has_errors", True)
                        else:
                            logger.info(f"Step 3.6: Extracted evidence from {acquired_pipeline_result.get('processed_count', 0)} sources")
                    except Exception as e:
                        import traceback
                        msg = f"Evidence pipeline on acquired sources failed: {e}"
                        logger.warning(msg)
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        result.errors.append(msg)
                        span.set_attribute("error", str(e))

            # Best-effort: build source -> citation mapping for deterministic writers.
            # Requires bibliography/citations.json (from literature synthesis) and
            # sources/<source_id>/raw/* (from evidence pipeline or manual placement).
            try:
                mapping = build_source_citation_map(project_folder)
                context["source_citation_map"] = mapping
                write_source_citation_map(project_folder, mapping)
            except Exception as e:
                msg = f"Failed to build source_citation_map: {e}"
                logger.warning(msg)
                result.errors.append(msg)
                context["source_citation_map"] = {}

            # Step 3.8: Analysis Script Execution (Issue #148)
            # Run any analysis scripts under analysis/ to produce outputs/metrics.json
            # This enables data-driven section writers (Results, Methods, Discussion).
            analysis_cfg = AnalysisExecutionConfig.from_context(context)
            if analysis_cfg.enabled:
                logger.info("Step 3.8: Running analysis scripts...")
                with self.tracer.start_as_current_span("analysis_execution") as span:
                    span.set_attribute("agent", "DataAnalysisExecution")
                    try:
                        analysis_result = await self.analysis_executor.execute(context)
                        result.analysis_execution_result = analysis_result
                        context["analysis_execution_result"] = analysis_result.to_dict()
                        span.set_attribute("success", analysis_result.success)

                        if analysis_result.success:
                            logger.info("Step 3.8: Analysis scripts completed successfully")
                            # Check if metrics.json was created
                            metrics_path = Path(project_folder) / "outputs" / "metrics.json"
                            span.set_attribute("metrics_created", metrics_path.exists())
                        else:
                            # Check if this was a downgrade (no scripts) or a hard failure
                            structured = analysis_result.structured_data or {}
                            metadata = structured.get("metadata", {})
                            if metadata.get("action") == "downgrade":
                                reason = metadata.get("reason", "unknown")
                                logger.info(f"Step 3.8: Analysis skipped (downgrade: {reason})")
                                result.degradations.append(
                                    make_degradation_event(
                                        stage="analysis",
                                        reason_code="analysis_skipped",
                                        message=f"Analysis execution skipped: {reason}",
                                        recommended_action="Add analysis scripts under analysis/ folder to generate metrics.",
                                        severity="info",
                                        details={"reason": reason},
                                    )
                                )
                            else:
                                msg = f"Analysis execution failed: {analysis_result.error}"
                                logger.warning(msg)
                                result.errors.append(msg)
                                span.set_attribute("error", analysis_result.error)
                                result.degradations.append(
                                    make_degradation_event(
                                        stage="analysis",
                                        reason_code="analysis_execution_failed",
                                        message=f"Analysis script execution failed: {analysis_result.error}",
                                        recommended_action="Check analysis scripts for errors; review outputs/artifacts.json for details.",
                                        severity="warning",
                                        details={"error": analysis_result.error},
                                    )
                                )
                    except Exception as e:
                        import traceback
                        msg = f"Analysis execution error: {e}"
                        logger.warning(msg)
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        result.errors.append(msg)
                        span.set_attribute("error", str(e))
            else:
                logger.info("Step 3.8: Analysis execution disabled in context")

            # Step 4: Paper Structure
            logger.info("Step 4/5: Creating paper structure...")
            with self.tracer.start_as_current_span("paper_structure") as span:
                span.set_attribute("agent", "PaperStructurer")
                span.set_attribute("model_tier", "sonnet")
                try:
                    is_cached, cached_data = (False, None)
                    if cache:
                        is_cached, cached_data = cache.get_if_valid("paper_structure", context)

                    if is_cached and cached_data:
                        paper_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 4/5: Using cached paper structure")
                    else:
                        paper_result = await self.paper_structurer.execute(context)
                        if cache and paper_result.success:
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

            # Optional Step 4.5: Writing + referee review integration (Sprint 4 Issue #54)
            writing_review_cfg = context.get("writing_review")
            if isinstance(writing_review_cfg, dict) and bool(writing_review_cfg.get("enabled", False)):
                logger.info("Optional step: Running writing + referee review stage...")
                with self.tracer.start_as_current_span("writing_review") as span:
                    span.set_attribute("enabled", True)
                    try:
                        writing_review_result = await run_writing_review_stage(context)
                        result.writing_review = writing_review_result.to_payload()
                        span.set_attribute("success", writing_review_result.success)
                        span.set_attribute("needs_revision", writing_review_result.needs_revision)

                        if writing_review_result.needs_revision:
                            result.errors.append(
                                f"Writing+review stage requires revision: {writing_review_result.error or 'needs_revision'}"
                            )
                            result.success = False
                            result.total_time = time.time() - start_time

                            # Save partial workflow results for debugging and iteration.
                            results_path = project_path / "literature_workflow_results.json"
                            with open(results_path, "w") as f:
                                json.dump(result.to_dict(), f, indent=2, default=str)

                            workflow_span.set_attribute("success", False)
                            workflow_span.set_attribute("total_tokens", result.total_tokens)
                            workflow_span.set_attribute("total_time", result.total_time)
                            return result
                    except Exception as e:
                        logger.error(f"Writing+review stage error: {e}")
                        result.errors.append(f"Writing+review stage error: {str(e)}")
                        result.writing_review = {
                            "success": False,
                            "needs_revision": True,
                            "written_section_relpaths": [],
                            "gates": {"enabled": True},
                            "review": None,
                            "error": str(e),
                        }
                        result.success = False
                        result.total_time = time.time() - start_time

                        results_path = project_path / "literature_workflow_results.json"
                        with open(results_path, "w") as f:
                            json.dump(result.to_dict(), f, indent=2, default=str)

                        workflow_span.set_attribute("success", False)
                        workflow_span.set_attribute("total_tokens", result.total_tokens)
                        workflow_span.set_attribute("total_time", result.total_time)
                        return result
            
            # Step 5: Project Planning
            logger.info("Step 5/5: Creating project plan...")
            with self.tracer.start_as_current_span("project_planner") as span:
                span.set_attribute("agent", "ProjectPlanner")
                span.set_attribute("model_tier", "opus")
                try:
                    is_cached, cached_data = (False, None)
                    if cache:
                        is_cached, cached_data = cache.get_if_valid("project_planner", context)
                    
                    if is_cached and cached_data:
                        plan_result = self._result_from_cache(cached_data)
                        span.set_attribute("cached", True)
                        logger.info("Step 5/5: Using cached project plan")
                    else:
                        plan_result = await self.project_planner.execute(context)
                        if cache and plan_result.success:
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
            
            # Step 6: Cross-Document Consistency Check (non-blocking)
            logger.info("Step 6/7: Running Consistency Check...")
            with self.tracer.start_as_current_span("consistency_check") as span:
                span.set_attribute("agent", "ConsistencyChecker")
                span.set_attribute("model_tier", "sonnet")
                try:
                    consistency_result = await self.consistency_checker.check_consistency(project_folder)
                    result.consistency_check_result = consistency_result
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
            
            # Step 7: Project Readiness Assessment (non-blocking)
            logger.info("Step 7/7: Running Readiness Assessment...")
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
            
            # Finalize
            result.total_time = time.time() - start_time
            
            # Determine overall success
            writing_review_ok = True
            if isinstance(result.writing_review, dict):
                gates = result.writing_review.get("gates")
                if isinstance(gates, dict) and gates.get("enabled") is True:
                    writing_review_ok = bool(result.writing_review.get("success", False)) and not bool(
                        result.writing_review.get("needs_revision", False)
                    )

            result.success = (
                result.hypothesis_result and result.hypothesis_result.success and
                result.literature_synthesis_result and result.literature_synthesis_result.success and
                result.paper_structure_result and result.paper_structure_result.success and
                result.project_plan_result and result.project_plan_result.success and
                writing_review_ok
            )
            
            # Save workflow results
            results_path = project_path / "literature_workflow_results.json"
            with open(results_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

            # Persist non-fatal issues for later fixes (non-blocking)
            try:
                write_workflow_issue_tracking(
                    project_folder,
                    result.to_dict(),
                    filename="literature_workflow_issues.json",
                )
            except Exception as e:
                logger.warning(f"Failed to write literature workflow issue tracking: {e}")
            
            workflow_span.set_attribute("success", result.success)
            workflow_span.set_attribute("total_tokens", result.total_tokens)
            workflow_span.set_attribute("total_time", result.total_time)
            
            logger.info(
                f"Literature workflow completed: success={result.success}, "
                f"tokens={result.total_tokens}, time={result.total_time:.1f}s"
            )
            
            return result
    
    def _result_from_cache(self, cached_data: dict) -> AgentResult:
        """Convert cached dictionary back to AgentResult.
        
        Note: cache.load() returns the agent_result dict directly (not wrapped),
        so cached_data IS the agent_result, not {agent_result: {...}}.
        """
        from src.llm.claude_client import TaskType, ModelTier
        
        # cached_data IS the agent_result (cache.load returns entry.agent_result directly)
        agent_result = cached_data
        logger.debug(f"_result_from_cache: agent_result keys = {list(agent_result.keys())}")
        logger.debug(f"_result_from_cache: structured_data = {agent_result.get('structured_data')}")
        
        # Parse task_type and model_tier from cached values
        task_type_str = agent_result.get("task_type", "data_analysis")
        model_tier_str = agent_result.get("model_tier", "sonnet")
        
        # Convert strings to enums
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.DATA_ANALYSIS
        
        try:
            model_tier = ModelTier(model_tier_str)
        except ValueError:
            model_tier = ModelTier.SONNET
        
        result = AgentResult(
            agent_name=agent_result.get("agent_name", "unknown"),
            task_type=task_type,
            model_tier=model_tier,
            success=agent_result.get("success", False),
            content=agent_result.get("content", ""),
            tokens_used=agent_result.get("tokens_used", 0),
            execution_time=agent_result.get("execution_time", 0),
            error=agent_result.get("error"),
            structured_data=agent_result.get("structured_data"),
            timestamp=agent_result.get("timestamp", datetime.now().isoformat()),
        )
        logger.debug(f"_result_from_cache: result.structured_data = {result.structured_data}")
        return result


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
