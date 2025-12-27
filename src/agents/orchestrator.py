"""
Agent Orchestrator
==================
Manages inter-agent communication, revision loops, and workflow
execution with iterative refinement.

Key responsibilities:
- Execute agents with permission enforcement
- Manage feedback and revision loops
- Track iteration state and convergence
- Handle inter-agent calls safely
- Coordinate quality reviews

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import asyncio
import json
import uuid
import time
from typing import Optional, Dict, List, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from loguru import logger

from .registry import AgentRegistry, AgentSpec, AgentCapability
from .feedback import (
    FeedbackResponse,
    RevisionTrigger,
    ConvergenceCriteria,
    QualityScore,
    AgentCallRequest,
    AgentCallResponse,
    Severity,
)
from .base import AgentResult, BaseAgent
from .cache import WorkflowCache
from .critical_review import CriticalReviewAgent
from .task_decomposition import (
    SubtaskRunRecord,
    aggregate_subtask_runs,
    decompose_task_via_llm,
    normalize_task_decomposition,
    validate_task_decomposition,
)
from src.llm.claude_client import ClaudeClient
from src.utils.project_layout import ensure_project_outputs_layout


class ExecutionMode(Enum):
    """Execution modes for the orchestrator."""
    SINGLE_PASS = "single_pass"     # Run once, no review
    WITH_REVIEW = "with_review"      # Run + critical review
    ITERATIVE = "iterative"          # Run + review + revise until converged


@dataclass
class ExecutionState:
    """Tracks the state of an agent execution."""
    agent_id: str
    agent_name: str
    iteration: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    quality_scores: List[float] = field(default_factory=list)
    results: List[AgentResult] = field(default_factory=list)
    feedback: List[FeedbackResponse] = field(default_factory=list)
    converged: bool = False
    convergence_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "iteration": self.iteration,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "quality_scores": self.quality_scores,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    # Execution settings
    default_mode: ExecutionMode = ExecutionMode.WITH_REVIEW
    enable_inter_agent_calls: bool = True
    max_call_depth: int = 2
    
    # Review settings
    auto_review: bool = True
    review_threshold: float = 0.7  # Skip review if self-critique > this
    
    # Convergence settings
    convergence: ConvergenceCriteria = field(default_factory=ConvergenceCriteria)
    
    # Caching settings
    cache_versions: bool = True
    cache_max_age_hours: int = 24

    # Evidence hook settings (off by default)
    enable_evidence_hook: bool = False
    evidence_hook_max_items: int = 25
    evidence_hook_min_excerpt_chars: int = 20
    evidence_hook_append_ledger: bool = False
    evidence_hook_require_evidence: bool = True
    evidence_hook_min_items_per_source: int = 1
    
    # Timeout settings
    agent_timeout: int = 600  # 10 minutes per agent
    review_timeout: int = 300  # 5 minutes for review


class AgentOrchestrator:
    """
    Orchestrates agent execution with iterative refinement.
    
    Features:
    - Permission-enforced inter-agent calls
    - Automatic quality review
    - Revision loops with convergence detection
    - Version tracking and caching
    - Call depth limiting
    
    Usage:
        orchestrator = AgentOrchestrator(project_folder)
        
        # Simple execution
        result = await orchestrator.execute_agent("A05", context)
        
        # With review
        result, feedback = await orchestrator.execute_with_review("A05", context)
        
        # Iterative refinement
        result = await orchestrator.execute_iterative("A05", context, max_iterations=3)
    """
    
    def __init__(
        self,
        project_folder: str,
        client: Optional[ClaudeClient] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            project_folder: Path to project folder for caching
            client: Shared ClaudeClient instance
            config: Orchestrator configuration
        """
        self.project_folder = project_folder
        self.client = client or ClaudeClient()
        self.config = config or OrchestratorConfig()
        
        # Initialize cache
        self.cache = WorkflowCache(
            project_folder,
            max_age_hours=self.config.cache_max_age_hours,
        )
        
        # Initialize critical reviewer
        self.reviewer = CriticalReviewAgent(client=self.client)
        
        # Track execution state
        self.execution_states: Dict[str, ExecutionState] = {}
        
        # Track call stack for depth limiting
        self._call_stack: List[str] = []
        
        # Agent instances cache
        self._agent_instances: Dict[str, BaseAgent] = {}
        
        logger.info(
            f"Orchestrator initialized for {project_folder} "
            f"(mode={self.config.default_mode.value})"
        )
    
    def _get_agent_instance(self, agent_id: str) -> Optional[BaseAgent]:
        """Get or create an agent instance."""
        if agent_id in self._agent_instances:
            return self._agent_instances[agent_id]
        
        agent = AgentRegistry.create_agent(agent_id, client=self.client)
        if agent:
            self._agent_instances[agent_id] = agent
        return agent
    
    def _check_permission(self, caller_id: str, target_id: str) -> bool:
        """Check if caller has permission to call target."""
        if not self.config.enable_inter_agent_calls:
            logger.warning(f"Inter-agent calls disabled")
            return False
        
        if not AgentRegistry.can_call(caller_id, target_id):
            logger.warning(
                f"Permission denied: {caller_id} cannot call {target_id}"
            )
            return False
        
        return True
    
    def _check_call_depth(self) -> bool:
        """Check if we've exceeded max call depth."""
        if len(self._call_stack) >= self.config.max_call_depth:
            logger.warning(
                f"Max call depth ({self.config.max_call_depth}) exceeded: "
                f"{' -> '.join(self._call_stack)}"
            )
            return False
        return True

    def _coerce_datetime_iso_utc(self, timestamp: Optional[str]) -> str:
        """Return an ISO timestamp with timezone info for schema date-time."""
        if isinstance(timestamp, str) and timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
            except ValueError:
                dt = datetime.now(timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.isoformat()

    def _find_first_str_by_key(self, obj: Any, *, keys: tuple[str, ...], max_depth: int = 8) -> Optional[str]:
        """Find the first non-empty string value for any key in a nested object."""
        if max_depth <= 0:
            return None

        if isinstance(obj, dict):
            for key in keys:
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val

            for val in obj.values():
                found = self._find_first_str_by_key(val, keys=keys, max_depth=max_depth - 1)
                if found:
                    return found

        if isinstance(obj, list):
            for val in obj:
                found = self._find_first_str_by_key(val, keys=keys, max_depth=max_depth - 1)
                if found:
                    return found

        return None

    def _maybe_run_evidence_hook(self, *, stage_name: str, result: AgentResult) -> None:
        """Optionally write evidence artifacts for a stage result.

        This is best-effort; failures are logged and never fail the workflow.
        """
        if not self.config.enable_evidence_hook:
            return
        if not getattr(result, "success", False):
            return

        try:
            from src.evidence.extraction import extract_evidence_items
            from src.evidence.gates import EvidenceGateConfig, check_evidence_gate
            from src.evidence.parser import MVPLineBlockParser
            from src.evidence.store import EvidenceStore
        except Exception as e:
            logger.warning(f"Evidence hook import failed: {e}")
            return

        try:
            text: Optional[str] = None
            structured = getattr(result, "structured_data", None)
            if isinstance(structured, dict):
                text = self._find_first_str_by_key(structured, keys=("formatted_answer", "content"))

            if not text:
                content = getattr(result, "content", None)
                text = content if isinstance(content, str) and content.strip() else None

            if not text:
                logger.warning(f"Evidence hook skipped; no text payload for stage {stage_name}")
                return

            source_id = f"cache:{stage_name}"
            store = EvidenceStore(str(self.project_folder))

            mvp_parser = MVPLineBlockParser()
            parsed_doc = mvp_parser.parse(text)
            parsed_payload: dict[str, Any] = {
                "parser": {"name": parsed_doc.parser_name, "version": parsed_doc.parser_version},
                "blocks": [
                    {
                        "kind": b.kind,
                        "span": {"start_line": b.span.start_line, "end_line": b.span.end_line},
                        "text": b.text,
                    }
                    for b in parsed_doc.blocks
                ],
            }
            store.write_parsed(source_id, parsed_payload)

            created_at = self._coerce_datetime_iso_utc(getattr(result, "timestamp", None))
            items = extract_evidence_items(
                parsed=parsed_payload,
                source_id=source_id,
                created_at=created_at,
                max_items=int(self.config.evidence_hook_max_items),
                min_excerpt_chars=int(self.config.evidence_hook_min_excerpt_chars),
            )
            store.write_evidence_items(source_id, items)

            if self.config.evidence_hook_append_ledger:
                try:
                    seen = any((it.get("source_id") == source_id) for it in store.iter_items(validate=False) or [])
                    if not seen:
                        store.append_many(items)
                except Exception as e:
                    logger.warning(f"Evidence hook ledger append failed for {stage_name}: {e}")

            gate_cfg = EvidenceGateConfig(
                require_evidence=bool(self.config.evidence_hook_require_evidence),
                min_items_per_source=int(self.config.evidence_hook_min_items_per_source),
            )
            gate_result = check_evidence_gate(
                project_folder=str(self.project_folder),
                source_ids=[source_id],
                config=gate_cfg,
            )
            if not gate_result.get("ok", False):
                logger.warning(f"Evidence gate failed for stage {stage_name}: {gate_result}")

        except Exception as e:
            logger.warning(f"Evidence hook failed for stage {stage_name}: {type(e).__name__}: {e}")
    
    async def execute_agent(
        self,
        agent_id: str,
        context: dict,
        use_cache: bool = True,
    ) -> AgentResult:
        """
        Execute an agent with optional caching.
        
        Args:
            agent_id: Agent ID to execute
            context: Execution context
            use_cache: Whether to use/update cache
            
        Returns:
            AgentResult from execution
        """
        spec = AgentRegistry.get(agent_id)
        if not spec:
            logger.error(f"Agent {agent_id} not found in registry")
            return AgentResult(
                agent_name="unknown",
                task_type=None,
                model_tier=None,
                success=False,
                content="",
                error=f"Agent {agent_id} not found",
            )
        
        # Check cache first
        stage_name = spec.name.lower()
        if use_cache:
            is_valid, cached_result = self.cache.get_if_valid(stage_name, context)
            if is_valid and cached_result:
                logger.info(f"Using cached result for {spec.name}")
                cached_obj = AgentResult.from_dict(cached_result)
                self._maybe_run_evidence_hook(stage_name=stage_name, result=cached_obj)
                return cached_obj
        
        # Get agent instance
        agent = self._get_agent_instance(agent_id)
        if not agent:
            return AgentResult(
                agent_name=spec.name,
                task_type=None,
                model_tier=None,
                success=False,
                content="",
                error=f"Failed to instantiate agent {agent_id}",
            )
        
        # Track call stack
        self._call_stack.append(agent_id)
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.execute(context),
                timeout=self.config.agent_timeout,
            )
            
            # Cache result if successful
            if use_cache and result.success:
                project_id = context.get("project_data", {}).get("id", "unknown")
                self.cache.save(stage_name, result.to_dict(), context, project_id)

            self._maybe_run_evidence_hook(stage_name=stage_name, result=result)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Agent {agent_id} timed out after {self.config.agent_timeout}s")
            return AgentResult(
                agent_name=spec.name,
                task_type=None,
                model_tier=None,
                success=False,
                content="",
                error=f"Execution timed out after {self.config.agent_timeout}s",
            )
        except Exception as e:
            logger.error(f"Agent {agent_id} execution error: {e}")
            return AgentResult(
                agent_name=spec.name,
                task_type=None,
                model_tier=None,
                success=False,
                content="",
                error=str(e),
            )
        finally:
            self._call_stack.pop()
    
    async def review_result(
        self,
        result: AgentResult,
        content_type: str = "general",
    ) -> FeedbackResponse:
        """
        Review an agent result using the critical reviewer.
        
        Args:
            result: AgentResult to review
            content_type: Type of content for criteria selection
            
        Returns:
            FeedbackResponse with quality assessment
        """
        try:
            feedback = await asyncio.wait_for(
                self.reviewer.review_agent_result(result, content_type),
                timeout=self.config.review_timeout,
            )
            return feedback
        except asyncio.TimeoutError:
            logger.error("Review timed out")
            return FeedbackResponse(
                request_id="timeout",
                reviewer_agent_id="A12",
                quality_score=QualityScore(overall=0.5),
                issues=[],
                summary="Review timed out",
                revision_required=True,
            )
        except Exception as e:
            logger.error(f"Review error: {e}")
            return FeedbackResponse(
                request_id="error",
                reviewer_agent_id="A12",
                quality_score=QualityScore(overall=0.5),
                issues=[],
                summary=f"Review failed: {e}",
                revision_required=True,
            )
    
    async def _self_critique_shortcut(
        self,
        agent_id: str,
        result: AgentResult,
    ) -> Optional[FeedbackResponse]:
        """Optionally skip formal review based on the agent's self-critique score."""
        if not self.config.auto_review:
            return None

        agent = self._get_agent_instance(agent_id)
        if not agent:
            return None

        self_critique = await agent.self_critique(result)
        self_score = self_critique.get("scores", {}).get("overall", 0)
        if self_score >= self.config.review_threshold:
            logger.info(
                f"Skipping review: self-critique score {self_score:.2f} "
                f">= threshold {self.config.review_threshold}"
            )
            return FeedbackResponse(
                request_id="self_critique",
                reviewer_agent_id=agent_id,
                quality_score=QualityScore(overall=self_score),
                issues=[],
                summary=self_critique.get("summary", "Self-critique passed"),
                revision_required=False,
            )

        return None

    async def execute_with_review(
        self,
        agent_id: str,
        context: dict,
        content_type: str = "general",
    ) -> tuple[AgentResult, FeedbackResponse]:
        """
        Execute an agent and review its output.
        
        Args:
            agent_id: Agent ID to execute
            context: Execution context
            content_type: Type of content for review criteria
            
        Returns:
            Tuple of (AgentResult, FeedbackResponse)
        """
        # Execute agent
        result = await self.execute_agent(agent_id, context)
        
        if not result.success:
            # Return without review if execution failed
            return result, FeedbackResponse(
                request_id="no_review",
                reviewer_agent_id="A12",
                quality_score=QualityScore(overall=0.0),
                issues=[],
                summary="Skipped review due to execution failure",
                revision_required=True,
            )
        
        shortcut = await self._self_critique_shortcut(agent_id, result)
        if shortcut:
            return result, shortcut
        
        # Formal review
        feedback = await self.review_result(result, content_type)
        
        return result, feedback

    async def _review_existing_result(
        self,
        agent_id: str,
        result: AgentResult,
        content_type: str,
    ) -> FeedbackResponse:
        """Review an already-produced result, preserving the self-critique shortcut."""
        if not result.success:
            return FeedbackResponse(
                request_id="no_review",
                reviewer_agent_id="A12",
                quality_score=QualityScore(overall=0.0),
                issues=[],
                summary="Skipped review due to execution failure",
                revision_required=True,
            )
        shortcut = await self._self_critique_shortcut(agent_id, result)
        if shortcut:
            return shortcut

        return await self.review_result(result, content_type)
    
    async def execute_iterative(
        self,
        agent_id: str,
        context: dict,
        content_type: str = "general",
        max_iterations: Optional[int] = None,
        convergence: Optional[ConvergenceCriteria] = None,
    ) -> AgentResult:
        """
        Execute an agent with iterative refinement until convergence.
        
        Args:
            agent_id: Agent ID to execute
            context: Execution context
            content_type: Type of content for review criteria
            max_iterations: Override default max iterations
            convergence: Override convergence criteria
            
        Returns:
            Final AgentResult after convergence
        """
        spec = AgentRegistry.get(agent_id)
        if not spec:
            return AgentResult(
                agent_name="unknown",
                task_type=None,
                model_tier=None,
                success=False,
                content="",
                error=f"Agent {agent_id} not found",
            )
        
        # Check if agent supports revision
        if not spec.supports_revision:
            logger.warning(f"Agent {agent_id} does not support revision, running single pass")
            result, _ = await self.execute_with_review(agent_id, context, content_type)
            return result
        
        # Initialize state
        conv = convergence or self.config.convergence
        max_iter = max_iterations or conv.max_iterations
        
        state = ExecutionState(
            agent_id=agent_id,
            agent_name=spec.name,
            started_at=datetime.now().isoformat(),
        )
        self.execution_states[agent_id] = state
        
        # Initial execution
        result, feedback = await self.execute_with_review(
            agent_id, context, content_type
        )
        
        state.results.append(result)
        state.feedback.append(feedback)
        state.quality_scores.append(feedback.quality_score.overall)
        
        # Cache version 0
        if self.config.cache_versions and result.success:
            project_id = context.get("project_data", {}).get("id", "unknown")
            self.cache.save_version(
                stage_name=spec.name.lower(),
                agent_result=result.to_dict(),
                context=context,
                project_id=project_id,
                version=0,
                quality_score=feedback.quality_score.overall,
            )
        
        # Iteration loop
        current_result = result
        previous_score = None
        
        for iteration in range(1, max_iter + 1):
            state.iteration = iteration
            
            # Check convergence
            should_stop, reason = conv.should_stop(
                current_score=feedback.quality_score.overall,
                previous_score=previous_score,
                iteration=iteration,
                critical_count=len(feedback.critical_issues),
                major_count=len(feedback.major_issues),
            )
            
            if should_stop:
                state.converged = True
                state.convergence_reason = reason
                logger.info(f"Converged at iteration {iteration}: {reason}")
                break
            
            # Check if revision is needed
            if not feedback.revision_required:
                state.converged = True
                state.convergence_reason = "No revision required"
                logger.info(f"No revision needed at iteration {iteration}")
                break
            
            # Build revision trigger
            trigger = RevisionTrigger(
                trigger_id=str(uuid.uuid4())[:8],
                target_agent_id=agent_id,
                original_content=current_result.content,
                feedback=feedback,
                iteration=iteration,
                max_iterations=max_iter,
                focus_areas=feedback.revision_priority,
            )
            
            # Revise
            agent = self._get_agent_instance(agent_id)
            if not agent:
                logger.error(f"Failed to get agent {agent_id} for revision")
                break
            
            logger.info(f"Starting revision {iteration} for {spec.name}")
            
            revised_result = await agent.revise(
                original_result=current_result,
                feedback=trigger.format_feedback_for_agent(),
                context=context,
            )

            # Review the revised content (do not re-execute the agent)
            new_feedback = await self._review_existing_result(
                agent_id=agent_id,
                result=revised_result,
                content_type=content_type,
            )
            
            # Update state
            state.results.append(revised_result)
            state.feedback.append(new_feedback)
            state.quality_scores.append(new_feedback.quality_score.overall)
            
            # Cache version
            if self.config.cache_versions and revised_result.success:
                project_id = context.get("project_data", {}).get("id", "unknown")
                self.cache.save_version(
                    stage_name=spec.name.lower(),
                    agent_result=revised_result.to_dict(),
                    context=context,
                    project_id=project_id,
                    version=iteration,
                    quality_score=new_feedback.quality_score.overall,
                    feedback_summary=new_feedback.summary[:200],
                )
            
            # Update for next iteration
            previous_score = feedback.quality_score.overall
            current_result = revised_result
            feedback = new_feedback
            
            logger.info(
                f"Iteration {iteration}: score {feedback.quality_score.overall:.2f} "
                f"(critical={len(feedback.critical_issues)}, major={len(feedback.major_issues)})"
            )
        
        # Finalize state
        state.completed_at = datetime.now().isoformat()
        
        if not state.converged:
            state.convergence_reason = f"Max iterations ({max_iter}) reached"
        
        # Return best result or latest
        if self.config.cache_versions:
            best = self.cache.get_best_version(spec.name.lower())
            if best:
                version, best_result = best
                logger.info(f"Returning best version {version}")
                return AgentResult.from_dict(best_result)
        
        return current_result
    
    async def handle_inter_agent_call(
        self,
        request: AgentCallRequest,
    ) -> AgentCallResponse:
        """
        Handle an inter-agent call request with permission checking.
        
        Args:
            request: AgentCallRequest specifying the call
            
        Returns:
            AgentCallResponse with result or error
        """
        start_time = time.time()
        
        # Check permission
        if not self._check_permission(request.caller_agent_id, request.target_agent_id):
            return AgentCallResponse(
                call_id=request.call_id,
                success=False,
                error=f"Permission denied: {request.caller_agent_id} cannot call {request.target_agent_id}",
                execution_time=time.time() - start_time,
            )
        
        # Check call depth
        if not self._check_call_depth():
            return AgentCallResponse(
                call_id=request.call_id,
                success=False,
                error=f"Max call depth exceeded",
                execution_time=time.time() - start_time,
            )
        
        # Execute target agent
        try:
            result = await asyncio.wait_for(
                self.execute_agent(
                    request.target_agent_id,
                    request.context,
                    use_cache=True,
                ),
                timeout=request.timeout_seconds,
            )
            
            return AgentCallResponse(
                call_id=request.call_id,
                success=result.success,
                result=result.to_dict() if result.success else None,
                error=result.error,
                execution_time=time.time() - start_time,
            )
            
        except asyncio.TimeoutError:
            return AgentCallResponse(
                call_id=request.call_id,
                success=False,
                error=f"Call timed out after {request.timeout_seconds}s",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return AgentCallResponse(
                call_id=request.call_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def execute_decomposed_task(
        self,
        *,
        task_text: str,
        context: Dict[str, Any],
        decomposition_override: Optional[Dict[str, Any]] = None,
        artifact_filename: str = "subtasks_aggregate.json",
    ) -> Dict[str, Any]:
        """Decompose a high-level task, execute subtasks, and aggregate results.

        The decomposition step is prompt-driven (LLM) unless `decomposition_override`
        is provided.

        Subtasks are routed through the agent registry and executed independently.
        Failures are isolated so one failing subtask does not crash the run.

        Args:
            task_text: High-level task description.
            context: Base context to provide to each subtask.
            decomposition_override: Optional decomposition dict (useful for tests).
            artifact_filename: Output artifact filename written under outputs/.

        Returns:
            Aggregate artifact payload dict (also written to disk).
        """

        if decomposition_override is not None:
            decomposition = normalize_task_decomposition(decomposition_override)
            validate_task_decomposition(decomposition)
        else:
            decomposition = await decompose_task_via_llm(
                client=self.client,
                task_text=task_text,
                available_agent_ids=AgentRegistry.list_ids(),
            )

        subtasks = list(decomposition.get("subtasks") or [])

        async def _run_one(subtask: Dict[str, Any]) -> SubtaskRunRecord:
            subtask_id = str(subtask.get("id") or "")
            agent_id = str(subtask.get("agent_id") or "")
            inputs = subtask.get("inputs")
            if not isinstance(inputs, dict):
                inputs = {}

            spec = AgentRegistry.get(agent_id)
            if spec is None:
                return SubtaskRunRecord(
                    subtask_id=subtask_id,
                    agent_id=agent_id,
                    success=False,
                    error=f"Unknown agent_id: {agent_id}",
                    result=None,
                )

            merged_context = dict(context)
            merged_context.update(inputs)

            missing: List[str] = []
            for key in (spec.input_schema.required or []):
                if key not in merged_context:
                    missing.append(key)

            if missing:
                return SubtaskRunRecord(
                    subtask_id=subtask_id,
                    agent_id=agent_id,
                    success=False,
                    error=f"Missing required inputs for {agent_id}: {', '.join(sorted(missing))}",
                    result=None,
                )

            try:
                result = await self.execute_agent(agent_id, merged_context, use_cache=True)
                if result.success:
                    return SubtaskRunRecord(
                        subtask_id=subtask_id,
                        agent_id=agent_id,
                        success=True,
                        error=None,
                        result=result.to_dict(),
                    )
                return SubtaskRunRecord(
                    subtask_id=subtask_id,
                    agent_id=agent_id,
                    success=False,
                    error=result.error or "Subtask failed",
                    result=None,
                )
            except Exception as e:
                return SubtaskRunRecord(
                    subtask_id=subtask_id,
                    agent_id=agent_id,
                    success=False,
                    error=str(e),
                    result=None,
                )

        runs = await asyncio.gather(*[_run_one(st) for st in subtasks])
        aggregate = aggregate_subtask_runs(decomposition=decomposition, runs=list(runs))

        paths = ensure_project_outputs_layout(self.project_folder)
        artifact_path = paths.outputs_dir / artifact_filename
        artifact_path.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        aggregate["artifact_path"] = str(artifact_path.relative_to(paths.project_folder))
        return aggregate
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions."""
        return {
            agent_id: state.to_dict()
            for agent_id, state in self.execution_states.items()
        }
    
    def get_agent_summary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get execution summary for a specific agent."""
        state = self.execution_states.get(agent_id)
        if state:
            return state.to_dict()
        return None


# Factory function for creating orchestrators
def create_orchestrator(
    project_folder: str,
    mode: ExecutionMode = ExecutionMode.WITH_REVIEW,
    max_iterations: int = 3,
    quality_threshold: float = 0.8,
) -> AgentOrchestrator:
    """
    Create an orchestrator with common configuration.
    
    Args:
        project_folder: Path to project folder
        mode: Default execution mode
        max_iterations: Maximum revision iterations
        quality_threshold: Quality score to stop iterating
        
    Returns:
        Configured AgentOrchestrator
    """
    config = OrchestratorConfig(
        default_mode=mode,
        convergence=ConvergenceCriteria(
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
        ),
    )
    
    return AgentOrchestrator(project_folder, config=config)
