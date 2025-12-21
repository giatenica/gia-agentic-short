"""
Base Agent Classes
==================
Foundation for all research agents using Claude API.

Implements best practices for:
- Current date awareness (models know today's date)
- Web search awareness (models flag when they need current info)
- Optimal model selection (Opus/Sonnet/Haiku per task type)
- Prompt caching (Anthropic cache control when enabled)
- Critical rules enforcement (no banned words, no hallucination)
- Iterative refinement with revision support
- Quality self-assessment

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Literal, List, Dict, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from src.agents.feedback import FeedbackResponse, QualityScore

from src.llm.claude_client import ClaudeClient, TaskType, ModelTier
from src.agents.best_practices import (
    build_enhanced_system_prompt,
    get_current_date_context,
    CachingGuidelines,
    CachingStrategy,
)
from loguru import logger


@dataclass
class AgentResult:
    """
    Result from an agent execution.
    
    Extended to support iterative refinement:
    - iteration: Current iteration number (0 for initial, 1+ for revisions)
    - quality_scores: Quality assessment from self-critique or reviewer
    - feedback_history: Record of feedback received
    - previous_versions: Content history for tracking changes
    """
    agent_name: str
    task_type: TaskType
    model_tier: ModelTier
    success: bool
    content: str
    structured_data: dict = field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Iteration tracking
    iteration: int = 0
    quality_scores: dict = field(default_factory=dict)
    feedback_history: List[str] = field(default_factory=list)
    previous_versions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert result to dictionary for JSON serialization."""
        return {
            "agent_name": self.agent_name,
            "task_type": self.task_type.value,
            "model_tier": self.model_tier.value,
            "success": self.success,
            "content": self.content,
            "structured_data": self.structured_data,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "quality_scores": self.quality_scores,
            "feedback_history": self.feedback_history,
            "previous_versions": self.previous_versions,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentResult":
        """Create AgentResult from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            task_type=TaskType(data["task_type"]),
            model_tier=ModelTier(data["model_tier"]),
            success=data["success"],
            content=data["content"],
            structured_data=data.get("structured_data", {}),
            error=data.get("error"),
            tokens_used=data.get("tokens_used", 0),
            execution_time=data.get("execution_time", 0.0),
            timestamp=data.get("timestamp", ""),
            iteration=data.get("iteration", 0),
            quality_scores=data.get("quality_scores", {}),
            feedback_history=data.get("feedback_history", []),
            previous_versions=data.get("previous_versions", []),
        )
    
    def with_revision(
        self,
        new_content: str,
        feedback: str,
        new_quality_scores: Optional[dict] = None,
    ) -> "AgentResult":
        """
        Create a new result representing a revision of this one.
        
        Args:
            new_content: Revised content
            feedback: Feedback that prompted the revision
            new_quality_scores: Updated quality scores
            
        Returns:
            New AgentResult with incremented iteration
        """
        return AgentResult(
            agent_name=self.agent_name,
            task_type=self.task_type,
            model_tier=self.model_tier,
            success=True,
            content=new_content,
            structured_data=self.structured_data,
            error=None,
            tokens_used=0,  # Will be updated by caller
            execution_time=0.0,  # Will be updated by caller
            iteration=self.iteration + 1,
            quality_scores=new_quality_scores or {},
            feedback_history=self.feedback_history + [feedback],
            previous_versions=self.previous_versions + [self.content],
        )


class BaseAgent(ABC):
    """
    Base class for all research agents.
    
    Features automatically included:
    - Current date context (models know today's date)
    - Web search awareness (models flag when they need current info)
    - Optimal model selection based on task type
    - Prompt caching with configurable TTL
    - Critical rules enforcement
    """
    
    def __init__(
        self,
        name: str,
        task_type: TaskType,
        system_prompt: str,
        client: Optional[ClaudeClient] = None,
        include_date: bool = True,
        include_web_awareness: bool = True,
        cache_ttl: Literal["ephemeral"] = "ephemeral",
        time_budget_seconds: Optional[int] = None,
    ):
        """
        Initialize agent with Claude client and configuration.
        
        Args:
            name: Agent identifier
            task_type: Type of task for model selection
            system_prompt: System instructions for the agent
            client: Optional ClaudeClient instance (creates new if not provided)
            include_date: Whether to add current date context to prompts
            include_web_awareness: Whether to add web search awareness
            cache_ttl: Cache duration ('ephemeral' = 5min)
            time_budget_seconds: Optional execution time budget (warn-only, does not abort)
        """
        self.name = name
        self.task_type = task_type
        self.cache_ttl = cache_ttl
        self.client = client or ClaudeClient()
        self.model_tier = self.client.get_model_for_task(task_type)
        self.time_budget_seconds = time_budget_seconds
        self._execution_start_time: Optional[float] = None
        
        # Build enhanced system prompt with best practices
        self.system_prompt = build_enhanced_system_prompt(
            base_prompt=system_prompt,
            include_date=include_date,
            include_web_awareness=include_web_awareness,
            include_model_context=True,
            model_tier=self.model_tier,
        )
        
        logger.info(f"Initialized {name} agent with {self.model_tier.value} model")
    
    @abstractmethod
    async def execute(self, context: dict) -> AgentResult:
        """
        Execute the agent's task.
        
        Args:
            context: Dictionary containing project data and any prior agent results
            
        Returns:
            AgentResult with the agent's findings
        """
        pass
    
    async def _call_claude(
        self,
        user_message: str,
        use_thinking: bool = False,
        max_tokens: int = 32000,
        budget_tokens: int = 16000,
    ) -> tuple[str, int]:
        """
        Call Claude API asynchronously with the agent's configuration.
        
        Uses async methods to avoid blocking the event loop during API calls.
        
        Args:
            user_message: The user message to send
            use_thinking: Whether to use extended thinking mode
            max_tokens: Maximum output tokens (used with thinking mode)
            budget_tokens: Token budget for extended thinking
            
        Returns:
            Tuple of (response content, tokens used)
        """
        import time
        start_time = time.time()
        
        # Format message as list for Claude API
        messages = [{"role": "user", "content": user_message}]
        
        try:
            if use_thinking:
                # Use async version to avoid blocking event loop
                thinking, response = await self.client.chat_with_thinking_async(
                    messages=messages,
                    system=self.system_prompt,
                    model=self.model_tier,
                    max_tokens=max_tokens,
                    budget_tokens=budget_tokens,
                )
                content = response
                tokens = self.client.usage.output_tokens
            else:
                # Use async chat method
                response = await self.client.chat_async(
                    messages=messages,
                    system=self.system_prompt,
                    task=self.task_type,
                )
                content = response
                tokens = self.client.usage.output_tokens
            
            elapsed = time.time() - start_time
            logger.debug(f"{self.name} completed in {elapsed:.2f}s, {tokens} tokens")
            
            # Check time budget (warn-only, does not abort)
            self._check_time_budget(elapsed)
            
            return content, tokens
            
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
            raise
    
    def _check_time_budget(self, elapsed_seconds: float) -> Optional[str]:
        """
        Check if execution time exceeds budget. Warn-only policy.
        
        Args:
            elapsed_seconds: Time taken so far
            
        Returns:
            Warning message if budget exceeded, None otherwise
        """
        if not self.time_budget_seconds:
            return None
        
        warning_threshold = 0.8  # Warn at 80% of budget
        
        if elapsed_seconds > self.time_budget_seconds:
            overage = elapsed_seconds - self.time_budget_seconds
            msg = (
                f"BUDGET EXCEEDED: {self.name} ran {elapsed_seconds:.1f}s "
                f"(budget: {self.time_budget_seconds}s, over by {overage:.1f}s)"
            )
            logger.warning(msg)
            return msg
        elif elapsed_seconds > self.time_budget_seconds * warning_threshold:
            remaining = self.time_budget_seconds - elapsed_seconds
            msg = (
                f"BUDGET WARNING: {self.name} at {elapsed_seconds:.1f}s "
                f"({remaining:.1f}s remaining of {self.time_budget_seconds}s budget)"
            )
            logger.warning(msg)
            return msg
        
        return None
    
    def start_execution_timer(self):
        """Start the execution timer for budget tracking."""
        import time
        self._execution_start_time = time.time()
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since execution started."""
        import time
        if self._execution_start_time is None:
            return 0.0
        return time.time() - self._execution_start_time
    
    def _build_result(
        self,
        success: bool,
        content: str,
        structured_data: Optional[dict] = None,
        error: Optional[str] = None,
        tokens_used: int = 0,
        execution_time: float = 0.0,
    ) -> AgentResult:
        """Build an AgentResult with common fields populated."""
        return AgentResult(
            agent_name=self.name,
            task_type=self.task_type,
            model_tier=self.model_tier,
            success=success,
            content=content,
            structured_data=structured_data or {},
            error=error,
            tokens_used=tokens_used,
            execution_time=execution_time,
        )
    
    async def revise(
        self,
        original_result: AgentResult,
        feedback: str,
        context: Optional[dict] = None,
    ) -> AgentResult:
        """
        Revise a previous output based on feedback.
        
        Default implementation re-runs execute() with feedback prepended.
        Subclasses can override for custom revision logic.
        
        Args:
            original_result: The result to revise
            feedback: Structured feedback on what to improve
            context: Original context (if needed for revision)
            
        Returns:
            New AgentResult with revision
        """
        import time
        start_time = time.time()
        
        # Build revision prompt
        revision_prompt = f"""## Revision Request

You previously produced the following output:

---
{original_result.content}
---

Please revise this output based on the following feedback:

{feedback}

Produce an improved version that addresses all the issues raised.
Maintain the same format and structure as the original.
"""
        
        try:
            response, tokens = await self._call_claude(
                user_message=revision_prompt,
                use_thinking=False,
                max_tokens=32000,
            )
            
            elapsed = time.time() - start_time
            
            # Create revised result
            revised = original_result.with_revision(
                new_content=response,
                feedback=feedback,
            )
            revised.tokens_used = tokens
            revised.execution_time = elapsed
            
            logger.info(
                f"{self.name} revision {revised.iteration} completed "
                f"in {elapsed:.2f}s, {tokens} tokens"
            )
            
            return revised
            
        except Exception as e:
            logger.error(f"{self.name} revision error: {e}")
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=f"Revision failed: {str(e)}",
                execution_time=time.time() - start_time,
                iteration=original_result.iteration + 1,
                previous_versions=original_result.previous_versions + [original_result.content],
            )
    
    async def self_critique(
        self,
        result: AgentResult,
        criteria: Optional[List[str]] = None,
    ) -> dict:
        """
        Self-assess the quality of output.
        
        Returns a dictionary of quality scores and identified issues.
        
        Args:
            result: The result to assess
            criteria: Specific criteria to evaluate (uses defaults if None)
            
        Returns:
            Dict with 'scores' and 'issues' keys
        """
        default_criteria = [
            "accuracy: Are all claims factually correct?",
            "completeness: Are all required elements present?",
            "clarity: Is the content clear and unambiguous?",
            "consistency: Is the content internally consistent?",
            "relevance: Is all content relevant to the task?",
        ]
        
        criteria_to_use = criteria or default_criteria
        criteria_text = "\n".join(f"- {c}" for c in criteria_to_use)
        
        critique_prompt = f"""## Self-Assessment Request

Please critically evaluate the following output:

---
{result.content}
---

Evaluate against these criteria:
{criteria_text}

Respond in this exact JSON format:
{{
    "scores": {{
        "accuracy": 0.0-1.0,
        "completeness": 0.0-1.0,
        "clarity": 0.0-1.0,
        "consistency": 0.0-1.0,
        "relevance": 0.0-1.0,
        "overall": 0.0-1.0
    }},
    "issues": [
        {{
            "category": "accuracy|completeness|clarity|consistency|relevance",
            "severity": "critical|major|minor",
            "description": "Brief description of the issue",
            "suggestion": "How to fix it"
        }}
    ],
    "summary": "One paragraph summary of quality assessment"
}}

Be honest and critical. Do not inflate scores.
"""
        
        try:
            response, _ = await self._call_claude(
                user_message=critique_prompt,
                use_thinking=False,
                max_tokens=4000,
            )
            
            # Parse JSON from response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                critique_data = json.loads(json_match.group())
                return critique_data
            else:
                logger.warning(f"{self.name} self-critique did not return valid JSON")
                return {
                    "scores": {"overall": 0.5},
                    "issues": [],
                    "summary": "Unable to parse self-critique",
                }
                
        except Exception as e:
            logger.error(f"{self.name} self-critique error: {e}")
            return {
                "scores": {"overall": 0.5},
                "issues": [],
                "summary": f"Self-critique failed: {str(e)}",
            }
    
    def supports_revision(self) -> bool:
        """Check if this agent supports revision."""
        return True  # Default to True; subclasses can override
    
    def get_agent_id(self) -> Optional[str]:
        """Get the agent's registry ID if registered."""
        from src.agents.registry import AgentRegistry
        spec = AgentRegistry.get_by_name(self.name)
        return spec.id if spec else None
