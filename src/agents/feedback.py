"""
Feedback Protocol
=================
Standardized data structures for inter-agent communication,
feedback requests, revision triggers, and quality assessments.

This module defines the protocol for:
- Structured feedback between agents
- Quality scoring with severity levels
- Revision triggers and convergence criteria
- Issue tracking and resolution

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class Severity(Enum):
    """Severity levels for identified issues."""
    CRITICAL = "critical"    # Must fix before proceeding
    MAJOR = "major"          # Should fix, significantly impacts quality
    MINOR = "minor"          # Nice to fix, improves quality
    SUGGESTION = "suggestion"  # Optional improvement


class IssueCategory(Enum):
    """Categories of issues that can be identified."""
    # Content issues
    ACCURACY = "accuracy"              # Factual errors, incorrect claims
    COMPLETENESS = "completeness"      # Missing required elements
    CONSISTENCY = "consistency"        # Internal contradictions
    CLARITY = "clarity"                # Unclear or ambiguous content
    
    # Academic quality issues
    CITATION = "citation"              # Missing/incorrect citations
    METHODOLOGY = "methodology"        # Methodological concerns
    LOGIC = "logic"                    # Logical flaws in reasoning
    CONTRIBUTION = "contribution"      # Unclear/weak contribution
    
    # Technical issues
    FORMATTING = "formatting"          # Style/format problems
    DATA = "data"                      # Data-related issues
    CODE = "code"                      # Code quality issues
    
    # Style issues (for writing validation)
    STYLE = "style"                    # Writing style violations
    BANNED_WORDS = "banned_words"      # Prohibited terminology
    WORD_COUNT = "word_count"          # Section length issues
    
    # Cross-document issues (for consistency validation)
    CROSS_DOCUMENT = "cross_document"  # Inconsistency across documents
    HYPOTHESIS_MISMATCH = "hypothesis_mismatch"  # Hypothesis differs across docs
    VARIABLE_MISMATCH = "variable_mismatch"      # Variable definition differs
    
    # Process issues
    SCOPE = "scope"                    # Out of scope content
    DEPENDENCY = "dependency"          # Missing dependencies


@dataclass
class Issue:
    """
    A specific issue identified in agent output.
    
    Attributes:
        category: Type of issue
        severity: How critical the issue is
        description: Detailed description of the issue
        location: Where in the output the issue occurs
        suggestion: How to fix the issue
        affects_downstream: Whether this impacts subsequent agents
    """
    category: IssueCategory
    severity: Severity
    description: str
    location: Optional[str] = None      # e.g., "Section 2.1", "Line 45"
    suggestion: Optional[str] = None    # Recommended fix
    affects_downstream: bool = False    # Impacts other agents?
    
    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "location": self.location,
            "suggestion": self.suggestion,
            "affects_downstream": self.affects_downstream,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Issue":
        return cls(
            category=IssueCategory(data["category"]),
            severity=Severity(data["severity"]),
            description=data["description"],
            location=data.get("location"),
            suggestion=data.get("suggestion"),
            affects_downstream=data.get("affects_downstream", False),
        )


@dataclass
class QualityScore:
    """
    Quality assessment scores for agent output.
    
    Scores are 0.0 to 1.0, where:
    - 0.0-0.3: Poor, requires major revision
    - 0.3-0.6: Acceptable, needs improvement
    - 0.6-0.8: Good, minor issues
    - 0.8-1.0: Excellent, publication ready
    """
    overall: float = 0.0
    accuracy: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    consistency: float = 0.0
    methodology: float = 0.0
    contribution: float = 0.0
    style: float = 0.0  # Writing style compliance
    
    # Threshold for "passing" quality
    PASSING_THRESHOLD: float = 0.7
    
    def passes(self) -> bool:
        """Check if overall score meets threshold."""
        return self.overall >= self.PASSING_THRESHOLD
    
    def lowest_dimension(self) -> tuple[str, float]:
        """Return the dimension with lowest score."""
        dimensions = {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "consistency": self.consistency,
            "methodology": self.methodology,
            "contribution": self.contribution,
            "style": self.style,
        }
        lowest = min(dimensions.items(), key=lambda x: x[1])
        return lowest
    
    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "consistency": self.consistency,
            "methodology": self.methodology,
            "contribution": self.contribution,
            "style": self.style,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QualityScore":
        return cls(
            overall=data.get("overall", 0.0),
            accuracy=data.get("accuracy", 0.0),
            completeness=data.get("completeness", 0.0),
            clarity=data.get("clarity", 0.0),
            consistency=data.get("consistency", 0.0),
            methodology=data.get("methodology", 0.0),
            contribution=data.get("contribution", 0.0),
            style=data.get("style", 0.0),
        )


@dataclass
class FeedbackRequest:
    """
    Request for feedback from one agent to another.
    
    Used when an agent needs another agent to review its output
    or when the orchestrator requests quality assessment.
    """
    request_id: str                        # Unique ID for tracking
    source_agent_id: str                   # Agent requesting feedback
    target_agent_id: str                   # Agent providing feedback
    content: str                           # Content to review
    content_type: str                      # Type of content (hypothesis, literature, etc.)
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    quality_criteria: List[str] = field(default_factory=list)  # Specific criteria to check
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "content": self.content,
            "content_type": self.content_type,
            "context": self.context,
            "quality_criteria": self.quality_criteria,
            "timestamp": self.timestamp,
        }


@dataclass
class FeedbackResponse:
    """
    Response to a feedback request.
    
    Contains quality assessment, identified issues, and
    whether revision is required.
    """
    request_id: str                        # Matches FeedbackRequest
    reviewer_agent_id: str                 # Agent that provided review
    quality_score: QualityScore            # Quality assessment
    issues: List[Issue] = field(default_factory=list)  # Identified issues
    summary: str = ""                      # Human-readable summary
    revision_required: bool = False        # Should source agent revise?
    revision_priority: List[str] = field(default_factory=list)  # What to fix first
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def critical_issues(self) -> List[Issue]:
        """Get only critical issues."""
        return [i for i in self.issues if i.severity == Severity.CRITICAL]
    
    @property
    def major_issues(self) -> List[Issue]:
        """Get only major issues."""
        return [i for i in self.issues if i.severity == Severity.MAJOR]
    
    @property
    def has_blocking_issues(self) -> bool:
        """Check if there are issues that block progress."""
        return len(self.critical_issues) > 0
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "reviewer_agent_id": self.reviewer_agent_id,
            "quality_score": self.quality_score.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
            "revision_required": self.revision_required,
            "revision_priority": self.revision_priority,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackResponse":
        return cls(
            request_id=data["request_id"],
            reviewer_agent_id=data["reviewer_agent_id"],
            quality_score=QualityScore.from_dict(data["quality_score"]),
            issues=[Issue.from_dict(i) for i in data.get("issues", [])],
            summary=data.get("summary", ""),
            revision_required=data.get("revision_required", False),
            revision_priority=data.get("revision_priority", []),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class RevisionTrigger:
    """
    Trigger for an agent to revise its output.
    
    Sent from orchestrator or reviewer to source agent
    with specific feedback to address.
    """
    trigger_id: str                        # Unique ID
    target_agent_id: str                   # Agent that should revise
    original_content: str                  # What was produced
    feedback: FeedbackResponse             # Review feedback
    iteration: int = 1                     # Current iteration number
    max_iterations: int = 3                # Maximum allowed iterations
    focus_areas: List[str] = field(default_factory=list)  # What to prioritize
    previous_revisions: List[str] = field(default_factory=list)  # History
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def can_iterate(self) -> bool:
        """Check if more iterations are allowed."""
        return self.iteration < self.max_iterations
    
    def format_feedback_for_agent(self) -> str:
        """Format feedback into a prompt-friendly string."""
        lines = [
            f"## Revision Request (Iteration {self.iteration}/{self.max_iterations})",
            "",
            f"### Quality Score: {self.feedback.quality_score.overall:.2f}",
            "",
            "### Issues to Address:",
        ]
        
        # Group issues by severity
        for severity in [Severity.CRITICAL, Severity.MAJOR, Severity.MINOR]:
            issues = [i for i in self.feedback.issues if i.severity == severity]
            if issues:
                lines.append(f"\n**{severity.value.upper()}:**")
                for issue in issues:
                    lines.append(f"- [{issue.category.value}] {issue.description}")
                    if issue.suggestion:
                        lines.append(f"  Suggestion: {issue.suggestion}")
        
        if self.focus_areas:
            lines.append("\n### Priority Focus Areas:")
            for area in self.focus_areas:
                lines.append(f"- {area}")
        
        lines.append(f"\n### Summary: {self.feedback.summary}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "trigger_id": self.trigger_id,
            "target_agent_id": self.target_agent_id,
            "original_content": self.original_content,
            "feedback": self.feedback.to_dict(),
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "focus_areas": self.focus_areas,
            "previous_revisions": self.previous_revisions,
            "timestamp": self.timestamp,
        }


@dataclass
class ConvergenceCriteria:
    """
    Criteria for determining when iteration should stop.
    
    Iteration stops when ANY of these conditions is met:
    - Quality score exceeds threshold
    - No critical/major issues remain
    - Maximum iterations reached
    - No improvement between iterations
    """
    quality_threshold: float = 0.8         # Stop if score >= this
    max_iterations: int = 3                # Hard stop after N iterations
    require_no_critical: bool = True       # Stop only if no critical issues
    require_no_major: bool = False         # Stop only if no major issues
    min_improvement: float = 0.05          # Stop if improvement < this
    
    def should_stop(
        self,
        current_score: float,
        previous_score: Optional[float],
        iteration: int,
        critical_count: int,
        major_count: int,
    ) -> tuple[bool, str]:
        """
        Determine if iteration should stop.
        
        Returns:
            (should_stop, reason)
        """
        # Check max iterations
        if iteration >= self.max_iterations:
            return True, f"Maximum iterations ({self.max_iterations}) reached"
        
        # Check quality threshold
        if current_score >= self.quality_threshold:
            return True, f"Quality threshold ({self.quality_threshold}) met"
        
        # Check critical issues
        if self.require_no_critical and critical_count > 0:
            return False, f"{critical_count} critical issues remain"
        
        # Check major issues
        if self.require_no_major and major_count > 0:
            return False, f"{major_count} major issues remain"
        
        # Check improvement
        if previous_score is not None:
            improvement = current_score - previous_score
            if improvement < self.min_improvement:
                return True, f"Insufficient improvement ({improvement:.3f} < {self.min_improvement})"
        
        # Default: continue if we haven't hit threshold
        if current_score < self.quality_threshold:
            return False, "Quality threshold not yet met"
        
        return True, "Convergence criteria satisfied"


@dataclass
class AgentCallRequest:
    """
    Request from one agent to invoke another agent.
    
    Used for inter-agent communication where one agent
    needs another agent's capability.
    """
    call_id: str                           # Unique call ID
    caller_agent_id: str                   # Agent making the request
    target_agent_id: str                   # Agent to invoke
    reason: str                            # Why this call is needed
    context: Dict[str, Any] = field(default_factory=dict)  # Context to pass
    priority: str = "normal"               # normal, high, critical
    timeout_seconds: int = 600             # Max time for call
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "caller_agent_id": self.caller_agent_id,
            "target_agent_id": self.target_agent_id,
            "reason": self.reason,
            "context": self.context,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentCallResponse:
    """
    Response from an inter-agent call.
    """
    call_id: str                           # Matches AgentCallRequest
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
        }
