"""
Time Tracking Utilities for Research Workflow.

Parses duration estimates from PROJECT_PLAN.md, tracks actual agent execution times,
compares estimates vs actuals, and provides comprehensive tracking at phase/step/substep levels.
"""

import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from .project_io import get_project_id


class TaskStatus(Enum):
    """Status of a tracked task."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskLevel(Enum):
    """Hierarchical level of a task."""
    PHASE = "phase"
    STEP = "step"
    SUBSTEP = "substep"


@dataclass
class TimeEstimate:
    """Parsed time estimate from PROJECT_PLAN.md."""
    min_hours: float
    max_hours: float
    source_text: str
    
    @property
    def avg_hours(self) -> float:
        """Average of min and max estimates."""
        return (self.min_hours + self.max_hours) / 2
    
    @property
    def min_seconds(self) -> float:
        """Min estimate in seconds."""
        return self.min_hours * 3600
    
    @property
    def max_seconds(self) -> float:
        """Max estimate in seconds."""
        return self.max_hours * 3600
    
    def to_dict(self) -> dict:
        return {
            "min_hours": self.min_hours,
            "max_hours": self.max_hours,
            "avg_hours": self.avg_hours,
            "source_text": self.source_text,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TimeEstimate":
        return cls(
            min_hours=data["min_hours"],
            max_hours=data["max_hours"],
            source_text=data.get("source_text", ""),
        )


@dataclass
class ExecutionBudget:
    """Time budget for agent execution with warn-only enforcement."""
    budget_seconds: float
    warning_threshold: float = 0.8  # Warn at 80% of budget
    
    def check_budget(self, elapsed_seconds: float, agent_name: str) -> Optional[str]:
        """
        Check if execution is within budget. Returns warning message if exceeded.
        Does NOT abort - warn-only policy.
        """
        if elapsed_seconds > self.budget_seconds:
            overage = elapsed_seconds - self.budget_seconds
            return (
                f"BUDGET EXCEEDED: {agent_name} ran {elapsed_seconds:.1f}s "
                f"(budget: {self.budget_seconds:.1f}s, over by {overage:.1f}s)"
            )
        elif elapsed_seconds > self.budget_seconds * self.warning_threshold:
            remaining = self.budget_seconds - elapsed_seconds
            return (
                f"BUDGET WARNING: {agent_name} at {elapsed_seconds:.1f}s "
                f"({remaining:.1f}s remaining of {self.budget_seconds:.1f}s budget)"
            )
        return None
    
    def to_dict(self) -> dict:
        return {
            "budget_seconds": self.budget_seconds,
            "warning_threshold": self.warning_threshold,
        }


@dataclass
class TrackedTask:
    """A tracked task at any level (phase, step, substep)."""
    task_id: str  # e.g., "Phase 1", "Step 1.1", "Substep 1.1.1"
    title: str
    level: TaskLevel
    parent_id: Optional[str] = None
    
    # Estimated time
    estimate: Optional[TimeEstimate] = None
    
    # Actual execution tracking
    status: TaskStatus = TaskStatus.NOT_STARTED
    actual_seconds: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    # Agent association
    agent_name: Optional[str] = None
    agent_executions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Acceptance criteria tracking
    acceptance_criteria: List[Dict[str, Any]] = field(default_factory=list)
    
    # Priority and dependencies
    priority: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    # Automation tracking
    automation_status: str = "pending"  # pending, automated, needs_capability
    automation_notes: Optional[str] = None
    
    def add_execution(
        self,
        agent_name: str,
        execution_time: float,
        tokens_used: int = 0,
        success: bool = True,
    ):
        """Record an agent execution against this task."""
        self.agent_executions.append({
            "agent_name": agent_name,
            "execution_time": execution_time,
            "tokens_used": tokens_used,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })
        self.actual_seconds += execution_time
        self.agent_name = agent_name
    
    def mark_started(self):
        """Mark task as in progress."""
        self.status = TaskStatus.IN_PROGRESS
        self.start_time = datetime.now().isoformat()
    
    def mark_completed(self):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.end_time = datetime.now().isoformat()
    
    @property
    def variance_percent(self) -> Optional[float]:
        """Percentage variance from estimate (positive = over, negative = under)."""
        if not self.estimate or self.actual_seconds == 0:
            return None
        expected = self.estimate.avg_hours * 3600
        if expected == 0:
            return None
        return ((self.actual_seconds - expected) / expected) * 100
    
    @property
    def actual_hours(self) -> float:
        """Actual execution time in hours."""
        return self.actual_seconds / 3600
    
    @property
    def criteria_completed(self) -> int:
        """Count of completed acceptance criteria."""
        return sum(1 for c in self.acceptance_criteria if c.get("completed", False))
    
    @property
    def criteria_total(self) -> int:
        """Total acceptance criteria."""
        return len(self.acceptance_criteria)
    
    @property
    def criteria_completion_rate(self) -> float:
        """Percentage of acceptance criteria completed."""
        if self.criteria_total == 0:
            return 1.0 if self.status == TaskStatus.COMPLETED else 0.0
        return self.criteria_completed / self.criteria_total
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "level": self.level.value,
            "parent_id": self.parent_id,
            "estimate": self.estimate.to_dict() if self.estimate else None,
            "status": self.status.value,
            "actual_seconds": self.actual_seconds,
            "actual_hours": self.actual_hours,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "agent_name": self.agent_name,
            "agent_executions": self.agent_executions,
            "acceptance_criteria": self.acceptance_criteria,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "automation_status": self.automation_status,
            "automation_notes": self.automation_notes,
            "variance_percent": self.variance_percent,
            "criteria_completed": self.criteria_completed,
            "criteria_total": self.criteria_total,
            "criteria_completion_rate": self.criteria_completion_rate,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrackedTask":
        task = cls(
            task_id=data["task_id"],
            title=data["title"],
            level=TaskLevel(data["level"]),
            parent_id=data.get("parent_id"),
            status=TaskStatus(data.get("status", "not_started")),
            actual_seconds=data.get("actual_seconds", 0.0),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            agent_name=data.get("agent_name"),
            agent_executions=data.get("agent_executions", []),
            acceptance_criteria=data.get("acceptance_criteria", []),
            priority=data.get("priority"),
            dependencies=data.get("dependencies", []),
            automation_status=data.get("automation_status", "pending"),
            automation_notes=data.get("automation_notes"),
        )
        if data.get("estimate"):
            task.estimate = TimeEstimate.from_dict(data["estimate"])
        return task


@dataclass
class TimeTrackingReport:
    """Comprehensive time tracking report for a project."""
    project_id: str
    project_folder: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # All tracked tasks
    tasks: List[TrackedTask] = field(default_factory=list)
    
    # Summary statistics
    total_estimated_hours: float = 0.0
    total_actual_hours: float = 0.0
    
    # Workflow execution tracking
    workflow_executions: List[Dict[str, Any]] = field(default_factory=list)
    
    # ⚡ Bolt Optimization: Use cached_property to avoid re-calculating task lists.
    # These properties filter the main `tasks` list. Without caching, each access
    # (e.g., in `to_dict` or `format_tracking_summary`) would create a new list,
    # leading to unnecessary computation, especially with many tasks.
    @cached_property
    def phases(self) -> List[TrackedTask]:
        """Get all phase-level tasks."""
        return [t for t in self.tasks if t.level == TaskLevel.PHASE]
    
    @cached_property
    def steps(self) -> List[TrackedTask]:
        """Get all step-level tasks."""
        return [t for t in self.tasks if t.level == TaskLevel.STEP]
    
    @cached_property
    def substeps(self) -> List[TrackedTask]:
        """Get all substep-level tasks."""
        return [t for t in self.tasks if t.level == TaskLevel.SUBSTEP]
    
    @cached_property
    def completed_tasks(self) -> List[TrackedTask]:
        """Get all completed tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
    
    @cached_property
    def in_progress_tasks(self) -> List[TrackedTask]:
        """Get tasks currently in progress."""
        return [t for t in self.tasks if t.status == TaskStatus.IN_PROGRESS]
    
    @cached_property
    def not_started_tasks(self) -> List[TrackedTask]:
        """Get tasks not yet started."""
        return [t for t in self.tasks if t.status == TaskStatus.NOT_STARTED]
    
    @property
    def overall_completion_rate(self) -> float:
        """Overall task completion rate."""
        if not self.tasks:
            return 0.0
        return len(self.completed_tasks) / len(self.tasks)
    
    @property
    def overall_variance_percent(self) -> Optional[float]:
        """Overall time variance from estimates."""
        if self.total_estimated_hours == 0:
            return None
        return ((self.total_actual_hours - self.total_estimated_hours) / self.total_estimated_hours) * 100
    
    @property
    def automation_ready_count(self) -> int:
        """Count of tasks marked as automated."""
        return sum(1 for t in self.tasks if t.automation_status == "automated")
    
    @property
    def needs_capability_count(self) -> int:
        """Count of tasks needing additional automation capability."""
        return sum(1 for t in self.tasks if t.automation_status == "needs_capability")
    
    def get_task(self, task_id: str) -> Optional[TrackedTask]:
        """Get task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_children(self, parent_id: str) -> List[TrackedTask]:
        """Get child tasks of a parent."""
        return [t for t in self.tasks if t.parent_id == parent_id]
    
    def add_workflow_execution(
        self,
        workflow_name: str,
        total_time: float,
        total_tokens: int,
        agent_times: Dict[str, float],
    ):
        """Record a workflow execution."""
        self.workflow_executions.append({
            "workflow_name": workflow_name,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "agent_times": agent_times,
            "timestamp": datetime.now().isoformat(),
        })
        self.total_actual_hours += total_time / 3600
    
    def calculate_totals(self):
        """Recalculate summary totals."""
        self.total_estimated_hours = sum(
            t.estimate.avg_hours for t in self.tasks 
            if t.estimate and t.level in [TaskLevel.STEP, TaskLevel.SUBSTEP]
        )
        self.total_actual_hours = sum(
            t.actual_hours for t in self.tasks
        )
    
    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "project_folder": self.project_folder,
            "generated_at": self.generated_at,
            "tasks": [t.to_dict() for t in self.tasks],
            "total_estimated_hours": self.total_estimated_hours,
            "total_actual_hours": self.total_actual_hours,
            "overall_completion_rate": self.overall_completion_rate,
            "overall_variance_percent": self.overall_variance_percent,
            "workflow_executions": self.workflow_executions,
            "summary": {
                "phases": {
                    "total": len(self.phases),
                    "completed": sum(1 for p in self.phases if p.status == TaskStatus.COMPLETED),
                },
                "steps": {
                    "total": len(self.steps),
                    "completed": sum(1 for s in self.steps if s.status == TaskStatus.COMPLETED),
                },
                "substeps": {
                    "total": len(self.substeps),
                    "completed": sum(1 for s in self.substeps if s.status == TaskStatus.COMPLETED),
                },
                "automation": {
                    "automated": self.automation_ready_count,
                    "needs_capability": self.needs_capability_count,
                    "pending": sum(1 for t in self.tasks if t.automation_status == "pending"),
                },
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TimeTrackingReport":
        report = cls(
            project_id=data["project_id"],
            project_folder=data["project_folder"],
            generated_at=data.get("generated_at", datetime.now().isoformat()),
            total_estimated_hours=data.get("total_estimated_hours", 0.0),
            total_actual_hours=data.get("total_actual_hours", 0.0),
            workflow_executions=data.get("workflow_executions", []),
        )
        report.tasks = [TrackedTask.from_dict(t) for t in data.get("tasks", [])]
        return report


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_duration(text: str) -> Optional[TimeEstimate]:
    """
    Parse duration text into TimeEstimate.
    
    Handles formats:
    - "12-15 hours"
    - "2.5 weeks (40-50 hours)"
    - "4 hours"
    - "(4 hours)"
    """
    # Pattern: X-Y hours or X hours
    hours_pattern = r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*hours?'
    single_hours_pattern = r'(\d+(?:\.\d+)?)\s*hours?'
    
    # Try range pattern first
    match = re.search(hours_pattern, text, re.IGNORECASE)
    if match:
        return TimeEstimate(
            min_hours=float(match.group(1)),
            max_hours=float(match.group(2)),
            source_text=text.strip(),
        )
    
    # Try single hours pattern
    match = re.search(single_hours_pattern, text, re.IGNORECASE)
    if match:
        hours = float(match.group(1))
        return TimeEstimate(
            min_hours=hours,
            max_hours=hours,
            source_text=text.strip(),
        )
    
    return None


def parse_acceptance_criteria(text: str) -> List[Dict[str, Any]]:
    """
    Parse acceptance criteria from markdown checkbox format.
    
    Format: - [ ] Criterion text  or  - [x] Completed criterion
    """
    criteria = []
    pattern = r'-\s*\[([ xX])\]\s*(.+?)(?=\n|$)'
    
    for match in re.finditer(pattern, text):
        completed = match.group(1).lower() == 'x'
        description = match.group(2).strip()
        criteria.append({
            "description": description,
            "completed": completed,
        })
    
    return criteria


def parse_project_plan(content: str, project_id: str, project_folder: str) -> TimeTrackingReport:
    """
    Parse PROJECT_PLAN.md content into a TimeTrackingReport.
    
    Extracts all phases, steps, substeps with their estimates and acceptance criteria.
    """
    report = TimeTrackingReport(
        project_id=project_id,
        project_folder=project_folder,
    )
    
    lines = content.split('\n')
    current_phase = None
    current_step = None
    substep_counter = 0
    in_acceptance_criteria = False
    current_criteria_text = []
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Phase detection: ## Phase N: Title
        phase_match = re.match(r'^##\s+(Phase\s+\d+):\s*(.+)$', line_stripped)
        if phase_match:
            # Save previous criteria
            if current_step and current_criteria_text:
                criteria = parse_acceptance_criteria('\n'.join(current_criteria_text))
                current_step.acceptance_criteria = criteria
                current_criteria_text = []
            
            phase_id = phase_match.group(1)
            phase_title = phase_match.group(2).strip()
            
            # Look for duration in next few lines
            estimate = None
            for j in range(i + 1, min(i + 10, len(lines))):
                if '**Duration:**' in lines[j]:
                    estimate = parse_duration(lines[j])
                    break
            
            current_phase = TrackedTask(
                task_id=phase_id,
                title=phase_title,
                level=TaskLevel.PHASE,
                estimate=estimate,
            )
            report.tasks.append(current_phase)
            current_step = None
            substep_counter = 0
            in_acceptance_criteria = False
            continue
        
        # Step detection: ### Step X.Y: Title
        step_match = re.match(r'^###\s+(Step\s+\d+\.\d+):\s*(.+)$', line_stripped)
        if step_match:
            # Save previous criteria
            if current_step and current_criteria_text:
                criteria = parse_acceptance_criteria('\n'.join(current_criteria_text))
                current_step.acceptance_criteria = criteria
                current_criteria_text = []
            
            step_id = step_match.group(1)
            step_title = step_match.group(2).strip()
            
            # Look for duration and priority in next few lines
            estimate = None
            priority = None
            dependencies = []
            
            for j in range(i + 1, min(i + 10, len(lines))):
                if '**Duration:**' in lines[j]:
                    estimate = parse_duration(lines[j])
                elif '**Priority:**' in lines[j]:
                    priority_match = re.search(r'\*\*Priority:\*\*\s*(\w+)', lines[j])
                    if priority_match:
                        priority = priority_match.group(1)
                elif '**Dependency:**' in lines[j] or '**Dependencies:**' in lines[j]:
                    dep_match = re.search(r'Requires?\s+(.+?)(?:\s+completion)?$', lines[j])
                    if dep_match:
                        dependencies.append(dep_match.group(1).strip())
            
            current_step = TrackedTask(
                task_id=step_id,
                title=step_title,
                level=TaskLevel.STEP,
                parent_id=current_phase.task_id if current_phase else None,
                estimate=estimate,
                priority=priority,
                dependencies=dependencies,
            )
            report.tasks.append(current_step)
            substep_counter = 0
            in_acceptance_criteria = False
            continue
        
        # Substep detection: N. **Title (X hours)**
        substep_match = re.match(r'^(\d+)\.\s+\*\*(.+?)\*\*', line_stripped)
        if substep_match and current_step:
            substep_num = substep_match.group(1)
            substep_text = substep_match.group(2)
            
            # Extract hours from substep text
            estimate = parse_duration(substep_text)
            
            # Clean title (remove hours part)
            title = re.sub(r'\s*\(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*hours?\)', '', substep_text).strip()
            
            substep_id = f"{current_step.task_id}.{substep_num}"
            
            substep = TrackedTask(
                task_id=substep_id,
                title=title,
                level=TaskLevel.SUBSTEP,
                parent_id=current_step.task_id,
                estimate=estimate,
            )
            report.tasks.append(substep)
            continue
        
        # Acceptance Criteria section
        if '**Acceptance Criteria:**' in line or 'Acceptance Criteria:' in line:
            in_acceptance_criteria = True
            current_criteria_text = []
            continue
        
        # Collect criteria lines
        if in_acceptance_criteria:
            if line_stripped.startswith('- ['):
                current_criteria_text.append(line_stripped)
            elif line_stripped.startswith('---') or line_stripped.startswith('##'):
                # End of criteria section
                if current_step and current_criteria_text:
                    criteria = parse_acceptance_criteria('\n'.join(current_criteria_text))
                    current_step.acceptance_criteria = criteria
                current_criteria_text = []
                in_acceptance_criteria = False
    
    # Handle final criteria
    if current_step and current_criteria_text:
        criteria = parse_acceptance_criteria('\n'.join(current_criteria_text))
        current_step.acceptance_criteria = criteria
    
    # Calculate totals
    report.calculate_totals()
    
    logger.info(
        f"Parsed PROJECT_PLAN.md: {len(report.phases)} phases, "
        f"{len(report.steps)} steps, {len(report.substeps)} substeps, "
        f"total estimate: {report.total_estimated_hours:.1f} hours"
    )
    
    return report


def load_tracking_report(project_folder: str) -> Optional[TimeTrackingReport]:
    """Load existing tracking report from project folder."""
    report_path = Path(project_folder) / "time_tracking_report.json"
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)
            return TimeTrackingReport.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load tracking report: {e}")
    return None


def save_tracking_report(report: TimeTrackingReport):
    """Save tracking report to project folder."""
    report_path = Path(report.project_folder) / "time_tracking_report.json"
    try:
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved time tracking report to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save tracking report: {e}")


def initialize_tracking(project_folder: str) -> TimeTrackingReport:
    """
    Initialize time tracking for a project.
    
    Loads existing report or creates new one from PROJECT_PLAN.md.
    """
    # Try to load existing report
    existing = load_tracking_report(project_folder)
    if existing:
        logger.info(f"Loaded existing tracking report for {project_folder}")
        return existing
    
    # Parse PROJECT_PLAN.md
    plan_path = Path(project_folder) / "PROJECT_PLAN.md"
    if not plan_path.exists():
        logger.warning(f"PROJECT_PLAN.md not found in {project_folder}")
        # Return empty report
        project_id = get_project_id(project_folder)
        return TimeTrackingReport(project_id=project_id, project_folder=project_folder)
    
    project_id = get_project_id(project_folder)
    
    with open(plan_path, 'r') as f:
        content = f.read()
    
    report = parse_project_plan(content, project_id, project_folder)
    
    # Save initial report
    save_tracking_report(report)
    
    return report


def update_task_status(
    report: TimeTrackingReport,
    task_id: str,
    status: TaskStatus,
    agent_name: Optional[str] = None,
    execution_time: float = 0.0,
    tokens_used: int = 0,
) -> bool:
    """Update a task's status and execution metrics."""
    task = report.get_task(task_id)
    if not task:
        logger.warning(f"Task {task_id} not found in report")
        return False
    
    task.status = status
    
    if status == TaskStatus.IN_PROGRESS:
        task.mark_started()
    elif status == TaskStatus.COMPLETED:
        task.mark_completed()
    
    if agent_name and execution_time > 0:
        task.add_execution(agent_name, execution_time, tokens_used)
    
    # Update parent status if all children complete
    if task.parent_id and status == TaskStatus.COMPLETED:
        children = report.get_children(task.parent_id)
        if all(c.status == TaskStatus.COMPLETED for c in children):
            parent = report.get_task(task.parent_id)
            if parent:
                parent.status = TaskStatus.COMPLETED
                parent.mark_completed()
    
    save_tracking_report(report)
    return True


def format_tracking_summary(report: TimeTrackingReport) -> str:
    """Format a human-readable tracking summary."""
    lines = [
        "# Time Tracking Summary",
        f"\n**Project:** {report.project_id}",
        f"**Generated:** {report.generated_at}",
        "",
        "## Overall Progress",
        f"- **Completion Rate:** {report.overall_completion_rate * 100:.1f}%",
        f"- **Estimated Hours:** {report.total_estimated_hours:.1f}",
        f"- **Actual Hours:** {report.total_actual_hours:.2f}",
    ]
    
    if report.overall_variance_percent is not None:
        variance = report.overall_variance_percent
        status = "under" if variance < 0 else "over"
        lines.append(f"- **Variance:** {abs(variance):.1f}% {status} estimate")
    
    lines.extend([
        "",
        "## Phase Progress",
    ])
    
    for phase in report.phases:
        status_icon = "✅" if phase.status == TaskStatus.COMPLETED else "🔄" if phase.status == TaskStatus.IN_PROGRESS else "⏳"
        lines.append(f"\n### {status_icon} {phase.task_id}: {phase.title}")
        
        if phase.estimate:
            lines.append(f"- Estimate: {phase.estimate.min_hours}-{phase.estimate.max_hours} hours")
        lines.append(f"- Actual: {phase.actual_hours:.2f} hours")
        lines.append(f"- Status: {phase.status.value}")
        
        # Show steps
        steps = report.get_children(phase.task_id)
        for step in steps:
            step_icon = "✅" if step.status == TaskStatus.COMPLETED else "🔄" if step.status == TaskStatus.IN_PROGRESS else "⏳"
            lines.append(f"\n  #### {step_icon} {step.task_id}: {step.title}")
            if step.estimate:
                lines.append(f"  - Estimate: {step.estimate.min_hours}-{step.estimate.max_hours} hours")
            lines.append(f"  - Actual: {step.actual_hours:.2f} hours")
            lines.append(f"  - Criteria: {step.criteria_completed}/{step.criteria_total} ({step.criteria_completion_rate*100:.0f}%)")
            if step.agent_name:
                lines.append(f"  - Agent: {step.agent_name}")
    
    lines.extend([
        "",
        "## Automation Status",
        f"- Automated: {report.automation_ready_count}",
        f"- Needs Capability: {report.needs_capability_count}",
        f"- Pending: {sum(1 for t in report.tasks if t.automation_status == 'pending')}",
    ])
    
    return '\n'.join(lines)
