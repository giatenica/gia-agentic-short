"""
Readiness Assessor Agent (A15).

Performs comprehensive project readiness assessment, tracks time against estimates,
and identifies automation gaps for full autonomous research capability.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from src.agents.base import BaseAgent, AgentResult, ModelTier
from src.llm.claude_client import ClaudeClient, TaskType
from src.utils.time_tracking import (
    TimeTrackingReport,
    TrackedTask,
    TaskStatus,
    TaskLevel,
    initialize_tracking,
    save_tracking_report,
    format_tracking_summary,
)
from src.utils.readiness_scoring import (
    ReadinessReport,
    ChecklistItem,
    PhaseReadiness,
    CheckStatus,
    AutomationCapability,
    assess_project_readiness,
    save_readiness_report,
    format_readiness_summary,
)


@dataclass
class AssessmentConfig:
    """Configuration for readiness assessment."""
    track_time: bool = True
    check_readiness: bool = True
    identify_automation_gaps: bool = True
    generate_summary: bool = True
    use_llm_analysis: bool = False  # Use LLM for deeper analysis
    

@dataclass
class AssessmentResult:
    """Combined assessment result."""
    time_tracking: Optional[TimeTrackingReport] = None
    readiness: Optional[ReadinessReport] = None
    summary: str = ""
    automation_coverage: float = 0.0
    blocking_gaps: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        if self.blocking_gaps is None:
            self.blocking_gaps = []
    
    def to_dict(self) -> dict:
        return {
            "time_tracking": self.time_tracking.to_dict() if self.time_tracking else None,
            "readiness": self.readiness.to_dict() if self.readiness else None,
            "summary": self.summary,
            "automation_coverage": self.automation_coverage,
            "blocking_gaps": self.blocking_gaps,
        }


class ReadinessAssessorAgent(BaseAgent):
    """
    Agent A15: Readiness Assessor.
    
    Performs comprehensive project assessment including:
    - Time tracking against PROJECT_PLAN.md estimates
    - Paper readiness scoring across all phases/steps/substeps
    - Automation gap identification
    - Progress reporting
    """
    
    SYSTEM_PROMPT = """You are a project readiness assessor for autonomous research projects.
Your task is to analyze project completion status, time tracking, and identify automation gaps.

Focus on:
1. Comparing actual execution times against estimated durations
2. Identifying checklist items that are not yet automated
3. Providing clear progress metrics
4. Flagging blocking issues that prevent autonomous completion

Be precise and systematic in your analysis. Report metrics clearly."""

    def __init__(
        self,
        client: Optional[ClaudeClient] = None,
        config: Optional[AssessmentConfig] = None,
    ):
        # Use Haiku tier via DATA_EXTRACTION task type for fast assessment
        super().__init__(
            name="ReadinessAssessor",
            task_type=TaskType.DATA_EXTRACTION,  # Maps to Haiku
            system_prompt=self.SYSTEM_PROMPT,
            client=client,
        )
        self.config = config or AssessmentConfig()
    
    def get_agent_id(self) -> str:
        return "A15"
    
    async def execute(self, context: Optional[dict] = None) -> AgentResult:
        """Execute assessment task."""
        context = context or {}
        project_folder = context.get("project_folder")
        
        if not project_folder:
            return AgentResult(
                agent_name=self.name,
                task_type=TaskType.DATA_ANALYSIS,
                model_tier=ModelTier.HAIKU,
                success=False,
                content="No project folder provided",
                error="project_folder required in context",
            )
        
        return await self.assess_project(project_folder)
    
    async def assess_project(
        self,
        project_folder: str,
        workflow_results: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Perform comprehensive project assessment.
        
        Args:
            project_folder: Path to project folder
            workflow_results: Optional workflow results to incorporate
            
        Returns:
            AgentResult with combined assessment
        """
        logger.info(f"A15 ReadinessAssessor: Assessing {project_folder}")
        
        folder = Path(project_folder)
        if not folder.exists():
            return AgentResult(
                agent_name=self.name,
                task_type=TaskType.DATA_ANALYSIS,
                model_tier=ModelTier.HAIKU,
                success=False,
                content=f"Project folder not found: {project_folder}",
                error="Project folder does not exist",
            )
        
        result = AssessmentResult()
        
        # Time tracking
        if self.config.track_time:
            result.time_tracking = initialize_tracking(project_folder)
            
            # Update with workflow results if provided
            if workflow_results:
                self._update_time_tracking(result.time_tracking, workflow_results)
        
        # Readiness assessment
        if self.config.check_readiness:
            result.readiness = assess_project_readiness(project_folder)
            
            # Update automation coverage
            if result.readiness.total_items > 0:
                result.automation_coverage = (
                    result.readiness.fully_automated_total / 
                    result.readiness.total_items * 100
                )
        
        # Identify blocking gaps
        if self.config.identify_automation_gaps and result.readiness:
            result.blocking_gaps = self._identify_blocking_gaps(result.readiness)
        
        # Generate summary
        if self.config.generate_summary:
            result.summary = self._generate_summary(result)
        
        # Optional LLM analysis for deeper insights
        if self.config.use_llm_analysis:
            llm_analysis = await self._llm_analyze(result)
            if llm_analysis:
                result.summary += f"\n\n## AI Analysis\n{llm_analysis}"
        
        # Save reports
        if result.time_tracking:
            save_tracking_report(result.time_tracking)
        if result.readiness:
            save_readiness_report(result.readiness)
        
        # Save combined assessment
        self._save_assessment(project_folder, result)
        
        return AgentResult(
            agent_name=self.name,
            task_type=TaskType.DATA_ANALYSIS,
            model_tier=ModelTier.HAIKU,
            success=True,
            content=result.summary,
            structured_data=result.to_dict(),
        )
    
    def _update_time_tracking(
        self,
        report: TimeTrackingReport,
        workflow_results: Dict[str, Any],
    ):
        """Update time tracking from workflow results."""
        # Extract agent execution times
        agents = workflow_results.get("agents") or {}
        
        for stage_name, agent_data in agents.items():
            if not isinstance(agent_data, dict):
                continue

            agent_name = agent_data.get("agent_name") or stage_name
            execution_time = agent_data.get("execution_time", 0.0)
            tokens_used = agent_data.get("tokens_used", 0)
            
            if execution_time > 0:
                # Map agent to tasks based on agent type
                task_mapping = self._get_agent_task_mapping()
                if agent_name in task_mapping:
                    for task_id in task_mapping[agent_name]:
                        task = report.get_task(task_id)
                        if task:
                            task.add_execution(agent_name, execution_time, tokens_used)
        
        # Add workflow execution record
        report.add_workflow_execution(
            workflow_name=workflow_results.get("workflow_name", "unknown"),
            total_time=workflow_results.get("total_time", 0.0),
            total_tokens=workflow_results.get("total_tokens", 0),
            agent_times={
                (data.get("agent_name") or name): data.get("execution_time", 0.0)
                for name, data in agents.items()
                if isinstance(data, dict)
            },
        )
    
    def _get_agent_task_mapping(self) -> Dict[str, List[str]]:
        """Map agent names to task IDs they contribute to."""
        return {
            "DataAnalyst": ["Step 1.4", "Step 2.1", "Step 2.2", "Step 2.4"],
            "ResearchExplorer": ["Step 1.1"],
            "GapAnalyst": ["Step 1.1", "Step 1.2"],
            "OverviewGenerator": ["Step 3.1", "Step 3.2", "Step 3.3", "Step 3.4"],
            "HypothesisDeveloper": ["Step 1.2"],
            "LiteratureSearch": ["Step 1.1"],
            "LiteratureSynthesis": ["Step 1.1"],
            "PaperStructure": ["Step 1.3", "Step 4.2"],
            "ProjectPlanner": ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
            "CriticalReviewer": ["Step 4.1"],
            "StyleEnforcer": ["Step 4.1", "Step 4.3"],
            "ConsistencyChecker": ["Step 4.1"],
        }
    
    def _identify_blocking_gaps(self, readiness: ReadinessReport) -> List[Dict[str, Any]]:
        """Identify gaps that block progress toward full automation."""
        blocking = []
        
        for gap in readiness.automation_gaps:
            # Check if this gap blocks downstream items
            item = readiness.get_item(gap["item_id"])
            if item and item.blocks:
                blocking.append({
                    **gap,
                    "blocking_items": item.blocks,
                    "priority": "high",
                })
            elif gap["automation_status"] == "needs_capability":
                blocking.append({
                    **gap,
                    "priority": "medium",
                })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        blocking.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
        
        return blocking
    
    def _generate_summary(self, result: AssessmentResult) -> str:
        """Generate combined assessment summary."""
        lines = [
            "# Project Assessment Summary",
            f"\n**Generated:** {datetime.now().isoformat()}",
            "",
        ]
        
        # Overall status
        if result.readiness:
            lines.extend([
                "## Overall Status",
                f"- **Completion:** {result.readiness.overall_completion:.1f}%",
                f"- **Automation Coverage:** {result.automation_coverage:.1f}%",
                f"- **Items Complete:** {result.readiness.complete_items}/{result.readiness.total_items}",
                "",
            ])
        
        # Time tracking summary
        if result.time_tracking:
            lines.extend([
                "## Time Tracking",
                f"- **Estimated Hours:** {result.time_tracking.total_estimated_hours:.1f}",
                f"- **Actual Hours:** {result.time_tracking.total_actual_hours:.2f}",
            ])
            if result.time_tracking.overall_variance_percent is not None:
                var = result.time_tracking.overall_variance_percent
                status = "under" if var < 0 else "over"
                lines.append(f"- **Variance:** {abs(var):.1f}% {status} estimate")
            lines.append("")
        
        # Phase breakdown
        if result.readiness:
            lines.append("## Phase Progress")
            for phase in result.readiness.phases:
                icon = "✅" if phase.completion_rate >= 1.0 else "🔄" if phase.completion_rate > 0 else "⏳"
                lines.append(
                    f"- {icon} **{phase.phase_id}:** {phase.phase_name} "
                    f"({phase.complete_items}/{phase.total_items}, {phase.completion_rate*100:.0f}%)"
                )
            lines.append("")
        
        # Category scores
        if result.readiness and result.readiness.category_scores:
            lines.append("## Category Scores")
            for category, score in sorted(result.readiness.category_scores.items()):
                bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
                lines.append(f"- **{category.title()}:** {bar} {score:.1f}%")
            lines.append("")
        
        # Automation gaps
        if result.blocking_gaps:
            lines.extend([
                "## Automation Gaps (Blocking)",
                f"**{len(result.blocking_gaps)} items need automation capability:**",
                "",
            ])
            for gap in result.blocking_gaps[:10]:
                priority = gap.get("priority", "medium")
                icon = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
                lines.append(f"- {icon} {gap['description']}")
                if gap.get("required_capabilities"):
                    lines.append(f"  - Needs: {', '.join(gap['required_capabilities'])}")
            if len(result.blocking_gaps) > 10:
                lines.append(f"- ... and {len(result.blocking_gaps) - 10} more")
        
        # Next actions
        lines.extend([
            "",
            "## Next Actions",
        ])
        
        if result.readiness:
            # Find first incomplete phase
            for phase in result.readiness.phases:
                if phase.completion_rate < 1.0:
                    incomplete_items = [
                        i for i in phase.items 
                        if i.status != CheckStatus.COMPLETE
                    ][:3]
                    lines.append(f"\n**Focus on {phase.phase_id}:**")
                    for item in incomplete_items:
                        agent = item.assigned_agent or "Unassigned"
                        lines.append(f"1. {item.description} (Agent: {agent})")
                    break
        
        return '\n'.join(lines)
    
    async def _llm_analyze(self, result: AssessmentResult) -> Optional[str]:
        """Use LLM for deeper analysis of assessment results."""
        if not self.client:
            return None
        
        gaps = result.blocking_gaps or []
        
        prompt = f"""Analyze this research project assessment and provide actionable insights:

## Current Status
- Overall Completion: {result.readiness.overall_completion if result.readiness else 'N/A'}%
- Automation Coverage: {result.automation_coverage:.1f}%
- Blocking Gaps: {len(gaps)}

## Category Scores
{json.dumps(result.readiness.category_scores if result.readiness else {}, indent=2)}

## Blocking Automation Gaps
{json.dumps(gaps[:5], indent=2)}

Provide:
1. Key risks to project completion
2. Priority actions for maximizing automation
3. Capability gaps that need immediate attention
4. Estimated effort to achieve full automation

Be concise and actionable. No preamble."""

        try:
            response = await self.client.chat_async(
                messages=[{"role": "user", "content": prompt}],
                system="You are a research project manager focused on AI automation.",
                max_tokens=1000,
            )
            return response if response else None
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return None
    
    def _save_assessment(self, project_folder: str, result: AssessmentResult):
        """Save combined assessment to project folder."""
        assessment_path = Path(project_folder) / "assessment_report.json"
        try:
            with open(assessment_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Saved assessment report to {assessment_path}")
        except Exception as e:
            logger.error(f"Failed to save assessment: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

async def assess_project(
    project_folder: str,
    workflow_results: Optional[Dict[str, Any]] = None,
    use_llm: bool = False,
) -> AssessmentResult:
    """
    Convenience function to assess a project.
    
    Args:
        project_folder: Path to project folder
        workflow_results: Optional workflow results
        use_llm: Whether to use LLM for deeper analysis
        
    Returns:
        AssessmentResult with time tracking and readiness
    """
    config = AssessmentConfig(use_llm_analysis=use_llm)
    agent = ReadinessAssessorAgent(config=config)
    result = await agent.assess_project(project_folder, workflow_results)
    
    if result.success and result.structured_data:
        # Reconstruct AssessmentResult from structured_data
        data = result.structured_data
        return AssessmentResult(
            time_tracking=TimeTrackingReport.from_dict(data["time_tracking"]) if data.get("time_tracking") else None,
            readiness=ReadinessReport.from_dict(data["readiness"]) if data.get("readiness") else None,
            summary=data.get("summary", ""),
            automation_coverage=data.get("automation_coverage", 0.0),
            blocking_gaps=data.get("blocking_gaps", []),
        )
    
    return AssessmentResult(summary=result.content)


def update_item_status(
    project_folder: str,
    item_id: str,
    status: CheckStatus,
    agent_name: Optional[str] = None,
    completion_percentage: float = 100.0,
):
    """
    Update a checklist item's status.
    
    Call this from agents after completing work.
    """
    from src.utils.readiness_scoring import load_readiness_report, save_readiness_report
    
    report = load_readiness_report(project_folder)
    if not report:
        report = assess_project_readiness(project_folder)
    
    item = report.get_item(item_id)
    if item:
        item.status = status
        item.completion_percentage = completion_percentage
        item.last_checked = datetime.now().isoformat()
        if agent_name:
            item.last_updated_by = agent_name
            if status == CheckStatus.COMPLETE:
                item.automation_capability = AutomationCapability.FULLY_AUTOMATED
        
        save_readiness_report(report)
        logger.info(f"Updated item {item_id} to {status.value} ({completion_percentage}%)")
    else:
        logger.warning(f"Item {item_id} not found in readiness report")
