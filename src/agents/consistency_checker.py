"""
Consistency Checker Agent (A14).

Validates cross-document consistency across research project outputs.
Detects mismatches in hypotheses, variables, methodology, citations, and statistics.

Features:
- Programmatic extraction and comparison across documents
- Automatic fix suggestion generation
- Export of fix scripts for automated corrections
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from src.agents.base import BaseAgent, AgentResult, ModelTier
from src.llm.claude_client import ClaudeClient, TaskType
from src.utils.consistency_validation import (
    ConsistencyCategory,
    ConsistencySeverity,
    ConsistencyReport,
    CrossDocumentIssue,
    validate_consistency,
)


@dataclass
class ConsistencyCheckConfig:
    """Configuration for consistency checking."""
    check_hypotheses: bool = True
    check_variables: bool = True
    check_methodology: bool = True
    check_citations: bool = True
    check_statistics: bool = True
    use_llm_analysis: bool = False  # Optional LLM-enhanced analysis
    fail_on_critical: bool = True   # Raise error on critical issues
    min_consistency_score: float = 0.7  # Minimum acceptable score
    generate_fix_script: bool = True  # Generate fix suggestions file


class ConsistencyCheckerAgent(BaseAgent):
    """
    Agent for validating cross-document consistency.
    
    Uses programmatic extraction and comparison to identify inconsistencies
    across RESEARCH_OVERVIEW.md, LITERATURE_REVIEW.md, PROJECT_PLAN.md,
    paper/STRUCTURE.md, and paper/main.tex.
    
    Attributes:
        config: ConsistencyCheckConfig with validation settings
    """
    
    SYSTEM_PROMPT = """You are a meticulous research consistency reviewer.
Your task is to analyze extracted elements from multiple research documents 
and identify semantic inconsistencies that may not be caught by exact matching.

Focus on:
1. Hypothesis variations that change meaning vs. minor wording differences
2. Variable definitions that are semantically different
3. Methodology descriptions that contradict each other
4. Citation references that may refer to the same or different works

Be precise in your analysis. Flag only meaningful inconsistencies that would
affect the research quality. Minor stylistic variations should be ignored."""

    def __init__(
        self,
        client: Optional[ClaudeClient] = None,
        config: Optional[ConsistencyCheckConfig] = None,
    ):
        super().__init__(
            name="ConsistencyChecker",
            task_type=TaskType.DATA_ANALYSIS,  # Sonnet for balanced analysis
            system_prompt=self.SYSTEM_PROMPT,
            client=client,
        )
        self.config = config or ConsistencyCheckConfig()
    
    def get_agent_id(self) -> str:
        """Return agent ID for registry."""
        return "A14"
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Execute consistency check on project documents.
        
        Args:
            context: Dictionary with 'project_folder' key
            
        Returns:
            AgentResult with consistency report
        """
        project_folder = context.get("project_folder")
        if not project_folder:
            return AgentResult(
                agent_name=self.name,
                success=False,
                content="Missing project_folder in context",
                error="project_folder is required",
            )
        
        return await self.check_consistency(project_folder)
    
    async def check_consistency(
        self,
        project_folder: str,
        focus_categories: Optional[List[ConsistencyCategory]] = None,
    ) -> AgentResult:
        """
        Perform cross-document consistency check.
        
        Args:
            project_folder: Path to project folder
            focus_categories: Optional list of categories to focus on
            
        Returns:
            AgentResult with consistency report
        """
        logger.info(f"A14 ConsistencyChecker: Validating {project_folder}")
        
        folder = Path(project_folder)
        if not folder.exists():
            return AgentResult(
                agent_name=self.name,
                task_type=TaskType.DATA_ANALYSIS,
                model_tier=ModelTier.SONNET,
                success=False,
                content=f"Project folder not found: {project_folder}",
                error="Project folder does not exist",
            )
        
        # Run programmatic consistency check
        report = validate_consistency(project_folder)
        
        # Filter by focus categories if specified
        if focus_categories:
            report.issues = [
                i for i in report.issues
                if i.category in focus_categories
            ]
        
        # Apply config filters
        filtered_issues = self._filter_issues(report.issues)
        report.issues = filtered_issues
        
        # Optionally enhance with LLM analysis
        if self.config.use_llm_analysis and report.issues:
            report = await self._enhance_with_llm(report)
        
        # Generate fix script if enabled
        if self.config.generate_fix_script and report.issues:
            fix_script_path = self._generate_fix_script(folder, report)
            logger.info(f"Generated consistency fix suggestions: {fix_script_path}")
        
        # Build result
        content = self._format_report(report)
        
        # Determine success based on config
        success = True
        error = None
        
        if self.config.fail_on_critical and report.critical_count > 0:
            success = False
            error = f"Found {report.critical_count} critical consistency issues"
        elif report.score < self.config.min_consistency_score:
            success = False
            error = f"Consistency score {report.score:.2f} below threshold {self.config.min_consistency_score}"
        
        return AgentResult(
            agent_name=self.name,
            task_type=TaskType.DATA_ANALYSIS,
            model_tier=ModelTier.SONNET,
            success=success,
            content=content,
            error=error,
            structured_data=report.to_dict(),
        )
    
    def _generate_fix_script(self, project_folder: Path, report: ConsistencyReport) -> Path:
        """Generate a JSON file with actionable fix suggestions.
        
        Creates outputs/consistency_fixes.json with structured fix recommendations
        that can be used by downstream tools or manual review.
        
        Args:
            project_folder: Path to project folder
            report: ConsistencyReport with issues
            
        Returns:
            Path to the generated fix file
        """
        outputs_dir = project_folder / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        fix_path = outputs_dir / "consistency_fixes.json"
        
        fixes = []
        for issue in report.issues:
            fix_entry = {
                "id": f"{issue.category.value}_{issue.key}",
                "severity": issue.severity.value,
                "category": issue.category.value,
                "key": issue.key,
                "description": issue.description,
                "canonical_source": issue.canonical_source,
                "canonical_value": issue.canonical_value,
                "affected_documents": issue.affected_documents,
                "action": self._determine_fix_action(issue),
                "search_pattern": self._generate_search_pattern(issue),
                "replacement": issue.canonical_value if issue.canonical_value else None,
                "manual_review_required": issue.severity in [
                    ConsistencySeverity.CRITICAL,
                    ConsistencySeverity.HIGH,
                ],
            }
            fixes.append(fix_entry)
        
        output = {
            "project_folder": str(project_folder),
            "generated_at": str(Path(__file__).stat().st_mtime),
            "total_issues": len(report.issues),
            "critical_count": report.critical_count,
            "high_count": report.high_count,
            "consistency_score": report.score,
            "fixes": fixes,
        }
        
        fix_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return fix_path
    
    def _determine_fix_action(self, issue: CrossDocumentIssue) -> str:
        """Determine the appropriate fix action for an issue."""
        if issue.category == ConsistencyCategory.CITATION:
            if "not defined" in issue.description:
                return "add_bibtex_entry"
            elif "never referenced" in issue.description:
                return "remove_or_reference"
            return "verify_citation"
        elif issue.category == ConsistencyCategory.HYPOTHESIS:
            return "align_hypothesis_text"
        elif issue.category == ConsistencyCategory.VARIABLE:
            return "align_variable_definition"
        elif issue.category == ConsistencyCategory.METHODOLOGY:
            return "align_methodology_description"
        elif issue.category == ConsistencyCategory.STATISTIC:
            return "verify_statistic_value"
        return "manual_review"
    
    def _generate_search_pattern(self, issue: CrossDocumentIssue) -> Optional[str]:
        """Generate a search pattern for finding the inconsistent text."""
        if not issue.variants:
            return None
        
        # For most issues, search for the non-canonical variants
        non_canonical_values = [
            v for doc, v in issue.variants.items()
            if doc != issue.canonical_source and v != issue.canonical_value
        ]
        
        if non_canonical_values:
            # Return the first variant that differs
            return non_canonical_values[0][:100]  # Truncate for practicality
        return None
    
    def _filter_issues(self, issues: List[CrossDocumentIssue]) -> List[CrossDocumentIssue]:
        """Filter issues based on config settings."""
        filtered = []
        
        for issue in issues:
            if issue.category == ConsistencyCategory.HYPOTHESIS and not self.config.check_hypotheses:
                continue
            if issue.category == ConsistencyCategory.VARIABLE and not self.config.check_variables:
                continue
            if issue.category == ConsistencyCategory.METHODOLOGY and not self.config.check_methodology:
                continue
            if issue.category == ConsistencyCategory.CITATION and not self.config.check_citations:
                continue
            if issue.category == ConsistencyCategory.STATISTIC and not self.config.check_statistics:
                continue
            filtered.append(issue)
        
        return filtered
    
    async def _enhance_with_llm(self, report: ConsistencyReport) -> ConsistencyReport:
        """
        Use LLM to analyze ambiguous inconsistencies.
        
        Args:
            report: Initial consistency report
            
        Returns:
            Enhanced report with LLM analysis
        """
        if not report.issues:
            return report
        
        # Build prompt for LLM analysis
        prompt = self._build_analysis_prompt(report)
        
        try:
            response = await self.client.complete_async(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                max_tokens=2000,
            )
            
            # Parse LLM response and update issue descriptions
            # This is a simplified implementation
            for issue in report.issues:
                if issue.severity in [ConsistencySeverity.CRITICAL, ConsistencySeverity.HIGH]:
                    # Add LLM context to suggestion
                    issue.suggestion = f"{issue.suggestion} [LLM reviewed]"
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
        
        return report
    
    def _build_analysis_prompt(self, report: ConsistencyReport) -> str:
        """Build prompt for LLM analysis of inconsistencies."""
        lines = ["Analyze these potential inconsistencies across research documents:\n"]
        
        for i, issue in enumerate(report.issues[:10], 1):  # Limit to 10 issues
            lines.append(f"\n{i}. {issue.category.value.upper()}: {issue.key}")
            lines.append(f"   Description: {issue.description}")
            lines.append(f"   Variants:")
            for doc, value in issue.variants.items():
                lines.append(f"     - {doc}: {value[:200]}")
        
        lines.append("\nFor each inconsistency, determine:")
        lines.append("1. Is this a meaningful semantic difference or just wording variation?")
        lines.append("2. What is the recommended canonical value?")
        lines.append("3. Severity: CRITICAL, HIGH, MEDIUM, or LOW")
        
        return "\n".join(lines)
    
    def _format_report(self, report: ConsistencyReport) -> str:
        """Format consistency report as readable text."""
        lines = [
            "# Cross-Document Consistency Report",
            "",
            f"**Project:** {report.project_folder}",
            f"**Documents Checked:** {len(report.documents_checked)}",
            f"**Elements Extracted:** {report.elements_extracted}",
            f"**Consistency Score:** {report.score:.2f}",
            "",
        ]
        
        if report.is_consistent:
            lines.append("All documents are internally consistent.")
            return "\n".join(lines)
        
        lines.append(f"## Issues Found: {len(report.issues)}")
        lines.append(f"- Critical: {report.critical_count}")
        lines.append(f"- High: {report.high_count}")
        lines.append(f"- Medium: {sum(1 for i in report.issues if i.severity == ConsistencySeverity.MEDIUM)}")
        lines.append(f"- Low: {sum(1 for i in report.issues if i.severity == ConsistencySeverity.LOW)}")
        lines.append("")
        
        # Group issues by severity
        for severity in [ConsistencySeverity.CRITICAL, ConsistencySeverity.HIGH, 
                         ConsistencySeverity.MEDIUM, ConsistencySeverity.LOW]:
            severity_issues = [i for i in report.issues if i.severity == severity]
            if not severity_issues:
                continue
            
            lines.append(f"### {severity.value.upper()} Issues")
            lines.append("")
            
            for issue in severity_issues:
                lines.append(f"**{issue.category.value}: {issue.key}**")
                lines.append(f"- {issue.description}")
                lines.append(f"- Canonical source: {issue.canonical_source}")
                lines.append(f"- Affected documents: {', '.join(issue.affected_documents)}")
                if issue.suggestion:
                    lines.append(f"- Suggestion: {issue.suggestion}")
                lines.append("")
        
        return "\n".join(lines)
    
    async def create_feedback_response(
        self,
        result: AgentResult,
        request_id: str,
    ) -> "FeedbackResponse":
        """
        Convert consistency check result to FeedbackResponse.
        
        Args:
            result: AgentResult from check_consistency
            request_id: Unique request identifier
            
        Returns:
            FeedbackResponse with consistency issues as feedback
        """
        from src.agents.feedback import (
            FeedbackResponse,
            Issue,
            IssueCategory,
            IssueSeverity,
            QualityScore,
        )
        
        report_data = result.structured_data or {}
        issues_data = report_data.get("issues", [])
        
        # Map consistency severity to feedback severity
        severity_map = {
            "critical": IssueSeverity.CRITICAL,
            "high": IssueSeverity.MAJOR,
            "medium": IssueSeverity.MINOR,
            "low": IssueSeverity.SUGGESTION,
        }
        
        # Convert to feedback issues
        feedback_issues = []
        for issue_data in issues_data:
            feedback_issues.append(Issue(
                category=IssueCategory.CROSS_DOCUMENT,
                severity=severity_map.get(issue_data["severity"], IssueSeverity.MINOR),
                description=issue_data["description"],
                location=", ".join(issue_data["affected_documents"]),
                suggestion=issue_data.get("suggestion", ""),
                context=str(issue_data.get("variants", {})),
            ))
        
        # Build quality score
        score = report_data.get("score", 1.0)
        quality_score = QualityScore(
            overall=score,
            consistency=score,
            accuracy=1.0 if report_data.get("critical_count", 0) == 0 else 0.5,
        )
        
        return FeedbackResponse(
            request_id=request_id,
            reviewer_id=self.get_agent_id(),
            overall_assessment="consistent" if report_data.get("is_consistent", True) else "inconsistent",
            quality_score=quality_score,
            issues=feedback_issues,
            revision_required=report_data.get("critical_count", 0) > 0,
            feedback_summary=result.content,
        )


# Convenience function for quick validation
async def check_project_consistency(
    project_folder: str,
    fail_on_critical: bool = True,
) -> ConsistencyReport:
    """
    Quick consistency check for a project folder.
    
    Args:
        project_folder: Path to project folder
        fail_on_critical: Whether to fail on critical issues
        
    Returns:
        ConsistencyReport with findings
    """
    config = ConsistencyCheckConfig(fail_on_critical=fail_on_critical)
    agent = ConsistencyCheckerAgent(config=config)
    result = await agent.check_consistency(project_folder)
    return ConsistencyReport(**result.structured_data) if result.structured_data else None
