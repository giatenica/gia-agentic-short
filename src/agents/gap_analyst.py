"""
Gap Analysis Agent
==================
Identifies missing elements needed for a complete research paper
and prioritizes what needs to be addressed.

Uses Opus 4.5 for complex reasoning about research completeness.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import time
import json
from typing import Optional

from .base import BaseAgent, AgentResult
from src.llm.claude_client import TaskType
from loguru import logger


# System prompt for gap analysis
GAP_ANALYSIS_PROMPT = """You are a gap analysis agent for academic finance research papers.

Your role is to identify what is missing or underdeveloped in a research project and create a prioritized action plan.

Based on the research exploration and data analysis provided, you should:

1. IDENTIFY CRITICAL GAPS: What must be addressed before the paper can proceed?
   - Missing research design elements
   - Data requirements not met
   - Unclear identification strategy
   - Undefined variables

2. IDENTIFY IMPORTANT GAPS: What significantly improves the paper but isn't blocking?
   - Literature positioning
   - Robustness considerations
   - Alternative explanations
   - Contribution clarity

3. IDENTIFY NICE-TO-HAVE: What would enhance but isn't essential?
   - Additional tests
   - Extended analysis
   - Supplementary data

4. CREATE ACTION ITEMS: For each gap, specify:
   - What needs to be done
   - Why it matters
   - Suggested approach
   - Priority (critical/important/nice-to-have)

Consider the target journal and paper type when assessing gaps.
A short article (5-10 pages) has different requirements than a full paper.

OUTPUT FORMAT:
Structure your response with clear sections:
- Critical Gaps (must address)
- Important Gaps (should address)
- Enhancement Opportunities (could address)
- Prioritized Action Plan

Be specific and actionable. Vague suggestions are not helpful.

IMPORTANT:
- Focus on what's actually missing based on the provided analysis
- Consider the specific journal requirements
- Provide concrete next steps"""


class GapAnalysisAgent(BaseAgent):
    """
    Agent that identifies gaps in research projects.
    
    Uses Opus 4.5 for complex reasoning about research completeness.
    """
    
    def __init__(self, client=None):
        super().__init__(
            name="GapAnalyst",
            task_type=TaskType.COMPLEX_REASONING,  # Uses Opus
            system_prompt=GAP_ANALYSIS_PROMPT,
            client=client,
        )
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Analyze gaps in the research project.
        
        Args:
            context: Must contain 'project_data' and results from prior agents
            
        Returns:
            AgentResult with gap analysis and action items
        """
        start_time = time.time()
        project_data = context.get("project_data", {})
        
        if not project_data:
            return self._build_result(
                success=False,
                content="",
                error="No project_data provided in context",
            )
        
        # Get prior agent results
        data_analysis = context.get("data_analysis", {})
        research_analysis = context.get("research_analysis", {})
        
        # Build comprehensive context for gap analysis
        user_message = self._build_gap_analysis_prompt(
            project_data, 
            data_analysis, 
            research_analysis
        )
        
        try:
            # Use extended thinking for complex reasoning
            content, tokens = await self._call_claude(user_message, use_thinking=True)
            
            # Extract structured gaps
            gaps = self._extract_gap_structure(
                project_data,
                data_analysis.get("structured_data", {}),
                research_analysis.get("structured_data", {}).get("assessment", {}),
            )
            
            return self._build_result(
                success=True,
                content=content,
                structured_data={
                    "gaps": gaps,
                    "completeness_score": self._calculate_completeness(gaps),
                    "ready_for_literature_review": self._check_lit_review_ready(gaps),
                },
                tokens_used=tokens,
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            return self._build_result(
                success=False,
                content="",
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _build_gap_analysis_prompt(
        self, 
        project_data: dict,
        data_analysis: dict,
        research_analysis: dict,
    ) -> str:
        """Build the prompt for gap analysis."""
        
        sections = []
        
        # Project overview
        sections.append(f"""PROJECT OVERVIEW:
Title: {project_data.get('title', 'Untitled')}
Target Journal: {project_data.get('target_journal', 'Not specified')}
Paper Type: {project_data.get('paper_type', 'Not specified')}
Research Type: {project_data.get('research_type', 'Not specified')}

Research Question:
{project_data.get('research_question', '[NOT PROVIDED]')}""")
        
        # Research exploration results
        if research_analysis.get("success"):
            sections.append(f"""RESEARCH EXPLORATION FINDINGS:
{research_analysis.get('content', 'No analysis available')}

Assessment Summary:
{json.dumps(research_analysis.get('structured_data', {}).get('assessment', {}), indent=2)}""")
        
        # Data analysis results
        if data_analysis.get("success"):
            has_data = data_analysis.get("structured_data", {}).get("has_data", False)
            sections.append(f"""DATA ANALYSIS FINDINGS:
Has Data: {'Yes' if has_data else 'No'}

{data_analysis.get('content', 'No data analysis available')}""")
        else:
            sections.append("DATA STATUS: No data uploaded or analyzed")
        
        # Specific elements status
        elements_status = []
        for element in ["methodology", "key_variables", "related_literature", 
                       "expected_contribution", "data_sources"]:
            value = project_data.get(element, "")
            status = "Provided" if value and len(value.strip()) > 10 else "Missing"
            elements_status.append(f"- {element.replace('_', ' ').title()}: {status}")
        
        sections.append(f"""ELEMENT STATUS:
{chr(10).join(elements_status)}""")
        
        # Constraints
        deadline = project_data.get("deadline", "")
        constraints = project_data.get("constraints", "")
        sections.append(f"""CONSTRAINTS:
Deadline: {deadline if deadline else 'Not specified'}
Other: {constraints if constraints else 'None specified'}""")
        
        prompt = "\n\n---\n\n".join(sections)
        prompt += """

Based on this information, identify all gaps and create a prioritized action plan.
Focus on what's needed to make this a publishable paper in the target journal."""
        
        return prompt
    
    def _extract_gap_structure(
        self,
        project_data: dict,
        data_structured: dict,
        assessment: dict,
    ) -> dict:
        """Extract structured gap information."""
        
        gaps = {
            "critical": [],
            "important": [],
            "nice_to_have": [],
        }
        
        # Check for critical gaps
        has_data = data_structured.get("has_data", False)
        research_type = project_data.get("research_type", "")
        
        # Data is critical for empirical research
        if research_type in ["Empirical", "Mixed", "Experimental"] and not has_data:
            gaps["critical"].append({
                "element": "Data",
                "issue": "No data available for empirical analysis",
                "action": "Acquire data from specified sources or identify alternative data sources",
            })
        
        # Methodology is critical
        if assessment.get("Methodology") == "missing":
            gaps["critical"].append({
                "element": "Methodology",
                "issue": "No methodology specified",
                "action": "Define the empirical or theoretical approach",
            })
        
        # Key variables are critical for empirical work
        if research_type in ["Empirical", "Mixed"] and assessment.get("Key Variables") == "missing":
            gaps["critical"].append({
                "element": "Key Variables",
                "issue": "Variables not defined",
                "action": "Specify dependent and independent variables",
            })
        
        # Important gaps
        if assessment.get("Related Literature") == "missing":
            gaps["important"].append({
                "element": "Literature Foundation",
                "issue": "Related literature not specified",
                "action": "Identify key papers and research streams to position the work",
            })
        
        if assessment.get("Expected Contribution") == "missing":
            gaps["important"].append({
                "element": "Contribution Statement",
                "issue": "Expected contribution not articulated",
                "action": "Define how this paper advances knowledge",
            })
        
        if assessment.get("Data Sources") == "missing" and not has_data:
            gaps["important"].append({
                "element": "Data Sources",
                "issue": "Data sources not identified",
                "action": "Specify where data will be obtained",
            })
        
        # Hypothesis handling
        hyp_status = assessment.get("Hypothesis", "missing")
        if hyp_status == "indicated_but_missing":
            gaps["important"].append({
                "element": "Hypothesis",
                "issue": "Hypothesis indicated but not provided",
                "action": "Formulate testable hypothesis with expected direction",
            })
        
        return gaps
    
    def _calculate_completeness(self, gaps: dict) -> int:
        """Calculate a completeness score (0-100)."""
        critical_count = len(gaps.get("critical", []))
        important_count = len(gaps.get("important", []))
        
        # Start at 100, deduct for gaps
        score = 100
        score -= critical_count * 25  # Critical gaps hurt a lot
        score -= important_count * 10  # Important gaps hurt less
        
        return max(0, min(100, score))
    
    def _check_lit_review_ready(self, gaps: dict) -> bool:
        """Check if ready to proceed to literature review."""
        # Can proceed to lit review if research question is clear
        # and there are no critical gaps related to the question itself
        critical_elements = [g["element"] for g in gaps.get("critical", [])]
        
        # These don't block literature review
        non_blocking = ["Data", "Key Variables"]
        
        blocking_critical = [e for e in critical_elements if e not in non_blocking]
        
        return len(blocking_critical) == 0
