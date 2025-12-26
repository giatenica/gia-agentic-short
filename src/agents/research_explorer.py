"""
Research Explorer Agent
=======================
Analyzes the project submission to understand what the user has provided
and extracts key research components.

Uses Sonnet 4.5 for balanced analysis of research design.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import time
import json
from typing import Optional, Any

from .base import BaseAgent, AgentResult
from src.llm.claude_client import TaskType
from loguru import logger


# System prompt for research exploration
RESEARCH_EXPLORER_PROMPT = """You are a research explorer agent for academic finance research.

Your role is to analyze research project submissions and understand what the researcher has provided.

When given a project submission, you should:

1. SUMMARIZE the research question and its scope
2. ASSESS the clarity and specificity of the hypothesis (if provided)
3. EVALUATE the proposed methodology's appropriateness
4. IDENTIFY the theoretical framework or literature foundation
5. UNDERSTAND the target contribution and novelty
6. NOTE the constraints (journal type, page limits, timeline)

Output a structured analysis with:
- Research Question Analysis: Is it clear, focused, and answerable?
- Hypothesis Assessment: Is it testable and well-grounded?
- Methodology Evaluation: Is the proposed approach suitable?
- Literature Positioning: What research stream does this fit?
- Contribution Clarity: Is the expected contribution well-defined?
- Feasibility Assessment: Given constraints, is this achievable?

Be analytical and constructive. Identify both strengths and areas needing development.

IMPORTANT:
- Base your analysis only on what is provided
- Do not make assumptions about missing information
- Note when key elements are undefined or unclear"""


class ResearchExplorerAgent(BaseAgent):
    """
    Agent that analyzes research project submissions.
    
    Uses Sonnet 4.5 for comprehensive research analysis.
    """
    
    def __init__(self, client: Optional[Any] = None):
        super().__init__(
            name="ResearchExplorer",
            task_type=TaskType.DATA_ANALYSIS,  # Uses Sonnet
            system_prompt=RESEARCH_EXPLORER_PROMPT,
            client=client,
        )
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Analyze the research project submission.
        
        Args:
            context: Must contain 'project_data' dictionary from project.json
            
        Returns:
            AgentResult with research analysis
        """
        start_time = time.time()
        project_data = context.get("project_data", {})
        
        if not project_data:
            return self._build_result(
                success=False,
                content="",
                error="No project_data provided in context",
            )
        
        # Format project data for analysis
        project_summary = self._format_project_for_analysis(project_data)
        
        # Include data analysis if available
        data_analysis = context.get("data_analysis", {})
        data_section = ""
        if data_analysis.get("success"):
            structured = data_analysis.get("structured_data", {}) or {}
            file_count = structured.get("file_count", 0)
            files = structured.get("files", []) or []

            file_lines = []
            for entry in files[:20]:
                rel_path = entry.get("file", "")
                kind = entry.get("type", "")
                rows = entry.get("rows")
                cols = entry.get("columns")
                meta_parts = [rel_path]
                if kind:
                    meta_parts.append(f"type={kind}")
                if rows is not None:
                    meta_parts.append(f"rows={rows}")
                if cols is not None:
                    meta_parts.append(f"cols={cols}")
                file_lines.append("- " + ", ".join([p for p in meta_parts if p]))

            data_text = data_analysis.get("content") or ""
            truncated_data_text = data_text[:1500]
            truncation_note = ""
            if len(data_text) > len(truncated_data_text):
                truncation_note = "\n[Data analyst narrative truncated for prompt size]"

            data_section = (
                "\n\nDATA AVAILABILITY (from DataAnalyst):\n"
                f"File count: {file_count}\n"
                + ("Files (up to 20):\n" + "\n".join(file_lines) + "\n" if file_lines else "")
                + ("\nData analyst narrative (truncated):\n" + truncated_data_text + truncation_note + "\n" if data_text else "")
            )
        
        user_message = f"""Analyze this research project submission:

{project_summary}
{data_section}

Provide a comprehensive analysis of what the researcher has provided and the quality of each element."""

        try:
            content, tokens = await self._call_claude(user_message)
            
            # Extract structured assessment
            assessment = self._extract_assessment(project_data)
            
            return self._build_result(
                success=True,
                content=content,
                structured_data={
                    "assessment": assessment,
                    "project_id": project_data.get("id"),
                    "target_journal": project_data.get("target_journal"),
                    "paper_type": project_data.get("paper_type"),
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
    
    def _format_project_for_analysis(self, project_data: dict) -> str:
        """Format project data for Claude analysis."""
        
        sections = []
        
        # Basic info
        sections.append(f"""PROJECT: {project_data.get('title', 'Untitled')}
ID: {project_data.get('id', 'N/A')}
Created: {project_data.get('created_at', 'N/A')[:10] if project_data.get('created_at') else 'N/A'}""")
        
        # Target publication
        sections.append(f"""TARGET PUBLICATION:
Journal: {project_data.get('target_journal', 'Not specified')}
Paper Type: {project_data.get('paper_type', 'Not specified')}
Research Type: {project_data.get('research_type', 'Not specified')}""")
        
        # Research question
        rq = project_data.get('research_question', '')
        sections.append(f"""RESEARCH QUESTION:
{rq if rq else '[NOT PROVIDED]'}""")
        
        # Hypothesis
        if project_data.get('has_hypothesis'):
            hyp = project_data.get('hypothesis', '')
            sections.append(f"""HYPOTHESIS:
{hyp if hyp else '[INDICATED BUT NOT PROVIDED]'}""")
        else:
            sections.append("HYPOTHESIS: [EXPLORATORY - NO HYPOTHESIS]")
        
        # Methodology
        method = project_data.get('methodology', '')
        sections.append(f"""METHODOLOGY:
{method if method else '[NOT PROVIDED]'}""")
        
        # Key variables
        vars = project_data.get('key_variables', '')
        sections.append(f"""KEY VARIABLES:
{vars if vars else '[NOT PROVIDED]'}""")
        
        # Data sources
        sources = project_data.get('data_sources', '')
        sections.append(f"""DATA SOURCES:
{sources if sources else '[NOT PROVIDED]'}""")
        
        # Related literature
        lit = project_data.get('related_literature', '')
        sections.append(f"""RELATED LITERATURE:
{lit if lit else '[NOT PROVIDED]'}""")
        
        # Expected contribution
        contrib = project_data.get('expected_contribution', '')
        sections.append(f"""EXPECTED CONTRIBUTION:
{contrib if contrib else '[NOT PROVIDED]'}""")
        
        # Constraints
        constraints = project_data.get('constraints', '')
        deadline = project_data.get('deadline', '')
        sections.append(f"""CONSTRAINTS:
{constraints if constraints else '[NONE SPECIFIED]'}
Deadline: {deadline if deadline else '[NOT SPECIFIED]'}""")
        
        # Additional notes
        notes = project_data.get('additional_notes', '')
        if notes:
            sections.append(f"""ADDITIONAL NOTES:
{notes}""")
        
        return "\n\n".join(sections)
    
    def _extract_assessment(self, project_data: dict) -> dict:
        """
        Extract a quick assessment of what's provided vs missing.
        
        Returns dict with element names and their status.
        """
        required_elements = {
            "research_question": "Research Question",
            "hypothesis": "Hypothesis",
            "methodology": "Methodology",
            "key_variables": "Key Variables",
            "data_sources": "Data Sources",
            "related_literature": "Related Literature",
            "expected_contribution": "Expected Contribution",
        }
        
        assessment = {}
        
        for key, name in required_elements.items():
            value = project_data.get(key, "")
            
            if key == "hypothesis":
                # Special handling for hypothesis
                if not project_data.get("has_hypothesis"):
                    assessment[name] = "exploratory"
                elif value and len(value.strip()) > 20:
                    assessment[name] = "provided"
                else:
                    assessment[name] = "indicated_but_missing"
            else:
                if value and len(value.strip()) > 10:
                    assessment[name] = "provided"
                else:
                    assessment[name] = "missing"
        
        # Count provided vs missing
        provided = sum(1 for v in assessment.values() if v == "provided")
        missing = sum(1 for v in assessment.values() if v == "missing")
        
        assessment["_summary"] = {
            "provided_count": provided,
            "missing_count": missing,
            "total": len(required_elements),
            "completeness_pct": round(provided / len(required_elements) * 100, 1),
        }
        
        return assessment
