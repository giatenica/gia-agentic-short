"""
Overview Generator Agent
========================
Creates a comprehensive research overview document that synthesizes
all agent findings and prepares for the literature review phase.

Uses Sonnet 4.5 for document generation.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import time
import json
from typing import Optional
from datetime import datetime

from .base import BaseAgent, AgentResult
from src.llm.claude_client import TaskType
from loguru import logger


# System prompt for overview generation
OVERVIEW_GENERATOR_PROMPT = """You are an overview generator agent for academic finance research.

Your role is to synthesize all analysis results into a clear, actionable research overview document.

The overview should serve as a comprehensive briefing for starting the literature review phase.

STRUCTURE YOUR OUTPUT AS:

# Research Project Overview

## Executive Summary
Brief summary of the project status and key findings.

## Research Design
- Research Question
- Hypothesis (if applicable)
- Methodology Overview
- Key Variables

## Data Status
- Current data availability
- Data requirements
- Quality assessment

## Gap Analysis Summary
- Critical gaps requiring attention
- Important gaps to address
- Completeness assessment

## Literature Review Preparation
- Key themes to investigate
- Suggested search terms
- Related research streams to explore
- Seminal papers to locate (if mentioned)

## Action Items
Prioritized list of next steps with clear ownership:
1. [Priority] Action item description

## Ready for Literature Review
Clear statement on whether the project is ready to proceed to literature review,
with any prerequisites that need to be addressed first.

---

Write in clear, professional academic language.
Be specific and actionable.
Do not use filler phrases or vague statements.

IMPORTANT:
- Synthesize, do not just repeat
- Highlight key insights and decisions needed
- Make it useful for someone starting the literature review"""


class OverviewGeneratorAgent(BaseAgent):
    """
    Agent that generates comprehensive research overviews.
    
    Uses Sonnet 4.5 for document generation.
    """
    
    def __init__(self, client=None):
        super().__init__(
            name="OverviewGenerator",
            task_type=TaskType.DOCUMENT_CREATION,  # Uses Sonnet
            system_prompt=OVERVIEW_GENERATOR_PROMPT,
            client=client,
        )
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Generate a comprehensive research overview.
        
        Args:
            context: Must contain project_data and all prior agent results
            
        Returns:
            AgentResult with the overview document
        """
        start_time = time.time()
        project_data = context.get("project_data", {})
        
        if not project_data:
            return self._build_result(
                success=False,
                content="",
                error="No project_data provided in context",
            )
        
        # Get all prior agent results
        data_analysis = context.get("data_analysis", {})
        research_analysis = context.get("research_analysis", {})
        gap_analysis = context.get("gap_analysis", {})
        
        # Build comprehensive prompt
        user_message = self._build_overview_prompt(
            project_data,
            data_analysis,
            research_analysis,
            gap_analysis,
        )
        
        try:
            content, tokens = await self._call_claude(user_message)
            
            # Extract literature review themes
            lit_themes = self._extract_literature_themes(project_data, gap_analysis)
            
            return self._build_result(
                success=True,
                content=content,
                structured_data={
                    "generated_at": datetime.now().isoformat(),
                    "project_id": project_data.get("id"),
                    "literature_themes": lit_themes,
                    "ready_for_lit_review": gap_analysis.get("structured_data", {}).get(
                        "ready_for_literature_review", False
                    ),
                    "completeness_score": gap_analysis.get("structured_data", {}).get(
                        "completeness_score", 0
                    ),
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
    
    def _build_overview_prompt(
        self,
        project_data: dict,
        data_analysis: dict,
        research_analysis: dict,
        gap_analysis: dict,
    ) -> str:
        """Build the prompt for overview generation."""
        
        sections = []
        
        # Project basics
        sections.append(f"""PROJECT INFORMATION:
Title: {project_data.get('title', 'Untitled')}
ID: {project_data.get('id', 'N/A')}
Target Journal: {project_data.get('target_journal', 'Not specified')}
Paper Type: {project_data.get('paper_type', 'Not specified')}
Research Type: {project_data.get('research_type', 'Not specified')}
Created: {project_data.get('created_at', 'N/A')[:10] if project_data.get('created_at') else 'N/A'}

Research Question:
{project_data.get('research_question', '[NOT PROVIDED]')}

Hypothesis:
{project_data.get('hypothesis', '[NOT PROVIDED / EXPLORATORY]') if project_data.get('has_hypothesis') else '[EXPLORATORY RESEARCH - NO HYPOTHESIS]'}

Methodology:
{project_data.get('methodology', '[NOT PROVIDED]')}

Key Variables:
{project_data.get('key_variables', '[NOT PROVIDED]')}

Related Literature:
{project_data.get('related_literature', '[NOT PROVIDED]')}

Expected Contribution:
{project_data.get('expected_contribution', '[NOT PROVIDED]')}""")
        
        # Data analysis results
        if data_analysis.get("success"):
            data_struct = data_analysis.get("structured_data", {})
            sections.append(f"""DATA ANALYSIS RESULTS:
Has Data: {'Yes' if data_struct.get('has_data') else 'No'}
Files: {data_struct.get('file_count', 0)}

Analysis:
{data_analysis.get('content', 'No analysis available')}""")
        else:
            sections.append("DATA STATUS: No data analyzed")
        
        # Research exploration results
        if research_analysis.get("success"):
            res_struct = research_analysis.get("structured_data", {})
            assessment = res_struct.get("assessment", {})
            summary = assessment.get("_summary", {})
            sections.append(f"""RESEARCH EXPLORATION RESULTS:
Completeness: {summary.get('provided_count', 0)}/{summary.get('total', 7)} elements provided ({summary.get('completeness_pct', 0)}%)

Analysis:
{research_analysis.get('content', 'No analysis available')}""")
        
        # Gap analysis results
        if gap_analysis.get("success"):
            gap_struct = gap_analysis.get("structured_data", {})
            gaps = gap_struct.get("gaps", {})
            sections.append(f"""GAP ANALYSIS RESULTS:
Completeness Score: {gap_struct.get('completeness_score', 0)}/100
Ready for Literature Review: {'Yes' if gap_struct.get('ready_for_literature_review') else 'No'}

Critical Gaps: {len(gaps.get('critical', []))}
Important Gaps: {len(gaps.get('important', []))}

Full Analysis:
{gap_analysis.get('content', 'No analysis available')}""")
        
        # Constraints
        deadline = project_data.get("deadline", "")
        constraints = project_data.get("constraints", "")
        sections.append(f"""CONSTRAINTS:
Deadline: {deadline if deadline else 'Not specified'}
Other: {constraints if constraints else 'None'}""")
        
        prompt = "\n\n===\n\n".join(sections)
        prompt += """

===

Based on all the above information, generate a comprehensive Research Project Overview document.
The document should synthesize all findings and prepare the project for the literature review phase."""
        
        return prompt
    
    def _extract_literature_themes(self, project_data: dict, gap_analysis: dict) -> list:
        """Extract suggested themes for literature review."""
        
        themes = []
        
        # From research question
        rq = project_data.get("research_question", "")
        if rq:
            # Extract key concepts (simplified extraction)
            # In production, this could use NLP
            themes.append({
                "source": "research_question",
                "theme": "Primary research question concepts",
                "search_focus": rq[:200],
            })
        
        # From related literature
        lit = project_data.get("related_literature", "")
        if lit and len(lit) > 20:
            themes.append({
                "source": "user_provided",
                "theme": "User-identified related literature",
                "search_focus": lit[:300],
            })
        
        # From methodology
        method = project_data.get("methodology", "")
        if method and len(method) > 20:
            themes.append({
                "source": "methodology",
                "theme": "Methodological approaches",
                "search_focus": method[:200],
            })
        
        # From target journal
        journal = project_data.get("target_journal", "")
        if journal and journal != "Other":
            themes.append({
                "source": "target_journal",
                "theme": f"Recent publications in {journal}",
                "search_focus": f"Similar topics published in {journal}",
            })
        
        return themes
    
    def generate_markdown_overview(self, result: AgentResult) -> str:
        """
        Generate a markdown file from the overview result.
        
        Adds metadata header and formatting.
        """
        if not result.success:
            return f"# Overview Generation Failed\n\nError: {result.error}"
        
        struct = result.structured_data
        
        header = f"""---
project_id: {struct.get('project_id', 'unknown')}
generated_at: {struct.get('generated_at', 'unknown')}
completeness_score: {struct.get('completeness_score', 0)}
ready_for_literature_review: {struct.get('ready_for_lit_review', False)}
---

"""
        return header + result.content
