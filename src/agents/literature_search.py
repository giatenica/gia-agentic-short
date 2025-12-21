"""src.agents.literature_search

Literature Search Agent
=======================

This agent is the integration point for the external Edison Scientific API.

What the Edison call does in this workflow
----------------------------------------
Edison receives a natural-language search query (plus optional background context)
and returns two main outputs:

- A narrative, citation-oriented response: Edison-generated text intended to
    resemble a short literature review synthesis.
- A structured citation list: paper metadata that can be converted into BibTeX
    and stored as JSON for later steps.

We persist both into `structured_data` so downstream agents can:
- Generate `LITERATURE_REVIEW.md` and `references.bib`.
- Track provenance and keep the workflow repeatable even though the retrieval is
    performed by an external service.

If Edison is unavailable, the agent exits early to avoid spending LLM tokens on
query formulation, and downstream synthesis generates a scaffold output instead.

Uses Sonnet 4.5 for formulating optimal search queries.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import time
import json
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

from .base import BaseAgent, AgentResult
from src.llm.claude_client import TaskType
from src.llm.edison_client import EdisonClient, LiteratureResult, JobStatus
from loguru import logger


# System prompt for query formulation
QUERY_FORMULATION_PROMPT = """You are a literature search query specialist for academic finance research.

Your role is to formulate optimal search queries for the Edison Scientific API to find relevant academic literature.

EDISON BEST PRACTICES FOR LITERATURE SEARCH:

1. QUERY STRUCTURE
   - Be specific and detailed in your questions
   - Include relevant academic terminology
   - Reference specific methodologies when applicable
   - Mention key variables or measures

2. CONTEXT ENRICHMENT
   - Provide background on the research area
   - Mention related concepts and theories
   - Include time periods if relevant
   - Note any specific journals or authors of interest

3. QUESTION FORMULATION
   - Ask questions that target peer-reviewed academic literature
   - Frame questions to elicit cited, evidence-based responses
   - Include questions about methodological approaches
   - Ask about debates and alternative perspectives

OUTPUT FORMAT:

## Primary Search Query
[A comprehensive, well-structured question for Edison that captures the main hypothesis and research focus]

## Supporting Queries
1. [Theoretical foundations query]
2. [Empirical evidence query]
3. [Methodology query]
4. [Alternative explanations query]

## Search Context
[Background context to provide Edison for more accurate results]

## Focus Areas
[List of specific research areas to prioritize]

IMPORTANT:
- Use precise academic language
- Be specific about what evidence you seek
- Include relevant finance terminology
- Frame queries to elicit comprehensive, cited responses"""


class LiteratureSearchAgent(BaseAgent):
    """
    Agent that searches literature via Edison Scientific API.
    
    Uses Sonnet 4.5 for query formulation and result interpretation.
    """
    
    def __init__(
        self,
        client=None,
        edison_client: Optional[EdisonClient] = None,
        search_timeout: int = 1200,  # 20 minutes default
        max_papers: int = 50,
    ):
        super().__init__(
            name="LiteratureSearcher",
            task_type=TaskType.DATA_ANALYSIS,  # Uses Sonnet
            system_prompt=QUERY_FORMULATION_PROMPT,
            client=client,
        )
        self.edison_client = edison_client or EdisonClient()
        self.search_timeout = search_timeout
        self.max_papers = max_papers
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Search literature based on hypothesis and research context.
        
        Args:
            context: Must contain:
                - 'hypothesis_result': Output from HypothesisDevelopmentAgent
                - OR 'research_overview' and 'literature_questions'
            
        Returns:
            AgentResult with literature search results
        """
        start_time = time.time()
        
        # Extract hypothesis and questions with defensive checks
        hypothesis_result = context.get("hypothesis_result") or {}
        logger.debug(f"hypothesis_result type: {type(hypothesis_result)}")
        logger.debug(f"hypothesis_result keys: {hypothesis_result.keys() if isinstance(hypothesis_result, dict) else 'N/A'}")
        
        structured_data = hypothesis_result.get("structured_data") if isinstance(hypothesis_result, dict) else {}
        structured_data = structured_data or {}
        
        logger.debug(f"structured_data: {structured_data}")
        
        # Get literature questions from hypothesis result or context
        literature_questions = (
            structured_data.get("literature_questions", []) or
            context.get("literature_questions", [])
        )
        
        main_hypothesis = (
            structured_data.get("main_hypothesis") or
            context.get("main_hypothesis", "")
        )
        
        logger.debug(f"main_hypothesis: {main_hypothesis[:100] if main_hypothesis else 'None'}...")
        logger.debug(f"literature_questions: {literature_questions}")
        
        research_overview = context.get("research_overview", "")
        
        if not main_hypothesis and not literature_questions:
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error="No hypothesis or literature questions provided",
                execution_time=time.time() - start_time,
            )

        # If Edison is unavailable, do not spend tokens on query formulation.
        # Let downstream synthesis create a scaffold/placeholder review.
        if hasattr(self.edison_client, "is_available") and not self.edison_client.is_available:
            init_error = getattr(self.edison_client, "init_error", None)
            init_error = init_error or "Edison API client not configured"
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=f"Edison search failed: {init_error}",
                execution_time=time.time() - start_time,
                structured_data={
                    "primary_query": "",
                    "supporting_queries": [],
                    "literature_result": {
                        "query": "",
                        "status": "failed",
                        "error": init_error,
                    },
                    "citations": [],
                    "total_papers": 0,
                },
            )
        
        try:
            # Step 1: Formulate optimal queries using Claude
            logger.info("Formulating search queries...")
            queries = await self._formulate_queries(
                main_hypothesis=main_hypothesis,
                literature_questions=literature_questions,
                research_overview=research_overview,
                context=context,
            )
            
            # Step 2: Search literature using Edison
            logger.info("Submitting literature search to Edison API...")
            primary_query = queries.get("primary_query", "")
            search_context = queries.get("search_context", "")
            
            # Edison is an external retrieval + synthesis service.
            # We store both the narrative response and the structured citations,
            # so later agents can write files (`LITERATURE_REVIEW.md`, BibTeX) and
            # reviewers can see what this stage contributed.
            literature_result = await self.edison_client.search_literature(
                query=primary_query,
                context=search_context,
            )
            
            # Step 3: Check for errors
            if literature_result.status == JobStatus.FAILED:
                return AgentResult(
                    agent_name=self.name,
                    task_type=self.task_type,
                    model_tier=self.model_tier,
                    success=False,
                    content="",
                    error=f"Edison search failed: {literature_result.error}",
                    execution_time=time.time() - start_time,
                )
            
            # Step 4: Return results
            logger.info(f"Edison search completed in {literature_result.processing_time:.1f}s with {len(literature_result.citations)} citations")
            
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content=literature_result.response,
                tokens_used=queries.get("tokens_used", 0),
                execution_time=time.time() - start_time,
                structured_data={
                    "primary_query": primary_query,
                    "supporting_queries": queries.get("supporting_queries", []),
                    "literature_result": literature_result.to_dict(),
                    "citations": [c.to_dict() for c in literature_result.citations],
                    "total_papers": len(literature_result.citations),
                    "edison_processing_time": literature_result.processing_time,
                },
            )
            
        except Exception as e:
            logger.error(f"Literature search error: {e}")
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    async def _formulate_queries(
        self,
        main_hypothesis: str,
        literature_questions: List[str],
        research_overview: str,
        context: dict,
    ) -> Dict[str, Any]:
        """Use Claude to formulate optimal Edison search queries."""
        
        user_message = f"""Please formulate optimal search queries for Edison Scientific API based on this research context.

## MAIN HYPOTHESIS
{main_hypothesis}

## LITERATURE QUESTIONS TO INVESTIGATE
{chr(10).join(f"- {q}" for q in literature_questions) if literature_questions else "None provided"}

## RESEARCH OVERVIEW SUMMARY
{research_overview[:3000] if research_overview else "Not provided"}

## PROJECT DATA
Target Journal: {context.get('project_data', {}).get('target_journal', 'Top finance journal')}
Paper Type: {context.get('project_data', {}).get('paper_type', 'Short article')}

Please generate:
1. A comprehensive primary search query
2. 3-4 supporting queries for different aspects
3. Context to provide Edison for better results
4. Focus areas to prioritize"""
        
        response, tokens = await self._call_claude(
            user_message=user_message,
            use_thinking=False,
            max_tokens=4000,
        )
        
        # Parse the response
        queries = self._parse_queries(response)
        queries["tokens_used"] = tokens
        
        return queries
    
    def _parse_queries(self, response: str) -> Dict[str, Any]:
        """Parse query formulation response."""
        result = {
            "primary_query": "",
            "supporting_queries": [],
            "search_context": "",
            "focus_areas": [],
        }
        
        lines = response.split("\n")
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect sections
            if "Primary Search Query" in line:
                if current_section and current_content:
                    self._save_section(result, current_section, current_content)
                current_section = "primary"
                current_content = []
            elif "Supporting Queries" in line:
                if current_section and current_content:
                    self._save_section(result, current_section, current_content)
                current_section = "supporting"
                current_content = []
            elif "Search Context" in line:
                if current_section and current_content:
                    self._save_section(result, current_section, current_content)
                current_section = "context"
                current_content = []
            elif "Focus Areas" in line:
                if current_section and current_content:
                    self._save_section(result, current_section, current_content)
                current_section = "focus"
                current_content = []
            elif line_stripped.startswith("##"):
                if current_section and current_content:
                    self._save_section(result, current_section, current_content)
                current_section = None
                current_content = []
            elif current_section and line_stripped:
                current_content.append(line_stripped)
        
        # Save final section
        if current_section and current_content:
            self._save_section(result, current_section, current_content)
        
        return result
    
    def _save_section(self, result: dict, section: str, content: List[str]):
        """Save parsed section content."""
        if section == "primary":
            result["primary_query"] = " ".join(content)
        elif section == "supporting":
            for line in content:
                # Remove numbering
                if line[0].isdigit() and ". " in line:
                    line = line.split(". ", 1)[1]
                if line.startswith("- "):
                    line = line[2:]
                if line:
                    result["supporting_queries"].append(line)
        elif section == "context":
            result["search_context"] = " ".join(content)
        elif section == "focus":
            for line in content:
                if line.startswith("- ") or line.startswith("* "):
                    line = line[2:]
                if line:
                    result["focus_areas"].append(line)
