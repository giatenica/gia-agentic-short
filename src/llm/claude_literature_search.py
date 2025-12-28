"""src.llm.claude_literature_search

Claude Literature Search
========================

A sophisticated, Claude-powered literature search implementation that replaces
Edison Scientific as the primary provider. This implements a 4-stage pipeline
inspired by PaperQA2's Ranking and Contextual Summarization (RCS) approach.

Pipeline Stages:
1. Query Decomposition - Break hypothesis into targeted aspect queries
2. Multi-Source Retrieval - Fetch papers from Semantic Scholar, OpenAlex, arXiv
3. Contextual Summarization - LLM evaluates each paper's relevance with scoring
4. Synthesis & Answer Generation - Generate literature review with citations

Key Features:
- Uses Claude Opus for maximum intelligence on synthesis tasks
- Deduplicates papers across sources
- Ranks papers by relevance before synthesis
- Extracts citable claims with evidence strength ratings
- Generates properly attributed academic prose

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import asyncio
import json
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from src.llm.claude_client import ClaudeClient, TaskType
from src.llm.semantic_scholar_client import (
    SemanticScholarClient,
    SemanticScholarPaper,
)
from src.llm.openalex_client import (
    OpenAlexClient,
    OpenAlexWork,
)
from src.agents.prompts.literature_search import (
    build_query_decomposition_prompt,
    build_contextual_summary_prompt,
    build_evidence_synthesis_prompt,
    build_literature_review_prompt,
)


@dataclass
class LiteratureSearchConfig:
    """Configuration for Claude Literature Search."""
    
    # Retrieval settings
    max_papers_per_source: int = 30  # Papers to fetch from each source
    max_papers_total: int = 50  # Total papers after deduplication
    evidence_k: int = 15  # Papers to evaluate for relevance (PaperQA2 default: 10-15)
    answer_max_sources: int = 8  # Max sources in final synthesis
    
    # Relevance filtering
    min_relevance_score: float = 5.0  # Minimum score to include (0-10 scale)
    
    # Time filters
    year_range: Optional[str] = None  # e.g., "2015-2024", ">2018"
    
    # Model settings
    use_opus_for_synthesis: bool = True  # Use Opus for final synthesis
    use_sonnet_for_evaluation: bool = True  # Use Sonnet for per-paper eval
    
    # Timeouts
    retrieval_timeout: float = 60.0  # Timeout for API calls
    evaluation_timeout: float = 120.0  # Timeout for LLM evaluation
    synthesis_timeout: float = 300.0  # Timeout for final synthesis


@dataclass
class EvaluatedPaper:
    """A paper with relevance evaluation."""
    
    # Basic metadata
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    abstract: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    source: str  # semantic_scholar, openalex, arxiv
    
    # Evaluation results
    relevance_score: float = 0.0
    relevance_rationale: str = ""
    key_findings: List[str] = field(default_factory=list)
    citable_claims: List[Dict[str, Any]] = field(default_factory=list)
    methodology: str = ""
    limitations: List[str] = field(default_factory=list)
    citation_recommendation: str = "optional"  # must-cite, should-cite, optional, exclude
    
    # Open access
    is_open_access: bool = False
    pdf_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "abstract": self.abstract,
            "doi": self.doi,
            "url": self.url,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "relevance_rationale": self.relevance_rationale,
            "key_findings": self.key_findings,
            "citable_claims": self.citable_claims,
            "methodology": self.methodology,
            "limitations": self.limitations,
            "citation_recommendation": self.citation_recommendation,
            "is_open_access": self.is_open_access,
            "pdf_url": self.pdf_url,
        }
    
    def to_citation_dict(self) -> Dict[str, Any]:
        """Convert to format compatible with existing Citation dataclass."""
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year or 0,
            "journal": self.venue,
            "doi": self.doi,
            "url": self.url,
            "abstract": self.abstract,
            "relevance_score": self.relevance_score,
            "paper_id": self.paper_id,
            "citations": None,
        }


@dataclass
class ClaudeLiteratureSearchResult:
    """Result from Claude Literature Search."""
    
    # Core outputs
    response: str  # Literature review text
    citations: List[Dict[str, Any]]  # Citation data for downstream use
    
    # Detailed outputs
    evaluated_papers: List[EvaluatedPaper] = field(default_factory=list)
    evidence_synthesis: Dict[str, Any] = field(default_factory=dict)
    aspect_queries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    total_papers_found: int = 0
    papers_evaluated: int = 0
    papers_included: int = 0
    processing_time: float = 0.0
    tokens_used: int = 0
    
    # Provider info
    provider: str = "claude_literature_search"
    sources_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": "",  # Set by caller
            "response": self.response,
            "citations": self.citations,
            "total_papers_searched": self.total_papers_found,
            "papers_evaluated": self.papers_evaluated,
            "papers_included": self.papers_included,
            "processing_time": self.processing_time,
            "job_id": None,
            "status": "completed",
            "error": None,
            "provider": self.provider,
            "sources_used": self.sources_used,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }


class ClaudeLiteratureSearch:
    """Claude-powered literature search with 4-stage pipeline.
    
    This implementation is designed to replace Edison Scientific as the primary
    literature search provider, offering:
    - Multi-source retrieval (Semantic Scholar, OpenAlex, arXiv)
    - LLM-based relevance evaluation (inspired by PaperQA2's RCS)
    - Evidence synthesis with source attribution
    - Literature review generation
    
    Usage:
        search = ClaudeLiteratureSearch()
        result = await search.search(
            hypothesis="Dual-class shares trade at a premium...",
            questions=["What drives voting premiums?", ...],
            domain="Finance",
        )
    """
    
    def __init__(
        self,
        config: Optional[LiteratureSearchConfig] = None,
        claude_client: Optional[ClaudeClient] = None,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        openalex_client: Optional[OpenAlexClient] = None,
    ):
        """Initialize Claude Literature Search.
        
        Args:
            config: Search configuration
            claude_client: Optional pre-configured Claude client
            semantic_scholar_client: Optional pre-configured S2 client
            openalex_client: Optional pre-configured OpenAlex client
        """
        self.config = config or LiteratureSearchConfig()
        self._claude_client = claude_client
        self._s2_client = semantic_scholar_client
        self._openalex_client = openalex_client
        
    @property
    def claude_client(self) -> ClaudeClient:
        """Get or create Claude client."""
        if self._claude_client is None:
            self._claude_client = ClaudeClient()
        return self._claude_client
    
    @property
    def s2_client(self) -> SemanticScholarClient:
        """Get or create Semantic Scholar client."""
        if self._s2_client is None:
            self._s2_client = SemanticScholarClient()
        return self._s2_client
    
    @property
    def openalex_client(self) -> OpenAlexClient:
        """Get or create OpenAlex client."""
        if self._openalex_client is None:
            self._openalex_client = OpenAlexClient()
        return self._openalex_client
    
    async def close(self) -> None:
        """Close all clients."""
        if self._s2_client:
            await self._s2_client.close()
        if self._openalex_client:
            await self._openalex_client.close()
    
    async def _llm_complete(
        self,
        system_prompt: str,
        user_message: str,
        task_type: TaskType,
        max_tokens: int = 4000,
    ) -> tuple[str, int]:
        """Helper method to call Claude and return (content, tokens).
        
        Wraps chat_async to provide a simple interface for the literature search.
        
        Args:
            system_prompt: System prompt for the LLM
            user_message: User message/query
            task_type: Task type for model selection
            max_tokens: Maximum tokens in response
            
        Returns:
            Tuple of (response_content, tokens_used)
        """
        messages = [{"role": "user", "content": user_message}]
        
        # Get tokens before call
        tokens_before = self.claude_client.usage.total_tokens
        
        content = await self.claude_client.chat_async(
            messages=messages,
            system=system_prompt,
            task=task_type,
            max_tokens=max_tokens,
        )
        
        # Calculate tokens used
        tokens_after = self.claude_client.usage.total_tokens
        tokens_used = tokens_after - tokens_before
        
        return content, tokens_used
    
    async def search(
        self,
        hypothesis: str,
        questions: Optional[List[str]] = None,
        domain: str = "Finance",
        context: str = "",
        journal_style: str = "Top finance journal",
    ) -> ClaudeLiteratureSearchResult:
        """Execute the full 4-stage literature search pipeline.
        
        Args:
            hypothesis: Main research hypothesis
            questions: Specific research questions to investigate
            domain: Research domain (e.g., "Finance", "Economics")
            context: Additional context for search
            journal_style: Target journal style for writing
            
        Returns:
            ClaudeLiteratureSearchResult with literature review and citations
        """
        start_time = time.time()
        total_tokens = 0
        sources_used: List[str] = []
        
        logger.info("Starting Claude Literature Search pipeline")
        
        # Stage 1: Query Decomposition
        logger.info("Stage 1: Decomposing research question into aspect queries")
        aspect_queries, tokens = await self._decompose_queries(
            hypothesis=hypothesis,
            questions=questions or [],
            domain=domain,
            context=context,
        )
        total_tokens += tokens
        logger.info(f"Generated {len(aspect_queries)} aspect queries")
        
        # Stage 2: Multi-Source Retrieval
        logger.info("Stage 2: Retrieving papers from multiple sources")
        papers, retrieved_sources = await self._retrieve_papers(aspect_queries)
        sources_used.extend(retrieved_sources)
        logger.info(f"Retrieved {len(papers)} unique papers from {retrieved_sources}")
        
        if not papers:
            logger.warning("No papers found; returning empty result")
            return ClaudeLiteratureSearchResult(
                response="No relevant literature found for the given research question.",
                citations=[],
                processing_time=time.time() - start_time,
                tokens_used=total_tokens,
                sources_used=sources_used,
            )
        
        # Stage 3: Contextual Summarization
        logger.info(f"Stage 3: Evaluating {min(len(papers), self.config.evidence_k)} papers for relevance")
        evaluated_papers, eval_tokens = await self._evaluate_papers(
            papers=papers[:self.config.evidence_k],
            hypothesis=hypothesis,
        )
        total_tokens += eval_tokens
        
        # Filter by relevance
        relevant_papers = [
            p for p in evaluated_papers
            if p.relevance_score >= self.config.min_relevance_score
        ]
        relevant_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        top_papers = relevant_papers[:self.config.answer_max_sources]
        
        logger.info(f"Selected {len(top_papers)} papers with relevance >= {self.config.min_relevance_score}")
        
        if not top_papers:
            logger.warning("No papers met relevance threshold")
            return ClaudeLiteratureSearchResult(
                response="Literature search found papers but none met the relevance threshold for the research question.",
                citations=[p.to_citation_dict() for p in evaluated_papers[:5]],
                evaluated_papers=evaluated_papers,
                aspect_queries=aspect_queries,
                total_papers_found=len(papers),
                papers_evaluated=len(evaluated_papers),
                papers_included=0,
                processing_time=time.time() - start_time,
                tokens_used=total_tokens,
                sources_used=sources_used,
            )
        
        # Stage 4a: Evidence Synthesis
        logger.info("Stage 4a: Synthesizing evidence")
        evidence_synthesis, synth_tokens = await self._synthesize_evidence(
            hypothesis=hypothesis,
            papers=top_papers,
            aspects=[q.get("aspect", "") for q in aspect_queries],
        )
        total_tokens += synth_tokens
        
        # Stage 4b: Literature Review Generation
        logger.info("Stage 4b: Generating literature review")
        literature_review, review_tokens = await self._generate_literature_review(
            hypothesis=hypothesis,
            evidence_synthesis=evidence_synthesis,
            papers=top_papers,
            journal_style=journal_style,
        )
        total_tokens += review_tokens
        
        processing_time = time.time() - start_time
        logger.info(f"Claude Literature Search completed in {processing_time:.1f}s, {total_tokens} tokens")
        
        return ClaudeLiteratureSearchResult(
            response=literature_review,
            citations=[p.to_citation_dict() for p in top_papers],
            evaluated_papers=evaluated_papers,
            evidence_synthesis=evidence_synthesis,
            aspect_queries=aspect_queries,
            total_papers_found=len(papers),
            papers_evaluated=len(evaluated_papers),
            papers_included=len(top_papers),
            processing_time=processing_time,
            tokens_used=total_tokens,
            sources_used=sources_used,
        )
    
    async def _decompose_queries(
        self,
        hypothesis: str,
        questions: List[str],
        domain: str,
        context: str,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Stage 1: Decompose research question into aspect queries."""
        system_prompt, user_message = build_query_decomposition_prompt(
            hypothesis=hypothesis,
            domain=domain,
            questions=questions,
            context=context,
        )
        
        content, tokens_used = await self._llm_complete(
            system_prompt=system_prompt,
            user_message=user_message,
            task_type=TaskType.DATA_ANALYSIS,  # Use Sonnet for query generation
            max_tokens=2000,
        )
        
        # Parse JSON response
        aspect_queries: List[Dict[str, Any]] = []
        try:
            # Try to extract JSON from response
            text = content
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]
            else:
                json_str = text
            
            data = json.loads(json_str)
            aspect_queries = data.get("aspect_queries", [])
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse query decomposition response: {e}")
            # Fallback: create queries from hypothesis
            aspect_queries = [
                {"aspect": "main", "query": hypothesis, "rationale": "Main hypothesis"},
            ]
            for i, q in enumerate(questions[:3]):
                aspect_queries.append({
                    "aspect": f"question_{i+1}",
                    "query": q,
                    "rationale": "Research question",
                })
        
        return aspect_queries, tokens_used
    
    async def _retrieve_papers(
        self,
        aspect_queries: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Stage 2: Retrieve papers from multiple sources."""
        all_papers: Dict[str, Dict[str, Any]] = {}  # Keyed by title for deduplication
        sources_used: List[str] = []
        
        # Build combined query from aspects
        queries = [q.get("query", "") for q in aspect_queries if q.get("query")]
        
        # Semantic Scholar search
        try:
            for query in queries[:3]:  # Limit to top 3 queries
                result = await self.s2_client.search_papers(
                    query=query,
                    limit=self.config.max_papers_per_source // len(queries[:3]),
                    year=self.config.year_range,
                )
                for paper in result.papers:
                    key = paper.title.lower().strip()
                    if key not in all_papers:
                        all_papers[key] = {
                            "paper_id": paper.paper_id,
                            "title": paper.title,
                            "authors": paper.authors,
                            "year": paper.year,
                            "venue": paper.venue,
                            "abstract": paper.abstract,
                            "doi": paper.doi,
                            "url": paper.url,
                            "citation_count": paper.citation_count,
                            "is_open_access": paper.is_open_access,
                            "pdf_url": paper.open_access_pdf_url,
                            "source": "semantic_scholar",
                        }
            sources_used.append("semantic_scholar")
            logger.debug(f"Semantic Scholar returned {len([p for p in all_papers.values() if p['source'] == 'semantic_scholar'])} papers")
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")
        
        # OpenAlex search
        try:
            for query in queries[:3]:
                result = await self.openalex_client.search_works(
                    query=query,
                    per_page=self.config.max_papers_per_source // len(queries[:3]),
                    publication_year=self.config.year_range,
                )
                for work in result.works:
                    key = work.title.lower().strip()
                    if key not in all_papers:
                        all_papers[key] = {
                            "paper_id": work.openalex_id,
                            "title": work.title,
                            "authors": work.authors,
                            "year": work.year,
                            "venue": work.venue,
                            "abstract": work.abstract,
                            "doi": work.doi,
                            "url": work.url,
                            "citation_count": work.citation_count,
                            "is_open_access": work.is_open_access,
                            "pdf_url": work.pdf_url,
                            "source": "openalex",
                        }
            sources_used.append("openalex")
            logger.debug(f"OpenAlex returned {len([p for p in all_papers.values() if p['source'] == 'openalex'])} papers")
        except Exception as e:
            logger.warning(f"OpenAlex search failed: {e}")
        
        # Sort by citation count (if available) to prioritize influential papers
        papers = list(all_papers.values())
        papers.sort(key=lambda p: p.get("citation_count") or 0, reverse=True)
        
        return papers[:self.config.max_papers_total], sources_used
    
    async def _evaluate_papers(
        self,
        papers: List[Dict[str, Any]],
        hypothesis: str,
    ) -> Tuple[List[EvaluatedPaper], int]:
        """Stage 3: Evaluate each paper for relevance using Claude."""
        evaluated: List[EvaluatedPaper] = []
        total_tokens = 0
        
        # Process papers in batches to avoid rate limits
        batch_size = 5
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            tasks = [
                self._evaluate_single_paper(paper, hypothesis)
                for paper in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for paper, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to evaluate paper '{paper.get('title', 'Unknown')}': {result}")
                    # Create minimal evaluation
                    evaluated.append(EvaluatedPaper(
                        paper_id=paper.get("paper_id", ""),
                        title=paper.get("title", ""),
                        authors=paper.get("authors", []),
                        year=paper.get("year"),
                        venue=paper.get("venue"),
                        abstract=paper.get("abstract"),
                        doi=paper.get("doi"),
                        url=paper.get("url"),
                        source=paper.get("source", "unknown"),
                        relevance_score=3.0,  # Default low score
                        relevance_rationale="Evaluation failed",
                        is_open_access=paper.get("is_open_access", False),
                        pdf_url=paper.get("pdf_url"),
                    ))
                else:
                    eval_paper, tokens = result
                    evaluated.append(eval_paper)
                    total_tokens += tokens
        
        return evaluated, total_tokens
    
    async def _evaluate_single_paper(
        self,
        paper: Dict[str, Any],
        hypothesis: str,
    ) -> Tuple[EvaluatedPaper, int]:
        """Evaluate a single paper for relevance."""
        system_prompt, user_message = build_contextual_summary_prompt(
            research_question=hypothesis,
            title=paper.get("title", ""),
            authors=paper.get("authors", []),
            year=paper.get("year"),
            venue=paper.get("venue", ""),
            abstract=paper.get("abstract", ""),
        )
        
        content, tokens_used = await self._llm_complete(
            system_prompt=system_prompt,
            user_message=user_message,
            task_type=TaskType.DATA_ANALYSIS,  # Use Sonnet for evaluation
            max_tokens=1500,
        )
        
        # Parse response
        eval_data: Dict[str, Any] = {}
        try:
            text = content
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]
            else:
                json_str = text
            eval_data = json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            logger.debug(f"Failed to parse evaluation response: {e}")
            eval_data = {"relevance_score": 5.0, "relevance_rationale": "Parse error"}
        
        return EvaluatedPaper(
            paper_id=paper.get("paper_id", ""),
            title=paper.get("title", ""),
            authors=paper.get("authors", []),
            year=paper.get("year"),
            venue=paper.get("venue"),
            abstract=paper.get("abstract"),
            doi=paper.get("doi"),
            url=paper.get("url"),
            source=paper.get("source", "unknown"),
            relevance_score=float(eval_data.get("relevance_score", 5.0)),
            relevance_rationale=eval_data.get("relevance_rationale", ""),
            key_findings=eval_data.get("key_findings", []),
            citable_claims=eval_data.get("citable_claims", []),
            methodology=eval_data.get("methodology", ""),
            limitations=eval_data.get("limitations", []),
            citation_recommendation=eval_data.get("citation_recommendation", "optional"),
            is_open_access=paper.get("is_open_access", False),
            pdf_url=paper.get("pdf_url"),
        ), tokens_used
    
    async def _synthesize_evidence(
        self,
        hypothesis: str,
        papers: List[EvaluatedPaper],
        aspects: List[str],
    ) -> Tuple[Dict[str, Any], int]:
        """Stage 4a: Synthesize evidence from evaluated papers."""
        papers_data = [p.to_dict() for p in papers]
        
        system_prompt, user_message = build_evidence_synthesis_prompt(
            hypothesis=hypothesis,
            evaluated_papers=papers_data,
            aspects=aspects,
        )
        
        # Use Opus for synthesis if configured
        task_type = TaskType.SCIENTIFIC_ANALYSIS if self.config.use_opus_for_synthesis else TaskType.DATA_ANALYSIS
        
        content, tokens_used = await self._llm_complete(
            system_prompt=system_prompt,
            user_message=user_message,
            task_type=task_type,
            max_tokens=4000,
        )
        
        # Parse response
        synthesis: Dict[str, Any] = {}
        try:
            text = content
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]
            else:
                json_str = text
            synthesis = json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse synthesis response: {e}")
            synthesis = {
                "evidence_for_hypothesis": [],
                "evidence_against_hypothesis": [],
                "synthesis_narrative": content[:2000],
            }
        
        return synthesis, tokens_used
    
    async def _generate_literature_review(
        self,
        hypothesis: str,
        evidence_synthesis: Dict[str, Any],
        papers: List[EvaluatedPaper],
        journal_style: str,
    ) -> Tuple[str, int]:
        """Stage 4b: Generate the final literature review."""
        papers_for_citation = [
            {
                "id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "venue": p.venue,
                "key_findings": p.key_findings,
                "citable_claims": p.citable_claims,
            }
            for p in papers
        ]
        
        system_prompt, user_message = build_literature_review_prompt(
            hypothesis=hypothesis,
            evidence_synthesis=evidence_synthesis,
            papers_for_citation=papers_for_citation,
            journal_style=journal_style,
            word_count=1500,
        )
        
        # Always use Opus for literature review writing
        content, tokens_used = await self._llm_complete(
            system_prompt=system_prompt,
            user_message=user_message,
            task_type=TaskType.ACADEMIC_WRITING,
            max_tokens=6000,
        )
        
        return content, tokens_used


# Convenience function for simple usage
async def claude_literature_search(
    hypothesis: str,
    questions: Optional[List[str]] = None,
    domain: str = "Finance",
) -> ClaudeLiteratureSearchResult:
    """Convenience function for Claude literature search.
    
    Args:
        hypothesis: Research hypothesis
        questions: Optional research questions
        domain: Research domain
        
    Returns:
        ClaudeLiteratureSearchResult
    """
    search = ClaudeLiteratureSearch()
    try:
        return await search.search(
            hypothesis=hypothesis,
            questions=questions,
            domain=domain,
        )
    finally:
        await search.close()
