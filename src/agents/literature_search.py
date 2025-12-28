"""src.agents.literature_search

Literature Search Agent
=======================

This agent provides literature search via multiple providers, with Claude Literature
Search as the primary provider and Edison Scientific as an expensive fallback.

Provider Priority:
1. Claude Literature Search (primary) - Free, uses Semantic Scholar + OpenAlex + Claude
2. Semantic Scholar (fallback) - Direct API access
3. arXiv (fallback) - Preprint search
4. Edison Scientific (expensive fallback) - Paid service
5. Manual sources list (final fallback)

What each provider does:
- Claude Literature Search: 4-stage pipeline with query decomposition, multi-source
    retrieval, contextual summarization, and evidence synthesis using Claude.
- Edison: External API returning narrative synthesis and structured citations.
- Semantic Scholar/arXiv: Direct paper metadata retrieval without synthesis.
- Manual: Local sources.json file for user-provided references.

We persist outputs into `structured_data` so downstream agents can:
- Generate `LITERATURE_REVIEW.md` and `references.bib`.
- Track provenance and keep the workflow repeatable.

Uses Opus 4.5 for literature synthesis, Sonnet 4.5 for query formulation.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import time
import json
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree

from .base import BaseAgent, AgentResult
from src.llm.claude_client import TaskType
from src.llm.edison_client import EdisonClient, LiteratureResult, JobStatus
from src.llm.claude_literature_search import (
    ClaudeLiteratureSearch,
    ClaudeLiteratureSearchResult,
    LiteratureSearchConfig,
)
from src.evidence.acquisition import find_default_sources_list_path
import httpx
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)


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


# Retry configuration for transient HTTP errors (429 rate limit, 500/503 server errors)
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Check if an exception is a retryable HTTP error."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_HTTP_STATUSES
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


class LiteratureSearchAgent(BaseAgent):
    """
    Agent that searches literature via multiple providers.
    
    Provider chain:
    1. Claude Literature Search (primary) - 4-stage pipeline with LLM synthesis
    2. Semantic Scholar (fallback)
    3. arXiv (fallback)  
    4. Edison Scientific (expensive fallback) - Only if configured and others fail
    5. Manual sources list (final fallback)
    
    Uses Opus 4.5 for synthesis, Sonnet 4.5 for query formulation.
    """
    
    def __init__(
        self,
        client=None,
        edison_client: Optional[EdisonClient] = None,
        claude_search: Optional[ClaudeLiteratureSearch] = None,
        search_timeout: int = 1200,  # 20 minutes default
        max_papers: int = 50,
        use_edison_fallback: bool = True,  # Whether to use Edison as expensive fallback
    ):
        super().__init__(
            name="LiteratureSearcher",
            task_type=TaskType.DATA_ANALYSIS,  # Uses Sonnet
            system_prompt=QUERY_FORMULATION_PROMPT,
            client=client,
        )
        self.edison_client = edison_client or EdisonClient()
        self.claude_search = claude_search or ClaudeLiteratureSearch(
            config=LiteratureSearchConfig(
                max_papers_total=max_papers,
                evidence_k=15,
                answer_max_sources=8,
            )
        )
        self.search_timeout = search_timeout
        self.max_papers = max_papers
        self.use_edison_fallback = use_edison_fallback

    def _build_fallback_query(self, *, main_hypothesis: str, literature_questions: List[str]) -> str:
        q = str((main_hypothesis or "").strip())
        if not q:
            for item in literature_questions:
                if isinstance(item, str) and item.strip():
                    q = item.strip()
                    break
        if not q:
            q = "literature search"
        return q

    def _provider_attempt(
        self,
        *,
        provider: str,
        ok: bool,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "provider": provider,
            "ok": bool(ok),
            "error_type": error_type,
            "error_message": error_message,
            "duration_seconds": duration_seconds,
        }

    def _as_citation_dict(self, *, title: str, authors: Optional[List[str]] = None, year: Optional[int] = None,
                          journal: Optional[str] = None, doi: Optional[str] = None, url: Optional[str] = None,
                          abstract: Optional[str] = None, paper_id: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(year, int):
            y = year
        elif isinstance(year, str):
            try:
                y = int(year)
            except ValueError:
                y = 0
        else:
            y = 0
        return {
            "title": str(title or ""),
            "authors": list(authors or []),
            "year": y,
            "journal": journal,
            "doi": doi,
            "url": url,
            "abstract": abstract,
            "relevance_score": None,
            "paper_id": paper_id,
            "citations": None,
        }

    async def _search_via_semantic_scholar(self, *, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Best-effort Semantic Scholar Graph API search with retry logic.

        Uses exponential backoff for rate limits (429) and server errors (500/503).
        Returns a tuple of (response_text, citations_data).
        """

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            before_sleep=lambda retry_state: logger.warning(
                f"Semantic Scholar retry {retry_state.attempt_number}/3 after error"
            ),
            reraise=True,
        )
        async def _do_request():
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": min(int(self.max_papers), 50),
                "fields": "title,authors,year,venue,url,abstract,externalIds,publicationVenue",
            }
            timeout = httpx.Timeout(20.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, params=params)
                # Handle rate limits with retry
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After", "5")
                    try:
                        wait_seconds = int(retry_after)
                    except ValueError:
                        wait_seconds = 5
                    logger.warning(f"Semantic Scholar rate limited, waiting {wait_seconds}s")
                    await asyncio.sleep(wait_seconds)
                    raise httpx.HTTPStatusError(
                        f"Rate limited (429)", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp.json()

        try:
            payload = await _do_request()
        except RetryError as e:
            logger.error(f"Semantic Scholar search failed after retries: {e}")
            raise

        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return ("", [])

        citations: List[Dict[str, Any]] = []
        for p in items:
            if not isinstance(p, dict):
                continue
            title = str(p.get("title") or "").strip()
            if not title:
                continue
            authors_payload = p.get("authors")
            authors: List[str] = []
            if isinstance(authors_payload, list):
                for a in authors_payload:
                    if isinstance(a, dict):
                        name = a.get("name")
                        if isinstance(name, str) and name.strip():
                            authors.append(name.strip())
            year = p.get("year")
            year_int = int(year) if isinstance(year, int) else 0
            doi = None
            ext = p.get("externalIds")
            if isinstance(ext, dict):
                d = ext.get("DOI")
                if isinstance(d, str) and d.strip():
                    doi = d.strip()
            venue = p.get("venue")
            journal = str(venue).strip() if isinstance(venue, str) and venue.strip() else None
            url_value = p.get("url")
            url_str = str(url_value).strip() if isinstance(url_value, str) and url_value.strip() else None
            abstract = p.get("abstract")
            abstract_str = str(abstract).strip() if isinstance(abstract, str) and abstract.strip() else None
            citations.append(
                self._as_citation_dict(
                    title=title,
                    authors=authors,
                    year=year_int,
                    journal=journal,
                    doi=doi,
                    url=url_str,
                    abstract=abstract_str,
                    paper_id=str(p.get("paperId")) if p.get("paperId") is not None else None,
                )
            )

        response_text = (
            "Semantic Scholar fallback search executed. This output is a provisional list of candidate papers; "
            "verify metadata against original sources before making definitive claims."
        )
        return (response_text, citations)

    async def _search_via_arxiv(self, *, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Best-effort arXiv API search with retry logic.

        Uses exponential backoff for server errors (500/503) and timeouts.
        Returns a tuple of (response_text, citations_data).
        """

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            before_sleep=lambda retry_state: logger.warning(
                f"arXiv retry {retry_state.attempt_number}/3 after error"
            ),
            reraise=True,
        )
        async def _do_request():
            url = "https://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": min(int(self.max_papers), 25),
            }
            timeout = httpx.Timeout(30.0, connect=15.0)  # Increased timeout for arXiv
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, params=params)
                # arXiv returns 500 for server overload; retry these
                if resp.status_code in RETRYABLE_HTTP_STATUSES:
                    logger.warning(f"arXiv returned {resp.status_code}, will retry")
                    raise httpx.HTTPStatusError(
                        f"arXiv server error ({resp.status_code})",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                return resp.text

        try:
            text = await _do_request()
        except RetryError as e:
            logger.error(f"arXiv search failed after retries: {e}")
            raise

        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError:
            return ("", [])

        ns = {"atom": "http://www.w3.org/2005/Atom"}

        citations: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            if not title:
                continue

            authors: List[str] = []
            for a in entry.findall("atom:author", ns):
                name = (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
                if name:
                    authors.append(name)

            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
            year_int = 0
            if published:
                try:
                    year_int = int(published[:4])
                except Exception:
                    year_int = 0

            url_str: Optional[str] = None
            id_text = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            if id_text:
                url_str = id_text

            abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()

            citations.append(
                self._as_citation_dict(
                    title=title,
                    authors=authors,
                    year=year_int,
                    journal="arXiv",
                    doi=None,
                    url=url_str,
                    abstract=abstract or None,
                )
            )

        response_text = (
            "arXiv fallback search executed. This output is a provisional list of candidate papers; "
            "verify metadata against original sources before making definitive claims."
        )
        return (response_text, citations)

    def _search_via_manual_sources_list(self, *, project_folder: Optional[str]) -> Tuple[str, List[Dict[str, Any]]]:
        """Fallback to a manual sources list under the project folder."""

        if not isinstance(project_folder, str) or not project_folder:
            return ("", [])

        rel = find_default_sources_list_path(project_folder)
        if not rel:
            return ("", [])

        pf = Path(project_folder).expanduser().resolve()
        list_path = (pf / rel).resolve()
        # Ensure the resolved list_path is still within the project folder to avoid path traversal
        if not list_path.is_relative_to(pf):
            return ("", [])
        if not list_path.exists() or not list_path.is_file():
            return ("", [])

        try:
            specs = json.loads(list_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed to load manual sources list from '{list_path}': {exc}")
            return ("", [])

        if not isinstance(specs, list):
            return ("", [])

        citations: List[Dict[str, Any]] = []
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            kind = str(spec.get("kind") or "").strip().lower()
            if kind == "arxiv":
                arxiv_id = str(spec.get("id") or "").strip()
                arxiv_url = str(spec.get("url") or "").strip()
                value = arxiv_id or arxiv_url
                if not value:
                    continue
                title = f"arXiv source: {value}"
                citations.append(self._as_citation_dict(title=title, authors=[], year=0, journal="arXiv", url=value))
                continue

            if kind in {"pdf_url", "html_url"}:
                u = str(spec.get("url") or "").strip()
                if not u:
                    continue
                title = f"Manual source: {Path(u).name or u}"
                citations.append(self._as_citation_dict(title=title, authors=[], year=0, journal=None, url=u))
                continue

        response_text = (
            "Manual sources list fallback used. This output reflects locally provided sources and may not be exhaustive."
        )
        return (response_text, citations)
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Search literature based on hypothesis and research context.
        
        Provider chain:
        1. Claude Literature Search (primary) - 4-stage pipeline with LLM synthesis
        2. Semantic Scholar (fallback)
        3. arXiv (fallback)
        4. Edison Scientific (expensive fallback) - Only if configured
        5. Manual sources list (final fallback)
        
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

        provider_attempts: List[Dict[str, Any]] = []
        used_provider: Optional[str] = None
        fallback_query = self._build_fallback_query(
            main_hypothesis=main_hypothesis,
            literature_questions=literature_questions,
        )
        
        try:
            queries: Dict[str, Any] = {
                "primary_query": main_hypothesis or fallback_query,
                "supporting_queries": literature_questions,
                "search_context": research_overview,
                "tokens_used": 0,
            }

            # ========================================================
            # Provider 1: Claude Literature Search (Primary)
            # ========================================================
            logger.info("Attempting Claude Literature Search (primary provider)...")
            t0 = time.time()
            try:
                claude_result = await self.claude_search.search(
                    hypothesis=main_hypothesis or fallback_query,
                    questions=literature_questions if isinstance(literature_questions, list) else [],
                    domain=context.get("project_data", {}).get("research_domain", "Finance"),
                    context=research_overview[:3000] if research_overview else "",
                    journal_style=context.get("project_data", {}).get("target_journal", "Top finance journal"),
                )
                
                provider_attempts.append(
                    self._provider_attempt(
                        provider="claude_literature_search",
                        ok=bool(claude_result.citations),
                        duration_seconds=time.time() - t0,
                    )
                )
                
                if claude_result.citations:
                    used_provider = "claude_literature_search"
                    logger.info(
                        f"Claude Literature Search completed in {claude_result.processing_time:.1f}s "
                        f"with {len(claude_result.citations)} citations"
                    )
                    
                    return AgentResult(
                        agent_name=self.name,
                        task_type=self.task_type,
                        model_tier=self.model_tier,
                        success=True,
                        content=claude_result.response,
                        tokens_used=claude_result.tokens_used,
                        execution_time=time.time() - start_time,
                        structured_data={
                            "primary_query": queries["primary_query"],
                            "supporting_queries": queries["supporting_queries"],
                            "literature_result": claude_result.to_dict(),
                            "citations": claude_result.citations,
                            "total_papers": claude_result.papers_included,
                            "processing_time": claude_result.processing_time,
                            "fallback_metadata": {
                                "used_provider": used_provider,
                                "attempts": provider_attempts,
                                "evidence_synthesis": claude_result.evidence_synthesis,
                                "aspect_queries": claude_result.aspect_queries,
                            },
                        },
                    )
                else:
                    logger.warning("Claude Literature Search returned no citations")
                    
            except Exception as e:
                logger.warning(f"Claude Literature Search failed: {type(e).__name__}: {e}")
                provider_attempts.append(
                    self._provider_attempt(
                        provider="claude_literature_search",
                        ok=False,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        duration_seconds=time.time() - t0,
                    )
                )

            # ========================================================
            # Provider 2: Semantic Scholar (Fallback)
            # ========================================================
            logger.info("Attempting Semantic Scholar fallback...")
            citations_data: List[Dict[str, Any]] = []
            content_text = ""

            try:
                t0 = time.time()
                content_text, citations_data = await self._search_via_semantic_scholar(query=fallback_query)
                provider_attempts.append(
                    self._provider_attempt(
                        provider="semantic_scholar",
                        ok=True,
                        duration_seconds=time.time() - t0,
                    )
                )
                if citations_data:
                    used_provider = "semantic_scholar"
            except Exception as e:
                logger.debug(f"Semantic Scholar fallback failed: {type(e).__name__}: {e}")
                provider_attempts.append(
                    self._provider_attempt(
                        provider="semantic_scholar",
                        ok=False,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                )

            # ========================================================
            # Provider 3: arXiv (Fallback)
            # ========================================================
            if used_provider is None:
                logger.info("Attempting arXiv fallback...")
                try:
                    t0 = time.time()
                    content_text, citations_data = await self._search_via_arxiv(query=fallback_query)
                    provider_attempts.append(
                        self._provider_attempt(
                            provider="arxiv",
                            ok=True,
                            duration_seconds=time.time() - t0,
                        )
                    )
                    if citations_data:
                        used_provider = "arxiv"
                except Exception as e:
                    logger.debug(f"arXiv fallback failed: {type(e).__name__}: {e}")
                    provider_attempts.append(
                        self._provider_attempt(
                            provider="arxiv",
                            ok=False,
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                    )

            # ========================================================
            # Provider 4: Edison Scientific (Expensive Fallback)
            # ========================================================
            if used_provider is None and self.use_edison_fallback:
                edison_available = hasattr(self.edison_client, "is_available") and self.edison_client.is_available
                if edison_available:
                    logger.info("Attempting Edison Scientific (expensive fallback)...")
                    t0 = time.time()
                    try:
                        # Formulate queries for Edison
                        edison_queries = await self._formulate_queries(
                            main_hypothesis=main_hypothesis,
                            literature_questions=literature_questions,
                            research_overview=research_overview,
                            context=context,
                        )
                        primary_query = str(edison_queries.get("primary_query", "") or fallback_query).strip()
                        search_context = str(edison_queries.get("search_context", "") or "").strip()
                        queries["tokens_used"] += edison_queries.get("tokens_used", 0)
                        
                        literature_result = await self.edison_client.search_literature(
                            query=primary_query,
                            context=search_context,
                        )
                        provider_attempts.append(
                            self._provider_attempt(
                                provider="edison",
                                ok=bool(literature_result.status != JobStatus.FAILED),
                                error_type="edison_failed" if literature_result.status == JobStatus.FAILED else None,
                                error_message=str(literature_result.error) if literature_result.status == JobStatus.FAILED else None,
                                duration_seconds=time.time() - t0,
                            )
                        )

                        if literature_result.status != JobStatus.FAILED:
                            used_provider = "edison"
                            logger.info(
                                f"Edison search completed in {literature_result.processing_time:.1f}s "
                                f"with {len(literature_result.citations)} citations"
                            )
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
                                    "literature_result": {
                                        **literature_result.to_dict(),
                                        "provider": "edison",
                                    },
                                    "citations": [c.to_dict() for c in literature_result.citations],
                                    "total_papers": len(literature_result.citations),
                                    "edison_processing_time": literature_result.processing_time,
                                    "fallback_metadata": {
                                        "used_provider": used_provider,
                                        "attempts": provider_attempts,
                                    },
                                },
                            )
                    except Exception as e:
                        logger.warning(f"Edison fallback failed: {type(e).__name__}: {e}")
                        provider_attempts.append(
                            self._provider_attempt(
                                provider="edison",
                                ok=False,
                                error_type=type(e).__name__,
                                error_message=str(e),
                                duration_seconds=time.time() - t0,
                            )
                        )
                else:
                    init_error = getattr(self.edison_client, "init_error", None)
                    provider_attempts.append(
                        self._provider_attempt(
                            provider="edison",
                            ok=False,
                            error_type="not_configured",
                            error_message=str(init_error or "Edison API client not configured"),
                        )
                    )

            # ========================================================
            # Provider 5: Manual Sources List (Final Fallback)
            # ========================================================
            if used_provider is None:
                logger.info("Attempting manual sources list fallback...")
                try:
                    t0 = time.time()
                    content_text, citations_data = self._search_via_manual_sources_list(
                        project_folder=context.get("project_folder")
                    )
                    provider_attempts.append(
                        self._provider_attempt(
                            provider="manual",
                            ok=True,
                            duration_seconds=time.time() - t0,
                        )
                    )
                    used_provider = "manual"
                except Exception as e:
                    logger.debug(f"Manual sources list fallback failed: {type(e).__name__}: {e}")
                    provider_attempts.append(
                        self._provider_attempt(
                            provider="manual",
                            ok=False,
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                    )

            # ========================================================
            # Return Result
            # ========================================================
            used_provider = used_provider or "none"
            degraded = used_provider == "none" or not citations_data
            if not content_text:
                content_text = (
                    "Literature search fallback chain completed, but no structured citations were found. "
                    "Provide a manual sources list or check API connectivity to improve results."
                )

            literature_dict: Dict[str, Any] = {
                "query": queries["primary_query"],
                "response": content_text,
                "citations": citations_data,
                "total_papers_searched": len(citations_data),
                "processing_time": time.time() - start_time,
                "job_id": None,
                "status": "completed",
                "error": None,
                "provider": used_provider,
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }

            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content=content_text,
                tokens_used=int(queries.get("tokens_used", 0) or 0),
                execution_time=time.time() - start_time,
                structured_data={
                    "primary_query": queries["primary_query"],
                    "supporting_queries": queries.get("supporting_queries", []),
                    "literature_result": literature_dict,
                    "citations": citations_data,
                    "total_papers": len(citations_data),
                    "fallback_metadata": {
                        "used_provider": used_provider,
                        "attempts": provider_attempts,
                        "degraded": bool(degraded),
                    },
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
