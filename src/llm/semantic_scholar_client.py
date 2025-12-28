"""src.llm.semantic_scholar_client

Semantic Scholar API Client
===========================

Provides async access to the Semantic Scholar Graph API for academic paper search,
metadata retrieval, and citation network exploration.

API Documentation: https://api.semanticscholar.org/api-docs/graph

Features:
- Paper search with keyword queries
- Bulk paper metadata retrieval
- Citation and reference network traversal
- Author information lookup
- Rate limiting and retry handling

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Semantic Scholar API base URL
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

# Default fields to retrieve for papers
DEFAULT_PAPER_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "venue",
    "publicationVenue",
    "authors",
    "externalIds",
    "url",
    "citationCount",
    "influentialCitationCount",
    "isOpenAccess",
    "openAccessPdf",
    "fieldsOfStudy",
    "tldr",
]

# Rate limit: 100 requests per 5 minutes for unauthenticated
# With API key: 1 request per second
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


@dataclass
class SemanticScholarPaper:
    """Represents a paper from Semantic Scholar."""
    
    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    is_open_access: bool = False
    open_access_pdf_url: Optional[str] = None
    fields_of_study: List[str] = field(default_factory=list)
    tldr: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with Citation dataclass."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year or 0,
            "abstract": self.abstract,
            "journal": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "citations": self.citation_count,
            "influential_citations": self.influential_citation_count,
            "is_open_access": self.is_open_access,
            "open_access_pdf_url": self.open_access_pdf_url,
            "fields_of_study": self.fields_of_study,
            "tldr": self.tldr,
            "relevance_score": None,
        }


@dataclass
class SemanticScholarSearchResult:
    """Result from a Semantic Scholar search."""
    
    papers: List[SemanticScholarPaper] = field(default_factory=list)
    total: int = 0
    offset: int = 0
    next_offset: Optional[int] = None
    query: str = ""
    search_time_ms: float = 0.0


class SemanticScholarClient:
    """Async client for Semantic Scholar Graph API.
    
    Usage:
        client = SemanticScholarClient()
        results = await client.search_papers("machine learning fairness")
        for paper in results.papers:
            print(paper.title, paper.citation_count)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: Optional[httpx.Timeout] = None,
    ):
        """Initialize Semantic Scholar client.
        
        Args:
            api_key: Optional S2 API key for higher rate limits
            timeout: Optional custom timeout configuration
        """
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.timeout = timeout or DEFAULT_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
        
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers, including API key if available."""
        headers = {
            "Accept": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self._get_headers(),
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    @staticmethod
    def _parse_paper(data: Dict[str, Any]) -> SemanticScholarPaper:
        """Parse API response into SemanticScholarPaper."""
        # Extract authors
        authors: List[str] = []
        authors_data = data.get("authors") or []
        for author in authors_data:
            if isinstance(author, dict):
                name = author.get("name")
                if name and isinstance(name, str):
                    authors.append(name.strip())
        
        # Extract external IDs
        external_ids = data.get("externalIds") or {}
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")
        
        # Extract venue
        venue = data.get("venue") or ""
        pub_venue = data.get("publicationVenue")
        if pub_venue and isinstance(pub_venue, dict):
            venue = pub_venue.get("name") or venue
        
        # Extract open access PDF
        oa_pdf = data.get("openAccessPdf")
        oa_pdf_url = None
        if oa_pdf and isinstance(oa_pdf, dict):
            oa_pdf_url = oa_pdf.get("url")
        
        # Extract TLDR
        tldr_data = data.get("tldr")
        tldr = None
        if tldr_data and isinstance(tldr_data, dict):
            tldr = tldr_data.get("text")
        
        # Extract fields of study
        fields = data.get("fieldsOfStudy") or []
        fields_of_study = [f for f in fields if isinstance(f, str)]
        
        return SemanticScholarPaper(
            paper_id=str(data.get("paperId") or ""),
            title=str(data.get("title") or "").strip(),
            authors=authors,
            year=data.get("year"),
            abstract=data.get("abstract"),
            venue=venue if venue else None,
            doi=str(doi).strip() if doi else None,
            arxiv_id=str(arxiv_id).strip() if arxiv_id else None,
            url=data.get("url"),
            citation_count=data.get("citationCount"),
            influential_citation_count=data.get("influentialCitationCount"),
            is_open_access=bool(data.get("isOpenAccess")),
            open_access_pdf_url=oa_pdf_url,
            fields_of_study=fields_of_study,
            tldr=tldr,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def search_papers(
        self,
        query: str,
        *,
        limit: int = 50,
        offset: int = 0,
        year: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
        open_access_only: bool = False,
        min_citation_count: Optional[int] = None,
        fields: Optional[List[str]] = None,
    ) -> SemanticScholarSearchResult:
        """Search for papers by keyword query.
        
        Args:
            query: Search query string
            limit: Maximum number of results (max 100)
            offset: Pagination offset
            year: Year filter (e.g., "2020", "2018-2022", "2020-")
            fields_of_study: Filter by fields (e.g., ["Economics", "Computer Science"])
            open_access_only: Only return open access papers
            min_citation_count: Minimum citation count filter
            fields: Fields to retrieve (uses defaults if not specified)
            
        Returns:
            SemanticScholarSearchResult with papers and pagination info
        """
        import time
        start_time = time.time()
        
        client = await self._get_client()
        
        # Build params
        params: Dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "offset": offset,
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        
        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if open_access_only:
            params["openAccessPdf"] = ""
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count
        
        url = f"{S2_API_BASE}/paper/search"
        logger.debug(f"Semantic Scholar search: {query[:50]}... (limit={limit})")
        
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit, backing off...")
                await asyncio.sleep(5)
                raise
            logger.error(f"Semantic Scholar API error: {e.response.status_code}")
            raise
        
        # Parse results
        papers: List[SemanticScholarPaper] = []
        items = data.get("data") or []
        for item in items:
            if isinstance(item, dict) and item.get("title"):
                papers.append(self._parse_paper(item))
        
        total = data.get("total", len(papers))
        next_offset = data.get("next")
        
        search_time = (time.time() - start_time) * 1000
        
        logger.info(f"Semantic Scholar found {len(papers)} papers for: {query[:50]}...")
        
        return SemanticScholarSearchResult(
            papers=papers,
            total=total,
            offset=offset,
            next_offset=next_offset,
            query=query,
            search_time_ms=search_time,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def get_paper(
        self,
        paper_id: str,
        *,
        fields: Optional[List[str]] = None,
    ) -> Optional[SemanticScholarPaper]:
        """Get a single paper by ID.
        
        Args:
            paper_id: Semantic Scholar paper ID, DOI, ArXiv ID, etc.
            fields: Fields to retrieve
            
        Returns:
            SemanticScholarPaper or None if not found
        """
        client = await self._get_client()
        
        params = {"fields": ",".join(fields or DEFAULT_PAPER_FIELDS)}
        url = f"{S2_API_BASE}/paper/{paper_id}"
        
        try:
            response = await client.get(url, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return self._parse_paper(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    async def get_papers_batch(
        self,
        paper_ids: List[str],
        *,
        fields: Optional[List[str]] = None,
    ) -> List[SemanticScholarPaper]:
        """Get multiple papers by ID in a single batch request.
        
        Args:
            paper_ids: List of paper IDs (max 500)
            fields: Fields to retrieve
            
        Returns:
            List of SemanticScholarPaper objects
        """
        if not paper_ids:
            return []
        
        client = await self._get_client()
        
        # Batch endpoint accepts up to 500 IDs
        batch_size = 500
        all_papers: List[SemanticScholarPaper] = []
        
        for i in range(0, len(paper_ids), batch_size):
            batch = paper_ids[i:i + batch_size]
            url = f"{S2_API_BASE}/paper/batch"
            params = {"fields": ",".join(fields or DEFAULT_PAPER_FIELDS)}
            
            try:
                response = await client.post(
                    url,
                    params=params,
                    json={"ids": batch},
                )
                response.raise_for_status()
                data = response.json()
                
                for item in data:
                    if item and isinstance(item, dict) and item.get("title"):
                        all_papers.append(self._parse_paper(item))
                        
            except httpx.HTTPStatusError as e:
                logger.warning(f"Batch paper fetch failed: {e}")
                # Continue with remaining batches
                
        return all_papers
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def get_paper_citations(
        self,
        paper_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        fields: Optional[List[str]] = None,
    ) -> List[SemanticScholarPaper]:
        """Get papers that cite the given paper.
        
        Args:
            paper_id: Paper ID to get citations for
            limit: Maximum citations to return
            offset: Pagination offset
            fields: Fields to retrieve
            
        Returns:
            List of citing papers
        """
        client = await self._get_client()
        
        params = {
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
            "limit": min(limit, 1000),
            "offset": offset,
        }
        url = f"{S2_API_BASE}/paper/{paper_id}/citations"
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        papers: List[SemanticScholarPaper] = []
        for item in data.get("data", []):
            citing = item.get("citingPaper")
            if citing and isinstance(citing, dict) and citing.get("title"):
                papers.append(self._parse_paper(citing))
        
        return papers
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def get_paper_references(
        self,
        paper_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        fields: Optional[List[str]] = None,
    ) -> List[SemanticScholarPaper]:
        """Get papers referenced by the given paper.
        
        Args:
            paper_id: Paper ID to get references for
            limit: Maximum references to return
            offset: Pagination offset
            fields: Fields to retrieve
            
        Returns:
            List of referenced papers
        """
        client = await self._get_client()
        
        params = {
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
            "limit": min(limit, 1000),
            "offset": offset,
        }
        url = f"{S2_API_BASE}/paper/{paper_id}/references"
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        papers: List[SemanticScholarPaper] = []
        for item in data.get("data", []):
            cited = item.get("citedPaper")
            if cited and isinstance(cited, dict) and cited.get("title"):
                papers.append(self._parse_paper(cited))
        
        return papers
    
    async def search_with_relevance_reranking(
        self,
        query: str,
        *,
        limit: int = 50,
        year_range: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
        boost_high_citations: bool = True,
    ) -> SemanticScholarSearchResult:
        """Search and rerank results by relevance signals.
        
        This method performs a search and then reranks results based on:
        - Citation count (if boost_high_citations is True)
        - Influential citation count
        - Recency (newer papers get slight boost)
        - Open access availability
        
        Args:
            query: Search query
            limit: Number of results to return
            year_range: Optional year filter
            fields_of_study: Optional field filter
            boost_high_citations: Whether to boost highly cited papers
            
        Returns:
            SemanticScholarSearchResult with reranked papers
        """
        # Fetch more results for reranking
        fetch_limit = min(limit * 2, 100)
        
        result = await self.search_papers(
            query,
            limit=fetch_limit,
            year=year_range,
            fields_of_study=fields_of_study,
        )
        
        if not result.papers:
            return result
        
        # Score and rerank papers
        scored_papers: List[tuple[float, SemanticScholarPaper]] = []
        current_year = datetime.now().year
        
        for paper in result.papers:
            score = 0.0
            
            # Citation score (log scale to prevent outlier dominance)
            if boost_high_citations and paper.citation_count:
                import math
                score += math.log1p(paper.citation_count) * 2
            
            # Influential citation bonus
            if paper.influential_citation_count:
                import math
                score += math.log1p(paper.influential_citation_count) * 3
            
            # Recency bonus (papers from last 5 years get boost)
            if paper.year and paper.year >= current_year - 5:
                score += (paper.year - (current_year - 5)) * 0.5
            
            # Open access bonus
            if paper.is_open_access:
                score += 2
            
            # TLDR presence indicates well-indexed paper
            if paper.tldr:
                score += 1
            
            scored_papers.append((score, paper))
        
        # Sort by score descending
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N
        reranked = [p for _, p in scored_papers[:limit]]
        
        return SemanticScholarSearchResult(
            papers=reranked,
            total=result.total,
            offset=result.offset,
            next_offset=result.next_offset,
            query=result.query,
            search_time_ms=result.search_time_ms,
        )


# Convenience function for one-off searches
async def search_semantic_scholar(
    query: str,
    *,
    limit: int = 50,
    year: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience function for simple paper searches.
    
    Args:
        query: Search query
        limit: Maximum results
        year: Year filter
        api_key: Optional API key
        
    Returns:
        List of paper dictionaries
    """
    client = SemanticScholarClient(api_key=api_key)
    try:
        result = await client.search_papers(query, limit=limit, year=year)
        return [p.to_dict() for p in result.papers]
    finally:
        await client.close()
