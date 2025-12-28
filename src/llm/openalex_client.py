"""src.llm.openalex_client

OpenAlex API Client
===================

Provides async access to the OpenAlex API for academic paper search,
open access metadata, concept enrichment, and author/institution data.

OpenAlex is a free, open catalog of the global research system, offering:
- 250M+ works with metadata and full-text links
- Open access URLs and PDF links
- Concept tagging and topic classification
- Citation networks
- Author disambiguation
- Institution and funder data

API Documentation: https://docs.openalex.org/

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote

import httpx
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# OpenAlex API base URL
OPENALEX_API_BASE = "https://api.openalex.org"

# Polite pool: include email for higher rate limits
DEFAULT_MAILTO = os.getenv("OPENALEX_EMAIL", "")

# Rate limits: 10 req/sec for polite pool, 100K req/day
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


@dataclass
class OpenAlexWork:
    """Represents a work (paper/article) from OpenAlex."""
    
    openalex_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    author_institutions: List[str] = field(default_factory=list)
    year: Optional[int] = None
    publication_date: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    open_access_url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    oa_status: Optional[str] = None  # gold, green, hybrid, bronze, closed
    citation_count: int = 0
    concepts: List[Dict[str, Any]] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    type: Optional[str] = None  # article, book-chapter, etc.
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with Citation dataclass."""
        return {
            "openalex_id": self.openalex_id,
            "title": self.title,
            "authors": self.authors,
            "author_institutions": self.author_institutions,
            "year": self.year or 0,
            "publication_date": self.publication_date,
            "abstract": self.abstract,
            "journal": self.venue,
            "doi": self.doi,
            "pmid": self.pmid,
            "url": self.url,
            "open_access_url": self.open_access_url,
            "pdf_url": self.pdf_url,
            "is_open_access": self.is_open_access,
            "oa_status": self.oa_status,
            "citations": self.citation_count,
            "concepts": self.concepts,
            "topics": self.topics,
            "type": self.type,
            "language": self.language,
            "relevance_score": None,
            "paper_id": self.openalex_id,
        }


@dataclass
class OpenAlexSearchResult:
    """Result from an OpenAlex search."""
    
    works: List[OpenAlexWork] = field(default_factory=list)
    total_count: int = 0
    page: int = 1
    per_page: int = 25
    query: str = ""
    search_time_ms: float = 0.0


class OpenAlexClient:
    """Async client for OpenAlex API.
    
    Usage:
        client = OpenAlexClient(email="your@email.com")
        results = await client.search_works("corporate governance")
        for work in results.works:
            print(work.title, work.citation_count, work.is_open_access)
    """
    
    def __init__(
        self,
        email: Optional[str] = None,
        timeout: Optional[httpx.Timeout] = None,
    ):
        """Initialize OpenAlex client.
        
        Args:
            email: Email for polite pool (higher rate limits)
            timeout: Optional custom timeout configuration
        """
        self.email = email or DEFAULT_MAILTO
        self.timeout = timeout or DEFAULT_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
        
    def _get_params(self) -> Dict[str, str]:
        """Get base request parameters including email for polite pool."""
        params: Dict[str, str] = {}
        if self.email:
            params["mailto"] = self.email
        return params
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    @staticmethod
    def _parse_work(data: Dict[str, Any]) -> OpenAlexWork:
        """Parse API response into OpenAlexWork."""
        # Extract authors and their institutions
        authors: List[str] = []
        institutions: List[str] = []
        authorships = data.get("authorships") or []
        for authorship in authorships:
            if isinstance(authorship, dict):
                author = authorship.get("author", {})
                if isinstance(author, dict):
                    name = author.get("display_name")
                    if name and isinstance(name, str):
                        authors.append(name.strip())
                # Get institutions
                inst_list = authorship.get("institutions") or []
                for inst in inst_list:
                    if isinstance(inst, dict):
                        inst_name = inst.get("display_name")
                        if inst_name and inst_name not in institutions:
                            institutions.append(inst_name)
        
        # Extract DOI (remove prefix)
        doi = None
        raw_doi = data.get("doi")
        if raw_doi and isinstance(raw_doi, str):
            doi = raw_doi.replace("https://doi.org/", "").strip()
        
        # Extract IDs
        ids = data.get("ids") or {}
        pmid = None
        if isinstance(ids, dict):
            pmid_raw = ids.get("pmid")
            if pmid_raw and isinstance(pmid_raw, str):
                pmid = pmid_raw.replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip()
        
        # Extract venue/source
        venue = None
        source = data.get("primary_location", {}) or {}
        if isinstance(source, dict):
            source_info = source.get("source") or {}
            if isinstance(source_info, dict):
                venue = source_info.get("display_name")
        
        # Extract open access info
        open_access = data.get("open_access") or {}
        is_oa = bool(open_access.get("is_oa"))
        oa_status = open_access.get("oa_status")
        oa_url = open_access.get("oa_url")
        
        # Find PDF URL
        pdf_url = None
        best_oa = data.get("best_oa_location") or {}
        if isinstance(best_oa, dict):
            pdf_url = best_oa.get("pdf_url")
        
        # Extract abstract (if available via inverted index)
        abstract = None
        abstract_inv = data.get("abstract_inverted_index")
        if abstract_inv and isinstance(abstract_inv, dict):
            # Reconstruct abstract from inverted index
            try:
                word_positions: List[Tuple[int, str]] = []
                for word, positions in abstract_inv.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort(key=lambda x: x[0])
                abstract = " ".join(w for _, w in word_positions)
            except Exception:
                pass
        
        # Extract concepts
        concepts: List[Dict[str, Any]] = []
        concepts_raw = data.get("concepts") or []
        for concept in concepts_raw:
            if isinstance(concept, dict):
                concepts.append({
                    "id": concept.get("id"),
                    "name": concept.get("display_name"),
                    "level": concept.get("level"),
                    "score": concept.get("score"),
                })
        
        # Extract topics
        topics: List[str] = []
        topics_raw = data.get("topics") or []
        for topic in topics_raw:
            if isinstance(topic, dict):
                name = topic.get("display_name")
                if name:
                    topics.append(name)
        
        return OpenAlexWork(
            openalex_id=str(data.get("id") or ""),
            title=str(data.get("title") or "").strip(),
            authors=authors,
            author_institutions=institutions,
            year=data.get("publication_year"),
            publication_date=data.get("publication_date"),
            abstract=abstract,
            venue=venue,
            doi=doi,
            pmid=pmid,
            url=data.get("id"),  # OpenAlex URL
            open_access_url=oa_url,
            pdf_url=pdf_url,
            is_open_access=is_oa,
            oa_status=oa_status,
            citation_count=data.get("cited_by_count") or 0,
            concepts=concepts,
            topics=topics,
            type=data.get("type"),
            language=data.get("language"),
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def search_works(
        self,
        query: str,
        *,
        page: int = 1,
        per_page: int = 50,
        publication_year: Optional[str] = None,
        type_filter: Optional[str] = None,
        is_open_access: Optional[bool] = None,
        concepts: Optional[List[str]] = None,
        sort: str = "relevance_score:desc",
    ) -> OpenAlexSearchResult:
        """Search for works (papers/articles) by text query.
        
        Args:
            query: Full-text search query
            page: Page number (1-indexed)
            per_page: Results per page (max 200)
            publication_year: Year filter (e.g., "2020", ">2018", "2018-2022")
            type_filter: Work type (article, book-chapter, etc.)
            is_open_access: Filter by open access status
            concepts: Filter by concept IDs
            sort: Sort order
            
        Returns:
            OpenAlexSearchResult with works and pagination info
        """
        import time
        start_time = time.time()
        
        client = await self._get_client()
        
        # Build params
        params = self._get_params()
        params["search"] = query
        params["page"] = str(page)
        params["per_page"] = str(min(per_page, 200))
        params["sort"] = sort
        
        # Build filter string
        filters: List[str] = []
        if publication_year:
            filters.append(f"publication_year:{publication_year}")
        if type_filter:
            filters.append(f"type:{type_filter}")
        if is_open_access is not None:
            filters.append(f"is_oa:{'true' if is_open_access else 'false'}")
        if concepts:
            for concept in concepts:
                filters.append(f"concepts.id:{concept}")
        
        if filters:
            params["filter"] = ",".join(filters)
        
        url = f"{OPENALEX_API_BASE}/works"
        logger.debug(f"OpenAlex search: {query[:50]}... (page={page})")
        
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("OpenAlex rate limit hit, backing off...")
                await asyncio.sleep(1)
                raise
            logger.error(f"OpenAlex API error: {e.response.status_code}")
            raise
        
        # Parse results
        works: List[OpenAlexWork] = []
        results = data.get("results") or []
        for item in results:
            if isinstance(item, dict) and item.get("title"):
                works.append(self._parse_work(item))
        
        total_count = data.get("meta", {}).get("count", len(works))
        
        search_time = (time.time() - start_time) * 1000
        
        logger.info(f"OpenAlex found {len(works)} works for: {query[:50]}...")
        
        return OpenAlexSearchResult(
            works=works,
            total_count=total_count,
            page=page,
            per_page=per_page,
            query=query,
            search_time_ms=search_time,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def get_work(
        self,
        work_id: str,
    ) -> Optional[OpenAlexWork]:
        """Get a single work by ID.
        
        Args:
            work_id: OpenAlex ID, DOI, PMID, etc.
            
        Returns:
            OpenAlexWork or None if not found
        """
        client = await self._get_client()
        
        params = self._get_params()
        
        # Handle different ID formats
        if work_id.startswith("10."):
            # DOI
            url = f"{OPENALEX_API_BASE}/works/doi:{work_id}"
        elif work_id.startswith("https://doi.org/"):
            # Full DOI URL
            doi = work_id.replace("https://doi.org/", "")
            url = f"{OPENALEX_API_BASE}/works/doi:{doi}"
        elif work_id.startswith("W"):
            # OpenAlex ID
            url = f"{OPENALEX_API_BASE}/works/{work_id}"
        elif work_id.startswith("https://openalex.org/"):
            # Full OpenAlex URL
            url = work_id
        else:
            # Try as-is
            url = f"{OPENALEX_API_BASE}/works/{work_id}"
        
        try:
            response = await client.get(url, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return self._parse_work(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    async def get_works_batch(
        self,
        work_ids: List[str],
    ) -> List[OpenAlexWork]:
        """Get multiple works by ID using filter.
        
        Args:
            work_ids: List of OpenAlex IDs (W...) or DOIs
            
        Returns:
            List of OpenAlexWork objects
        """
        if not work_ids:
            return []
        
        client = await self._get_client()
        
        # OpenAlex supports OR filters for batch lookups
        all_works: List[OpenAlexWork] = []
        batch_size = 50  # Filter URL length limits
        
        for i in range(0, len(work_ids), batch_size):
            batch = work_ids[i:i + batch_size]
            
            # Build filter for OpenAlex IDs
            openalex_ids = [w for w in batch if w.startswith("W") or w.startswith("https://openalex.org/")]
            dois = [w for w in batch if w.startswith("10.") or w.startswith("https://doi.org/")]
            
            filter_parts: List[str] = []
            if openalex_ids:
                ids_str = "|".join(w.replace("https://openalex.org/", "") for w in openalex_ids)
                filter_parts.append(f"openalex_id:{ids_str}")
            if dois:
                dois_clean = [d.replace("https://doi.org/", "") for d in dois]
                dois_str = "|".join(dois_clean)
                filter_parts.append(f"doi:{dois_str}")
            
            if not filter_parts:
                continue
            
            params = self._get_params()
            params["filter"] = ",".join(filter_parts)
            params["per_page"] = "100"
            
            url = f"{OPENALEX_API_BASE}/works"
            
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for item in data.get("results", []):
                    if item and isinstance(item, dict) and item.get("title"):
                        all_works.append(self._parse_work(item))
                        
            except httpx.HTTPStatusError as e:
                logger.warning(f"Batch work fetch failed: {e}")
        
        return all_works
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def get_work_citations(
        self,
        work_id: str,
        *,
        page: int = 1,
        per_page: int = 50,
    ) -> List[OpenAlexWork]:
        """Get works that cite the given work.
        
        Args:
            work_id: OpenAlex ID or DOI
            page: Page number
            per_page: Results per page
            
        Returns:
            List of citing works
        """
        client = await self._get_client()
        
        # Get the OpenAlex ID
        if work_id.startswith("10.") or work_id.startswith("https://doi.org/"):
            doi = work_id.replace("https://doi.org/", "")
            filter_str = f"cites:doi:{doi}"
        else:
            openalex_id = work_id.replace("https://openalex.org/", "")
            filter_str = f"cites:{openalex_id}"
        
        params = self._get_params()
        params["filter"] = filter_str
        params["page"] = str(page)
        params["per_page"] = str(min(per_page, 200))
        params["sort"] = "cited_by_count:desc"
        
        url = f"{OPENALEX_API_BASE}/works"
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        works: List[OpenAlexWork] = []
        for item in data.get("results", []):
            if item and isinstance(item, dict) and item.get("title"):
                works.append(self._parse_work(item))
        
        return works
    
    async def search_by_concept(
        self,
        concept_name: str,
        *,
        page: int = 1,
        per_page: int = 50,
        publication_year: Optional[str] = None,
        min_citations: Optional[int] = None,
    ) -> OpenAlexSearchResult:
        """Search works by concept/topic.
        
        Args:
            concept_name: Concept to search for
            page: Page number
            per_page: Results per page
            publication_year: Year filter
            min_citations: Minimum citation count
            
        Returns:
            OpenAlexSearchResult
        """
        import time
        start_time = time.time()
        
        client = await self._get_client()
        
        # First, find the concept ID
        params = self._get_params()
        params["search"] = concept_name
        
        concept_url = f"{OPENALEX_API_BASE}/concepts"
        response = await client.get(concept_url, params=params)
        response.raise_for_status()
        concept_data = response.json()
        
        concept_id = None
        for c in concept_data.get("results", []):
            if isinstance(c, dict) and c.get("id"):
                concept_id = c.get("id")
                break
        
        if not concept_id:
            return OpenAlexSearchResult(
                query=concept_name,
                search_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Search works with this concept
        filters: List[str] = [f"concepts.id:{concept_id}"]
        if publication_year:
            filters.append(f"publication_year:{publication_year}")
        if min_citations is not None:
            filters.append(f"cited_by_count:>{min_citations}")
        
        params = self._get_params()
        params["filter"] = ",".join(filters)
        params["page"] = str(page)
        params["per_page"] = str(min(per_page, 200))
        params["sort"] = "cited_by_count:desc"
        
        url = f"{OPENALEX_API_BASE}/works"
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        works: List[OpenAlexWork] = []
        for item in data.get("results", []):
            if item and isinstance(item, dict) and item.get("title"):
                works.append(self._parse_work(item))
        
        return OpenAlexSearchResult(
            works=works,
            total_count=data.get("meta", {}).get("count", len(works)),
            page=page,
            per_page=per_page,
            query=concept_name,
            search_time_ms=(time.time() - start_time) * 1000,
        )
    
    async def enrich_with_open_access(
        self,
        dois: List[str],
    ) -> Dict[str, Optional[str]]:
        """Get open access PDF URLs for a list of DOIs.
        
        Args:
            dois: List of DOIs
            
        Returns:
            Dict mapping DOI to PDF URL (or None if not available)
        """
        works = await self.get_works_batch(dois)
        
        result: Dict[str, Optional[str]] = {}
        for work in works:
            if work.doi:
                result[work.doi] = work.pdf_url or work.open_access_url
        
        # Fill in missing DOIs
        for doi in dois:
            clean_doi = doi.replace("https://doi.org/", "")
            if clean_doi not in result:
                result[clean_doi] = None
        
        return result


# Convenience function for one-off searches
async def search_openalex(
    query: str,
    *,
    per_page: int = 50,
    publication_year: Optional[str] = None,
    email: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience function for simple work searches.
    
    Args:
        query: Search query
        per_page: Maximum results
        publication_year: Year filter
        email: Email for polite pool
        
    Returns:
        List of work dictionaries
    """
    client = OpenAlexClient(email=email)
    try:
        result = await client.search_works(
            query,
            per_page=per_page,
            publication_year=publication_year,
        )
        return [w.to_dict() for w in result.works]
    finally:
        await client.close()
