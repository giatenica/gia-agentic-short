"""src.llm.edison_client

Edison Scientific API Client
============================

This module provides a small wrapper around the official `edison-client` library.

What the Edison "literature" stage does
--------------------------------------
The Edison API call used by this repo is a literature search + synthesis step.
Given a natural-language query (and optional context), Edison returns:

- A narrative response (`LiteratureResult.response`): a written, citation-oriented
    literature review style synthesis of relevant academic work.
- A structured list of citations (`LiteratureResult.citations`): metadata for the
    papers Edison considered relevant (title, authors, year, DOI/URL when present,
    and optional relevance signals).

This repo treats Edison as an external source and uses its outputs as inputs to
later agents:

- `LiteratureSearchAgent` stores the Edison response and citations in
    `structured_data`.
- `LiteratureSynthesisAgent` converts those inputs into project files such as
    `LITERATURE_REVIEW.md`, `references.bib`, and `citations_data.json`.

If Edison is unavailable (missing key, auth failure, network issues), workflows
are designed to keep running and can generate scaffold outputs downstream.

API Documentation: https://edisonscientific.gitbook.io/edison-cookbook/edison-client
Package: https://pypi.org/project/edison-client/

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum

import hashlib
from edison_client import EdisonClient as OfficialEdisonClient
from edison_client import JobNames as EdisonJobNames
from loguru import logger

# Track active/recent Edison requests to prevent duplicates
_active_requests: dict[str, float] = {}  # query_hash -> start_time
_request_lock = None  # Lazy-init asyncio.Lock


class JobStatus(Enum):
    """Edison job status states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Citation:
    """A literature citation from Edison."""
    title: str
    authors: List[str]
    year: int
    journal: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    relevance_score: Optional[float] = None
    paper_id: Optional[str] = None
    citations: Optional[int] = None
    
    def to_bibtex(self, key: Optional[str] = None) -> str:
        """Convert citation to BibTeX format."""
        if not key:
            first_author = self.authors[0].split()[-1] if self.authors else "Unknown"
            key = f"{first_author}{self.year}"
        
        authors_str = " and ".join(self.authors)
        
        bibtex = f"@article{{{key},\n"
        bibtex += f"  title = {{{self.title}}},\n"
        bibtex += f"  author = {{{authors_str}}},\n"
        bibtex += f"  year = {{{self.year}}},\n"
        
        if self.journal:
            bibtex += f"  journal = {{{self.journal}}},\n"
        if self.doi:
            bibtex += f"  doi = {{{self.doi}}},\n"
        if self.url:
            bibtex += f"  url = {{{self.url}}},\n"
        
        bibtex += "}\n"
        return bibtex
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
            "doi": self.doi,
            "url": self.url,
            "abstract": self.abstract,
            "relevance_score": self.relevance_score,
            "paper_id": self.paper_id,
            "citations": self.citations,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Citation":
        return cls(
            title=data.get("title", ""),
            authors=data.get("authors", []),
            year=data.get("year", 0),
            journal=data.get("journal"),
            doi=data.get("doi"),
            url=data.get("url"),
            abstract=data.get("abstract"),
            relevance_score=data.get("relevance_score"),
            paper_id=data.get("paper_id"),
            citations=data.get("citations"),
        )


@dataclass
class LiteratureResult:
    """Result from a literature search."""
    query: str = ""
    response: str = ""
    citations: List[Citation] = field(default_factory=list)
    total_papers_searched: int = 0
    processing_time: float = 0.0
    job_id: Optional[str] = None
    status: JobStatus = JobStatus.COMPLETED
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "response": self.response,
            "citations": [c.to_dict() for c in self.citations],
            "total_papers_searched": self.total_papers_searched,
            "processing_time": self.processing_time,
            "job_id": self.job_id,
            "status": self.status.value,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LiteratureResult":
        return cls(
            query=data.get("query", ""),
            response=data.get("response", ""),
            citations=[Citation.from_dict(c) for c in data.get("citations", [])],
            total_papers_searched=data.get("total_papers_searched", 0),
            processing_time=data.get("processing_time", 0.0),
            job_id=data.get("job_id"),
            status=JobStatus(data.get("status", "completed")),
            error=data.get("error"),
        )
    
    def to_bibtex(self) -> str:
        """Generate BibTeX file content from all citations."""
        bibtex_entries = []
        used_keys = set()
        
        for citation in self.citations:
            first_author = citation.authors[0].split()[-1] if citation.authors else "Unknown"
            base_key = f"{first_author}{citation.year}"
            key = base_key
            suffix = ord('a')
            
            while key in used_keys:
                key = f"{base_key}{chr(suffix)}"
                suffix += 1
            
            used_keys.add(key)
            bibtex_entries.append(citation.to_bibtex(key))
        
        return "\n".join(bibtex_entries)


class EdisonClient:
    """
    Wrapper client for Edison Scientific API using official edison-client package.

    Notes for reviewers
    -------------------
    This client is intentionally thin. It does not implement its own retrieval or
    ranking; it delegates to Edison and normalizes the result into `LiteratureResult`.
    In this codebase, the Edison stage is used to obtain a draft, citation-oriented
    literature synthesis plus a machine-readable citation list.
    
    Usage:
        client = EdisonClient()
        
        # Simple synchronous search (blocks until complete)
        result = client.search_literature_sync(
            query="What are the effects of voting rights on option pricing?"
        )
        
        # Async search
        result = await client.search_literature(
            query="What are the effects of voting rights on option pricing?"
        )
    """
    
    _UNSET = object()

    def __init__(self, api_key: Optional[str] | object = _UNSET):
        """
        Initialize Edison client.
        
        Args:
            api_key: Edison API key.
                - If omitted, defaults to EDISON_API_KEY env var.
                - If explicitly set to None, disables Edison usage (no env fallback).
        """
        if api_key is self._UNSET:
            resolved_key = os.getenv("EDISON_API_KEY")
        else:
            resolved_key = api_key

        self.api_key = resolved_key
        self._client = None
        self._init_error: Optional[str] = None

        if not self.api_key:
            logger.warning("EDISON_API_KEY not set; Edison API calls will fail")
            return

        try:
            self._client = OfficialEdisonClient(api_key=self.api_key)
            logger.info("Edison client initialized with official edison-client package")
        except Exception as e:
            # Official client may authenticate during construction; keep workflows runnable
            # and surface the error on actual calls.
            self._client = None
            self._init_error = str(e)
            logger.warning(f"Edison client initialization failed: {e}")

    @property
    def is_available(self) -> bool:
        """Whether the underlying Edison client is ready for API calls."""
        return self._client is not None

    @property
    def init_error(self) -> Optional[str]:
        """Initialization error captured during construction, if any."""
        return self._init_error
    
    async def search_literature(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> LiteratureResult:
        """
        Search literature asynchronously using Edison API.

                What this returns
                -----------------
                The returned `LiteratureResult` contains:
                - `response`: Edison-generated narrative synthesis intended to read like a
                    compact literature review with citations.
                - `citations`: a list of structured citation metadata used downstream to
                    generate `references.bib` and `citations_data.json`.
        
                The workflow treats the response as external output; it may require human
                verification before being used as final academic writing.
        
        Includes request deduplication to prevent duplicate API calls when
        workflow restarts or multiple instances run concurrently.
        
        Args:
            query: The research question to search for
            context: Optional additional context
            
        Returns:
            LiteratureResult with answer and citations
        """
        global _request_lock, _active_requests
        import asyncio
        import time
        
        if not self._client:
            return LiteratureResult(
                query=query,
                status=JobStatus.FAILED,
                error=self._init_error or "Edison API client not configured",
            )
        
        # Build the full query with context if provided
        full_query = query
        if context:
            full_query = f"{query}\n\nContext:\n{context}"
        
        # Generate hash for deduplication (based on query content)
        query_hash = hashlib.sha256(full_query.encode()).hexdigest()[:16]
        
        # Lazy-init lock for thread safety
        if _request_lock is None:
            _request_lock = asyncio.Lock()
        
        # Check for duplicate/concurrent requests
        async with _request_lock:
            current_time = time.time()
            
            # Clean up old entries (older than 30 minutes)
            expired = [h for h, t in _active_requests.items() if current_time - t > 1800]
            for h in expired:
                del _active_requests[h]
            
            # Check if this query is already in progress or was recently made
            if query_hash in _active_requests:
                elapsed = current_time - _active_requests[query_hash]
                logger.warning(
                    f"DUPLICATE Edison request detected (hash={query_hash[:8]}, "
                    f"started {elapsed:.1f}s ago). Blocking duplicate call."
                )
                return LiteratureResult(
                    query=query,
                    status=JobStatus.FAILED,
                    error=f"Duplicate request blocked. A similar query was submitted {elapsed:.1f}s ago.",
                )
            
            # Mark this request as active
            _active_requests[query_hash] = current_time
            logger.info(f"Edison request registered (hash={query_hash[:8]})")
        
        start_time = time.time()
        
        try:
            logger.info(f"Submitting literature search to Edison: {query[:100]}...")
            
            # Use async method from edison-client
            task_data = {
                "name": EdisonJobNames.LITERATURE,
                "query": full_query,
            }
            
            # arun_tasks_until_done waits for completion
            task_response = await self._client.arun_tasks_until_done(task_data)
            
            processing_time = time.time() - start_time
            logger.info(f"Edison search completed in {processing_time:.1f}s (hash={query_hash[:8]})")
            
            # Parse response
            answer = getattr(task_response, 'answer', str(task_response))
            
            # Extract citations if available
            citations = self._parse_citations(task_response)
            
            return LiteratureResult(
                query=query,
                response=answer,
                citations=citations,
                total_papers_searched=len(citations),
                processing_time=processing_time,
                job_id=getattr(task_response, 'id', None),
                status=JobStatus.COMPLETED,
            )
            
        except Exception as e:
            logger.error(f"Edison search error: {e}")
            return LiteratureResult(
                query=query,
                status=JobStatus.FAILED,
                error=str(e),
                processing_time=time.time() - start_time,
            )
        finally:
            # Remove from active requests after completion (success or failure)
            # Keep in tracking for 5 minutes to prevent rapid re-submission
            async with _request_lock:
                if query_hash in _active_requests:
                    # Update timestamp to mark completion time for dedup window
                    _active_requests[query_hash] = time.time()
    
    def search_literature_sync(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> LiteratureResult:
        """
        Search literature synchronously (blocks until complete).
        
        Args:
            query: The research question to search for
            context: Optional additional context
            
        Returns:
            LiteratureResult with answer and citations
        """
        if not self._client:
            return LiteratureResult(
                query=query,
                status=JobStatus.FAILED,
                error=self._init_error or "Edison API client not configured",
            )
        
        import time
        start_time = time.time()
        
        try:
            full_query = query
            if context:
                full_query = f"{query}\n\nContext:\n{context}"
            
            logger.info(f"Submitting literature search to Edison: {query[:100]}...")
            
            task_data = {
                "name": EdisonJobNames.LITERATURE,
                "query": full_query,
            }
            
            # Synchronous method - blocks until done
            task_response = self._client.run_tasks_until_done(task_data)
            
            processing_time = time.time() - start_time
            logger.info(f"Edison search completed in {processing_time:.1f}s")
            
            answer = getattr(task_response, 'answer', str(task_response))
            citations = self._parse_citations(task_response)
            
            return LiteratureResult(
                query=query,
                response=answer,
                citations=citations,
                total_papers_searched=len(citations),
                processing_time=processing_time,
                job_id=getattr(task_response, 'id', None),
                status=JobStatus.COMPLETED,
            )
            
        except Exception as e:
            logger.error(f"Edison search error: {e}")
            return LiteratureResult(
                query=query,
                status=JobStatus.FAILED,
                error=str(e),
                processing_time=time.time() - start_time,
            )
    
    def _parse_citations(self, task_response) -> List[Citation]:
        """Parse citations from Edison task response."""
        citations = []
        
        # Try different possible attributes where citations might be stored
        raw_citations = None
        
        # PQATaskResponse has citations attribute
        if hasattr(task_response, 'citations'):
            raw_citations = task_response.citations
        elif hasattr(task_response, 'references'):
            raw_citations = task_response.references
        elif hasattr(task_response, 'papers'):
            raw_citations = task_response.papers
        
        if not raw_citations:
            logger.debug("No citations found in Edison response")
            return citations
        
        for ref in raw_citations:
            try:
                # Handle different citation formats
                if isinstance(ref, dict):
                    citation = Citation(
                        title=ref.get('title', ''),
                        authors=ref.get('authors', []),
                        year=ref.get('year', 0),
                        journal=ref.get('journal'),
                        doi=ref.get('doi'),
                        url=ref.get('url'),
                        abstract=ref.get('abstract'),
                    )
                elif hasattr(ref, 'title'):
                    # Object with attributes
                    citation = Citation(
                        title=getattr(ref, 'title', ''),
                        authors=getattr(ref, 'authors', []),
                        year=getattr(ref, 'year', 0),
                        journal=getattr(ref, 'journal', None),
                        doi=getattr(ref, 'doi', None),
                        url=getattr(ref, 'url', None),
                        abstract=getattr(ref, 'abstract', None),
                    )
                else:
                    # String citation - parse as best as possible
                    citation = Citation(
                        title=str(ref),
                        authors=[],
                        year=0,
                    )
                
                citations.append(citation)
                
            except Exception as e:
                logger.warning(f"Failed to parse citation: {e}")
                continue
        
        logger.info(f"Parsed {len(citations)} citations from Edison response")
        return citations
