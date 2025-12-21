"""
Edison Scientific API Client
=============================
Client for interacting with Edison Scientific's research platform API.
Supports literature search and other research tasks.

API Documentation: https://platform.edisonscientific.com/profile (for API key)
FAQ: https://edisonscientific.com/faqs

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import asyncio
import httpx
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger


class JobNames(Enum):
    """Edison API job types."""
    LITERATURE = "literature"  # Literature search with citations
    DATA_ANALYSIS = "data_analysis"  # Data synthesis and analysis
    TARGET_DISCOVERY = "target_discovery"  # Target identification
    KOSMOS = "kosmos"  # Deep research (12-48 hours)


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
    
    def to_bibtex(self, key: Optional[str] = None) -> str:
        """Convert citation to BibTeX format."""
        if not key:
            # Generate key from first author and year
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
        )


@dataclass
class LiteratureResult:
    """Result from a literature search."""
    query: str
    response: str
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
            # Generate unique key
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
    Client for Edison Scientific API.
    
    Usage:
        async with EdisonClient() as client:
            # Submit literature search
            job_id = await client.submit_literature_search(
                query="What are the effects of algorithmic trading on market quality?",
                max_papers=50
            )
            
            # Wait for results (can take up to 20 minutes)
            result = await client.wait_for_result(job_id, timeout=1200)
    
    Or for simple usage:
        client = EdisonClient()
        job_id = await client.submit_literature_search(query)
        result = await client.wait_for_result(job_id)
    """
    
    BASE_URL = "https://api.edisonscientific.com/v1"
    DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)  # 60s total, 10s connect
    
    def __init__(self, api_key: Optional[str] = None, timeout: Optional[httpx.Timeout] = None):
        """
        Initialize Edison client.
        
        Args:
            api_key: Edison API key (defaults to EDISON_API_KEY env var)
            timeout: Optional custom timeout configuration
        """
        self.api_key = api_key or os.getenv("EDISON_API_KEY")
        if not self.api_key:
            logger.warning("EDISON_API_KEY not set - Edison API calls will fail")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }
        
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
        
        logger.info("Edison client initialized")
    
    async def __aenter__(self) -> "EdisonClient":
        """Async context manager entry - creates connection pool."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - closes connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client, creating one if needed (for non-context-manager usage)."""
        if self._client is None:
            # Create a single-use client for backward compatibility
            return httpx.AsyncClient(timeout=self.timeout, headers=self.headers)
        return self._client
    
    async def submit_literature_search(
        self,
        query: str,
        context: Optional[str] = None,
        max_papers: int = 50,
        focus_areas: Optional[List[str]] = None,
    ) -> str:
        """
        Submit a literature search job.
        
        Args:
            query: The research question or topic to search
            context: Additional context about the research project
            max_papers: Maximum number of papers to return
            focus_areas: Specific research areas to focus on
            
        Returns:
            Job ID for tracking the search
        """
        payload = {
            "job_type": JobNames.LITERATURE.value,
            "query": query,
            "parameters": {
                "max_results": max_papers,
                "include_abstracts": True,
                "include_citations": True,
            }
        }
        
        if context:
            payload["context"] = context
        
        if focus_areas:
            payload["parameters"]["focus_areas"] = focus_areas
        
        client = await self._get_client()
        should_close = self._client is None  # Close if we created a new client
        
        try:
            response = await client.post(
                f"{self.BASE_URL}/jobs",
                json=payload,
            )
            
            if response.status_code == 401:
                raise ValueError("Invalid Edison API key")
            elif response.status_code == 429:
                raise ValueError("Edison API rate limit exceeded")
            elif response.status_code not in (200, 201):
                raise ValueError(f"Edison API error: {response.status_code} - {response.text}")
            
            data = response.json()
            job_id = data.get("job_id") or data.get("id")
            
            logger.info(f"Submitted literature search job: {job_id}")
            return job_id
        finally:
            if should_close:
                await client.aclose()
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a job.
        
        Args:
            job_id: The job ID to check
            
        Returns:
            Job status information
        """
        client = await self._get_client()
        should_close = self._client is None
        
        try:
            response = await client.get(f"{self.BASE_URL}/jobs/{job_id}")
            
            if response.status_code != 200:
                raise ValueError(f"Failed to get job status: {response.status_code}")
            
            return response.json()
        finally:
            if should_close:
                await client.aclose()
    
    async def get_job_result(self, job_id: str) -> LiteratureResult:
        """
        Get the result of a completed job.
        
        Args:
            job_id: The job ID to get results for
            
        Returns:
            LiteratureResult with response and citations
        """
        client = await self._get_client()
        should_close = self._client is None
        
        try:
            response = await client.get(f"{self.BASE_URL}/jobs/{job_id}/result")
            
            if response.status_code != 200:
                raise ValueError(f"Failed to get job result: {response.status_code}")
            
            data = response.json()
            
            # Parse citations from response
            citations = []
            for paper in data.get("papers", data.get("citations", [])):
                citations.append(Citation(
                    title=paper.get("title", ""),
                    authors=paper.get("authors", []),
                    year=paper.get("year", 0),
                    journal=paper.get("journal"),
                    doi=paper.get("doi"),
                    url=paper.get("url"),
                    abstract=paper.get("abstract"),
                    relevance_score=paper.get("relevance_score"),
                ))
            
            return LiteratureResult(
                query=data.get("query", ""),
                response=data.get("response", data.get("answer", "")),
                citations=citations,
                total_papers_searched=data.get("total_searched", len(citations)),
                processing_time=data.get("processing_time", 0.0),
                job_id=job_id,
                status=JobStatus.COMPLETED,
            )
        finally:
            if should_close:
                await client.aclose()
    
    async def wait_for_result(
        self,
        job_id: str,
        timeout: int = 1200,  # 20 minutes default
        poll_interval: int = 30,
        callback: Optional[callable] = None,
    ) -> LiteratureResult:
        """
        Wait for a job to complete and return results.
        
        Args:
            job_id: The job ID to wait for
            timeout: Maximum wait time in seconds (default: 20 minutes)
            poll_interval: How often to check status in seconds
            callback: Optional callback function called with status updates
            
        Returns:
            LiteratureResult when job completes
        """
        start_time = datetime.now()
        elapsed = 0
        
        logger.info(f"Waiting for job {job_id} (timeout: {timeout}s)")
        
        while elapsed < timeout:
            status_data = await self.get_job_status(job_id)
            status = status_data.get("status", "unknown")
            
            if callback:
                callback(status, elapsed, status_data)
            
            logger.debug(f"Job {job_id} status: {status} (elapsed: {elapsed}s)")
            
            if status == "completed":
                return await self.get_job_result(job_id)
            elif status == "failed":
                error_msg = status_data.get("error", "Unknown error")
                return LiteratureResult(
                    query="",
                    response="",
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error=error_msg,
                )
            
            await asyncio.sleep(poll_interval)
            elapsed = (datetime.now() - start_time).total_seconds()
        
        # Timeout
        logger.warning(f"Job {job_id} timed out after {timeout}s")
        return LiteratureResult(
            query="",
            response="",
            job_id=job_id,
            status=JobStatus.TIMEOUT,
            error=f"Job timed out after {timeout} seconds",
        )
    
    async def health_check(self) -> bool:
        """Check if Edison API is accessible."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.BASE_URL}/health",
                    headers=self.headers,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Edison health check failed: {e}")
            return False


# Convenience function for quick literature search
async def search_literature(
    query: str,
    context: Optional[str] = None,
    max_papers: int = 50,
    timeout: int = 1200,
) -> LiteratureResult:
    """
    Convenience function to perform a literature search.
    
    Args:
        query: Research question or topic
        context: Additional context
        max_papers: Maximum papers to return
        timeout: Maximum wait time in seconds
        
    Returns:
        LiteratureResult with response and citations
    """
    client = EdisonClient()
    job_id = await client.submit_literature_search(query, context, max_papers)
    return await client.wait_for_result(job_id, timeout)
