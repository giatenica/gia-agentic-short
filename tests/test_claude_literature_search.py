"""Tests for Claude Literature Search.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json

from src.llm.claude_literature_search import (
    ClaudeLiteratureSearch,
    ClaudeLiteratureSearchResult,
    LiteratureSearchConfig,
    EvaluatedPaper,
)


class TestLiteratureSearchConfig:
    """Test LiteratureSearchConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LiteratureSearchConfig()
        
        assert config.max_papers_per_source == 30
        assert config.max_papers_total == 50
        assert config.evidence_k == 15
        assert config.answer_max_sources == 8
        assert config.min_relevance_score == 5.0
        assert config.use_opus_for_synthesis is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = LiteratureSearchConfig(
            max_papers_total=100,
            evidence_k=20,
            answer_max_sources=10,
            min_relevance_score=7.0,
        )
        
        assert config.max_papers_total == 100
        assert config.evidence_k == 20
        assert config.answer_max_sources == 10
        assert config.min_relevance_score == 7.0


class TestEvaluatedPaper:
    """Test EvaluatedPaper dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        paper = EvaluatedPaper(
            paper_id="abc123",
            title="Test Paper",
            authors=["John Doe"],
            year=2023,
            venue="Test Journal",
            abstract="Abstract text",
            doi="10.1234/test",
            url="https://example.com",
            source="semantic_scholar",
            relevance_score=8.5,
            relevance_rationale="Highly relevant",
            key_findings=["Finding 1", "Finding 2"],
            citation_recommendation="must-cite",
        )
        
        result = paper.to_dict()
        
        assert result["paper_id"] == "abc123"
        assert result["title"] == "Test Paper"
        assert result["relevance_score"] == 8.5
        assert result["citation_recommendation"] == "must-cite"
    
    def test_to_citation_dict(self):
        """Test conversion to citation format."""
        paper = EvaluatedPaper(
            paper_id="abc123",
            title="Test Paper",
            authors=["John Doe"],
            year=2023,
            venue="Test Journal",
            abstract="Abstract text",
            doi="10.1234/test",
            url="https://example.com",
            source="semantic_scholar",
            relevance_score=8.5,
        )
        
        result = paper.to_citation_dict()
        
        assert result["title"] == "Test Paper"
        assert result["year"] == 2023
        assert result["journal"] == "Test Journal"
        assert result["doi"] == "10.1234/test"
        assert result["relevance_score"] == 8.5


class TestClaudeLiteratureSearchResult:
    """Test ClaudeLiteratureSearchResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ClaudeLiteratureSearchResult(
            response="Literature review text",
            citations=[{"title": "Paper 1"}],
            total_papers_found=50,
            papers_evaluated=15,
            papers_included=8,
            processing_time=120.5,
            tokens_used=5000,
            sources_used=["semantic_scholar", "openalex"],
        )
        
        data = result.to_dict()
        
        assert data["response"] == "Literature review text"
        assert len(data["citations"]) == 1
        assert data["total_papers_searched"] == 50
        assert data["provider"] == "claude_literature_search"
        assert data["status"] == "completed"


class TestClaudeLiteratureSearch:
    """Test ClaudeLiteratureSearch class."""
    
    @pytest.fixture
    def mock_claude_client(self):
        """Create mock Claude client."""
        mock = MagicMock()
        mock.complete = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_s2_client(self):
        """Create mock Semantic Scholar client."""
        mock = MagicMock()
        mock.search_papers = AsyncMock()
        mock.close = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_openalex_client(self):
        """Create mock OpenAlex client."""
        mock = MagicMock()
        mock.search_works = AsyncMock()
        mock.close = AsyncMock()
        return mock
    
    @pytest.fixture
    def search(self, mock_claude_client, mock_s2_client, mock_openalex_client):
        """Create search instance with mocked clients."""
        return ClaudeLiteratureSearch(
            config=LiteratureSearchConfig(
                evidence_k=3,
                answer_max_sources=2,
            ),
            claude_client=mock_claude_client,
            semantic_scholar_client=mock_s2_client,
            openalex_client=mock_openalex_client,
        )
    
    @pytest.mark.asyncio
    async def test_search_returns_result(self, search, mock_claude_client, mock_s2_client, mock_openalex_client):
        """Test that search returns a ClaudeLiteratureSearchResult."""
        # Mock query decomposition
        mock_claude_client.complete.side_effect = [
            # Query decomposition response
            MagicMock(
                content=json.dumps({
                    "aspect_queries": [
                        {"aspect": "main", "query": "test query", "rationale": "test"}
                    ],
                    "key_terms": ["term1"],
                }),
                usage=MagicMock(total_tokens=100),
            ),
            # Paper evaluation responses
            MagicMock(
                content=json.dumps({
                    "relevance_score": 8.0,
                    "relevance_rationale": "Very relevant",
                    "key_findings": ["finding1"],
                    "citable_claims": [],
                }),
                usage=MagicMock(total_tokens=200),
            ),
            MagicMock(
                content=json.dumps({
                    "relevance_score": 6.0,
                    "relevance_rationale": "Moderately relevant",
                    "key_findings": ["finding2"],
                    "citable_claims": [],
                }),
                usage=MagicMock(total_tokens=200),
            ),
            # Synthesis response
            MagicMock(
                content=json.dumps({
                    "evidence_for_hypothesis": [],
                    "evidence_against_hypothesis": [],
                    "synthesis_narrative": "Synthesis text",
                }),
                usage=MagicMock(total_tokens=300),
            ),
            # Literature review response
            MagicMock(
                content="This is the literature review text.",
                usage=MagicMock(total_tokens=400),
            ),
        ]
        
        # Mock Semantic Scholar results
        from src.llm.semantic_scholar_client import SemanticScholarPaper, SemanticScholarSearchResult
        mock_s2_client.search_papers.return_value = SemanticScholarSearchResult(
            papers=[
                SemanticScholarPaper(
                    paper_id="s2_1",
                    title="Paper from S2",
                    authors=["Author 1"],
                    year=2023,
                    abstract="Abstract 1",
                ),
            ],
            total=1,
            query="test",
        )
        
        # Mock OpenAlex results
        from src.llm.openalex_client import OpenAlexWork, OpenAlexSearchResult
        mock_openalex_client.search_works.return_value = OpenAlexSearchResult(
            works=[
                OpenAlexWork(
                    openalex_id="oa_1",
                    title="Paper from OpenAlex",
                    authors=["Author 2"],
                    year=2022,
                    abstract="Abstract 2",
                ),
            ],
            total_count=1,
            query="test",
        )
        
        result = await search.search(
            hypothesis="Test hypothesis about something",
            questions=["Question 1?"],
            domain="Finance",
        )
        
        assert isinstance(result, ClaudeLiteratureSearchResult)
        assert result.response == "This is the literature review text."
        assert result.provider == "claude_literature_search"
        assert result.tokens_used > 0
    
    @pytest.mark.asyncio
    async def test_search_handles_no_papers(self, search, mock_claude_client, mock_s2_client, mock_openalex_client):
        """Test search handles case when no papers are found."""
        # Mock query decomposition
        mock_claude_client.complete.return_value = MagicMock(
            content=json.dumps({
                "aspect_queries": [
                    {"aspect": "main", "query": "test query", "rationale": "test"}
                ],
            }),
            usage=MagicMock(total_tokens=100),
        )
        
        # Mock empty results from both sources
        from src.llm.semantic_scholar_client import SemanticScholarSearchResult
        from src.llm.openalex_client import OpenAlexSearchResult
        
        mock_s2_client.search_papers.return_value = SemanticScholarSearchResult(
            papers=[],
            total=0,
            query="test",
        )
        mock_openalex_client.search_works.return_value = OpenAlexSearchResult(
            works=[],
            total_count=0,
            query="test",
        )
        
        result = await search.search(
            hypothesis="Test hypothesis",
        )
        
        assert isinstance(result, ClaudeLiteratureSearchResult)
        assert "No relevant literature found" in result.response
        assert len(result.citations) == 0
    
    @pytest.mark.asyncio
    async def test_search_handles_api_errors(self, search, mock_claude_client, mock_s2_client, mock_openalex_client):
        """Test search handles API errors gracefully."""
        # Mock query decomposition
        mock_claude_client.complete.return_value = MagicMock(
            content=json.dumps({
                "aspect_queries": [
                    {"aspect": "main", "query": "test query", "rationale": "test"}
                ],
            }),
            usage=MagicMock(total_tokens=100),
        )
        
        # Mock API errors
        mock_s2_client.search_papers.side_effect = Exception("API Error")
        mock_openalex_client.search_works.side_effect = Exception("API Error")
        
        result = await search.search(
            hypothesis="Test hypothesis",
        )
        
        assert isinstance(result, ClaudeLiteratureSearchResult)
        assert len(result.citations) == 0
    
    @pytest.mark.asyncio
    async def test_close(self, search, mock_s2_client, mock_openalex_client):
        """Test closing the search instance."""
        await search.close()
        
        mock_s2_client.close.assert_called_once()
        mock_openalex_client.close.assert_called_once()
    
    def test_property_creates_clients_lazily(self):
        """Test that clients are created lazily."""
        search = ClaudeLiteratureSearch()
        
        # Initially no clients
        assert search._claude_client is None
        assert search._s2_client is None
        assert search._openalex_client is None


class TestQueryDecomposition:
    """Test query decomposition stage."""
    
    @pytest.mark.asyncio
    async def test_decompose_queries_parses_json(self):
        """Test that query decomposition parses JSON response."""
        mock_claude = MagicMock()
        mock_claude.complete = AsyncMock(return_value=MagicMock(
            content="""```json
{
    "aspect_queries": [
        {"aspect": "theoretical", "query": "voting premium theory", "rationale": "Find foundational work"},
        {"aspect": "empirical", "query": "dual class shares empirical evidence", "rationale": "Find evidence"}
    ],
    "key_terms": ["voting premium", "dual-class"],
    "related_fields": ["corporate governance"]
}
```""",
            usage=MagicMock(total_tokens=150),
        ))
        
        search = ClaudeLiteratureSearch(claude_client=mock_claude)
        
        queries, tokens = await search._decompose_queries(
            hypothesis="Dual-class shares trade at a premium",
            questions=["What drives the premium?"],
            domain="Finance",
            context="",
        )
        
        assert len(queries) == 2
        assert queries[0]["aspect"] == "theoretical"
        assert queries[1]["aspect"] == "empirical"
        assert tokens == 150
    
    @pytest.mark.asyncio
    async def test_decompose_queries_handles_parse_error(self):
        """Test fallback when JSON parsing fails."""
        mock_claude = MagicMock()
        mock_claude.complete = AsyncMock(return_value=MagicMock(
            content="This is not valid JSON",
            usage=MagicMock(total_tokens=50),
        ))
        
        search = ClaudeLiteratureSearch(claude_client=mock_claude)
        
        queries, tokens = await search._decompose_queries(
            hypothesis="Test hypothesis",
            questions=["Question 1", "Question 2"],
            domain="Finance",
            context="",
        )
        
        # Should fall back to hypothesis and questions
        assert len(queries) >= 1
        assert queries[0]["query"] == "Test hypothesis"


class TestPaperRetrieval:
    """Test paper retrieval stage."""
    
    @pytest.mark.asyncio
    async def test_retrieve_deduplicates_papers(self):
        """Test that papers are deduplicated across sources."""
        from src.llm.semantic_scholar_client import SemanticScholarPaper, SemanticScholarSearchResult
        from src.llm.openalex_client import OpenAlexWork, OpenAlexSearchResult
        
        mock_s2 = MagicMock()
        mock_s2.search_papers = AsyncMock(return_value=SemanticScholarSearchResult(
            papers=[
                SemanticScholarPaper(
                    paper_id="s2_1",
                    title="Same Paper Title",  # Same title
                    authors=["Author 1"],
                    year=2023,
                ),
            ],
            total=1,
            query="test",
        ))
        
        mock_openalex = MagicMock()
        mock_openalex.search_works = AsyncMock(return_value=OpenAlexSearchResult(
            works=[
                OpenAlexWork(
                    openalex_id="oa_1",
                    title="Same Paper Title",  # Same title (duplicate)
                    authors=["Author 1"],
                    year=2023,
                ),
                OpenAlexWork(
                    openalex_id="oa_2",
                    title="Different Paper",  # Different title
                    authors=["Author 2"],
                    year=2022,
                ),
            ],
            total_count=2,
            query="test",
        ))
        
        search = ClaudeLiteratureSearch(
            semantic_scholar_client=mock_s2,
            openalex_client=mock_openalex,
        )
        
        aspect_queries = [{"query": "test query", "aspect": "main"}]
        papers, sources = await search._retrieve_papers(aspect_queries)
        
        # Should have 2 unique papers (deduplicated by title)
        assert len(papers) == 2
        assert "semantic_scholar" in sources
        assert "openalex" in sources


class TestPaperEvaluation:
    """Test paper evaluation stage."""
    
    @pytest.mark.asyncio
    async def test_evaluate_paper_parses_response(self):
        """Test that paper evaluation parses response correctly."""
        mock_claude = MagicMock()
        mock_claude.complete = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "relevance_score": 8.5,
                "relevance_rationale": "Directly relevant to hypothesis",
                "key_findings": ["Finding 1", "Finding 2"],
                "citable_claims": [
                    {"claim": "Claim text", "evidence_type": "empirical", "strength": "strong"}
                ],
                "methodology": "Panel regression",
                "limitations": ["Small sample"],
                "citation_recommendation": "must-cite",
            }),
            usage=MagicMock(total_tokens=300),
        ))
        
        search = ClaudeLiteratureSearch(claude_client=mock_claude)
        
        paper_data = {
            "paper_id": "abc123",
            "title": "Test Paper",
            "authors": ["John Doe"],
            "year": 2023,
            "venue": "Test Journal",
            "abstract": "This is the abstract",
            "doi": "10.1234/test",
            "url": "https://example.com",
            "source": "semantic_scholar",
        }
        
        evaluated, tokens = await search._evaluate_single_paper(
            paper=paper_data,
            hypothesis="Test hypothesis",
        )
        
        assert evaluated.relevance_score == 8.5
        assert evaluated.citation_recommendation == "must-cite"
        assert len(evaluated.key_findings) == 2
        assert tokens == 300
