"""Tests for Semantic Scholar API client.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

from src.llm.semantic_scholar_client import (
    SemanticScholarClient,
    SemanticScholarPaper,
    SemanticScholarSearchResult,
    search_semantic_scholar,
)


class TestSemanticScholarPaper:
    """Test SemanticScholarPaper dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        paper = SemanticScholarPaper(
            paper_id="abc123",
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            year=2023,
            venue="Test Journal",
            abstract="This is a test abstract.",
            doi="10.1234/test",
            url="https://example.com/paper",
            citation_count=100,
        )
        
        result = paper.to_dict()
        
        assert result["paper_id"] == "abc123"
        assert result["title"] == "Test Paper"
        assert result["authors"] == ["John Doe", "Jane Smith"]
        assert result["year"] == 2023
        assert result["journal"] == "Test Journal"
        assert result["doi"] == "10.1234/test"
        assert result["citations"] == 100
    
    def test_to_dict_defaults(self):
        """Test dictionary conversion with default values."""
        paper = SemanticScholarPaper(
            paper_id="abc123",
            title="Test Paper",
        )
        
        result = paper.to_dict()
        
        assert result["year"] == 0
        assert result["authors"] == []
        assert result["doi"] is None


class TestSemanticScholarClient:
    """Test SemanticScholarClient."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return SemanticScholarClient(api_key="test-key")
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = SemanticScholarClient(api_key="my-key")
        assert client.api_key == "my-key"
    
    @patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": "env-key"})
    def test_init_from_env(self):
        """Test API key from environment."""
        client = SemanticScholarClient()
        assert client.api_key == "env-key"
    
    def test_get_headers_with_key(self, client):
        """Test headers include API key."""
        headers = client._get_headers()
        assert headers["x-api-key"] == "test-key"
        assert headers["Accept"] == "application/json"
    
    def test_get_headers_without_key(self):
        """Test headers without API key."""
        client = SemanticScholarClient(api_key=None)
        client.api_key = None
        headers = client._get_headers()
        assert "x-api-key" not in headers
    
    def test_parse_paper_basic(self):
        """Test parsing basic paper data."""
        data = {
            "paperId": "abc123",
            "title": "Test Paper",
            "year": 2023,
            "citationCount": 50,
        }
        
        paper = SemanticScholarClient._parse_paper(data)
        
        assert paper.paper_id == "abc123"
        assert paper.title == "Test Paper"
        assert paper.year == 2023
        assert paper.citation_count == 50
    
    def test_parse_paper_with_authors(self):
        """Test parsing paper with authors."""
        data = {
            "paperId": "abc123",
            "title": "Test Paper",
            "authors": [
                {"name": "John Doe"},
                {"name": "Jane Smith"},
            ],
        }
        
        paper = SemanticScholarClient._parse_paper(data)
        
        assert paper.authors == ["John Doe", "Jane Smith"]
    
    def test_parse_paper_with_external_ids(self):
        """Test parsing paper with external IDs."""
        data = {
            "paperId": "abc123",
            "title": "Test Paper",
            "externalIds": {
                "DOI": "10.1234/test",
                "ArXiv": "2301.12345",
            },
        }
        
        paper = SemanticScholarClient._parse_paper(data)
        
        assert paper.doi == "10.1234/test"
        assert paper.arxiv_id == "2301.12345"
    
    def test_parse_paper_with_tldr(self):
        """Test parsing paper with TLDR."""
        data = {
            "paperId": "abc123",
            "title": "Test Paper",
            "tldr": {"text": "This is a summary."},
        }
        
        paper = SemanticScholarClient._parse_paper(data)
        
        assert paper.tldr == "This is a summary."
    
    @pytest.mark.asyncio
    async def test_search_papers_success(self, client):
        """Test successful paper search."""
        mock_response = {
            "data": [
                {
                    "paperId": "abc123",
                    "title": "Test Paper 1",
                    "year": 2023,
                    "authors": [{"name": "John Doe"}],
                },
                {
                    "paperId": "def456",
                    "title": "Test Paper 2",
                    "year": 2022,
                    "authors": [{"name": "Jane Smith"}],
                },
            ],
            "total": 100,
        }
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            ))
            mock_get_client.return_value = mock_http
            
            result = await client.search_papers("machine learning")
            
            assert len(result.papers) == 2
            assert result.total == 100
            assert result.papers[0].title == "Test Paper 1"
            assert result.papers[1].title == "Test Paper 2"
    
    @pytest.mark.asyncio
    async def test_search_papers_empty(self, client):
        """Test search with no results."""
        mock_response = {"data": [], "total": 0}
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            ))
            mock_get_client.return_value = mock_http
            
            result = await client.search_papers("nonexistent topic xyz")
            
            assert len(result.papers) == 0
            assert result.total == 0
    
    @pytest.mark.asyncio
    async def test_get_paper_success(self, client):
        """Test getting a single paper."""
        mock_response = {
            "paperId": "abc123",
            "title": "Test Paper",
            "year": 2023,
        }
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            ))
            mock_get_client.return_value = mock_http
            
            paper = await client.get_paper("abc123")
            
            assert paper is not None
            assert paper.paper_id == "abc123"
            assert paper.title == "Test Paper"
    
    @pytest.mark.asyncio
    async def test_get_paper_not_found(self, client):
        """Test getting a paper that doesn't exist."""
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http
            
            paper = await client.get_paper("nonexistent")
            
            assert paper is None
    
    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test closing the client."""
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.aclose = AsyncMock()
        client._client = mock_http
        
        await client.close()
        
        mock_http.aclose.assert_called_once()
        assert client._client is None
    
    @pytest.mark.asyncio
    async def test_search_with_relevance_reranking(self, client):
        """Test search with relevance reranking."""
        mock_response = {
            "data": [
                {
                    "paperId": "1",
                    "title": "Low Citations",
                    "year": 2020,
                    "citationCount": 10,
                },
                {
                    "paperId": "2",
                    "title": "High Citations",
                    "year": 2023,
                    "citationCount": 1000,
                    "isOpenAccess": True,
                },
            ],
            "total": 2,
        }
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            ))
            mock_get_client.return_value = mock_http
            
            result = await client.search_with_relevance_reranking(
                "test query",
                limit=2,
            )
            
            # High citations paper should be ranked first
            assert result.papers[0].title == "High Citations"


class TestConvenienceFunction:
    """Test convenience search function."""
    
    @pytest.mark.asyncio
    async def test_search_semantic_scholar(self):
        """Test convenience search function."""
        mock_response = {
            "data": [
                {
                    "paperId": "abc123",
                    "title": "Test Paper",
                    "year": 2023,
                },
            ],
            "total": 1,
        }
        
        with patch("src.llm.semantic_scholar_client.SemanticScholarClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.search_papers = AsyncMock(return_value=SemanticScholarSearchResult(
                papers=[SemanticScholarPaper(
                    paper_id="abc123",
                    title="Test Paper",
                    year=2023,
                )],
                total=1,
                query="test",
            ))
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance
            
            result = await search_semantic_scholar("test query")
            
            assert len(result) == 1
            assert result[0]["title"] == "Test Paper"
