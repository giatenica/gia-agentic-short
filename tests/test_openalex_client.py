"""Tests for OpenAlex API client.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.llm.openalex_client import (
    OpenAlexClient,
    OpenAlexWork,
    OpenAlexSearchResult,
    search_openalex,
)


class TestOpenAlexWork:
    """Test OpenAlexWork dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        work = OpenAlexWork(
            openalex_id="W123456",
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            year=2023,
            venue="Test Journal",
            abstract="This is a test abstract.",
            doi="10.1234/test",
            url="https://openalex.org/W123456",
            citation_count=100,
            is_open_access=True,
        )
        
        result = work.to_dict()
        
        assert result["openalex_id"] == "W123456"
        assert result["title"] == "Test Paper"
        assert result["authors"] == ["John Doe", "Jane Smith"]
        assert result["year"] == 2023
        assert result["journal"] == "Test Journal"
        assert result["doi"] == "10.1234/test"
        assert result["citations"] == 100
        assert result["is_open_access"] is True
    
    def test_to_dict_defaults(self):
        """Test dictionary conversion with default values."""
        work = OpenAlexWork(
            openalex_id="W123456",
            title="Test Paper",
        )
        
        result = work.to_dict()
        
        assert result["year"] == 0
        assert result["authors"] == []
        assert result["doi"] is None
        assert result["is_open_access"] is False


class TestOpenAlexClient:
    """Test OpenAlexClient."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return OpenAlexClient(email="test@example.com")
    
    def test_init_with_email(self):
        """Test initialization with email."""
        client = OpenAlexClient(email="my@email.com")
        assert client.email == "my@email.com"
    
    @patch.dict("os.environ", {"OPENALEX_EMAIL": "env@email.com"}, clear=True)
    def test_init_from_env(self):
        """Test email from environment."""
        # Need to re-import to get fresh DEFAULT_MAILTO
        import importlib
        import src.llm.openalex_client as oac
        importlib.reload(oac)
        client = oac.OpenAlexClient()
        assert client.email == "env@email.com"
    
    def test_get_params_with_email(self, client):
        """Test params include mailto."""
        params = client._get_params()
        assert params["mailto"] == "test@example.com"
    
    def test_get_params_without_email(self):
        """Test params without email."""
        client = OpenAlexClient(email=None)
        client.email = ""
        params = client._get_params()
        assert "mailto" not in params
    
    def test_parse_work_basic(self):
        """Test parsing basic work data."""
        data = {
            "id": "https://openalex.org/W123456",
            "title": "Test Paper",
            "publication_year": 2023,
            "cited_by_count": 50,
        }
        
        work = OpenAlexClient._parse_work(data)
        
        assert work.openalex_id == "https://openalex.org/W123456"
        assert work.title == "Test Paper"
        assert work.year == 2023
        assert work.citation_count == 50
    
    def test_parse_work_with_authors(self):
        """Test parsing work with authors."""
        data = {
            "id": "https://openalex.org/W123456",
            "title": "Test Paper",
            "authorships": [
                {"author": {"display_name": "John Doe"}},
                {"author": {"display_name": "Jane Smith"}},
            ],
        }
        
        work = OpenAlexClient._parse_work(data)
        
        assert work.authors == ["John Doe", "Jane Smith"]
    
    def test_parse_work_with_institutions(self):
        """Test parsing work with institutions."""
        data = {
            "id": "https://openalex.org/W123456",
            "title": "Test Paper",
            "authorships": [
                {
                    "author": {"display_name": "John Doe"},
                    "institutions": [
                        {"display_name": "MIT"},
                        {"display_name": "Harvard"},
                    ],
                },
            ],
        }
        
        work = OpenAlexClient._parse_work(data)
        
        assert "MIT" in work.author_institutions
        assert "Harvard" in work.author_institutions
    
    def test_parse_work_with_doi(self):
        """Test parsing work with DOI."""
        data = {
            "id": "https://openalex.org/W123456",
            "title": "Test Paper",
            "doi": "https://doi.org/10.1234/test",
        }
        
        work = OpenAlexClient._parse_work(data)
        
        assert work.doi == "10.1234/test"  # Prefix should be removed
    
    def test_parse_work_with_open_access(self):
        """Test parsing work with open access info."""
        data = {
            "id": "https://openalex.org/W123456",
            "title": "Test Paper",
            "open_access": {
                "is_oa": True,
                "oa_status": "gold",
                "oa_url": "https://example.com/paper.pdf",
            },
        }
        
        work = OpenAlexClient._parse_work(data)
        
        assert work.is_open_access is True
        assert work.oa_status == "gold"
        assert work.open_access_url == "https://example.com/paper.pdf"
    
    def test_parse_work_with_concepts(self):
        """Test parsing work with concepts."""
        data = {
            "id": "https://openalex.org/W123456",
            "title": "Test Paper",
            "concepts": [
                {"id": "C123", "display_name": "Machine Learning", "level": 1, "score": 0.9},
                {"id": "C456", "display_name": "Finance", "level": 0, "score": 0.7},
            ],
        }
        
        work = OpenAlexClient._parse_work(data)
        
        assert len(work.concepts) == 2
        assert work.concepts[0]["name"] == "Machine Learning"
        assert work.concepts[1]["name"] == "Finance"
    
    @pytest.mark.asyncio
    async def test_search_works_success(self, client):
        """Test successful work search."""
        mock_response = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Test Paper 1",
                    "publication_year": 2023,
                },
                {
                    "id": "https://openalex.org/W2",
                    "title": "Test Paper 2",
                    "publication_year": 2022,
                },
            ],
            "meta": {"count": 100},
        }
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            ))
            mock_get_client.return_value = mock_http
            
            result = await client.search_works("corporate governance")
            
            assert len(result.works) == 2
            assert result.total_count == 100
            assert result.works[0].title == "Test Paper 1"
            assert result.works[1].title == "Test Paper 2"
    
    @pytest.mark.asyncio
    async def test_search_works_empty(self, client):
        """Test search with no results."""
        mock_response = {"results": [], "meta": {"count": 0}}
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            ))
            mock_get_client.return_value = mock_http
            
            result = await client.search_works("nonexistent topic xyz")
            
            assert len(result.works) == 0
            assert result.total_count == 0
    
    @pytest.mark.asyncio
    async def test_get_work_success(self, client):
        """Test getting a single work."""
        mock_response = {
            "id": "https://openalex.org/W123456",
            "title": "Test Paper",
            "publication_year": 2023,
        }
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            ))
            mock_get_client.return_value = mock_http
            
            work = await client.get_work("W123456")
            
            assert work is not None
            assert work.title == "Test Paper"
    
    @pytest.mark.asyncio
    async def test_get_work_by_doi(self, client):
        """Test getting a work by DOI."""
        mock_response = {
            "id": "https://openalex.org/W123456",
            "title": "Test Paper",
            "doi": "https://doi.org/10.1234/test",
        }
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            ))
            mock_get_client.return_value = mock_http
            
            work = await client.get_work("10.1234/test")
            
            assert work is not None
            assert work.doi == "10.1234/test"
    
    @pytest.mark.asyncio
    async def test_get_work_not_found(self, client):
        """Test getting a work that doesn't exist."""
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http
            
            work = await client.get_work("nonexistent")
            
            assert work is None
    
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
    async def test_enrich_with_open_access(self, client):
        """Test enriching DOIs with open access URLs."""
        mock_response = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Paper 1",
                    "doi": "https://doi.org/10.1234/a",
                    "best_oa_location": {"pdf_url": "https://example.com/a.pdf"},
                },
            ],
            "meta": {"count": 1},
        }
        
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
                raise_for_status=MagicMock(),
            ))
            mock_get_client.return_value = mock_http
            
            result = await client.enrich_with_open_access(["10.1234/a", "10.1234/b"])
            
            assert result["10.1234/a"] == "https://example.com/a.pdf"
            assert result["10.1234/b"] is None


class TestConvenienceFunction:
    """Test convenience search function."""
    
    @pytest.mark.asyncio
    async def test_search_openalex(self):
        """Test convenience search function."""
        with patch("src.llm.openalex_client.OpenAlexClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.search_works = AsyncMock(return_value=OpenAlexSearchResult(
                works=[OpenAlexWork(
                    openalex_id="W123456",
                    title="Test Paper",
                    year=2023,
                )],
                total_count=1,
                query="test",
            ))
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance
            
            result = await search_openalex("test query")
            
            assert len(result) == 1
            assert result[0]["title"] == "Test Paper"
