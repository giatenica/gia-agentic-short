"""
Edison Client Unit Tests
========================
Tests for Edison Scientific API client, particularly citation extraction.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.edison_client import EdisonClient, Citation, LiteratureResult, JobStatus


class TestCitationExtraction:
    """Tests for citation extraction from Edison responses."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'EDISON_API_KEY': ''}, clear=True)
    def test_extract_citations_from_formatted_answer(self):
        """Should extract citations from Edison's formatted_answer References section."""
        # Create client (will be disabled due to empty key, but we can test parsing)
        client = EdisonClient(api_key=None)
        
        # Sample formatted_answer with References section (simplified from real Edison output)
        formatted_answer = '''
Question: What is the evidence for voting rights value?

The literature shows significant voting premiums in dual-class structures.

References

1. (zingales1994thevalueof pages 1-4): Luigi Zingales. The value of the voting right: a study of the milan stock exchange experience. Review of Financial Studies, 7:125-148, Jan 1994. URL: https://doi.org/10.1093/rfs/7.1.125, doi:10.1093/rfs/7.1.125. This article has 1484 citations.

2. (merton1976optionpricingwhen pages 1-5): Robert C. Merton. Option pricing when underlying stock returns are discontinuous. Journal of Financial Economics, 3:125-144, Jan 1976. URL: https://doi.org/10.1016/0304-405x(76)90022-2, doi:10.1016/0304-405x(76)90022-2. This article has 9250 citations.

3. (masulis2009agencyproblemsat pages 1-3): RONALD W. MASULIS, CONG WANG, and FEI XIE. Agency problems at dual-class companies. The Journal of Finance, 64:1697-1727, Jul 2009. URL: https://doi.org/10.1111/j.1540-6261.2009.01477.x, doi:10.1111/j.1540-6261.2009.01477.x. This article has 886 citations.
'''
        
        citations = client._extract_citations_from_text(formatted_answer)
        
        assert len(citations) >= 3, f"Expected at least 3 citations, got {len(citations)}"
        
        # Check first citation
        zingales = next((c for c in citations if 'Zingales' in str(c.authors)), None)
        assert zingales is not None, "Should find Zingales citation"
        assert zingales.year == 1994
        assert '10.1093/rfs/7.1.125' in (zingales.doi or '')
        assert 'voting' in zingales.title.lower()
        
        # Check Merton citation
        merton = next((c for c in citations if 'Merton' in str(c.authors)), None)
        assert merton is not None, "Should find Merton citation"
        assert merton.year == 1976
        assert 'option' in merton.title.lower()
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'EDISON_API_KEY': ''}, clear=True)
    def test_extract_citations_deduplicates(self):
        """Should deduplicate citations with same title."""
        client = EdisonClient(api_key=None)
        
        formatted_answer = '''
References

1. (key1 pages 1-4): Author One. Same Title Here. Journal One, 10:1-10, 2020. doi:10.1000/test1

2. (key2 pages 5-8): Author One. Same Title Here. Journal One, 10:1-10, 2020. doi:10.1000/test1
'''
        
        citations = client._extract_citations_from_text(formatted_answer)
        
        # Should deduplicate based on title
        assert len(citations) == 1, f"Expected 1 deduplicated citation, got {len(citations)}"
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'EDISON_API_KEY': ''}, clear=True)
    def test_extract_citations_handles_no_references_section(self):
        """Should return empty list when no References section."""
        client = EdisonClient(api_key=None)
        
        formatted_answer = '''
This is just regular text without a references section.
It discusses some topics but has no citations.
'''
        
        citations = client._extract_citations_from_text(formatted_answer)
        
        assert len(citations) == 0, "Should return empty list when no References"
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'EDISON_API_KEY': ''}, clear=True)
    def test_parse_citations_prefers_structured(self):
        """Should use structured citations when available."""
        client = EdisonClient(api_key=None)
        
        # Mock response with structured citations
        @dataclass
        class MockResponse:
            citations: list
            formatted_answer: str = ""
        
        mock_response = MockResponse(
            citations=[
                {'title': 'Structured Paper', 'authors': ['Test Author'], 'year': 2023, 'doi': '10.1000/struct'}
            ],
            formatted_answer='''
References

1. (key): Text Author. Text Paper. Text Journal, 2020. doi:10.1000/text
'''
        )
        
        citations = client._parse_citations(mock_response)
        
        assert len(citations) == 1
        assert citations[0].title == 'Structured Paper'
        assert citations[0].year == 2023
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'EDISON_API_KEY': ''}, clear=True)
    def test_parse_citations_falls_back_to_text(self):
        """Should fall back to text extraction when structured citations empty."""
        client = EdisonClient(api_key=None)
        
        # Mock response with empty structured citations but text references
        @dataclass
        class MockResponse:
            citations: list
            formatted_answer: str
        
        mock_response = MockResponse(
            citations=[],
            formatted_answer='''
Some answer text.

References

1. (key): Fallback Author. Fallback Paper Title. Fallback Journal, 2021. doi:10.1000/fallback
'''
        )
        
        citations = client._parse_citations(mock_response)
        
        assert len(citations) >= 1
        assert any('Fallback' in c.title for c in citations)


class TestEdisonClientInitialization:
    """Tests for Edison client initialization."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'EDISON_API_KEY': ''}, clear=True)
    def test_client_disabled_without_key(self):
        """Client should be disabled when no API key."""
        client = EdisonClient(api_key=None)
        assert not client.is_available
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'EDISON_API_KEY': 'test-key'}, clear=True)
    @patch('src.llm.edison_client.OfficialEdisonClient')
    def test_client_initializes_with_env_key(self, mock_official_client):
        """Client should initialize with env var key."""
        client = EdisonClient()
        assert client.api_key == 'test-key'
        mock_official_client.assert_called_once_with(api_key='test-key')


class TestCitationModel:
    """Tests for Citation dataclass."""
    
    @pytest.mark.unit
    def test_citation_to_dict(self):
        """Citation.to_dict should include all fields."""
        citation = Citation(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            year=2023,
            journal="Test Journal",
            doi="10.1000/test",
            url="https://example.com",
        )
        
        d = citation.to_dict()
        
        assert d['title'] == "Test Paper"
        assert d['authors'] == ["Author One", "Author Two"]
        assert d['year'] == 2023
        assert d['journal'] == "Test Journal"
        assert d['doi'] == "10.1000/test"
    
    @pytest.mark.unit
    def test_citation_from_dict(self):
        """Citation.from_dict should reconstruct citation."""
        data = {
            'title': "Test Paper",
            'authors': ["Author One"],
            'year': 2023,
            'doi': "10.1000/test",
        }
        
        citation = Citation.from_dict(data)
        
        assert citation.title == "Test Paper"
        assert citation.year == 2023
    
    @pytest.mark.unit
    def test_citation_to_bibtex(self):
        """Citation.to_bibtex should generate valid BibTeX."""
        citation = Citation(
            title="Test Paper Title",
            authors=["John Smith", "Jane Doe"],
            year=2023,
            journal="Test Journal",
            doi="10.1000/test",
        )
        
        bibtex = citation.to_bibtex("Smith2023")
        
        assert "@article{Smith2023," in bibtex
        assert "title = {Test Paper Title}" in bibtex
        assert "author = {John Smith and Jane Doe}" in bibtex
        assert "year = {2023}" in bibtex
        assert "journal = {Test Journal}" in bibtex
        assert "doi = {10.1000/test}" in bibtex


class TestLiteratureResult:
    """Tests for LiteratureResult dataclass."""
    
    @pytest.mark.unit
    def test_literature_result_to_dict(self):
        """LiteratureResult.to_dict should serialize properly."""
        result = LiteratureResult(
            query="test query",
            response="test response",
            citations=[Citation(title="Test", authors=["Author"], year=2023)],
            status=JobStatus.COMPLETED,
        )
        
        d = result.to_dict()
        
        assert d['query'] == "test query"
        assert d['response'] == "test response"
        assert len(d['citations']) == 1
        assert d['status'] == "completed"
    
    @pytest.mark.unit
    def test_literature_result_from_dict(self):
        """LiteratureResult.from_dict should reconstruct result."""
        data = {
            'query': "test query",
            'response': "test response",
            'citations': [{'title': 'Test', 'authors': ['Author'], 'year': 2023}],
            'status': 'completed',
        }
        
        result = LiteratureResult.from_dict(data)
        
        assert result.query == "test query"
        assert len(result.citations) == 1
        assert result.status == JobStatus.COMPLETED
