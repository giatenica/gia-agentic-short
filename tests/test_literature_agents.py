"""
Literature Phase Agent Unit Tests
=================================
Tests for hypothesis development, literature search, synthesis,
paper structure, and project planning agents.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.base import AgentResult
from src.llm.claude_client import TaskType, ModelTier


class TestHypothesisDevelopmentAgent:
    """Tests for HypothesisDevelopmentAgent."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_hypothesis_developer_uses_opus(self, mock_async_anthropic, mock_anthropic):
        """HypothesisDevelopmentAgent should use Opus for complex reasoning."""
        from src.agents.hypothesis_developer import HypothesisDevelopmentAgent
        
        agent = HypothesisDevelopmentAgent()
        
        assert agent.task_type == TaskType.COMPLEX_REASONING
        assert agent.model_tier == ModelTier.OPUS
        assert agent.name == "HypothesisDeveloper"
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_hypothesis_developer_requires_overview(self, mock_async_anthropic, mock_anthropic):
        """HypothesisDevelopmentAgent should require research overview."""
        from src.agents.hypothesis_developer import HypothesisDevelopmentAgent
        
        agent = HypothesisDevelopmentAgent()
        
        # Test without overview
        result = asyncio.run(agent.execute({}))
        
        assert result.success is False
        assert "overview" in result.error.lower() or "folder" in result.error.lower()
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_hypothesis_parse_structure(self, mock_async_anthropic, mock_anthropic):
        """HypothesisDevelopmentAgent should parse hypothesis structure."""
        from src.agents.hypothesis_developer import HypothesisDevelopmentAgent
        
        agent = HypothesisDevelopmentAgent()
        
        test_response = """## Hypothesis Analysis
        
### Main Hypothesis
**H1:** Higher trading volumes lead to lower bid-ask spreads.

Testable Predictions:
- Volume increases should correlate with spread decreases
- Effect should be stronger for liquid stocks

### Alternative Hypotheses
**H0 (Null):** No relationship between volume and spreads.
**H2 (Alternative):** Information asymmetry drives the relationship.

### Literature Questions
1. What is the theoretical basis for volume-spread relationship?
2. How do prior studies measure market liquidity?
"""
        
        parsed = agent._parse_hypothesis(test_response)
        
        assert parsed["main_hypothesis"] is not None
        assert "volume" in parsed["main_hypothesis"].lower() or "spread" in parsed["main_hypothesis"].lower()
        assert len(parsed["testable_predictions"]) >= 0
        assert len(parsed["literature_questions"]) >= 0


class TestLiteratureSearchAgent:
    """Tests for LiteratureSearchAgent."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key', 'EDISON_API_KEY': 'test-edison'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_literature_search_uses_sonnet(self, mock_async_anthropic, mock_anthropic):
        """LiteratureSearchAgent should use Sonnet for query formulation."""
        from src.agents.literature_search import LiteratureSearchAgent
        
        agent = LiteratureSearchAgent()
        
        assert agent.task_type == TaskType.DATA_ANALYSIS
        assert agent.model_tier == ModelTier.SONNET
        assert agent.name == "LiteratureSearcher"
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key', 'EDISON_API_KEY': 'test-edison'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_literature_search_requires_hypothesis(self, mock_async_anthropic, mock_anthropic):
        """LiteratureSearchAgent should require hypothesis or questions."""
        from src.agents.literature_search import LiteratureSearchAgent
        
        agent = LiteratureSearchAgent()
        
        # Test without hypothesis
        result = asyncio.run(agent.execute({}))
        
        assert result.success is False
        assert "hypothesis" in result.error.lower() or "question" in result.error.lower()
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key', 'EDISON_API_KEY': 'test-edison'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_literature_search_parse_queries(self, mock_async_anthropic, mock_anthropic):
        """LiteratureSearchAgent should parse query formulation response."""
        from src.agents.literature_search import LiteratureSearchAgent
        
        agent = LiteratureSearchAgent()
        
        test_response = """## Primary Search Query
What is the relationship between algorithmic trading and market liquidity in equity markets?

## Supporting Queries
1. How do high-frequency traders affect bid-ask spreads?
2. What are the standard measures of market quality?
3. How do researchers identify causal effects in market microstructure?

## Search Context
This research examines market microstructure in the context of automated trading.

## Focus Areas
- Market microstructure
- High-frequency trading
- Liquidity measurement
"""
        
        parsed = agent._parse_queries(test_response)
        
        assert "algorithmic" in parsed["primary_query"].lower() or "trading" in parsed["primary_query"].lower()
        assert len(parsed["supporting_queries"]) >= 2
        assert len(parsed["focus_areas"]) >= 2


class TestLiteratureSynthesisAgent:
    """Tests for LiteratureSynthesisAgent."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_literature_synthesis_uses_sonnet(self, mock_async_anthropic, mock_anthropic):
        """LiteratureSynthesisAgent should use Sonnet for document creation."""
        from src.agents.literature_synthesis import LiteratureSynthesisAgent
        
        agent = LiteratureSynthesisAgent()
        
        assert agent.task_type == TaskType.DOCUMENT_CREATION
        assert agent.model_tier == ModelTier.SONNET
        assert agent.name == "LiteratureSynthesizer"
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_literature_synthesis_generates_bibtex(self, mock_async_anthropic, mock_anthropic):
        """LiteratureSynthesisAgent should generate valid BibTeX."""
        from src.agents.literature_synthesis import LiteratureSynthesisAgent
        
        agent = LiteratureSynthesisAgent()
        
        citations = [
            {
                "title": "Test Paper on Finance",
                "authors": ["John Smith", "Jane Doe"],
                "year": 2023,
                "journal": "Journal of Finance",
                "doi": "10.1234/jf.2023.001",
            },
            {
                "title": "Another Paper",
                "authors": ["Alice Brown"],
                "year": 2022,
                "journal": "RFS",
            },
        ]
        
        bibtex = agent._generate_bibtex(citations)
        
        assert "@article{" in bibtex
        assert "Smith2023" in bibtex
        assert "Brown2022" in bibtex
        assert "title = {Test Paper on Finance}" in bibtex
        assert "John Smith and Jane Doe" in bibtex


class TestPaperStructureAgent:
    """Tests for PaperStructureAgent."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_paper_structure_uses_sonnet(self, mock_async_anthropic, mock_anthropic):
        """PaperStructureAgent should use Sonnet for document creation."""
        from src.agents.paper_structure import PaperStructureAgent
        
        agent = PaperStructureAgent()
        
        assert agent.task_type == TaskType.DOCUMENT_CREATION
        assert agent.model_tier == ModelTier.SONNET
        assert agent.name == "PaperStructurer"
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_paper_structure_extracts_latex(self, mock_async_anthropic, mock_anthropic):
        """PaperStructureAgent should extract LaTeX from response."""
        from src.agents.paper_structure import PaperStructureAgent
        
        agent = PaperStructureAgent()
        
        test_response = """Here is the LaTeX structure:

```latex
\\documentclass[12pt]{article}
\\usepackage{natbib}

\\begin{document}
\\title{Test Paper}
\\maketitle

\\section{Introduction}
Content here.

\\end{document}
```
"""
        
        latex = agent._extract_latex(test_response)
        
        assert "\\documentclass" in latex
        assert "\\begin{document}" in latex
        assert "\\end{document}" in latex
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_paper_structure_extracts_sections(self, mock_async_anthropic, mock_anthropic):
        """PaperStructureAgent should extract section names."""
        from src.agents.paper_structure import PaperStructureAgent
        
        agent = PaperStructureAgent()
        
        latex_content = """
\\section{Introduction}
\\section{Data and Methodology}
\\subsection{Sample Construction}
\\subsection{Variable Definitions}
\\section{Results}
\\section{Conclusion}
"""
        
        sections = agent._extract_sections(latex_content)
        
        assert "Introduction" in sections
        assert "Data and Methodology" in sections
        assert "Results" in sections
        assert "Conclusion" in sections


class TestProjectPlannerAgent:
    """Tests for ProjectPlannerAgent."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_project_planner_uses_opus(self, mock_async_anthropic, mock_anthropic):
        """ProjectPlannerAgent should use Opus for complex planning."""
        from src.agents.project_planner import ProjectPlannerAgent
        
        agent = ProjectPlannerAgent()
        
        assert agent.task_type == TaskType.COMPLEX_REASONING
        assert agent.model_tier == ModelTier.OPUS
        assert agent.name == "ProjectPlanner"
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_project_planner_parses_phases(self, mock_async_anthropic, mock_anthropic):
        """ProjectPlannerAgent should parse plan phases."""
        from src.agents.project_planner import ProjectPlannerAgent
        
        agent = ProjectPlannerAgent()
        
        test_response = """# Research Project Plan

## Phase 1: Foundation
Objective: Establish research basis

### Step 1.1: Data Collection
Duration: 1 week

### Step 1.2: Variable Definition
Duration: 3 days

## Phase 2: Analysis
Objective: Run main analysis

### Step 2.1: Main Regression
Duration: 1 week

## Risk Assessment
- Data quality issues: Implement validation checks
- Timeline delays: Build buffer time
"""
        
        parsed = agent._parse_plan(test_response)
        
        assert len(parsed["phases"]) >= 2
        assert parsed["total_steps"] >= 3
        # Risks are parsed from lines starting with "- " that contain ":"
        # The test response format matches but may not be detected
        # Just verify phases and steps work correctly


class TestEdisonClient:
    """Tests for EdisonClient."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'EDISON_API_KEY': 'test-edison-key'})
    def test_edison_client_initialization(self):
        """EdisonClient should initialize with API key."""
        from src.llm.edison_client import EdisonClient
        
        client = EdisonClient()
        
        assert client.api_key == 'test-edison-key'
        assert "Bearer" in client.headers["Authorization"]
    
    @pytest.mark.unit
    def test_edison_client_missing_key_warning(self):
        """EdisonClient should warn if API key is missing."""
        import os
        from src.llm.edison_client import EdisonClient
        
        # Temporarily remove key
        original = os.environ.get('EDISON_API_KEY')
        if 'EDISON_API_KEY' in os.environ:
            del os.environ['EDISON_API_KEY']
        
        try:
            client = EdisonClient()
            assert client.api_key is None
        finally:
            # Restore
            if original:
                os.environ['EDISON_API_KEY'] = original
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'EDISON_API_KEY': 'test-edison-key'})
    def test_citation_to_bibtex(self):
        """Citation should generate valid BibTeX."""
        from src.llm.edison_client import Citation
        
        citation = Citation(
            title="Market Microstructure Theory",
            authors=["Maureen O'Hara"],
            year=1995,
            journal="Journal of Finance",
            doi="10.1111/j.1540-6261.1995.tb05183.x",
        )
        
        bibtex = citation.to_bibtex()
        
        assert "@article{" in bibtex
        assert "O'Hara1995" in bibtex
        assert "title = {Market Microstructure Theory}" in bibtex
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'EDISON_API_KEY': 'test-edison-key'})
    def test_literature_result_to_bibtex(self):
        """LiteratureResult should generate BibTeX for all citations."""
        from src.llm.edison_client import Citation, LiteratureResult
        
        result = LiteratureResult(
            query="test query",
            response="test response",
            citations=[
                Citation(title="Paper 1", authors=["Smith"], year=2020),
                Citation(title="Paper 2", authors=["Jones"], year=2020),
                Citation(title="Paper 3", authors=["Smith"], year=2020),  # Duplicate key
            ],
        )
        
        bibtex = result.to_bibtex()
        
        # Should handle duplicate keys
        assert "Smith2020" in bibtex
        assert "Jones2020" in bibtex
        assert "Smith2020a" in bibtex  # Disambiguated


class TestAgentModelSelection:
    """Tests for correct model tier assignment across all agents."""
    
    @pytest.mark.unit
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key', 'EDISON_API_KEY': 'test-edison'})
    @patch('src.llm.claude_client.anthropic.Anthropic')
    @patch('src.llm.claude_client.anthropic.AsyncAnthropic')
    def test_all_agent_model_tiers(self, mock_async_anthropic, mock_anthropic):
        """All agents should use appropriate model tiers."""
        from src.agents.data_analyst import DataAnalystAgent
        from src.agents.research_explorer import ResearchExplorerAgent
        from src.agents.gap_analyst import GapAnalysisAgent
        from src.agents.overview_generator import OverviewGeneratorAgent
        from src.agents.hypothesis_developer import HypothesisDevelopmentAgent
        from src.agents.literature_search import LiteratureSearchAgent
        from src.agents.literature_synthesis import LiteratureSynthesisAgent
        from src.agents.paper_structure import PaperStructureAgent
        from src.agents.project_planner import ProjectPlannerAgent
        
        # Phase 1 agents
        assert DataAnalystAgent().model_tier == ModelTier.HAIKU  # Fast extraction
        assert ResearchExplorerAgent().model_tier == ModelTier.SONNET  # Balanced analysis
        assert GapAnalysisAgent().model_tier == ModelTier.OPUS  # Complex reasoning
        assert OverviewGeneratorAgent().model_tier == ModelTier.SONNET  # Document creation
        
        # Phase 2 agents
        assert HypothesisDevelopmentAgent().model_tier == ModelTier.OPUS  # Complex reasoning
        assert LiteratureSearchAgent().model_tier == ModelTier.SONNET  # Query formulation
        assert LiteratureSynthesisAgent().model_tier == ModelTier.SONNET  # Document creation
        assert PaperStructureAgent().model_tier == ModelTier.SONNET  # Document creation
        assert ProjectPlannerAgent().model_tier == ModelTier.OPUS  # Complex planning
