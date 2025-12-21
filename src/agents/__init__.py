"""
Research Agents Module
======================
Multi-agent system for research paper analysis and preparation.

Phase 1 Agents (Initial Analysis):
- DataAnalystAgent: Examines uploaded data files with Python
- ResearchExplorerAgent: Analyzes what the user has provided
- GapAnalysisAgent: Identifies missing elements for research
- OverviewGeneratorAgent: Creates comprehensive research overview

Phase 2 Agents (Literature and Planning):
- HypothesisDevelopmentAgent: Formulates testable research hypotheses
- LiteratureSearchAgent: Searches literature via Edison Scientific API
- LiteratureSynthesisAgent: Synthesizes literature and creates .bib file
- PaperStructureAgent: Creates LaTeX paper structure
- ProjectPlannerAgent: Creates detailed project plan

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from .base import BaseAgent, AgentResult
from .data_analyst import DataAnalystAgent
from .research_explorer import ResearchExplorerAgent
from .gap_analyst import GapAnalysisAgent
from .overview_generator import OverviewGeneratorAgent
from .hypothesis_developer import HypothesisDevelopmentAgent
from .literature_search import LiteratureSearchAgent
from .literature_synthesis import LiteratureSynthesisAgent
from .paper_structure import PaperStructureAgent
from .project_planner import ProjectPlannerAgent
from .workflow import ResearchWorkflow
from .literature_workflow import LiteratureWorkflow

__all__ = [
    # Base
    "BaseAgent",
    "AgentResult",
    # Phase 1: Initial Analysis
    "DataAnalystAgent",
    "ResearchExplorerAgent",
    "GapAnalysisAgent",
    "OverviewGeneratorAgent",
    "ResearchWorkflow",
    # Phase 2: Literature and Planning
    "HypothesisDevelopmentAgent",
    "LiteratureSearchAgent",
    "LiteratureSynthesisAgent",
    "PaperStructureAgent",
    "ProjectPlannerAgent",
    "LiteratureWorkflow",
]
