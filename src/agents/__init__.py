"""
Research Agents Module
======================
Multi-agent system for research paper analysis and preparation.

Agents:
- DataAnalystAgent: Examines uploaded data files with Python
- ResearchExplorerAgent: Analyzes what the user has provided
- GapAnalysisAgent: Identifies missing elements for research
- OverviewGeneratorAgent: Creates comprehensive research overview

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from .base import BaseAgent, AgentResult
from .data_analyst import DataAnalystAgent
from .research_explorer import ResearchExplorerAgent
from .gap_analyst import GapAnalysisAgent
from .overview_generator import OverviewGeneratorAgent
from .workflow import ResearchWorkflow

__all__ = [
    "BaseAgent",
    "AgentResult",
    "DataAnalystAgent",
    "ResearchExplorerAgent",
    "GapAnalysisAgent",
    "OverviewGeneratorAgent",
    "ResearchWorkflow",
]
