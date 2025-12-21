"""
Agent Registry
==============
Centralized registry of all agents with unique IDs, capabilities,
input/output schemas, and inter-agent call permissions.

This enables:
- Agent discovery and invocation by ID
- Inter-agent communication with permission enforcement
- Capability-based agent selection
- Documentation of the agent ecosystem

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Type, Any, Callable
from enum import Enum
import importlib
from loguru import logger


class AgentCapability(Enum):
    """Capabilities that agents can have."""
    # Data capabilities
    DATA_ANALYSIS = "data_analysis"
    DATA_VALIDATION = "data_validation"
    CODE_EXECUTION = "code_execution"
    
    # Research capabilities
    HYPOTHESIS_DEVELOPMENT = "hypothesis_development"
    LITERATURE_SEARCH = "literature_search"
    LITERATURE_SYNTHESIS = "literature_synthesis"
    GAP_ANALYSIS = "gap_analysis"
    
    # Writing capabilities
    DOCUMENT_GENERATION = "document_generation"
    PAPER_STRUCTURING = "paper_structuring"
    PROJECT_PLANNING = "project_planning"
    
    # Quality capabilities
    CRITICAL_REVIEW = "critical_review"
    CONSISTENCY_CHECK = "consistency_check"
    STYLE_ENFORCEMENT = "style_enforcement"  # Writing style validation
    
    # Meta capabilities
    OVERVIEW_GENERATION = "overview_generation"
    RESEARCH_EXPLORATION = "research_exploration"

class ModelTier(Enum):
    """Model tiers for agent execution."""
    OPUS = "opus"      # Complex reasoning, critical review
    SONNET = "sonnet"  # Balanced tasks, document creation
    HAIKU = "haiku"    # Fast, simple tasks


@dataclass
class AgentInputSchema:
    """Defines required and optional inputs for an agent."""
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class AgentOutputSchema:
    """Defines what an agent produces."""
    content_type: str = "text"  # text, structured, file
    structured_fields: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class AgentSpec:
    """Complete specification for an agent."""
    id: str                              # Unique ID (A01, A02, etc.)
    name: str                            # Human-readable name
    class_name: str                      # Python class name
    module_path: str                     # Module path for import
    model_tier: ModelTier                # Which model tier it uses
    capabilities: List[AgentCapability]  # What it can do
    input_schema: AgentInputSchema       # What it needs
    output_schema: AgentOutputSchema     # What it produces
    description: str                     # What it does
    can_call: List[str] = field(default_factory=list)  # Agent IDs it can invoke
    max_iterations: int = 3              # Max self-revision iterations
    supports_revision: bool = True       # Whether it can revise its output
    uses_extended_thinking: bool = False # Whether it uses thinking mode


# Agent Registry Definition
AGENT_REGISTRY: Dict[str, AgentSpec] = {
    # ========== Phase 1: Initial Analysis Agents ==========
    "A01": AgentSpec(
        id="A01",
        name="DataAnalyst",
        class_name="DataAnalystAgent",
        module_path="src.agents.data_analyst",
        model_tier=ModelTier.HAIKU,
        capabilities=[
            AgentCapability.DATA_ANALYSIS,
            AgentCapability.DATA_VALIDATION,
            AgentCapability.CODE_EXECUTION,
        ],
        input_schema=AgentInputSchema(
            required=["project_folder"],
            optional=["project_data"],
            description="Analyzes data files in the project folder",
        ),
        output_schema=AgentOutputSchema(
            content_type="structured",
            structured_fields=["data_summary", "schema_info", "quality_metrics"],
            description="Data analysis report with statistics and quality assessment",
        ),
        description="Examines data files using Python, generates statistics, schema info, and data quality assessments",
        can_call=[],  # Leaf agent, doesn't call others
        supports_revision=True,
        uses_extended_thinking=False,
    ),
    
    "A02": AgentSpec(
        id="A02",
        name="ResearchExplorer",
        class_name="ResearchExplorerAgent",
        module_path="src.agents.research_explorer",
        model_tier=ModelTier.SONNET,
        capabilities=[
            AgentCapability.RESEARCH_EXPLORATION,
        ],
        input_schema=AgentInputSchema(
            required=["project_folder"],
            optional=["project_data", "data_analysis"],
            description="Analyzes project submissions to extract research components",
        ),
        output_schema=AgentOutputSchema(
            content_type="structured",
            structured_fields=["research_question", "hypothesis_clarity", "methodology_assessment"],
            description="Research analysis with component extraction",
        ),
        description="Analyzes project submissions, extracts research components, assesses hypothesis clarity",
        can_call=[],
        supports_revision=True,
        uses_extended_thinking=False,
    ),
    
    "A03": AgentSpec(
        id="A03",
        name="GapAnalyst",
        class_name="GapAnalysisAgent",
        module_path="src.agents.gap_analyst",
        model_tier=ModelTier.OPUS,
        capabilities=[
            AgentCapability.GAP_ANALYSIS,
            AgentCapability.CRITICAL_REVIEW,
        ],
        input_schema=AgentInputSchema(
            required=["project_folder"],
            optional=["project_data", "data_analysis", "research_analysis"],
            description="Identifies gaps in research plan",
        ),
        output_schema=AgentOutputSchema(
            content_type="structured",
            structured_fields=["critical_gaps", "action_plan", "priority_ranking"],
            description="Gap analysis with prioritized action plan",
        ),
        description="Identifies missing research elements, creates prioritized action plans",
        can_call=["A01", "A02"],  # Can request data/research analysis
        supports_revision=True,
        uses_extended_thinking=True,
    ),
    
    "A04": AgentSpec(
        id="A04",
        name="OverviewGenerator",
        class_name="OverviewGeneratorAgent",
        module_path="src.agents.overview_generator",
        model_tier=ModelTier.SONNET,
        capabilities=[
            AgentCapability.OVERVIEW_GENERATION,
            AgentCapability.DOCUMENT_GENERATION,
        ],
        input_schema=AgentInputSchema(
            required=["project_folder"],
            optional=["project_data", "data_analysis", "research_analysis", "gap_analysis"],
            description="Synthesizes findings into overview document",
        ),
        output_schema=AgentOutputSchema(
            content_type="file",
            files_created=["RESEARCH_OVERVIEW.md"],
            description="Research overview document",
        ),
        description="Synthesizes all findings into RESEARCH_OVERVIEW.md",
        can_call=["A03"],  # Can request gap analysis review
        supports_revision=True,
        uses_extended_thinking=False,
    ),
    
    # ========== Phase 2: Literature Agents ==========
    "A05": AgentSpec(
        id="A05",
        name="HypothesisDeveloper",
        class_name="HypothesisDevelopmentAgent",
        module_path="src.agents.hypothesis_developer",
        model_tier=ModelTier.OPUS,
        capabilities=[
            AgentCapability.HYPOTHESIS_DEVELOPMENT,
        ],
        input_schema=AgentInputSchema(
            required=["project_folder"],
            optional=["research_overview", "project_data"],
            description="Develops testable hypotheses from research overview",
        ),
        output_schema=AgentOutputSchema(
            content_type="structured",
            structured_fields=["main_hypothesis", "alternative_hypotheses", "literature_questions"],
            description="Hypothesis formulation with literature questions",
        ),
        description="Formulates testable hypotheses, identifies literature questions",
        can_call=["A03"],  # Can request gap check
        supports_revision=True,
        uses_extended_thinking=True,
    ),
    
    "A06": AgentSpec(
        id="A06",
        name="LiteratureSearcher",
        class_name="LiteratureSearchAgent",
        module_path="src.agents.literature_search",
        model_tier=ModelTier.SONNET,
        capabilities=[
            AgentCapability.LITERATURE_SEARCH,
        ],
        input_schema=AgentInputSchema(
            required=["hypothesis_result"],
            optional=["research_overview", "literature_questions"],
            description="Searches literature via Edison API",
        ),
        output_schema=AgentOutputSchema(
            content_type="structured",
            structured_fields=["search_results", "citations", "relevance_scores"],
            description="Literature search results with citations",
        ),
        description="Formulates Edison API queries, executes searches, returns citations",
        can_call=[],  # Calls external API, not other agents
        supports_revision=True,
        uses_extended_thinking=False,
    ),
    
    "A07": AgentSpec(
        id="A07",
        name="LiteratureSynthesizer",
        class_name="LiteratureSynthesisAgent",
        module_path="src.agents.literature_synthesis",
        model_tier=ModelTier.SONNET,
        capabilities=[
            AgentCapability.LITERATURE_SYNTHESIS,
            AgentCapability.DOCUMENT_GENERATION,
        ],
        input_schema=AgentInputSchema(
            required=["literature_result", "hypothesis_result", "project_folder"],
            optional=[],
            description="Synthesizes literature search results",
        ),
        output_schema=AgentOutputSchema(
            content_type="file",
            files_created=["LITERATURE_SUMMARY.md", "references.bib"],
            description="Literature synthesis documents",
        ),
        description="Processes search results, creates LITERATURE_SUMMARY.md and references.bib",
        can_call=["A06"],  # Can request additional searches
        supports_revision=True,
        uses_extended_thinking=False,
    ),
    
    "A08": AgentSpec(
        id="A08",
        name="PaperStructurer",
        class_name="PaperStructureAgent",
        module_path="src.agents.paper_structure",
        model_tier=ModelTier.SONNET,
        capabilities=[
            AgentCapability.PAPER_STRUCTURING,
            AgentCapability.DOCUMENT_GENERATION,
        ],
        input_schema=AgentInputSchema(
            required=["project_folder"],
            optional=["hypothesis_result", "literature_synthesis", "project_data"],
            description="Creates paper structure based on style guide",
        ),
        output_schema=AgentOutputSchema(
            content_type="file",
            files_created=["paper/main.tex", "paper/STRUCTURE.md"],
            description="LaTeX paper structure",
        ),
        description="Creates LaTeX paper structure based on style guide",
        can_call=[],
        supports_revision=True,
        uses_extended_thinking=False,
    ),
    
    "A09": AgentSpec(
        id="A09",
        name="ProjectPlanner",
        class_name="ProjectPlannerAgent",
        module_path="src.agents.project_planner",
        model_tier=ModelTier.OPUS,
        capabilities=[
            AgentCapability.PROJECT_PLANNING,
            AgentCapability.DOCUMENT_GENERATION,
        ],
        input_schema=AgentInputSchema(
            required=["project_folder"],
            optional=["hypothesis_result", "literature_result", "literature_synthesis", 
                     "paper_structure", "project_data"],
            description="Creates detailed project plan",
        ),
        output_schema=AgentOutputSchema(
            content_type="file",
            files_created=["PROJECT_PLAN.md"],
            description="Detailed project plan with phases and timeline",
        ),
        description="Creates detailed project plans with phases, steps, timelines",
        can_call=["A03", "A05"],  # Can check gaps, verify hypothesis
        supports_revision=True,
        uses_extended_thinking=True,
    ),
    
    # ========== Phase 3: Gap Resolution Agents ==========
    "A10": AgentSpec(
        id="A10",
        name="GapResolver",
        class_name="GapResolverAgent",
        module_path="src.agents.gap_resolver",
        model_tier=ModelTier.SONNET,
        capabilities=[
            AgentCapability.GAP_ANALYSIS,
            AgentCapability.CODE_EXECUTION,
            AgentCapability.DATA_ANALYSIS,
        ],
        input_schema=AgentInputSchema(
            required=["project_folder", "gap_analysis"],
            optional=["project_data", "data_analysis"],
            description="Resolves identified gaps",
        ),
        output_schema=AgentOutputSchema(
            content_type="structured",
            structured_fields=["resolved_gaps", "execution_results", "remaining_gaps"],
            description="Gap resolution results",
        ),
        description="Generates and executes Python code to resolve data gaps",
        can_call=["A01", "A03", "A13"],  # Can request data analysis, gap recheck, style validation
        supports_revision=True,
        uses_extended_thinking=False,
    ),
    
    "A11": AgentSpec(
        id="A11",
        name="OverviewUpdater",
        class_name="OverviewUpdaterAgent",
        module_path="src.agents.gap_resolution_workflow",  # Part of gap resolution
        model_tier=ModelTier.OPUS,
        capabilities=[
            AgentCapability.OVERVIEW_GENERATION,
            AgentCapability.DOCUMENT_GENERATION,
        ],
        input_schema=AgentInputSchema(
            required=["project_folder", "gap_resolution_results"],
            optional=["original_overview"],
            description="Updates overview with resolved gaps",
        ),
        output_schema=AgentOutputSchema(
            content_type="file",
            files_created=["UPDATED_RESEARCH_OVERVIEW.md"],
            description="Updated research overview",
        ),
        description="Creates UPDATED_RESEARCH_OVERVIEW.md with gap resolutions",
        can_call=["A03", "A13"],  # Can verify gaps are resolved, validate style
        supports_revision=True,
        uses_extended_thinking=True,
    ),
    
    # ========== Quality Assurance Agents ==========
    "A12": AgentSpec(
        id="A12",
        name="CriticalReviewer",
        class_name="CriticalReviewAgent",
        module_path="src.agents.critical_review",
        model_tier=ModelTier.OPUS,
        capabilities=[
            AgentCapability.CRITICAL_REVIEW,
            AgentCapability.CONSISTENCY_CHECK,
        ],
        input_schema=AgentInputSchema(
            required=["content", "content_type"],
            optional=["quality_criteria", "source_agent_id"],
            description="Reviews any agent output for quality",
        ),
        output_schema=AgentOutputSchema(
            content_type="structured",
            structured_fields=["quality_scores", "issues", "feedback", "revision_required"],
            description="Quality assessment with feedback",
        ),
        description="Evaluates agent outputs against quality criteria, generates structured feedback",
        can_call=["A14"],  # Can call ConsistencyChecker for cross-document validation
        supports_revision=False,  # Reviewer doesn't revise its own output
        uses_extended_thinking=True,
    ),
    
    "A13": AgentSpec(
        id="A13",
        name="StyleEnforcer",
        class_name="StyleEnforcerAgent",
        module_path="src.agents.style_enforcer",
        model_tier=ModelTier.HAIKU,  # Fast validation, no complex reasoning needed
        capabilities=[
            AgentCapability.STYLE_ENFORCEMENT,
        ],
        input_schema=AgentInputSchema(
            required=["text"],
            optional=["content_type", "auto_fix", "is_final_output"],
            description="Validates text against writing style guide",
        ),
        output_schema=AgentOutputSchema(
            content_type="structured",
            structured_fields=["banned_words", "word_counts", "style_score", "suggestions"],
            files_created=[],
            description="Style validation report with issues and fixes",
        ),
        description="Validates LaTeX output against writing style guide; checks banned words, word counts, formatting",
        can_call=[],  # Pure validation, doesn't call other agents
        supports_revision=False,  # Validator doesn't produce revisable content
        uses_extended_thinking=False,
    ),
    
    "A14": AgentSpec(
        id="A14",
        name="ConsistencyChecker",
        class_name="ConsistencyCheckerAgent",
        module_path="src.agents.consistency_checker",
        model_tier=ModelTier.SONNET,  # Needs reasoning for comparison
        capabilities=[
            AgentCapability.CONSISTENCY_CHECK,
            AgentCapability.CRITICAL_REVIEW,
        ],
        input_schema=AgentInputSchema(
            required=["project_folder"],
            optional=["focus_categories"],
            description="Validates consistency across all project documents",
        ),
        output_schema=AgentOutputSchema(
            content_type="structured",
            structured_fields=["issues", "element_mapping", "consistency_score", "is_consistent"],
            files_created=[],
            description="Cross-document consistency report",
        ),
        description="Validates hypotheses, variables, methodology, citations, statistics across RESEARCH_OVERVIEW.md, LITERATURE_SUMMARY.md, PROJECT_PLAN.md, paper/",
        can_call=["A12"],  # Can request deeper review from CriticalReviewer
        supports_revision=False,  # Validator doesn't produce revisable content
        uses_extended_thinking=False,
    ),
}


class AgentRegistry:
    """
    Centralized registry for agent management and discovery.
    
    Provides:
    - Agent lookup by ID, name, or capability
    - Permission checking for inter-agent calls
    - Agent instantiation
    - Registry introspection
    """
    
    _instance = None
    _agents: Dict[str, AgentSpec] = AGENT_REGISTRY
    _loaded_classes: Dict[str, Type] = {}
    
    def __new__(cls):
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get(cls, agent_id: str) -> Optional[AgentSpec]:
        """Get agent spec by ID."""
        return cls._agents.get(agent_id)
    
    @classmethod
    def get_by_name(cls, name: str) -> Optional[AgentSpec]:
        """Get agent spec by name."""
        for spec in cls._agents.values():
            if spec.name == name or spec.class_name == name:
                return spec
        return None
    
    @classmethod
    def get_by_capability(cls, capability: AgentCapability) -> List[AgentSpec]:
        """Get all agents with a specific capability."""
        return [
            spec for spec in cls._agents.values()
            if capability in spec.capabilities
        ]
    
    @classmethod
    def get_by_model_tier(cls, tier: ModelTier) -> List[AgentSpec]:
        """Get all agents using a specific model tier."""
        return [
            spec for spec in cls._agents.values()
            if spec.model_tier == tier
        ]
    
    @classmethod
    def can_call(cls, caller_id: str, target_id: str) -> bool:
        """Check if caller agent is permitted to call target agent."""
        caller = cls.get(caller_id)
        if not caller:
            return False
        return target_id in caller.can_call
    
    @classmethod
    def get_callable_agents(cls, agent_id: str) -> List[AgentSpec]:
        """Get list of agents that can be called by given agent."""
        agent = cls.get(agent_id)
        if not agent:
            return []
        return [cls.get(aid) for aid in agent.can_call if cls.get(aid)]
    
    @classmethod
    def load_agent_class(cls, agent_id: str) -> Optional[Type]:
        """
        Dynamically load and return the agent class.
        
        Caches loaded classes for reuse.
        """
        if agent_id in cls._loaded_classes:
            return cls._loaded_classes[agent_id]
        
        spec = cls.get(agent_id)
        if not spec:
            logger.error(f"Agent {agent_id} not found in registry")
            return None
        
        try:
            module = importlib.import_module(spec.module_path)
            agent_class = getattr(module, spec.class_name)
            cls._loaded_classes[agent_id] = agent_class
            return agent_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load agent {agent_id}: {e}")
            return None
    
    @classmethod
    def create_agent(cls, agent_id: str, **kwargs) -> Optional[Any]:
        """
        Create an instance of an agent by ID.
        
        Args:
            agent_id: Agent ID (e.g., "A01")
            **kwargs: Arguments to pass to agent constructor
            
        Returns:
            Agent instance or None if not found/failed
        """
        agent_class = cls.load_agent_class(agent_id)
        if not agent_class:
            return None
        
        try:
            return agent_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate agent {agent_id}: {e}")
            return None
    
    @classmethod
    def list_all(cls) -> List[AgentSpec]:
        """Get all registered agents."""
        return list(cls._agents.values())
    
    @classmethod
    def list_ids(cls) -> List[str]:
        """Get all agent IDs."""
        return list(cls._agents.keys())
    
    @classmethod
    def get_permissions_matrix(cls) -> Dict[str, List[str]]:
        """Get the complete inter-agent call permissions matrix."""
        return {
            agent_id: spec.can_call
            for agent_id, spec in cls._agents.items()
        }
    
    @classmethod
    def summary(cls) -> str:
        """Generate a human-readable summary of all agents."""
        lines = ["# Agent Registry Summary", ""]
        
        # Group by phase
        phases = {
            "Phase 1 - Initial Analysis": ["A01", "A02", "A03", "A04"],
            "Phase 2 - Literature": ["A05", "A06", "A07", "A08", "A09"],
            "Phase 3 - Gap Resolution": ["A10", "A11"],
            "Quality Assurance": ["A12", "A13", "A14"],
        }
        
        for phase_name, agent_ids in phases.items():
            lines.append(f"## {phase_name}")
            lines.append("")
            lines.append("| ID | Name | Model | Capabilities | Can Call |")
            lines.append("|-----|------|-------|--------------|----------|")
            
            for aid in agent_ids:
                spec = cls.get(aid)
                if spec:
                    caps = ", ".join(c.value for c in spec.capabilities[:2])
                    if len(spec.capabilities) > 2:
                        caps += "..."
                    calls = ", ".join(spec.can_call) if spec.can_call else "-"
                    lines.append(
                        f"| {spec.id} | {spec.name} | {spec.model_tier.value} | {caps} | {calls} |"
                    )
            lines.append("")
        
        return "\n".join(lines)


# Convenience function for quick access
def get_agent(agent_id: str) -> Optional[AgentSpec]:
    """Get agent specification by ID."""
    return AgentRegistry.get(agent_id)


def can_agent_call(caller_id: str, target_id: str) -> bool:
    """Check if caller can invoke target agent."""
    return AgentRegistry.can_call(caller_id, target_id)
