"""
Readiness Scoring Utilities for Research Projects.

Comprehensive tracking of paper readiness at phase/step/substep levels.
Tracks automation status and identifies gaps requiring additional capability.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from loguru import logger

from .project_io import get_project_id
from src.config import INTAKE_SERVER


class ReadinessCategory(Enum):
    """Categories of readiness assessment."""
    DATA = "data"
    LITERATURE = "literature"
    HYPOTHESES = "hypotheses"
    METHODOLOGY = "methodology"
    ANALYSIS = "analysis"
    WRITING = "writing"
    REVIEW = "review"
    STRUCTURE = "structure"
    AUTOMATION = "automation"


class CheckStatus(Enum):
    """Status of a readiness check."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_STARTED = "not_started"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class AutomationCapability(Enum):
    """Automation capability status for a task."""
    FULLY_AUTOMATED = "fully_automated"  # Agent can complete autonomously
    PARTIALLY_AUTOMATED = "partially_automated"  # Agent can assist but needs enhancement
    NEEDS_CAPABILITY = "needs_capability"  # Requires new agent/capability
    PENDING_IMPLEMENTATION = "pending_implementation"  # Capability exists, not yet implemented


@dataclass
class ChecklistItem:
    """Individual readiness checklist item."""
    item_id: str
    category: ReadinessCategory
    description: str
    status: CheckStatus = CheckStatus.NOT_STARTED
    
    # Evidence tracking
    evidence_file: Optional[str] = None
    evidence_section: Optional[str] = None
    evidence_notes: Optional[str] = None
    
    # Automation tracking
    automation_capability: AutomationCapability = AutomationCapability.PENDING_IMPLEMENTATION
    assigned_agent: Optional[str] = None
    required_capabilities: List[str] = field(default_factory=list)
    
    # Progress tracking
    completion_percentage: float = 0.0
    last_checked: Optional[str] = None
    last_updated_by: Optional[str] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    
    def mark_complete(self, agent_name: Optional[str] = None, evidence: Optional[str] = None):
        """Mark item as complete."""
        self.status = CheckStatus.COMPLETE
        self.completion_percentage = 100.0
        self.last_checked = datetime.now().isoformat()
        if agent_name:
            self.last_updated_by = agent_name
            self.automation_capability = AutomationCapability.FULLY_AUTOMATED
        if evidence:
            self.evidence_notes = evidence
    
    def mark_partial(self, percentage: float, agent_name: Optional[str] = None):
        """Mark item as partially complete."""
        self.status = CheckStatus.PARTIAL
        self.completion_percentage = min(100.0, max(0.0, percentage))
        self.last_checked = datetime.now().isoformat()
        if agent_name:
            self.last_updated_by = agent_name
    
    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "category": self.category.value,
            "description": self.description,
            "status": self.status.value,
            "evidence_file": self.evidence_file,
            "evidence_section": self.evidence_section,
            "evidence_notes": self.evidence_notes,
            "automation_capability": self.automation_capability.value,
            "assigned_agent": self.assigned_agent,
            "required_capabilities": self.required_capabilities,
            "completion_percentage": self.completion_percentage,
            "last_checked": self.last_checked,
            "last_updated_by": self.last_updated_by,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ChecklistItem":
        return cls(
            item_id=data["item_id"],
            category=ReadinessCategory(data["category"]),
            description=data["description"],
            status=CheckStatus(data.get("status", "not_started")),
            evidence_file=data.get("evidence_file"),
            evidence_section=data.get("evidence_section"),
            evidence_notes=data.get("evidence_notes"),
            automation_capability=AutomationCapability(data.get("automation_capability", "pending_implementation")),
            assigned_agent=data.get("assigned_agent"),
            required_capabilities=data.get("required_capabilities", []),
            completion_percentage=data.get("completion_percentage", 0.0),
            last_checked=data.get("last_checked"),
            last_updated_by=data.get("last_updated_by"),
            depends_on=data.get("depends_on", []),
            blocks=data.get("blocks", []),
        )


@dataclass
class PhaseReadiness:
    """Readiness assessment for a project phase."""
    phase_id: str
    phase_name: str
    
    # Checklist items for this phase
    items: List[ChecklistItem] = field(default_factory=list)
    
    # Phase-level tracking
    status: CheckStatus = CheckStatus.NOT_STARTED
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Automation summary
    fully_automated_count: int = 0
    needs_capability_count: int = 0
    
    @property
    def completion_rate(self) -> float:
        """Overall completion rate for this phase."""
        if not self.items:
            return 0.0
        return sum(i.completion_percentage for i in self.items) / len(self.items) / 100
    
    @property
    def complete_items(self) -> int:
        """Count of complete items."""
        return sum(1 for i in self.items if i.status == CheckStatus.COMPLETE)
    
    @property
    def total_items(self) -> int:
        """Total items in this phase."""
        return len(self.items)
    
    def update_automation_counts(self):
        """Update automation summary counts."""
        self.fully_automated_count = sum(
            1 for i in self.items 
            if i.automation_capability == AutomationCapability.FULLY_AUTOMATED
        )
        self.needs_capability_count = sum(
            1 for i in self.items 
            if i.automation_capability == AutomationCapability.NEEDS_CAPABILITY
        )
    
    def to_dict(self) -> dict:
        self.update_automation_counts()
        return {
            "phase_id": self.phase_id,
            "phase_name": self.phase_name,
            "items": [i.to_dict() for i in self.items],
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "completion_rate": self.completion_rate,
            "complete_items": self.complete_items,
            "total_items": self.total_items,
            "fully_automated_count": self.fully_automated_count,
            "needs_capability_count": self.needs_capability_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PhaseReadiness":
        phase = cls(
            phase_id=data["phase_id"],
            phase_name=data["phase_name"],
            status=CheckStatus(data.get("status", "not_started")),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )
        phase.items = [ChecklistItem.from_dict(i) for i in data.get("items", [])]
        phase.update_automation_counts()
        return phase


@dataclass
class ReadinessReport:
    """Comprehensive paper readiness report."""
    project_id: str
    project_folder: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Phase-level readiness
    phases: List[PhaseReadiness] = field(default_factory=list)
    
    # Category-level summaries
    category_scores: Dict[str, float] = field(default_factory=dict)
    
    # Overall metrics
    overall_completion: float = 0.0
    
    # File existence tracking
    required_files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Automation gap analysis
    automation_gaps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Agent execution history
    agent_contributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def total_items(self) -> int:
        """Total checklist items across all phases."""
        return sum(p.total_items for p in self.phases)
    
    @property
    def complete_items(self) -> int:
        """Total complete items across all phases."""
        return sum(p.complete_items for p in self.phases)
    
    @property
    def fully_automated_total(self) -> int:
        """Total fully automated items."""
        return sum(p.fully_automated_count for p in self.phases)
    
    @property
    def needs_capability_total(self) -> int:
        """Total items needing new capability."""
        return sum(p.needs_capability_count for p in self.phases)
    
    def calculate_overall_completion(self):
        """Recalculate overall completion percentage."""
        if not self.phases:
            self.overall_completion = 0.0
            return
        
        total_weighted = 0.0
        total_weight = 0.0
        
        for phase in self.phases:
            weight = phase.total_items
            total_weighted += phase.completion_rate * weight
            total_weight += weight
        
        self.overall_completion = (total_weighted / total_weight * 100) if total_weight > 0 else 0.0
    
    def calculate_category_scores(self):
        """Calculate completion scores by category."""
        category_items: Dict[ReadinessCategory, List[ChecklistItem]] = {}
        
        for phase in self.phases:
            for item in phase.items:
                if item.category not in category_items:
                    category_items[item.category] = []
                category_items[item.category].append(item)
        
        self.category_scores = {}
        for category, items in category_items.items():
            if items:
                avg_completion = sum(i.completion_percentage for i in items) / len(items)
                self.category_scores[category.value] = avg_completion
    
    def get_phase(self, phase_id: str) -> Optional[PhaseReadiness]:
        """Get phase by ID."""
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        return None
    
    def get_item(self, item_id: str) -> Optional[ChecklistItem]:
        """Get checklist item by ID."""
        for phase in self.phases:
            for item in phase.items:
                if item.item_id == item_id:
                    return item
        return None
    
    def add_agent_contribution(
        self,
        agent_name: str,
        items_completed: List[str],
        execution_time: float,
        tokens_used: int,
    ):
        """Record an agent's contribution to readiness."""
        if agent_name not in self.agent_contributions:
            self.agent_contributions[agent_name] = {
                "items_completed": [],
                "total_execution_time": 0.0,
                "total_tokens": 0,
                "executions": [],
            }
        
        self.agent_contributions[agent_name]["items_completed"].extend(items_completed)
        self.agent_contributions[agent_name]["total_execution_time"] += execution_time
        self.agent_contributions[agent_name]["total_tokens"] += tokens_used
        self.agent_contributions[agent_name]["executions"].append({
            "timestamp": datetime.now().isoformat(),
            "items": items_completed,
            "execution_time": execution_time,
            "tokens": tokens_used,
        })
    
    def identify_automation_gaps(self):
        """Identify areas needing automation capability."""
        self.automation_gaps = []
        
        for phase in self.phases:
            for item in phase.items:
                if item.automation_capability in [
                    AutomationCapability.NEEDS_CAPABILITY,
                    AutomationCapability.PARTIALLY_AUTOMATED,
                ]:
                    self.automation_gaps.append({
                        "item_id": item.item_id,
                        "phase_id": phase.phase_id,
                        "description": item.description,
                        "category": item.category.value,
                        "automation_status": item.automation_capability.value,
                        "required_capabilities": item.required_capabilities,
                        "current_completion": item.completion_percentage,
                    })
    
    def to_dict(self) -> dict:
        self.calculate_overall_completion()
        self.calculate_category_scores()
        self.identify_automation_gaps()
        
        return {
            "project_id": self.project_id,
            "project_folder": self.project_folder,
            "generated_at": self.generated_at,
            "phases": [p.to_dict() for p in self.phases],
            "category_scores": self.category_scores,
            "overall_completion": self.overall_completion,
            "required_files": self.required_files,
            "automation_gaps": self.automation_gaps,
            "agent_contributions": self.agent_contributions,
            "summary": {
                "total_items": self.total_items,
                "complete_items": self.complete_items,
                "completion_rate": self.overall_completion,
                "fully_automated": self.fully_automated_total,
                "needs_capability": self.needs_capability_total,
                "automation_coverage": (
                    self.fully_automated_total / self.total_items * 100 
                    if self.total_items > 0 else 0.0
                ),
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ReadinessReport":
        report = cls(
            project_id=data["project_id"],
            project_folder=data["project_folder"],
            generated_at=data.get("generated_at", datetime.now().isoformat()),
            overall_completion=data.get("overall_completion", 0.0),
            required_files=data.get("required_files", {}),
            automation_gaps=data.get("automation_gaps", []),
            agent_contributions=data.get("agent_contributions", {}),
            category_scores=data.get("category_scores", {}),
        )
        report.phases = [PhaseReadiness.from_dict(p) for p in data.get("phases", [])]
        return report


# =============================================================================
# Readiness Check Functions
# =============================================================================

def check_file_exists(project_folder: str, relative_path: str) -> Dict[str, Any]:
    """Check if a required file exists and has content."""
    full_path = Path(project_folder) / relative_path
    result = {
        "path": relative_path,
        "exists": full_path.exists(),
        "size_bytes": 0,
        "has_content": False,
        "last_modified": None,
    }
    
    if full_path.exists():
        stat = full_path.stat()
        result["size_bytes"] = stat.st_size
        result["has_content"] = stat.st_size > 100  # More than minimal content
        result["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
    
    return result


def check_markdown_sections(file_path: str, required_sections: List[str]) -> Dict[str, Any]:
    """Check if a markdown file contains required sections."""
    result = {
        "file_path": file_path,
        "sections_found": [],
        "sections_missing": [],
        "total_sections": len(required_sections),
    }
    
    if not Path(file_path).exists():
        result["sections_missing"] = required_sections
        return result
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        for section in required_sections:
            # Check for section header
            pattern = rf'^#+\s+{re.escape(section)}'
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                result["sections_found"].append(section)
            else:
                result["sections_missing"].append(section)
    except Exception as e:
        logger.warning(f"Error checking sections in {file_path}: {e}")
        result["sections_missing"] = required_sections
    
    return result


def check_latex_components(file_path: str) -> Dict[str, Any]:
    """Check LaTeX paper for required components."""
    result = {
        "file_path": file_path,
        "has_title": False,
        "has_abstract": False,
        "has_introduction": False,
        "has_methodology": False,
        "has_results": False,
        "has_conclusion": False,
        "has_bibliography": False,
        "sections_count": 0,
        "tables_count": 0,
        "figures_count": 0,
        "equations_count": 0,
    }
    
    if not Path(file_path).exists():
        return result
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check components
        result["has_title"] = r'\title{' in content
        result["has_abstract"] = r'\begin{abstract}' in content
        result["has_introduction"] = bool(re.search(r'\\section\{.*[Ii]ntroduction', content))
        result["has_methodology"] = bool(re.search(r'\\section\{.*(Method|Data|Empirical)', content))
        result["has_results"] = bool(re.search(r'\\section\{.*[Rr]esult', content))
        result["has_conclusion"] = bool(re.search(r'\\section\{.*[Cc]onclusion', content))
        result["has_bibliography"] = r'\bibliography{' in content or r'\printbibliography' in content
        
        # Count elements
        result["sections_count"] = len(re.findall(r'\\section\{', content))
        result["tables_count"] = len(re.findall(r'\\begin\{table', content))
        result["figures_count"] = len(re.findall(r'\\begin\{figure', content))
        result["equations_count"] = len(re.findall(r'\\begin\{equation', content))
    except Exception as e:
        logger.warning(f"Error checking LaTeX file {file_path}: {e}")
    
    return result


def check_data_readiness(project_folder: str) -> Dict[str, Any]:
    """Check data readiness for analysis."""
    data_folder = Path(project_folder) / "data"
    result = {
        "data_folder_exists": data_folder.exists(),
        "raw_data_exists": False,
        "processed_data_exists": False,
        "data_files": [],
        "total_files": 0,
        "total_size_mb": 0.0,
    }
    
    if not data_folder.exists():
        return result
    
    # Check for raw data
    raw_folder = data_folder / "raw data"
    result["raw_data_exists"] = raw_folder.exists() and any(raw_folder.iterdir()) if raw_folder.exists() else False
    
    # Check for processed data
    processed_folder = data_folder / "processed"
    result["processed_data_exists"] = processed_folder.exists() and any(processed_folder.iterdir()) if processed_folder.exists() else False
    
    # List data files (capped to avoid unbounded scans on large trees)
    max_files = int(INTAKE_SERVER.MAX_ZIP_FILES)
    exclude_dirs = {"__pycache__", ".venv", ".git", "node_modules", "temp", "tmp", ".workflow_cache", ".evidence"}

    total_size = 0
    accepted = 0

    # Optimization: Use os.walk to efficiently prune excluded directories,
    # preventing traversal into large, irrelevant folders like .git or node_modules.
    # This is significantly faster than rglob("*") followed by filtering.
    for root, dirs, files in os.walk(data_folder, topdown=True):
        # Prune directories to avoid traversing them
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

        if accepted >= max_files:
            break

        for filename in files:
            if filename.startswith('.'):
                continue

            try:
                file_path = Path(root) / filename
                size = file_path.stat().st_size
            except OSError:
                continue

            accepted += 1
            if accepted >= max_files:
                break

            total_size += size
            result["data_files"].append(
                {
                    "name": filename,
                    "path": str(file_path.relative_to(project_folder)),
                    "size_bytes": size,
                }
            )

    result["total_files"] = len(result["data_files"])
    result["total_size_mb"] = total_size / (1024 * 1024)
    
    return result


def check_literature_readiness(project_folder: str) -> Dict[str, Any]:
    """Check literature review readiness."""
    result = {
        "literature_folder_exists": False,
        "bibtex_exists": False,
        "bibtex_entries": 0,
        "literature_summary_exists": False,
        "literature_review_exists": False,
        "papers_count": 0,
    }
    
    project_path = Path(project_folder)

    # Current literature synthesis artifact lives at project root.
    review_path = project_path / "LITERATURE_REVIEW.md"
    result["literature_review_exists"] = review_path.exists()
    result["literature_summary_exists"] = result["literature_review_exists"]

    # Bibliography can exist at project root or under bibliography/.
    bib_candidates = [
        project_path / "references.bib",
        project_path / "bibliography" / "references.bib",
    ]
    bib_path = next((p for p in bib_candidates if p.exists()), None)
    result["bibtex_exists"] = bib_path is not None

    if bib_path is not None:
        try:
            content = bib_path.read_text(encoding="utf-8")
            result["bibtex_entries"] = len(re.findall(r"@\w+\{", content))
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning(
                "Failed to read BibTeX file '%s': %s",
                bib_path,
                exc,
            )
            result["bibtex_entries"] = 0

    # Backward-compat: if a legacy literature/ folder exists, keep the flag.
    lit_folder = project_path / "literature"
    result["literature_folder_exists"] = lit_folder.exists()
    if lit_folder.exists():
        result["papers_count"] = len(list(lit_folder.glob("*.pdf")))
    
    return result


# =============================================================================
# Report Generation
# =============================================================================

# Standard checklist items mapped to agents
STANDARD_CHECKLIST = [
    # Phase 1: Foundation
    {
        "phase_id": "Phase 1",
        "phase_name": "Foundation Completion",
        "items": [
            {
                "item_id": "1.1.1",
                "category": "literature",
                "description": "Core papers read and synthesized (15+ papers)",
                "assigned_agent": "A6",  # LiteratureSearch
                "required_capabilities": ["literature_search", "literature_synthesis"],
            },
            {
                "item_id": "1.1.2",
                "category": "literature",
                "description": "Literature matrix created mapping papers to streams",
                "assigned_agent": "A7",  # LiteratureSynthesis
                "required_capabilities": ["literature_synthesis"],
            },
            {
                "item_id": "1.1.3",
                "category": "literature",
                "description": "Gap statement articulated with citations",
                "assigned_agent": "A7",
                "required_capabilities": ["literature_synthesis"],
            },
            {
                "item_id": "1.2.1",
                "category": "hypotheses",
                "description": "H1 formalized with testable prediction",
                "assigned_agent": "A5",  # HypothesisDeveloper
                "required_capabilities": ["hypothesis_development"],
            },
            {
                "item_id": "1.2.2",
                "category": "hypotheses",
                "description": "H2 formalized with testable prediction",
                "assigned_agent": "A5",
                "required_capabilities": ["hypothesis_development"],
            },
            {
                "item_id": "1.2.3",
                "category": "hypotheses",
                "description": "H3 formalized with testable prediction",
                "assigned_agent": "A5",
                "required_capabilities": ["hypothesis_development"],
            },
            {
                "item_id": "1.2.4",
                "category": "hypotheses",
                "description": "Theoretical mechanisms documented",
                "assigned_agent": "A5",
                "required_capabilities": ["hypothesis_development"],
            },
            {
                "item_id": "1.3.1",
                "category": "methodology",
                "description": "Variable definitions complete",
                "assigned_agent": "A8",  # PaperStructure
                "required_capabilities": ["paper_structuring"],
            },
            {
                "item_id": "1.3.2",
                "category": "methodology",
                "description": "Regression specifications written",
                "assigned_agent": "A8",
                "required_capabilities": ["paper_structuring"],
            },
            {
                "item_id": "1.3.3",
                "category": "methodology",
                "description": "Identification strategy documented",
                "assigned_agent": "A8",
                "required_capabilities": ["paper_structuring"],
            },
            {
                "item_id": "1.4.1",
                "category": "data",
                "description": "Raw data loaded and validated",
                "assigned_agent": "A1",  # DataAnalyst
                "required_capabilities": ["data_analysis", "code_execution"],
            },
            {
                "item_id": "1.4.2",
                "category": "data",
                "description": "Matched pairs created",
                "assigned_agent": "A1",
                "required_capabilities": ["data_analysis", "code_execution"],
            },
            {
                "item_id": "1.4.3",
                "category": "data",
                "description": "Summary statistics computed",
                "assigned_agent": "A1",
                "required_capabilities": ["data_analysis", "code_execution"],
            },
        ],
    },
    # Phase 2: Analysis
    {
        "phase_id": "Phase 2",
        "phase_name": "Empirical Analysis",
        "items": [
            {
                "item_id": "2.1.1",
                "category": "analysis",
                "description": "Baseline regression (H1) estimated",
                "assigned_agent": "A1",
                "required_capabilities": ["data_analysis", "code_execution"],
            },
            {
                "item_id": "2.1.2",
                "category": "analysis",
                "description": "Liquidity controls added (H2)",
                "assigned_agent": "A1",
                "required_capabilities": ["data_analysis", "code_execution"],
            },
            {
                "item_id": "2.1.3",
                "category": "analysis",
                "description": "Moneyness interactions tested (H3)",
                "assigned_agent": "A1",
                "required_capabilities": ["data_analysis", "code_execution"],
            },
            {
                "item_id": "2.1.4",
                "category": "analysis",
                "description": "Results table created (Table 2)",
                "assigned_agent": "A1",
                "required_capabilities": ["data_analysis"],
            },
            {
                "item_id": "2.2.1",
                "category": "analysis",
                "description": "Robustness checks completed (5+ specs)",
                "assigned_agent": "A1",
                "required_capabilities": ["data_analysis", "code_execution"],
            },
            {
                "item_id": "2.2.2",
                "category": "analysis",
                "description": "Robustness table created (Table 3)",
                "assigned_agent": "A1",
                "required_capabilities": ["data_analysis"],
            },
            {
                "item_id": "2.4.1",
                "category": "analysis",
                "description": "Figure 1 (time series) created",
                "assigned_agent": "A1",
                "required_capabilities": ["data_analysis", "code_execution"],
            },
            {
                "item_id": "2.4.2",
                "category": "analysis",
                "description": "Figure 2 (moneyness pattern) created",
                "assigned_agent": "A1",
                "required_capabilities": ["data_analysis", "code_execution"],
            },
        ],
    },
    # Phase 3: Writing
    {
        "phase_id": "Phase 3",
        "phase_name": "Paper Writing",
        "items": [
            {
                "item_id": "3.1.1",
                "category": "writing",
                "description": "Introduction drafted (2-2.5 pages)",
                "assigned_agent": "A4",  # OverviewGenerator
                "required_capabilities": ["document_generation"],
            },
            {
                "item_id": "3.1.2",
                "category": "writing",
                "description": "Contribution statement clear",
                "assigned_agent": "A4",
                "required_capabilities": ["document_generation"],
            },
            {
                "item_id": "3.2.1",
                "category": "writing",
                "description": "Data section drafted",
                "assigned_agent": "A4",
                "required_capabilities": ["document_generation"],
            },
            {
                "item_id": "3.2.2",
                "category": "writing",
                "description": "Methodology section drafted",
                "assigned_agent": "A4",
                "required_capabilities": ["document_generation"],
            },
            {
                "item_id": "3.3.1",
                "category": "writing",
                "description": "Results section drafted",
                "assigned_agent": "A4",
                "required_capabilities": ["document_generation"],
            },
            {
                "item_id": "3.4.1",
                "category": "writing",
                "description": "Conclusion drafted",
                "assigned_agent": "A4",
                "required_capabilities": ["document_generation"],
            },
        ],
    },
    # Phase 4: Review
    {
        "phase_id": "Phase 4",
        "phase_name": "Polish and Submit",
        "items": [
            {
                "item_id": "4.1.1",
                "category": "review",
                "description": "Internal review completed",
                "assigned_agent": "A12",  # CriticalReviewer
                "required_capabilities": ["critical_review"],
            },
            {
                "item_id": "4.1.2",
                "category": "review",
                "description": "Cross-document consistency verified",
                "assigned_agent": "A14",  # ConsistencyChecker
                "required_capabilities": ["consistency_check"],
            },
            {
                "item_id": "4.1.3",
                "category": "review",
                "description": "Style guide compliance checked",
                "assigned_agent": "A13",  # StyleEnforcer
                "required_capabilities": ["style_enforcement"],
            },
            {
                "item_id": "4.2.1",
                "category": "structure",
                "description": "Paper structure finalized",
                "assigned_agent": "A8",
                "required_capabilities": ["paper_structuring"],
            },
            {
                "item_id": "4.2.2",
                "category": "structure",
                "description": "All tables formatted for submission",
                "assigned_agent": "A8",
                "required_capabilities": ["paper_structuring"],
            },
            {
                "item_id": "4.2.3",
                "category": "structure",
                "description": "All figures formatted for submission",
                "assigned_agent": "A8",
                "required_capabilities": ["paper_structuring"],
            },
            {
                "item_id": "4.3.1",
                "category": "review",
                "description": "Final proofreading complete",
                "assigned_agent": "A13",
                "required_capabilities": ["style_enforcement"],
            },
        ],
    },
]


def initialize_readiness_report(project_folder: str, project_id: str) -> ReadinessReport:
    """Initialize a readiness report from standard checklist."""
    report = ReadinessReport(
        project_id=project_id,
        project_folder=project_folder,
    )
    
    # Create phases from standard checklist
    for phase_def in STANDARD_CHECKLIST:
        phase = PhaseReadiness(
            phase_id=phase_def["phase_id"],
            phase_name=phase_def["phase_name"],
        )
        
        for item_def in phase_def["items"]:
            item = ChecklistItem(
                item_id=item_def["item_id"],
                category=ReadinessCategory(item_def["category"]),
                description=item_def["description"],
                assigned_agent=item_def.get("assigned_agent"),
                required_capabilities=item_def.get("required_capabilities", []),
            )
            phase.items.append(item)
        
        report.phases.append(phase)
    
    # Check file existence
    report.required_files = {
        "RESEARCH_OVERVIEW.md": check_file_exists(project_folder, "RESEARCH_OVERVIEW.md"),
        "PROJECT_PLAN.md": check_file_exists(project_folder, "PROJECT_PLAN.md"),
        "paper/main.tex": check_file_exists(project_folder, "paper/main.tex"),
        "paper/STRUCTURE.md": check_file_exists(project_folder, "paper/STRUCTURE.md"),
        "literature/references.bib": check_file_exists(project_folder, "literature/references.bib"),
    }
    
    report.calculate_overall_completion()
    report.identify_automation_gaps()
    
    return report


def assess_project_readiness(project_folder: str) -> ReadinessReport:
    """
    Perform comprehensive readiness assessment of a project.
    
    Checks files, content, and updates checklist status.
    """
    project_id = get_project_id(project_folder)
    
    # Load existing report or create new one
    report_path = Path(project_folder) / "readiness_report.json"
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                report = ReadinessReport.from_dict(json.load(f))
        except (json.JSONDecodeError, IOError, OSError):
            report = initialize_readiness_report(project_folder, project_id)
    else:
        report = initialize_readiness_report(project_folder, project_id)
    
    # Update file checks
    report.required_files = {
        "RESEARCH_OVERVIEW.md": check_file_exists(project_folder, "RESEARCH_OVERVIEW.md"),
        "PROJECT_PLAN.md": check_file_exists(project_folder, "PROJECT_PLAN.md"),
        "paper/main.tex": check_file_exists(project_folder, "paper/main.tex"),
        "paper/STRUCTURE.md": check_file_exists(project_folder, "paper/STRUCTURE.md"),
        "literature/references.bib": check_file_exists(project_folder, "literature/references.bib"),
    }
    
    # Check data readiness
    data_status = check_data_readiness(project_folder)
    
    # Update data-related items based on actual state
    for phase in report.phases:
        for item in phase.items:
            if item.category == ReadinessCategory.DATA:
                if "raw data" in item.description.lower() and data_status["raw_data_exists"]:
                    item.mark_partial(50.0)
                elif "summary statistics" in item.description.lower():
                    # Check if summary stats exist in workflow results
                    workflow_results_path = Path(project_folder) / "workflow_results.json"
                    if workflow_results_path.exists():
                        item.mark_partial(30.0)
    
    # Check literature readiness
    lit_status = check_literature_readiness(project_folder)
    
    for phase in report.phases:
        for item in phase.items:
            if item.category == ReadinessCategory.LITERATURE:
                if "papers read" in item.description.lower():
                    if lit_status["bibtex_entries"] >= 15:
                        item.mark_complete(evidence=f"{lit_status['bibtex_entries']} BibTeX entries")
                    elif lit_status["bibtex_entries"] > 0:
                        item.mark_partial(lit_status["bibtex_entries"] / 15 * 100)
    
    # Check LaTeX paper
    latex_path = Path(project_folder) / "paper" / "main.tex"
    if latex_path.exists():
        latex_status = check_latex_components(str(latex_path))
        
        for phase in report.phases:
            for item in phase.items:
                if item.category == ReadinessCategory.WRITING:
                    if "introduction" in item.description.lower() and latex_status["has_introduction"]:
                        item.mark_partial(50.0)
                    if "methodology" in item.description.lower() and latex_status["has_methodology"]:
                        item.mark_partial(50.0)
                    if "results" in item.description.lower() and latex_status["has_results"]:
                        item.mark_partial(50.0)
                    if "conclusion" in item.description.lower() and latex_status["has_conclusion"]:
                        item.mark_partial(50.0)
    
    # Recalculate
    report.calculate_overall_completion()
    report.calculate_category_scores()
    report.identify_automation_gaps()
    
    # Save updated report
    save_readiness_report(report)
    
    logger.info(
        f"Readiness assessment complete: {report.overall_completion:.1f}% overall, "
        f"{report.complete_items}/{report.total_items} items complete"
    )
    
    return report


def save_readiness_report(report: ReadinessReport):
    """Save readiness report to project folder."""
    report_path = Path(report.project_folder) / "readiness_report.json"
    try:
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved readiness report to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save readiness report: {e}")


def load_readiness_report(project_folder: str) -> Optional[ReadinessReport]:
    """Load existing readiness report."""
    report_path = Path(project_folder) / "readiness_report.json"
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                return ReadinessReport.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load readiness report: {e}")
    return None


def format_readiness_summary(report: ReadinessReport) -> str:
    """Format a human-readable readiness summary."""
    report.calculate_overall_completion()
    report.calculate_category_scores()
    
    lines = [
        "# Paper Readiness Report",
        f"\n**Project:** {report.project_id}",
        f"**Generated:** {report.generated_at}",
        f"**Overall Completion:** {report.overall_completion:.1f}%",
        "",
        "## Category Scores",
    ]
    
    for category, score in sorted(report.category_scores.items()):
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        lines.append(f"- **{category.title()}:** {bar} {score:.1f}%")
    
    lines.extend([
        "",
        "## Phase Progress",
    ])
    
    for phase in report.phases:
        status_icon = "✅" if phase.status == CheckStatus.COMPLETE else "🔄" if phase.status == CheckStatus.PARTIAL else "⏳"
        lines.append(f"\n### {status_icon} {phase.phase_id}: {phase.phase_name}")
        lines.append(f"**Progress:** {phase.complete_items}/{phase.total_items} ({phase.completion_rate*100:.0f}%)")
        
        # Group items by status
        complete = [i for i in phase.items if i.status == CheckStatus.COMPLETE]
        partial = [i for i in phase.items if i.status == CheckStatus.PARTIAL]
        not_started = [i for i in phase.items if i.status == CheckStatus.NOT_STARTED]
        
        if complete:
            lines.append("\n**Complete:**")
            for item in complete:
                lines.append(f"- ✅ {item.description}")
        
        if partial:
            lines.append("\n**In Progress:**")
            for item in partial:
                lines.append(f"- 🔄 {item.description} ({item.completion_percentage:.0f}%)")
        
        if not_started:
            lines.append("\n**Not Started:**")
            for item in not_started[:5]:  # Show first 5
                lines.append(f"- ⏳ {item.description}")
            if len(not_started) > 5:
                lines.append(f"- ... and {len(not_started) - 5} more")
    
    # Automation gaps
    if report.automation_gaps:
        lines.extend([
            "",
            "## Automation Gaps",
            f"**Items needing capability:** {len(report.automation_gaps)}",
            "",
        ])
        for gap in report.automation_gaps[:10]:
            lines.append(f"- {gap['description']} ({gap['automation_status']})")
    
    # Agent contributions
    if report.agent_contributions:
        lines.extend([
            "",
            "## Agent Contributions",
        ])
        for agent, contrib in report.agent_contributions.items():
            lines.append(
                f"- **{agent}:** {len(contrib['items_completed'])} items, "
                f"{contrib['total_execution_time']:.1f}s execution time"
            )
    
    return '\n'.join(lines)
