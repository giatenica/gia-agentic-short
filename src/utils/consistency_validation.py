"""
Cross-document consistency validation utilities.

Extracts and compares key elements across research documents to identify
inconsistencies in hypotheses, variables, methodology, citations, and statistics.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from loguru import logger


class ConsistencyCategory(Enum):
    """Categories of consistency checks."""
    HYPOTHESIS = "hypothesis"
    VARIABLE = "variable"
    METHODOLOGY = "methodology"
    CITATION = "citation"
    STATISTIC = "statistic"
    TERMINOLOGY = "terminology"


class ConsistencySeverity(Enum):
    """Severity levels for consistency issues."""
    CRITICAL = "critical"  # Blocks workflow - hypothesis/variable mismatch
    HIGH = "high"          # Should be fixed - methodology/citation issues
    MEDIUM = "medium"      # Warning - statistic/terminology drift
    LOW = "low"            # Informational


# Document hierarchy for canonical source determination
DOCUMENT_PRIORITY = {
    "RESEARCH_OVERVIEW.md": 1,      # Primary source of truth
    "UPDATED_RESEARCH_OVERVIEW.md": 2,
    "LITERATURE_REVIEW.md": 3,
    "PROJECT_PLAN.md": 4,
    "paper/STRUCTURE.md": 5,
    "paper/main.tex": 6,            # Final output
}


@dataclass
class ConsistencyElement:
    """A single element extracted from a document for consistency checking."""
    category: ConsistencyCategory
    key: str                  # Identifier (e.g., "H1", "voting_premium", "sample_size")
    value: str                # The actual content
    document: str             # Source document path
    location: str             # Line number or section reference
    context: str = ""         # Surrounding context for disambiguation
    
    def __hash__(self):
        return hash((self.category.value, self.key, self.document))
    
    def __eq__(self, other):
        if not isinstance(other, ConsistencyElement):
            return False
        return (self.category == other.category and 
                self.key == other.key and 
                self.document == other.document)


@dataclass
class CrossDocumentIssue:
    """An inconsistency found across multiple documents."""
    category: ConsistencyCategory
    severity: ConsistencySeverity
    key: str                          # Element identifier
    description: str                  # Human-readable description
    affected_documents: List[str]     # Documents with the inconsistency
    canonical_value: str              # Recommended correct value
    canonical_source: str             # Document providing canonical value
    variants: Dict[str, str]          # Document -> actual value mapping
    suggestion: str = ""              # Fix suggestion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "key": self.key,
            "description": self.description,
            "affected_documents": self.affected_documents,
            "canonical_value": self.canonical_value,
            "canonical_source": self.canonical_source,
            "variants": self.variants,
            "suggestion": self.suggestion,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossDocumentIssue":
        """Create from dictionary."""
        return cls(
            category=ConsistencyCategory(data["category"]),
            severity=ConsistencySeverity(data["severity"]),
            key=data["key"],
            description=data["description"],
            affected_documents=data["affected_documents"],
            canonical_value=data["canonical_value"],
            canonical_source=data["canonical_source"],
            variants=data["variants"],
            suggestion=data.get("suggestion", ""),
        )


@dataclass
class ConsistencyReport:
    """Complete consistency validation report."""
    project_folder: str
    documents_checked: List[str]
    elements_extracted: int
    issues: List[CrossDocumentIssue] = field(default_factory=list)
    element_mapping: Dict[str, List[ConsistencyElement]] = field(default_factory=dict)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ConsistencySeverity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ConsistencySeverity.HIGH)
    
    @property
    def is_consistent(self) -> bool:
        """True if no critical or high severity issues."""
        return self.critical_count == 0 and self.high_count == 0
    
    @property
    def score(self) -> float:
        """Consistency score 0.0-1.0."""
        if not self.elements_extracted:
            return 1.0
        # Weight issues by severity
        penalty = (
            self.critical_count * 0.3 +
            self.high_count * 0.15 +
            sum(1 for i in self.issues if i.severity == ConsistencySeverity.MEDIUM) * 0.05 +
            sum(1 for i in self.issues if i.severity == ConsistencySeverity.LOW) * 0.02
        )
        return max(0.0, 1.0 - penalty)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_folder": self.project_folder,
            "documents_checked": self.documents_checked,
            "elements_extracted": self.elements_extracted,
            "issues": [i.to_dict() for i in self.issues],
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "is_consistent": self.is_consistent,
            "score": self.score,
        }


# =============================================================================
# Extraction Functions
# =============================================================================

def extract_hypotheses_markdown(content: str, document: str) -> List[ConsistencyElement]:
    """Extract hypothesis definitions from Markdown documents."""
    elements = []
    
    # Pattern 1: H1, H2, H3, H4 format (numbered hypotheses)
    hypothesis_pattern = r'(?:^|\n)\s*(?:\*\*)?H(\d+)(?:\*\*)?[:\s]+(.+?)(?=\n(?:\s*(?:\*\*)?H\d|\n\n|\Z))'
    matches = re.findall(hypothesis_pattern, content, re.MULTILINE | re.DOTALL)
    
    for num, text in matches:
        # Clean up the text
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\*\*', '', text)  # Remove bold markers
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.HYPOTHESIS,
            key=f"H{num}",
            value=text[:500],  # Truncate long hypotheses
            document=document,
            location=f"Hypothesis H{num}",
        ))
    
    # Pattern 2: "Hypothesis:" or "Main Hypothesis:" sections
    section_pattern = r'(?:Main\s+)?Hypothesis[:\s]+(.+?)(?=\n\n|\n#+|\Z)'
    section_matches = re.findall(section_pattern, content, re.IGNORECASE | re.DOTALL)
    
    for i, text in enumerate(section_matches):
        text = re.sub(r'\s+', ' ', text.strip())
        if text and not any(e.key.startswith("H") for e in elements):
            elements.append(ConsistencyElement(
                category=ConsistencyCategory.HYPOTHESIS,
                key=f"MAIN_H{i+1}" if i > 0 else "MAIN_HYPOTHESIS",
                value=text[:500],
                document=document,
                location="Main Hypothesis section",
            ))
    
    return elements


def extract_hypotheses_latex(content: str, document: str) -> List[ConsistencyElement]:
    """Extract hypothesis definitions from LaTeX documents."""
    elements = []
    
    # Pattern: \hypothesis{H1}{text} or similar
    cmd_pattern = r'\\hypothesis\{(H?\d+)\}\{(.+?)\}'
    matches = re.findall(cmd_pattern, content, re.DOTALL)
    
    for key, text in matches:
        key = key if key.startswith("H") else f"H{key}"
        text = re.sub(r'\s+', ' ', text.strip())
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.HYPOTHESIS,
            key=key,
            value=text[:500],
            document=document,
            location=f"\\hypothesis command",
        ))
    
    # Pattern: H1: text in comments or text
    text_pattern = r'H(\d+)[:\s]+(.+?)(?=H\d|\\|$)'
    text_matches = re.findall(text_pattern, content)
    
    for num, text in text_matches:
        key = f"H{num}"
        if not any(e.key == key for e in elements):
            text = re.sub(r'\s+', ' ', text.strip())
            elements.append(ConsistencyElement(
                category=ConsistencyCategory.HYPOTHESIS,
                key=key,
                value=text[:500],
                document=document,
                location=f"Inline hypothesis H{num}",
            ))
    
    return elements


def extract_variables_markdown(content: str, document: str) -> List[ConsistencyElement]:
    """Extract variable definitions from Markdown documents."""
    elements = []
    
    # Pattern 1: Variable definitions in lists or tables
    # - **Variable Name**: definition
    var_pattern = r'(?:^|\n)\s*[-*]\s*\*\*([^*]+)\*\*[:\s]+(.+?)(?=\n[-*]|\n\n|\Z)'
    matches = re.findall(var_pattern, content, re.MULTILINE)
    
    for name, definition in matches:
        name = name.strip()
        definition = re.sub(r'\s+', ' ', definition.strip())
        # Normalize variable name to key
        key = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.VARIABLE,
            key=key,
            value=f"{name}: {definition[:200]}",
            document=document,
            location="Variable definition",
            context=name,
        ))
    
    # Pattern 2: Key Variables section
    key_var_section = re.search(
        r'(?:Key\s+)?Variables?[:\s]*\n((?:[-*].+\n?)+)',
        content, re.IGNORECASE
    )
    if key_var_section:
        var_text = key_var_section.group(1)
        var_items = re.findall(r'[-*]\s*(.+)', var_text)
        for item in var_items:
            # Extract variable name from item
            var_match = re.match(r'([^:]+)[:\s]+(.+)', item)
            if var_match:
                name, definition = var_match.groups()
                key = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip().lower())
                if not any(e.key == key for e in elements):
                    elements.append(ConsistencyElement(
                        category=ConsistencyCategory.VARIABLE,
                        key=key,
                        value=f"{name.strip()}: {definition.strip()[:200]}",
                        document=document,
                        location="Key Variables section",
                        context=name.strip(),
                    ))
    
    return elements


def extract_variables_latex(content: str, document: str) -> List[ConsistencyElement]:
    """Extract variable definitions from LaTeX documents."""
    elements = []
    
    # Pattern 1: \newcommand definitions
    newcmd_pattern = r'\\newcommand\{\\([^}]+)\}\{(.+?)\}'
    matches = re.findall(newcmd_pattern, content)
    
    for name, definition in matches:
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.VARIABLE,
            key=name.lower(),
            value=f"\\{name}: {definition}",
            document=document,
            location="\\newcommand definition",
            context=name,
        ))
    
    # Pattern 2: Variable descriptions in comments
    comment_pattern = r'%\s*([A-Za-z_]+)\s*[=:]\s*(.+)'
    comment_matches = re.findall(comment_pattern, content)
    
    for name, definition in comment_matches:
        key = name.lower()
        if not any(e.key == key for e in elements):
            elements.append(ConsistencyElement(
                category=ConsistencyCategory.VARIABLE,
                key=key,
                value=f"{name}: {definition.strip()[:200]}",
                document=document,
                location="LaTeX comment",
                context=name,
            ))
    
    return elements


def extract_methodology_markdown(content: str, document: str) -> List[ConsistencyElement]:
    """Extract methodology descriptions from Markdown documents."""
    elements = []
    
    # Pattern 1: Methodology/Methods section
    method_section = re.search(
        r'(?:##?\s*)?(?:Methodology|Methods?|Approach|Analysis)\s*\n((?:.+\n?)+?)(?=\n##|\Z)',
        content, re.IGNORECASE
    )
    if method_section:
        text = re.sub(r'\s+', ' ', method_section.group(1).strip())
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.METHODOLOGY,
            key="main_methodology",
            value=text[:1000],
            document=document,
            location="Methodology section",
        ))
    
    # Pattern 2: Regression specifications
    regression_pattern = r'(?:regression|model)[:\s]+(.+?)(?=\n\n|\Z)'
    regression_matches = re.findall(regression_pattern, content, re.IGNORECASE)
    
    for i, spec in enumerate(regression_matches):
        spec = re.sub(r'\s+', ' ', spec.strip())
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.METHODOLOGY,
            key=f"regression_spec_{i+1}",
            value=spec[:500],
            document=document,
            location=f"Regression specification {i+1}",
        ))
    
    # Pattern 3: Fixed effects mentions
    fe_pattern = r'fixed\s+effects?[:\s]+([^.\n]+)'
    fe_matches = re.findall(fe_pattern, content, re.IGNORECASE)
    
    for fe in fe_matches:
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.METHODOLOGY,
            key="fixed_effects",
            value=fe.strip(),
            document=document,
            location="Fixed effects specification",
        ))
    
    return elements


def extract_methodology_latex(content: str, document: str) -> List[ConsistencyElement]:
    """Extract methodology from LaTeX documents."""
    elements = []
    
    # Pattern 1: Equation environments
    equation_pattern = r'\\begin\{(?:equation|align)\*?\}(.+?)\\end\{(?:equation|align)\*?\}'
    matches = re.findall(equation_pattern, content, re.DOTALL)
    
    for i, eq in enumerate(matches):
        eq = re.sub(r'\s+', ' ', eq.strip())
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.METHODOLOGY,
            key=f"equation_{i+1}",
            value=eq[:500],
            document=document,
            location=f"Equation {i+1}",
        ))
    
    # Pattern 2: Inline math with regression-like content
    inline_pattern = r'\$([^$]*(?:beta|alpha|gamma|epsilon|Y_|X_)[^$]*)\$'
    inline_matches = re.findall(inline_pattern, content)
    
    for i, math in enumerate(inline_matches[:5]):  # Limit to first 5
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.METHODOLOGY,
            key=f"inline_math_{i+1}",
            value=math.strip(),
            document=document,
            location=f"Inline math {i+1}",
        ))
    
    return elements


def extract_citations_markdown(content: str, document: str) -> List[ConsistencyElement]:
    """Extract citation references from Markdown documents."""
    elements = []
    
    # Pattern 1: Author (Year) format
    author_year_pattern = r'([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?)\s*\((\d{4})\)'
    matches = re.findall(author_year_pattern, content)
    
    for author, year in matches:
        key = f"{author.split()[0].lower()}_{year}"
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.CITATION,
            key=key,
            value=f"{author} ({year})",
            document=document,
            location="In-text citation",
        ))
    
    # Pattern 2: [@citation_key] format
    cite_key_pattern = r'\[@?([a-zA-Z0-9_-]+)\]'
    key_matches = re.findall(cite_key_pattern, content)
    
    for key in key_matches:
        if not any(e.key == key.lower() for e in elements):
            elements.append(ConsistencyElement(
                category=ConsistencyCategory.CITATION,
                key=key.lower(),
                value=f"[@{key}]",
                document=document,
                location="Citation key reference",
            ))
    
    return elements


def extract_citations_latex(content: str, document: str) -> List[ConsistencyElement]:
    """Extract citations from LaTeX documents."""
    elements = []
    
    # Pattern: \cite{key}, \citep{key}, \citet{key}
    cite_pattern = r'\\cite[pt]?\{([^}]+)\}'
    matches = re.findall(cite_pattern, content)
    
    for keys in matches:
        for key in keys.split(','):
            key = key.strip()
            if not any(e.key == key.lower() for e in elements):
                elements.append(ConsistencyElement(
                    category=ConsistencyCategory.CITATION,
                    key=key.lower(),
                    value=f"\\cite{{{key}}}",
                    document=document,
                    location="LaTeX citation",
                ))
    
    return elements


def extract_citations_bibtex(content: str, document: str) -> List[ConsistencyElement]:
    """Extract citation definitions from BibTeX files."""
    elements = []
    
    # Pattern: @type{key,
    entry_pattern = r'@\w+\{([^,]+),'
    matches = re.findall(entry_pattern, content)
    
    for key in matches:
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.CITATION,
            key=key.strip().lower(),
            value=f"@entry{{{key}}}",
            document=document,
            location="BibTeX entry",
        ))
    
    return elements


def extract_statistics_markdown(content: str, document: str) -> List[ConsistencyElement]:
    """Extract statistical values from Markdown documents."""
    elements = []
    
    # Pattern 1: Sample size mentions
    sample_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:observations?|samples?|rows?|records?)'
    matches = re.findall(sample_pattern, content, re.IGNORECASE)
    
    for value in matches:
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.STATISTIC,
            key="sample_size",
            value=value.replace(',', ''),
            document=document,
            location="Sample size mention",
        ))
    
    # Pattern 2: Date ranges
    date_range_pattern = r'(\d{4})\s*[-–to]+\s*(\d{4})'
    date_matches = re.findall(date_range_pattern, content)
    
    for start, end in date_matches:
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.STATISTIC,
            key="date_range",
            value=f"{start}-{end}",
            document=document,
            location="Date range",
        ))
    
    # Pattern 3: Percentages
    pct_pattern = r'(\d+(?:\.\d+)?)\s*%'
    pct_matches = re.findall(pct_pattern, content)
    
    for pct in pct_matches[:10]:  # Limit to first 10
        elements.append(ConsistencyElement(
            category=ConsistencyCategory.STATISTIC,
            key="percentage",
            value=f"{pct}%",
            document=document,
            location="Percentage value",
        ))
    
    return elements


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_all_elements(project_folder: str) -> Tuple[List[ConsistencyElement], List[str]]:
    """
    Extract all consistency elements from project documents.
    
    Returns:
        Tuple of (elements list, documents checked list)
    """
    folder = Path(project_folder)
    elements = []
    documents_checked = []
    
    # Define documents to check with their extractors
    document_configs = [
        ("RESEARCH_OVERVIEW.md", [
            extract_hypotheses_markdown,
            extract_variables_markdown,
            extract_methodology_markdown,
            extract_citations_markdown,
            extract_statistics_markdown,
        ]),
        ("UPDATED_RESEARCH_OVERVIEW.md", [
            extract_hypotheses_markdown,
            extract_variables_markdown,
            extract_methodology_markdown,
            extract_citations_markdown,
            extract_statistics_markdown,
        ]),
        ("LITERATURE_SUMMARY.md", [
            extract_hypotheses_markdown,
            extract_citations_markdown,
            extract_methodology_markdown,
        ]),
        ("PROJECT_PLAN.md", [
            extract_hypotheses_markdown,
            extract_variables_markdown,
            extract_methodology_markdown,
            extract_statistics_markdown,
        ]),
        ("paper/STRUCTURE.md", [
            extract_hypotheses_markdown,
            extract_variables_markdown,
            extract_methodology_markdown,
        ]),
        ("paper/main.tex", [
            extract_hypotheses_latex,
            extract_variables_latex,
            extract_methodology_latex,
            extract_citations_latex,
        ]),
        ("literature/references.bib", [
            extract_citations_bibtex,
        ]),
    ]
    
    for doc_path, extractors in document_configs:
        full_path = folder / doc_path
        if full_path.exists():
            try:
                content = full_path.read_text(encoding='utf-8')
                documents_checked.append(doc_path)
                
                for extractor in extractors:
                    extracted = extractor(content, doc_path)
                    elements.extend(extracted)
                    
                logger.debug(f"Extracted {len([e for e in elements if e.document == doc_path])} elements from {doc_path}")
                
            except Exception as e:
                logger.warning(f"Failed to extract from {doc_path}: {e}")
    
    return elements, documents_checked


# =============================================================================
# Comparison Functions
# =============================================================================

def get_canonical_source(documents: List[str]) -> str:
    """Determine the canonical source document from a list."""
    sorted_docs = sorted(
        documents,
        key=lambda d: DOCUMENT_PRIORITY.get(d, 100)
    )
    return sorted_docs[0] if sorted_docs else documents[0]


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip().lower())
    # Remove punctuation at end
    text = re.sub(r'[.,;:!?]+$', '', text)
    # Remove common variations
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    return text


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity between two texts."""
    t1 = set(normalize_text(text1).split())
    t2 = set(normalize_text(text2).split())
    
    if not t1 or not t2:
        return 0.0
    
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    
    return intersection / union if union > 0 else 0.0


def compare_elements(elements: List[ConsistencyElement]) -> List[CrossDocumentIssue]:
    """
    Compare elements across documents to find inconsistencies.
    
    Args:
        elements: All extracted elements
        
    Returns:
        List of cross-document issues
    """
    issues = []
    
    # Group elements by category and key
    grouped: Dict[Tuple[ConsistencyCategory, str], List[ConsistencyElement]] = {}
    for element in elements:
        group_key = (element.category, element.key)
        if group_key not in grouped:
            grouped[group_key] = []
        grouped[group_key].append(element)
    
    # Check each group for inconsistencies
    for (category, key), group in grouped.items():
        if len(group) < 2:
            continue  # Need at least 2 elements to compare
        
        # Get unique values
        values = {e.value for e in group}
        
        if len(values) == 1:
            continue  # All consistent
        
        # Compare each pair for similarity
        unique_elements = list({e.value: e for e in group}.values())
        
        # Check if differences are significant
        significant_diff = False
        for i, e1 in enumerate(unique_elements):
            for e2 in unique_elements[i+1:]:
                similarity = calculate_similarity(e1.value, e2.value)
                if similarity < 0.8:  # Less than 80% similar
                    significant_diff = True
                    break
            if significant_diff:
                break
        
        if not significant_diff:
            continue  # Differences are minor
        
        # Determine canonical source and value
        documents = [e.document for e in group]
        canonical_source = get_canonical_source(documents)
        canonical_element = next(
            (e for e in group if e.document == canonical_source),
            group[0]
        )
        
        # Build variants dict
        variants = {e.document: e.value for e in group}
        
        # Determine severity based on category
        severity_map = {
            ConsistencyCategory.HYPOTHESIS: ConsistencySeverity.CRITICAL,
            ConsistencyCategory.VARIABLE: ConsistencySeverity.CRITICAL,
            ConsistencyCategory.METHODOLOGY: ConsistencySeverity.HIGH,
            ConsistencyCategory.CITATION: ConsistencySeverity.HIGH,
            ConsistencyCategory.STATISTIC: ConsistencySeverity.MEDIUM,
            ConsistencyCategory.TERMINOLOGY: ConsistencySeverity.MEDIUM,
        }
        
        issues.append(CrossDocumentIssue(
            category=category,
            severity=severity_map.get(category, ConsistencySeverity.MEDIUM),
            key=key,
            description=f"Inconsistent {category.value} '{key}' across {len(documents)} documents",
            affected_documents=list(set(documents)),
            canonical_value=canonical_element.value,
            canonical_source=canonical_source,
            variants=variants,
            suggestion=f"Align all occurrences with the definition in {canonical_source}",
        ))
    
    return issues


def check_citation_orphans(elements: List[ConsistencyElement]) -> List[CrossDocumentIssue]:
    """Check for citations referenced but not defined in BibTeX."""
    issues = []
    
    # Get all citation references (from .md and .tex files)
    references = {
        e.key for e in elements
        if e.category == ConsistencyCategory.CITATION
        and not e.document.endswith('.bib')
    }
    
    # Get all citation definitions (from .bib files)
    definitions = {
        e.key for e in elements
        if e.category == ConsistencyCategory.CITATION
        and e.document.endswith('.bib')
    }
    
    # Find orphans (referenced but not defined)
    orphans = references - definitions
    
    for key in orphans:
        # Find which documents reference this citation
        docs = [
            e.document for e in elements
            if e.category == ConsistencyCategory.CITATION
            and e.key == key
            and not e.document.endswith('.bib')
        ]
        
        issues.append(CrossDocumentIssue(
            category=ConsistencyCategory.CITATION,
            severity=ConsistencySeverity.HIGH,
            key=key,
            description=f"Citation '{key}' referenced but not defined in references.bib",
            affected_documents=docs,
            canonical_value="",
            canonical_source="literature/references.bib",
            variants={d: f"[@{key}]" for d in docs},
            suggestion=f"Add entry for '{key}' to references.bib or remove citation",
        ))
    
    # Find unused (defined but not referenced)
    unused = definitions - references
    
    for key in unused:
        issues.append(CrossDocumentIssue(
            category=ConsistencyCategory.CITATION,
            severity=ConsistencySeverity.LOW,
            key=key,
            description=f"Citation '{key}' defined in references.bib but never referenced",
            affected_documents=["literature/references.bib"],
            canonical_value="",
            canonical_source="literature/references.bib",
            variants={"literature/references.bib": f"@entry{{{key}}}"},
            suggestion=f"Consider removing unused citation '{key}' from references.bib",
        ))
    
    return issues


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_consistency(project_folder: str) -> ConsistencyReport:
    """
    Validate cross-document consistency for a project.
    
    Args:
        project_folder: Path to project folder
        
    Returns:
        ConsistencyReport with all findings
    """
    logger.info(f"Validating cross-document consistency for {project_folder}")
    
    # Extract all elements
    elements, documents_checked = extract_all_elements(project_folder)
    
    # Build element mapping for report
    element_mapping: Dict[str, List[ConsistencyElement]] = {}
    for element in elements:
        if element.key not in element_mapping:
            element_mapping[element.key] = []
        element_mapping[element.key].append(element)
    
    # Run comparisons
    issues = []
    issues.extend(compare_elements(elements))
    issues.extend(check_citation_orphans(elements))
    
    # Sort by severity
    severity_order = {
        ConsistencySeverity.CRITICAL: 0,
        ConsistencySeverity.HIGH: 1,
        ConsistencySeverity.MEDIUM: 2,
        ConsistencySeverity.LOW: 3,
    }
    issues.sort(key=lambda i: severity_order[i.severity])
    
    logger.info(
        f"Consistency check complete: {len(elements)} elements, "
        f"{len(issues)} issues ({sum(1 for i in issues if i.severity == ConsistencySeverity.CRITICAL)} critical)"
    )
    
    return ConsistencyReport(
        project_folder=project_folder,
        documents_checked=documents_checked,
        elements_extracted=len(elements),
        issues=issues,
        element_mapping=element_mapping,
    )
