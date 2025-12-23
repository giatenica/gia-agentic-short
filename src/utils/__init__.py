"""
Utility Functions
=================
Common utilities for validation, security, time tracking, readiness scoring,
consistency checking, and style validation.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from .validation import (
    validate_path,
    validate_project_folder,
    sanitize_filename,
    safe_json_loads,
    is_safe_path,
)

from .project_io import (
    load_project_json,
    get_project_id,
)

from .schema_validation import (
    validate_against_schema,
    validate_evidence_item,
    is_valid_evidence_item,
    validate_citation_record,
    is_valid_citation_record,
)

# Time tracking utilities
from .time_tracking import (
    TimeEstimate,
    ExecutionBudget,
    TrackedTask,
    TaskStatus,
    TaskLevel,
    TimeTrackingReport,
    parse_duration,
    parse_project_plan,
    save_tracking_report,
    load_tracking_report,
    initialize_tracking,
    update_task_status,
)

# Readiness scoring utilities
from .readiness_scoring import (
    ReadinessCategory,
    CheckStatus,
    AutomationCapability,
    ChecklistItem,
    PhaseReadiness,
    ReadinessReport,
    STANDARD_CHECKLIST,
    initialize_readiness_report,
    assess_project_readiness,
    save_readiness_report,
    load_readiness_report,
    check_file_exists,
)

# Consistency validation utilities
from .consistency_validation import (
    ConsistencyCategory,
    ConsistencySeverity,
    ConsistencyElement,
    CrossDocumentIssue,
    ConsistencyReport,
    extract_all_elements,
    compare_elements,
    check_citation_orphans,
)

# Style validation utilities
from .style_validation import (
    BannedWordMatch,
    SectionWordCount,
    StyleValidationResult,
    check_banned_words,
    count_words_by_section,
    validate_style,
)

__all__ = [
    # Validation
    "validate_path",
    "validate_project_folder",
    "sanitize_filename",
    "safe_json_loads",
    "is_safe_path",
    # Project IO
    "load_project_json",
    "get_project_id",
    # Schema validation
    "validate_against_schema",
    "validate_evidence_item",
    "is_valid_evidence_item",
    "validate_citation_record",
    "is_valid_citation_record",
    # Time tracking
    "TimeEstimate",
    "ExecutionBudget",
    "TrackedTask",
    "TaskStatus",
    "TaskLevel",
    "TimeTrackingReport",
    "parse_duration",
    "parse_project_plan",
    "save_tracking_report",
    "load_tracking_report",
    "initialize_tracking",
    "update_task_status",
    # Readiness scoring
    "ReadinessCategory",
    "CheckStatus",
    "AutomationCapability",
    "ChecklistItem",
    "PhaseReadiness",
    "ReadinessReport",
    "STANDARD_CHECKLIST",
    "initialize_readiness_report",
    "assess_project_readiness",
    "save_readiness_report",
    "load_readiness_report",
    "check_file_exists",
    # Consistency validation
    "ConsistencyCategory",
    "ConsistencySeverity",
    "ConsistencyElement",
    "CrossDocumentIssue",
    "ConsistencyReport",
    "extract_all_elements",
    "compare_elements",
    "check_citation_orphans",
    # Style validation
    "BannedWordMatch",
    "SectionWordCount",
    "StyleValidationResult",
    "check_banned_words",
    "count_words_by_section",
    "validate_style",
]
