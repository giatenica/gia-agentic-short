"""src.agents.prompts

Centralized prompts for GIA Research Pipeline agents.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from .literature_search import (
    QUERY_DECOMPOSITION_SYSTEM_PROMPT,
    QUERY_DECOMPOSITION_USER_TEMPLATE,
    CONTEXTUAL_SUMMARY_SYSTEM_PROMPT,
    CONTEXTUAL_SUMMARY_USER_TEMPLATE,
    EVIDENCE_SYNTHESIS_SYSTEM_PROMPT,
    EVIDENCE_SYNTHESIS_USER_TEMPLATE,
    LITERATURE_REVIEW_SYSTEM_PROMPT,
    LITERATURE_REVIEW_USER_TEMPLATE,
    build_query_decomposition_prompt,
    build_contextual_summary_prompt,
    build_evidence_synthesis_prompt,
    build_literature_review_prompt,
)

__all__ = [
    "QUERY_DECOMPOSITION_SYSTEM_PROMPT",
    "QUERY_DECOMPOSITION_USER_TEMPLATE",
    "CONTEXTUAL_SUMMARY_SYSTEM_PROMPT",
    "CONTEXTUAL_SUMMARY_USER_TEMPLATE",
    "EVIDENCE_SYNTHESIS_SYSTEM_PROMPT",
    "EVIDENCE_SYNTHESIS_USER_TEMPLATE",
    "LITERATURE_REVIEW_SYSTEM_PROMPT",
    "LITERATURE_REVIEW_USER_TEMPLATE",
    "build_query_decomposition_prompt",
    "build_contextual_summary_prompt",
    "build_evidence_synthesis_prompt",
    "build_literature_review_prompt",
]
