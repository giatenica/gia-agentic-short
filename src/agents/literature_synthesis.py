"""
Literature Synthesis Agent
==========================
Processes Edison API responses to create incremental research files
with literature summaries and generates a template .bib file for
the paper's bibliography.

Uses Sonnet 4.5 for document synthesis.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import asyncio
import time
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone

from .base import BaseAgent, AgentResult
from src.llm.claude_client import TaskType
from src.llm.edison_client import Citation, LiteratureResult
from src.citations.populate import build_and_write_bibliography_from_citations_data
from src.citations.verification import resolve_doi_to_record_with_fallback
from loguru import logger


# System prompt for literature synthesis
LITERATURE_SYNTHESIS_PROMPT = """You are a literature synthesis agent for academic finance research.

Your role is to process literature search results and create comprehensive research documentation that will be used for writing academic papers.

SYNTHESIS PROCESS:

1. ORGANIZE BY THEME
   - Group papers by research stream
   - Identify theoretical foundations
   - Note methodological approaches
   - Highlight key findings and debates

2. ASSESS RELEVANCE
   - How directly does each paper relate to the hypothesis?
   - What evidence supports or challenges the hypothesis?
   - What methodological approaches are most applicable?
   - Which papers should be cited prominently?

3. IDENTIFY KEY PAPERS
   - Seminal papers that must be cited
   - Recent papers with similar methodology
   - Papers with conflicting findings
   - Methodological reference papers

4. EXTRACT INSIGHTS
   - What does the literature consensus say?
   - What gaps exist in prior research?
   - What methodological best practices emerge?
   - How should results be positioned relative to prior work?

OUTPUT FORMAT:

# Literature Review Summary

## Overview
[Brief summary of the literature landscape]

## Key Research Streams

### [Stream 1 Name]
**Key Papers:**
- Author (Year): [Key finding]
- Author (Year): [Key finding]

**Implications for This Research:**
[How this stream relates to the hypothesis]

### [Stream 2 Name]
[Continue for each major stream]

## Methodological Insights
[What methods are standard in this literature?]

## Evidence Summary

### Supporting Evidence
[Papers/findings that support the hypothesis]

### Contrasting Evidence
[Papers/findings that challenge or complicate the hypothesis]

## Literature Gaps
[What has not been studied that this paper will address]

## Citation Priority List
1. **Must Cite:** [Papers essential to reference]
2. **Should Cite:** [Papers important for positioning]
3. **Consider Citing:** [Papers for robustness/completeness]

## Recommended Positioning
[How to position the paper relative to prior literature]

IMPORTANT:
- Be accurate in representing prior findings
- Note when claims come from Edison's synthesis vs direct paper content
- Distinguish between well-established findings and contested claims
- Do not overstate the novelty of the contribution"""


class LiteratureSynthesisAgent(BaseAgent):
    """
    Agent that synthesizes literature search results.
    
    Uses Sonnet 4.5 for document synthesis.
    """
    
    def __init__(self, client: Optional[Any] = None):
        super().__init__(
            name="LiteratureSynthesizer",
            task_type=TaskType.DOCUMENT_CREATION,  # Uses Sonnet
            system_prompt=LITERATURE_SYNTHESIS_PROMPT,
            client=client,
        )
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Synthesize literature and create research documentation.
        
        Args:
            context: Must contain:
                - 'literature_result': Output from LiteratureSearchAgent
                - 'hypothesis_result': Output from HypothesisDevelopmentAgent
                - 'project_folder': Path to save output files
            
        Returns:
            AgentResult with synthesis and file paths
        """
        start_time = time.time()
        
        # Get inputs
        literature_result = context.get("literature_result", {})
        hypothesis_result = context.get("hypothesis_result", {})
        project_folder = context.get("project_folder")
        
        # Extract literature data
        lit_structured = literature_result.get("structured_data", {}) or {}
        if not isinstance(lit_structured, dict):
            lit_structured = {}

        lit_response = literature_result.get("content", lit_structured.get("literature_result", {}).get("response", ""))
        citations_data = lit_structured.get("citations") or []
        
        if not lit_response and not citations_data:
            # Allow the workflow to proceed even when Edison is unavailable.
            # This generates a preliminary literature review scaffold that can be
            # replaced once literature search is configured.
            lit_response = (
                "No literature search response is available. Edison may be unavailable "
                "or not configured. Create a preliminary literature review scaffold with "
                "explicit placeholders for citations and papers to verify."
            )
        
        # Extract hypothesis data
        hyp_structured = hypothesis_result.get("structured_data", {})
        main_hypothesis = hyp_structured.get("main_hypothesis", "")
        
        try:
            # Step 1: Synthesize literature using Claude (with timeout + scaffold fallback)
            logger.info("Synthesizing literature review...")
            synthesis_response = ""
            tokens = 0
            timed_out = False
            try:
                synthesis_response, tokens = await asyncio.wait_for(
                    self._call_claude(
                        user_message=self._build_synthesis_message(
                            lit_response=lit_response,
                            citations_data=citations_data,
                            main_hypothesis=main_hypothesis,
                            hypothesis_content=hypothesis_result.get("content", ""),
                        ),
                        use_thinking=False,
                        max_tokens=16000,
                    ),
                    timeout=240,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                timed_out = True
                logger.warning("Literature synthesis LLM call timed out; generating scaffold output instead.")
                synthesis_response = self._build_timeout_scaffold(
                    lit_response=lit_response,
                    citations_data=citations_data,
                    main_hypothesis=main_hypothesis,
                )
            
            # Step 2: Generate BibTeX file
            logger.info("Generating BibTeX file...")
            bibliography_info: Dict[str, Any] = {}
            bibtex_content = self._generate_bibtex(citations_data)

            # If we have a project folder, generate canonical bibliography outputs under
            # bibliography/ and reuse that BibTeX for backward-compatible references.bib.
            if project_folder:
                try:
                    bibliography_info = self._build_and_write_bibliography(
                        project_folder=str(project_folder),
                        citations_data=citations_data,
                    )
                    canonical_bib = bibliography_info.get("references_bib")
                    if isinstance(canonical_bib, str) and canonical_bib:
                        bibtex_content = Path(canonical_bib).read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Failed to build canonical bibliography outputs; using legacy BibTeX. err={e}")
            
            # Step 3: Save files if project folder provided
            files_saved = {}
            if project_folder:
                project_path = Path(project_folder)

                verification_note = ""
                verification = bibliography_info.get("verification") if isinstance(bibliography_info, dict) else None
                if isinstance(verification, dict):
                    status = str(verification.get("status") or "")
                    if status and status != "verified":
                        verification_note = (
                            "Note: Citation metadata in this document is provisional. "
                            "Automated verification failed or was incomplete; verify against original sources "
                            "before making definitive citation claims."
                        )
                
                # Save literature synthesis
                lit_review_path = project_path / "LITERATURE_REVIEW.md"
                lit_review_path.write_text(self._format_literature_review(
                    synthesis=synthesis_response,
                    citations_count=len(citations_data),
                    hypothesis=main_hypothesis,
                    verification_note=verification_note,
                ))
                files_saved["literature_review"] = str(lit_review_path)
                logger.info(f"Saved literature review to {lit_review_path}")
                
                # Save BibTeX
                bib_path = project_path / "references.bib"
                bib_path.write_text(bibtex_content)
                files_saved["bibtex"] = str(bib_path)
                logger.info(f"Saved BibTeX to {bib_path}")

                if isinstance(bibliography_info, dict):
                    canonical_citations = bibliography_info.get("citations_json")
                    canonical_bib = bibliography_info.get("references_bib")
                    if isinstance(canonical_citations, str) and canonical_citations:
                        files_saved["bibliography_citations_json"] = canonical_citations
                    if isinstance(canonical_bib, str) and canonical_bib:
                        files_saved["bibliography_references_bib"] = canonical_bib
                
                # Save citations data as JSON for later use
                citations_json_path = project_path / "citations_data.json"
                with open(citations_json_path, "w") as f:
                    json.dump({
                        "citations": citations_data,
                        "search_query": lit_structured.get("primary_query", ""),
                        "total_papers": len(citations_data),
                        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "bibliography": bibliography_info,
                    }, f, indent=2)
                files_saved["citations_json"] = str(citations_json_path)
            
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content=synthesis_response,
                tokens_used=tokens,
                execution_time=time.time() - start_time,
                structured_data={
                    "files_saved": files_saved,
                    "bibtex_content": bibtex_content,
                    "citations_count": len(citations_data),
                    "research_streams": self._extract_streams(synthesis_response),
                    "timed_out": timed_out,
                    "bibliography": bibliography_info,
                },
            )
            
        except Exception as e:
            logger.error(f"Literature synthesis error: {e}")
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _build_synthesis_message(
        self,
        lit_response: str,
        citations_data: List[dict],
        main_hypothesis: str,
        hypothesis_content: str,
    ) -> str:
        """Build the user message for synthesis."""
        
        # Format citations list
        citations_summary = ""
        for i, c in enumerate(citations_data[:30], 1):  # Limit to top 30
            authors = ", ".join(c.get("authors", [])[:3])
            if len(c.get("authors", [])) > 3:
                authors += " et al."
            year = c.get("year", "n.d.")
            title = c.get("title", "Untitled")
            journal = c.get("journal", "")
            
            citations_summary += f"{i}. {authors} ({year}). \"{title}\""
            if journal:
                citations_summary += f". {journal}"
            citations_summary += "\n"
        
        return f"""Please synthesize the following literature search results for academic research documentation.

## MAIN HYPOTHESIS
{main_hypothesis}

## HYPOTHESIS ANALYSIS
{hypothesis_content[:2000] if hypothesis_content else "Not provided"}

## EDISON LITERATURE RESPONSE
{lit_response}

## PAPERS FOUND ({len(citations_data)} total)
{citations_summary}

Please create a comprehensive literature review summary that:
1. Organizes papers by research stream
2. Identifies key findings relevant to the hypothesis
3. Notes methodological approaches used
4. Highlights gaps this research can fill
5. Provides citation priority recommendations"""
    
    def _generate_bibtex(self, citations_data: List[dict]) -> str:
        """Generate BibTeX file from citations."""
        bibtex_entries = []
        used_keys = set()
        
        header = f"""% Bibliography for Research Paper
% Generated by Gia Tenica - Literature Synthesis Agent
% Date: {datetime.now().strftime("%Y-%m-%d")}
% Papers: {len(citations_data)}
%
% Note: This is a template bibliography. Please verify entries
% against original sources before final submission.

"""
        bibtex_entries.append(header)
        
        for citation in citations_data:
            # Generate unique key
            authors = citation.get("authors", ["Unknown"])
            first_author = authors[0].split()[-1] if authors else "Unknown"
            year = citation.get("year", 0)
            base_key = f"{first_author}{year}"
            key = base_key
            suffix = ord('a')
            
            while key in used_keys:
                key = f"{base_key}{chr(suffix)}"
                suffix += 1
            
            used_keys.add(key)
            
            # Format authors
            authors_str = " and ".join(authors)
            
            # Build entry
            entry = f"@article{{{key},\n"
            entry += f"  title = {{{citation.get('title', '')}}},\n"
            entry += f"  author = {{{authors_str}}},\n"
            entry += f"  year = {{{year}}},\n"
            
            if citation.get("journal"):
                entry += f"  journal = {{{citation['journal']}}},\n"
            if citation.get("doi"):
                entry += f"  doi = {{{citation['doi']}}},\n"
            if citation.get("url"):
                entry += f"  url = {{{citation['url']}}},\n"
            
            entry += "}\n"
            bibtex_entries.append(entry)
        
        return "\n".join(bibtex_entries)

    def _build_and_write_bibliography(self, *, project_folder: str, citations_data: List[dict]) -> Dict[str, Any]:
        return build_and_write_bibliography_from_citations_data(
            project_folder=project_folder,
            citations_data=citations_data,
            resolve_doi_fn=resolve_doi_to_record_with_fallback,
        )
    
    def _format_literature_review(
        self,
        synthesis: str,
        citations_count: int,
        hypothesis: str,
        verification_note: str = "",
    ) -> str:
        """Format the complete literature review document."""
        
        header = f"""---
title: Literature Review Summary
generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
papers_reviewed: {citations_count}
hypothesis: "{hypothesis[:100]}..."
author: Gia Tenica*
---

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher, for more information see: https://giatenica.com

---

"""
        if verification_note:
            return header + verification_note + "\n\n" + synthesis
        return header + synthesis
    
    def _extract_streams(self, synthesis: str) -> List[str]:
        """Extract research stream names from synthesis."""
        streams = []
        lines = synthesis.split("\n")
        
        for line in lines:
            # Look for ### headers that likely indicate research streams
            if line.startswith("### ") and "Stream" not in line:
                stream_name = line.replace("###", "").strip()
                if stream_name and stream_name not in ["Key Papers", "Implications for This Research"]:
                    streams.append(stream_name)
        
        return streams
