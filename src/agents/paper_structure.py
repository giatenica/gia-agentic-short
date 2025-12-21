"""
Paper Structure Agent
=====================
Creates a LaTeX paper structure based on the research overview,
literature review, and the writing style guide. Generates a
complete paper template ready for content.

Uses Sonnet 4.5 for document generation.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import time
import json
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from .base import BaseAgent, AgentResult
from src.llm.claude_client import TaskType
from loguru import logger


# Load writing style guide at module level for system prompt
WRITING_STYLE_GUIDE_PATH = Path(__file__).parent.parent.parent / "docs" / "writing_style_guide.md"

# System prompt for paper structure generation
PAPER_STRUCTURE_PROMPT = """You are a paper structure agent for academic finance research.

Your role is to create well-organized LaTeX paper structures that conform to top finance journal standards. You have expertise in:
- Journal formatting requirements (RFS, JFE, JF, JFQA)
- Academic writing conventions
- LaTeX document preparation
- Short article structure (5-10 pages)

STRUCTURE REQUIREMENTS:

For short finance papers (5-10 pages):
1. Title Page (separate)
2. Abstract (50-100 words)
3. Introduction (1-2 pages, includes brief literature)
4. Data and Methodology (1-2 pages)
5. Results (2-3 pages)
6. Conclusion (0.5-1 page)
7. References
8. Tables and Figures (separate pages)
9. Online Appendix (if needed)

FORMATTING STANDARDS:
- 12pt Times New Roman
- Double-spaced
- 1-inch margins
- Chicago Author-Date citations
- No vertical rules in tables
- booktabs package for tables

LaTeX PACKAGES TO INCLUDE:
- geometry (margins)
- setspace (double spacing)
- natbib (citations)
- graphicx (figures)
- booktabs (tables)
- amsmath (equations)
- hyperref (links)
- appendix (for appendices)

OUTPUT FORMAT:

Generate a complete LaTeX document structure with:
1. Document preamble with all necessary packages
2. Title page template (anonymous for submission)
3. All section scaffolding with placeholder content
4. Table and figure templates
5. Bibliography setup
6. Appendix structure if needed

The LaTeX code should be immediately compilable and follow all journal conventions.

IMPORTANT:
- Do not include author-identifying information in main document
- Include placeholders where content will be added
- Add comments explaining what goes in each section
- Follow exact formatting requirements for target journal
- Keep structure appropriate for short article format"""


class PaperStructureAgent(BaseAgent):
    """
    Agent that creates LaTeX paper structures.
    
    Uses Sonnet 4.5 for document generation.
    """
    
    def __init__(self, client=None):
        super().__init__(
            name="PaperStructurer",
            task_type=TaskType.DOCUMENT_CREATION,  # Uses Sonnet
            system_prompt=PAPER_STRUCTURE_PROMPT,
            client=client,
        )
        
        # Load writing style guide
        self.style_guide = ""
        if WRITING_STYLE_GUIDE_PATH.exists():
            self.style_guide = WRITING_STYLE_GUIDE_PATH.read_text()
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Create LaTeX paper structure based on research context.
        
        Args:
            context: Should contain:
                - 'research_overview': Research overview content
                - 'literature_result': Literature synthesis results
                - 'hypothesis_result': Hypothesis development results
                - 'project_folder': Path to save output files
                - 'project_data': Project metadata (journal, paper type)
            
        Returns:
            AgentResult with LaTeX structure
        """
        start_time = time.time()
        
        # Get inputs
        research_overview = context.get("research_overview", "")
        hypothesis_result = context.get("hypothesis_result", {})
        literature_result = context.get("literature_result", {})
        project_data = context.get("project_data", {})
        project_folder = context.get("project_folder")
        
        try:
            # Build context for Claude
            user_message = self._build_structure_message(
                research_overview=research_overview,
                hypothesis_result=hypothesis_result,
                literature_result=literature_result,
                project_data=project_data,
            )
            
            # Generate LaTeX structure
            logger.info("Generating LaTeX paper structure...")
            response, tokens = await self._call_claude(
                user_message=user_message,
                use_thinking=False,
                max_tokens=20000,
            )
            
            # Extract LaTeX code
            latex_content = self._extract_latex(response)
            
            # Save files if project folder provided
            files_saved = {}
            if project_folder:
                project_path = Path(project_folder)
                
                # Create paper directory
                paper_dir = project_path / "paper"
                paper_dir.mkdir(exist_ok=True)
                
                # Save main LaTeX file
                main_tex_path = paper_dir / "main.tex"
                main_tex_path.write_text(latex_content)
                files_saved["main_tex"] = str(main_tex_path)
                logger.info(f"Saved main.tex to {main_tex_path}")
                
                # Copy references.bib if it exists
                bib_source = project_path / "references.bib"
                if bib_source.exists():
                    bib_dest = paper_dir / "references.bib"
                    bib_dest.write_text(bib_source.read_text())
                    files_saved["references_bib"] = str(bib_dest)
                
                # Save structure overview
                structure_md = self._generate_structure_overview(response, project_data)
                structure_path = paper_dir / "STRUCTURE.md"
                structure_path.write_text(structure_md)
                files_saved["structure_md"] = str(structure_path)
            
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content=latex_content,
                tokens_used=tokens,
                execution_time=time.time() - start_time,
                structured_data={
                    "files_saved": files_saved,
                    "sections": self._extract_sections(latex_content),
                    "target_journal": project_data.get("target_journal", ""),
                    "paper_type": project_data.get("paper_type", "short article"),
                },
            )
            
        except Exception as e:
            logger.error(f"Paper structure error: {e}")
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _build_structure_message(
        self,
        research_overview: str,
        hypothesis_result: dict,
        literature_result: dict,
        project_data: dict,
    ) -> str:
        """Build the user message for structure generation."""
        
        # Extract key information
        hyp_data = hypothesis_result.get("structured_data", {})
        main_hypothesis = hyp_data.get("main_hypothesis", "")
        
        lit_data = literature_result.get("structured_data", {})
        research_streams = lit_data.get("research_streams", [])
        
        target_journal = project_data.get("target_journal", "RFS/JFE/JF/JFQA")
        paper_type = project_data.get("paper_type", "short article (5-10 pages)")
        
        message = f"""Please create a complete LaTeX paper structure for this academic finance research project.

## TARGET SPECIFICATIONS
- Journal: {target_journal}
- Paper Type: {paper_type}
- Format: Double-anonymous peer review

## RESEARCH OVERVIEW
{research_overview[:4000] if research_overview else "Not yet available - create general structure"}

## MAIN HYPOTHESIS
{main_hypothesis if main_hypothesis else "To be determined"}

## LITERATURE STREAMS TO ADDRESS
{chr(10).join(f"- {s}" for s in research_streams) if research_streams else "- Pending literature review"}

## RELEVANT STYLE GUIDE EXCERPTS

### Section Word Counts (Short Paper)
| Section | Target |
|---------|--------|
| Abstract | 50-75 words |
| Introduction | 500-800 words |
| Data/Methods | 400-700 words |
| Results | 800-1200 words |
| Conclusion | 200-400 words |
| **Total** | ~2000-3200 words |

### LaTeX Template Requirements
- documentclass: article, 12pt
- Packages: setspace, geometry, natbib, graphicx, booktabs, amsmath
- Bibliography style: chicago
- Double-spaced throughout
- 1-inch margins

Please generate a complete, compilable LaTeX document with:
1. Full preamble with all necessary packages
2. Title page (anonymous)
3. All sections with detailed placeholder content
4. Table and figure templates
5. Bibliography setup pointing to references.bib
6. Appendix template

The document should be immediately usable as a starting point for writing."""
        
        return message
    
    def _extract_latex(self, response: str) -> str:
        """Extract LaTeX code from response."""
        # Look for code block
        if "```latex" in response:
            start = response.find("```latex") + 8
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        if "```tex" in response:
            start = response.find("```tex") + 6
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        # Look for \documentclass
        if "\\documentclass" in response:
            start = response.find("\\documentclass")
            end = response.rfind("\\end{document}") + len("\\end{document}")
            if end > start:
                return response[start:end].strip()
        
        # Return full response if no code block found
        return response
    
    def _extract_sections(self, latex_content: str) -> list:
        """Extract section names from LaTeX."""
        sections = []
        for line in latex_content.split("\n"):
            if "\\section{" in line:
                start = line.find("\\section{") + 9
                end = line.find("}", start)
                if end > start:
                    sections.append(line[start:end])
            elif "\\subsection{" in line:
                start = line.find("\\subsection{") + 12
                end = line.find("}", start)
                if end > start:
                    sections.append("  - " + line[start:end])
        return sections
    
    def _generate_structure_overview(self, response: str, project_data: dict) -> str:
        """Generate markdown overview of paper structure."""
        
        return f"""# Paper Structure Overview

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Target Journal: {project_data.get("target_journal", "Top Finance Journal")}
Paper Type: {project_data.get("paper_type", "Short Article")}

## Files Generated

- `main.tex` - Main LaTeX document
- `references.bib` - Bibliography file
- `STRUCTURE.md` - This overview

## Compilation

To compile the paper:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk:

```bash
latexmk -pdf main.tex
```

## Word Count Targets

| Section | Target Words |
|---------|-------------|
| Abstract | 50-75 |
| Introduction | 500-800 |
| Data and Methodology | 400-700 |
| Results | 800-1200 |
| Conclusion | 200-400 |
| **Total** | 2000-3200 |

## Next Steps

1. Fill in abstract with key findings preview
2. Complete introduction with research question and contribution
3. Add data description and methodology details
4. Present results with tables and figures
5. Write conclusion with implications
6. Verify all citations are in references.bib
7. Review formatting against journal guidelines

---

Author: Gia Tenica*

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher, for more information see: https://giatenica.com
"""
