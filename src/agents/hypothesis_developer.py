"""
Hypothesis Development Agent
============================
Analyzes the research overview to formulate the best hypothesis
for the research project. Considers the data available, literature
positioning, and methodological constraints.

Uses Opus 4.5 for complex reasoning about hypothesis formulation.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import time
import json
from typing import Optional
from pathlib import Path

from .base import BaseAgent, AgentResult
from src.llm.claude_client import TaskType
from loguru import logger


# System prompt for hypothesis development
HYPOTHESIS_DEVELOPMENT_PROMPT = """You are a hypothesis development agent for academic finance research.

Your role is to analyze research overviews and formulate well-structured, testable hypotheses that:
1. Are grounded in economic theory or established empirical patterns
2. Can be tested with the available data
3. Make a clear contribution to the literature
4. Are appropriate for the target journal and paper format

ANALYSIS PROCESS:

1. REVIEW THE RESEARCH CONTEXT
   - What is the core research question?
   - What data is available?
   - What methodology is proposed or feasible?
   - What is the target journal and format?

2. EVALUATE POTENTIAL HYPOTHESES
   - What predictions follow from economic theory?
   - What empirical patterns might exist?
   - What alternative explanations should be considered?
   - How do these relate to prior literature?

3. FORMULATE THE HYPOTHESIS
   - State the main hypothesis clearly and precisely
   - Explain the economic intuition behind it
   - Identify testable predictions
   - Note alternative hypotheses to consider

OUTPUT FORMAT:

## Hypothesis Analysis

### Research Context Summary
[Brief summary of the project context]

### Economic Foundation
[The theoretical or empirical basis for the hypothesis]

### Main Hypothesis
**H1:** [Clear, testable statement]

Economic Intuition: [Why this prediction follows from theory or evidence]

Testable Predictions:
- [Specific prediction 1]
- [Specific prediction 2]
- [Specific prediction 3]

### Alternative Hypotheses
**H0 (Null):** [The null hypothesis]
**H2 (Alternative):** [An alternative explanation to rule out]

### Data Requirements
- Variables needed to test the hypothesis
- Identification strategy considerations
- Potential limitations

### Literature Questions
[3-5 specific questions to search in the literature that will:
1. Establish the theoretical foundation
2. Identify prior empirical findings
3. Reveal methodological best practices
4. Locate comparable studies]

IMPORTANT:
- Be specific and precise; avoid vague statements
- Ground hypotheses in theory or established patterns
- Consider what the data can actually test
- Frame predictions in terms of testable relationships
- Do not overstate expected findings"""


class HypothesisDevelopmentAgent(BaseAgent):
    """
    Agent that formulates research hypotheses.
    
    Uses Opus 4.5 for complex reasoning about hypothesis development.
    """
    
    def __init__(self, client=None):
        super().__init__(
            name="HypothesisDeveloper",
            task_type=TaskType.COMPLEX_REASONING,  # Uses Opus
            system_prompt=HYPOTHESIS_DEVELOPMENT_PROMPT,
            client=client,
        )
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Develop hypothesis from research overview.
        
        Args:
            context: Must contain 'project_folder' with RESEARCH_OVERVIEW.md
                    or 'research_overview' content directly
            
        Returns:
            AgentResult with hypothesis development
        """
        start_time = time.time()
        
        # Get research overview
        overview_content = context.get("research_overview")
        
        if not overview_content:
            project_folder = context.get("project_folder")
            if project_folder:
                overview_path = Path(project_folder) / "RESEARCH_OVERVIEW.md"
                if overview_path.exists():
                    overview_content = overview_path.read_text()
                else:
                    return AgentResult(
                        agent_name=self.name,
                        task_type=self.task_type,
                        model_tier=self.model_tier,
                        success=False,
                        content="",
                        error="RESEARCH_OVERVIEW.md not found",
                        execution_time=time.time() - start_time,
                    )
            else:
                return AgentResult(
                    agent_name=self.name,
                    task_type=self.task_type,
                    model_tier=self.model_tier,
                    success=False,
                    content="",
                    error="No research overview or project folder provided",
                    execution_time=time.time() - start_time,
                )
        
        # Get project data for additional context
        project_data = context.get("project_data", {})
        
        # Build user message
        user_message = self._build_user_message(overview_content, project_data)
        
        try:
            # Call Claude with extended thinking for complex reasoning
            response, tokens = await self._call_claude(
                user_message=user_message,
                use_thinking=True,  # Enable extended thinking for hypothesis development
                max_tokens=16000,
                budget_tokens=10000,
            )
            
            # Parse hypothesis from response
            hypothesis_data = self._parse_hypothesis(response)
            
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content=response,
                tokens_used=tokens,
                execution_time=time.time() - start_time,
                structured_data=hypothesis_data,
            )
            
        except Exception as e:
            logger.error(f"Hypothesis development error: {e}")
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _build_user_message(self, overview_content: str, project_data: dict) -> str:
        """Build the user message for hypothesis development."""
        message = f"""Please analyze this research overview and develop a well-formulated hypothesis.

## RESEARCH OVERVIEW

{overview_content}

"""
        
        # Add project context if available
        if project_data:
            message += "## PROJECT CONTEXT\n\n"
            
            if project_data.get("research_question"):
                message += f"**Research Question:** {project_data['research_question']}\n\n"
            
            if project_data.get("hypothesis"):
                message += f"**Initial Hypothesis (if any):** {project_data['hypothesis']}\n\n"
            
            if project_data.get("methodology"):
                message += f"**Proposed Methodology:** {project_data['methodology']}\n\n"
            
            if project_data.get("target_journal"):
                message += f"**Target Journal:** {project_data['target_journal']}\n"
            
            if project_data.get("paper_type"):
                message += f"**Paper Type:** {project_data['paper_type']}\n"
        
        message += """
Based on this information, please:
1. Analyze the research context
2. Formulate a clear, testable main hypothesis
3. Identify alternative hypotheses
4. Generate specific literature search questions

Focus on what can actually be tested with the available data and methodology."""
        
        return message
    
    def _parse_hypothesis(self, response: str) -> dict:
        """Parse structured data from the hypothesis response."""
        data = {
            "main_hypothesis": None,
            "null_hypothesis": None,
            "alternative_hypotheses": [],
            "testable_predictions": [],
            "literature_questions": [],
        }
        
        lines = response.split("\n")
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect main hypothesis
            if line_stripped.startswith("**H1:**") or line_stripped.startswith("H1:"):
                data["main_hypothesis"] = line_stripped.replace("**H1:**", "").replace("H1:", "").strip()
            
            # Detect null hypothesis
            elif line_stripped.startswith("**H0") or line_stripped.startswith("H0"):
                data["null_hypothesis"] = line_stripped.replace("**H0 (Null):**", "").replace("H0:", "").strip()
            
            # Detect alternative hypothesis
            elif line_stripped.startswith("**H2") or line_stripped.startswith("H2"):
                alt = line_stripped.replace("**H2 (Alternative):**", "").replace("H2:", "").strip()
                if alt:
                    data["alternative_hypotheses"].append(alt)
            
            # Section detection
            if "Testable Predictions" in line:
                current_section = "predictions"
            elif "Literature Questions" in line:
                current_section = "questions"
            elif line.startswith("###"):
                current_section = None
            
            # Collect list items
            if line_stripped.startswith("- ") or line_stripped.startswith("* "):
                item = line_stripped[2:].strip()
                if current_section == "predictions" and item:
                    data["testable_predictions"].append(item)
                elif current_section == "questions" and item:
                    data["literature_questions"].append(item)
            
            # Numbered list items
            if current_section == "questions":
                for i in range(1, 10):
                    if line_stripped.startswith(f"{i}. "):
                        item = line_stripped[3:].strip()
                        if item:
                            data["literature_questions"].append(item)
        
        return data
