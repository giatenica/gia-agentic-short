"""
Project Planner Agent
=====================
Creates a detailed project plan with phases, steps, and substeps
for completing the research paper. Plans are based on the current
research state and remaining work.

Uses Opus 4.5 for complex planning and reasoning.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import time
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta

from .base import BaseAgent, AgentResult
from src.llm.claude_client import TaskType
from loguru import logger


# System prompt for project planning
PROJECT_PLANNER_PROMPT = """You are a project planning agent for AUTONOMOUS AI research workflows.

IMPORTANT: You are planning for AI AGENTS, not human researchers. All time estimates must be in AGENTIC EXECUTION TIME:
- Agents execute tasks in MINUTES, not hours
- A task that takes a human 1 hour typically takes an agent 5-15 minutes
- Use format: "X-Y minutes" or "X-Y hours" (max 2-3 hours per step)
- Account for API latency, file I/O, and processing time

TIME ESTIMATION GUIDE (Agent vs Human):
| Human Time | Agent Time | Task Type |
|------------|------------|-----------|  
| 1 hour | 5-15 min | Reading/synthesis |
| 2-3 hours | 15-30 min | Analysis/coding |
| 4-6 hours | 30-60 min | Complex reasoning |
| 8-12 hours | 1-2 hours | Full workflow step |
| 1 day | 2-4 hours | Major deliverable |

Your role is to create detailed, actionable project plans for autonomous AI agent execution. You have expertise in:
- AI agent workflow orchestration
- Agentic execution time estimation
- Finance paper requirements
- Task dependencies and parallel execution opportunities

PLANNING PRINCIPLES:

1. PHASED APPROACH
   - Phase 1: Foundation (hypothesis, data, methodology)
   - Phase 2: Analysis (main results, robustness checks)
   - Phase 3: Writing (draft sections, integrate findings)
   - Phase 4: Polish (revision, formatting, submission)

2. TASK STRUCTURE
   - Each phase has clear deliverables
   - Tasks have specific, measurable outcomes
   - Dependencies are explicitly noted
   - Time estimates are in AGENT EXECUTION TIME (minutes/hours)

3. ACADEMIC WORKFLOW
   - Literature review informs methodology
   - Data cleaning precedes analysis
   - Results drive writing
   - Multiple revision rounds expected

4. QUALITY GATES
   - Each phase has acceptance criteria
   - Clear checkpoints for review
   - Go/no-go decisions at phase boundaries

OUTPUT FORMAT:

# Research Project Plan

## Project Overview
- Title: [Research title]
- Target Journal: [Journal]
- Target Submission Date: [Date if known]
- Current Status: [Assessment of progress]

## Phase 1: Foundation
**Objective:** [Phase goal]
**Duration:** [Estimated time]
**Prerequisites:** [What must be done first]

### Step 1.1: [Step name]
**Duration:** [Time estimate]
**Deliverables:**
- [Specific output 1]
- [Specific output 2]

**Substeps:**
1. [Detailed action]
2. [Detailed action]

**Acceptance Criteria:**
- [ ] [Checkable criterion]
- [ ] [Checkable criterion]

### Step 1.2: [Next step]
[Continue pattern]

## Phase 2: Analysis
[Similar structure]

## Phase 3: Writing
[Similar structure]

## Phase 4: Polish and Submit
[Similar structure]

## Timeline Summary
| Phase | Start | End | Status |
|-------|-------|-----|--------|
| [Phase] | [Date] | [Date] | [Status] |

## Risk Assessment
- [Risk 1]: [Mitigation]
- [Risk 2]: [Mitigation]

## Resource Requirements
- [Resource type]: [Details]

IMPORTANT:
- Base plans on actual current state
- Be specific about deliverables
- Use AGENTIC time estimates (minutes/hours, not human days)
- Note dependencies between tasks
- Identify parallel execution opportunities
- Make tasks actionable and measurable
- Format durations as "X-Y minutes" or "X-Y hours" for time tracker compatibility"""


class ProjectPlannerAgent(BaseAgent):
    """
    Agent that creates detailed project plans.
    
    Uses Opus 4.5 for complex planning and reasoning.
    """
    
    def __init__(self, client=None):
        super().__init__(
            name="ProjectPlanner",
            task_type=TaskType.COMPLEX_REASONING,  # Uses Opus
            system_prompt=PROJECT_PLANNER_PROMPT,
            client=client,
        )
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Create project plan based on current research state.
        
        Args:
            context: Should contain:
                - 'research_overview': Current research overview
                - 'hypothesis_result': Hypothesis development output
                - 'literature_result': Literature search output (Edison papers)
                - 'literature_synthesis': Literature synthesis output (LITERATURE_REVIEW.md)
                - 'paper_structure': Paper structure output
                - 'project_data': Project metadata
                - 'project_folder': Path to save output files
            
        Returns:
            AgentResult with project plan
        """
        start_time = time.time()
        
        # Get inputs
        research_overview = context.get("research_overview", "")
        hypothesis_result = context.get("hypothesis_result", {})
        literature_result = context.get("literature_result", {})
        literature_synthesis = context.get("literature_synthesis", {})
        paper_structure = context.get("paper_structure", {})
        project_data = context.get("project_data", {})
        project_folder = context.get("project_folder")
        
        try:
            # Build context for Claude
            user_message = self._build_planning_message(
                research_overview=research_overview,
                hypothesis_result=hypothesis_result,
                literature_result=literature_result,
                literature_synthesis=literature_synthesis,
                paper_structure=paper_structure,
                project_data=project_data,
            )
            
            # Generate project plan with extended thinking
            logger.info("Generating project plan...")
            response, tokens = await self._call_claude(
                user_message=user_message,
                use_thinking=True,  # Complex planning benefits from thinking
                max_tokens=20000,
                budget_tokens=12000,
            )
            
            # Parse plan structure
            plan_data = self._parse_plan(response)
            
            # Save files if project folder provided
            files_saved = {}
            if project_folder:
                project_path = Path(project_folder)
                
                # Save project plan
                plan_path = project_path / "PROJECT_PLAN.md"
                plan_path.write_text(self._format_plan(response))
                files_saved["project_plan"] = str(plan_path)
                logger.info(f"Saved project plan to {plan_path}")
                
                # Save plan data as JSON
                plan_json_path = project_path / "project_plan.json"
                with open(plan_json_path, "w") as f:
                    json.dump({
                        "generated_at": datetime.now().isoformat(),
                        "phases": plan_data.get("phases", []),
                        "timeline": plan_data.get("timeline", {}),
                        "risks": plan_data.get("risks", []),
                    }, f, indent=2)
                files_saved["plan_json"] = str(plan_json_path)
            
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=True,
                content=response,
                tokens_used=tokens,
                execution_time=time.time() - start_time,
                structured_data={
                    "files_saved": files_saved,
                    "phases": plan_data.get("phases", []),
                    "total_steps": plan_data.get("total_steps", 0),
                    "estimated_duration": plan_data.get("estimated_duration", ""),
                },
            )
            
        except Exception as e:
            logger.error(f"Project planning error: {e}")
            return AgentResult(
                agent_name=self.name,
                task_type=self.task_type,
                model_tier=self.model_tier,
                success=False,
                content="",
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _build_planning_message(
        self,
        research_overview: str,
        hypothesis_result: dict,
        literature_result: dict,
        literature_synthesis: dict,
        paper_structure: dict,
        project_data: dict,
    ) -> str:
        """Build the user message for planning."""
        
        # Assess current state
        current_state = self._assess_current_state(
            research_overview=research_overview,
            hypothesis_result=hypothesis_result,
            literature_result=literature_result,
            literature_synthesis=literature_synthesis,
            paper_structure=paper_structure,
        )
        
        # Extract key information
        target_journal = project_data.get("target_journal", "Top Finance Journal")
        paper_type = project_data.get("paper_type", "short article")
        timeline = project_data.get("timeline", "")
        
        # Get literature stats from synthesis if available, else from search
        lit_data = literature_synthesis.get("structured_data", {}) if literature_synthesis else {}
        if not lit_data:
            lit_data = literature_result.get("structured_data", {}) if literature_result else {}
        papers_count = lit_data.get("citations_count", 0) or lit_data.get("papers_reviewed", 0)
        research_streams = lit_data.get("research_streams", [])
        
        message = f"""Please create a detailed project plan for this academic finance research paper.

## TARGET SPECIFICATIONS
- Journal: {target_journal}
- Paper Type: {paper_type}
- Timeline: {timeline if timeline else "Flexible"}

## CURRENT PROJECT STATE

### Completed Steps
{chr(10).join(f"- {item}" for item in current_state["completed"])}

### In Progress
{chr(10).join(f"- {item}" for item in current_state["in_progress"])}

### Remaining Work
{chr(10).join(f"- {item}" for item in current_state["remaining"])}

## RESEARCH OVERVIEW SUMMARY
{research_overview[:3000] if research_overview else "Not yet generated"}

## HYPOTHESIS STATUS
{hypothesis_result.get("content", "")[:1500] if hypothesis_result else "Not yet developed"}

## LITERATURE STATUS
Papers reviewed: {papers_count}
Research streams identified: {len(research_streams)}
Literature review file: {"Generated (LITERATURE_REVIEW.md)" if literature_synthesis and literature_synthesis.get("success") else "Not yet generated"}

## PAPER STRUCTURE STATUS
{"Paper structure created" if paper_structure else "Paper structure not yet created"}
Sections defined: {len(paper_structure.get("structured_data", {}).get("sections", [])) if paper_structure else 0}

Please create a comprehensive project plan that:
1. Acknowledges current progress
2. Defines clear phases with deliverables
3. Provides specific, actionable steps
4. Includes realistic time estimates
5. Identifies dependencies and risks
6. Sets quality gates between phases

Consider this is a short paper (5-10 pages) for a top finance journal."""
        
        return message
    
    def _assess_current_state(
        self,
        research_overview: str,
        hypothesis_result: dict,
        literature_result: dict,
        literature_synthesis: dict,
        paper_structure: dict,
    ) -> Dict[str, List[str]]:
        """Assess current project state."""
        
        completed = []
        in_progress = []
        remaining = []
        
        # Check research overview
        if research_overview:
            completed.append("Initial research overview generated")
        else:
            remaining.append("Generate research overview")
        
        # Check hypothesis
        if hypothesis_result and hypothesis_result.get("success"):
            completed.append("Hypothesis developed")
        elif hypothesis_result:
            in_progress.append("Hypothesis development")
        else:
            remaining.append("Develop testable hypothesis")
        
        # Check literature - synthesis is the primary indicator (contains LITERATURE_REVIEW.md)
        # Literature search (literature_result) is intermediate step
        if literature_synthesis and literature_synthesis.get("success"):
            # Synthesis completed - this means LITERATURE_REVIEW.md was generated
            lit_data = literature_synthesis.get("structured_data", {})
            papers_count = lit_data.get("citations_count", 0) or lit_data.get("papers_reviewed", 0)
            completed.append(f"Literature review completed ({papers_count} papers synthesized)")
            # Check if files were saved
            files_saved = lit_data.get("files_saved", {})
            if files_saved.get("literature_review"):
                completed.append("LITERATURE_REVIEW.md generated")
            if files_saved.get("references"):
                completed.append("References bibliography generated")
        elif literature_result and literature_result.get("success"):
            # Search completed but synthesis not yet done
            citations_count = literature_result.get("structured_data", {}).get("citations_count", 0)
            in_progress.append(f"Literature search completed ({citations_count} papers found), synthesis pending")
        elif literature_result:
            in_progress.append("Literature search in progress")
        else:
            remaining.append("Complete literature review")
        
        # Check paper structure
        if paper_structure and paper_structure.get("success"):
            completed.append("Paper structure created")
            # Check for LaTeX template
            files_saved = paper_structure.get("structured_data", {}).get("files_saved", {})
            if files_saved.get("main_tex"):
                completed.append("LaTeX template generated")
        else:
            remaining.append("Create paper structure")
        
        # Always remaining for a new project
        remaining.extend([
            "Data analysis and results",
            "Write introduction section",
            "Write methodology section",
            "Write results section",
            "Write conclusion",
            "Create tables and figures",
            "Format for submission",
            "Final review and polish",
        ])
        
        return {
            "completed": completed,
            "in_progress": in_progress,
            "remaining": remaining,
        }
    
    def _parse_plan(self, response: str) -> Dict[str, Any]:
        """Parse plan structure from response."""
        result = {
            "phases": [],
            "timeline": {},
            "risks": [],
            "total_steps": 0,
        }
        
        lines = response.split("\n")
        current_phase = None
        
        for i, line in enumerate(lines):
            # Detect phases
            if line.startswith("## Phase"):
                phase_name = line.replace("##", "").strip()
                current_phase = {
                    "name": phase_name,
                    "steps": [],
                }
                result["phases"].append(current_phase)
            
            # Detect steps
            elif line.startswith("### Step") and current_phase:
                step_name = line.replace("###", "").strip()
                current_phase["steps"].append(step_name)
                result["total_steps"] += 1
            
            # Detect risks
            elif "Risk" in line and line.startswith("-"):
                risk = line.replace("-", "").strip()
                if risk and ":" in risk:
                    result["risks"].append(risk)
        
        # Estimate duration based on phases
        phase_count = len(result["phases"])
        if phase_count > 0:
            result["estimated_duration"] = f"{phase_count * 2}-{phase_count * 4} weeks"
        
        return result
    
    def _format_plan(self, response: str) -> str:
        """Format the complete project plan document."""
        
        header = f"""---
title: Research Project Plan
generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
type: project_plan
author: Gia Tenica*
---

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher, for more information see: https://giatenica.com

---

"""
        return header + response
