"""
Gap Resolver Agent
==================
Analyzes research overview to identify gaps, generates and executes
Python code to resolve data-related issues, and produces an updated
research overview ready for hypothesis development.

Uses Sonnet 4.5 for code generation and Opus 4.5 for synthesis.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import sys
import time
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

from .base import BaseAgent, AgentResult
from src.llm.claude_client import ClaudeClient, TaskType
from loguru import logger


@dataclass
class CodeExecutionResult:
    """Result from executing generated Python code."""
    success: bool
    code: str
    stdout: str
    stderr: str
    execution_time: float
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "error": self.error,
        }


@dataclass
class GapResolution:
    """Resolution attempt for a specific gap."""
    gap_id: str
    gap_type: str  # critical, important, enhancement
    gap_description: str
    resolution_approach: str
    execution_attempts: List[dict] = field(default_factory=list)
    code_generated: Optional[str] = None
    execution_result: Optional[CodeExecutionResult] = None
    findings: str = ""
    resolved: bool = False
    
    def to_dict(self) -> dict:
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type,
            "gap_description": self.gap_description,
            "resolution_approach": self.resolution_approach,
            "execution_attempts": self.execution_attempts,
            "code_generated": self.code_generated,
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "findings": self.findings,
            "resolved": self.resolved,
        }


# Defensive coding rules for LLM-generated pandas code
DEFENSIVE_PANDAS_RULES = """
DEFENSIVE PANDAS CODING REQUIREMENTS (MANDATORY):
================================================
All generated code MUST follow these defensive patterns to handle edge cases:

1. EMPTY DATAFRAME CHECK (required before any operation):
   if df.empty:
       print("WARNING: DataFrame is empty, skipping analysis")
       # Return early or handle gracefully

2. COLUMN EXISTENCE CHECK (required before column access):
   required_cols = ['col1', 'col2', 'col3']
   missing = [c for c in required_cols if c not in df.columns]
   if missing:
       print(f"ERROR: Missing columns: {missing}")
       print(f"Available columns: {list(df.columns)}")
       # Return early or handle gracefully

3. NULL-AWARE OPERATIONS:
   # Always use dropna() before operations that fail on NaN
   valid_data = df[column].dropna()
   if len(valid_data) == 0:
       print(f"WARNING: No valid (non-null) data in {column}")
   else:
       result = valid_data.describe()

4. SAFE QUANTILE CUTS (qcut can fail with duplicate edges):
   # Prefer the shared helper SmartDataLoader.qcut_safe, which returns
   # (bins, warning_message) instead of printing inside the helper.
   bins, warning = SmartDataLoader.qcut_safe(
       series.dropna(),
       q=4,
       labels=['Q1','Q2','Q3','Q4'],
   )
   if warning:
       print(warning)

   # If you cannot use SmartDataLoader.qcut_safe, fall back to this pattern:
   try:
       bins = pd.qcut(series.dropna(), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
   except ValueError as e:
       error_msg = f"WARNING: qcut failed ({e}), using cut() with fixed bins"
       print(error_msg)
       bins = pd.cut(series.dropna(), bins=4, labels=['Q1','Q2','Q3','Q4'])

5. LARGE DATASET SAMPLING (for datasets > 1M rows):
   MAX_ROWS = 500_000
   if len(df) > MAX_ROWS:
       print(f"Sampling {MAX_ROWS:,} from {len(df):,} rows")
       df = df.sample(n=MAX_ROWS, random_state=42)

6. DESCRIBE SAFEGUARD:
   numeric_df = df.select_dtypes(include=['number'])
   if numeric_df.empty:
       print("WARNING: No numeric columns to describe")
   else:
       print(numeric_df.describe())

7. ERROR CONTEXT IN EXCEPTIONS:
   try:
       # operation
   except Exception as e:
       print(f"ERROR in [operation_name]: {type(e).__name__}: {e}")
       print(f"DataFrame shape: {df.shape if 'df' in dir() else 'N/A'}")
       print(f"Columns: {list(df.columns) if 'df' in dir() and hasattr(df, 'columns') else 'N/A'}")
"""

# System prompt for gap parsing and code generation
GAP_RESOLVER_SYSTEM_PROMPT = f"""You are a research gap resolver agent that analyzes research overviews and generates Python code to address data-related gaps.

Your role is to:
1. Parse the RESEARCH_OVERVIEW.md to identify actionable gaps
2. Generate Python code to address data verification and analysis gaps
3. Interpret code execution results
4. Update the research status based on findings

CAPABILITIES:
- You can generate Python code that will be executed in a secure environment
- You have access to pandas, numpy, pyarrow for data analysis
- The code should be self-contained and output results via print()
- Focus on gaps that can be resolved through data analysis

WORKFLOW:
1. First, analyze the overview to extract gaps that require code execution
2. For each data-related gap, generate diagnostic Python code
3. Review execution results and summarize findings
4. Determine which gaps were resolved

CODE GENERATION RULES:
- Always use print() to output results - this is how you receive information
- Handle exceptions gracefully with try/except
- Include summary statistics and key findings in output
- Use absolute file paths provided in context
- Import all required libraries at the top
- Limit output to essential information (avoid printing entire dataframes)
{DEFENSIVE_PANDAS_RULES}

OUTPUT FORMAT:
When generating code, wrap it in triple backticks with python:
```python
# Your code here
```

When interpreting results, structure your response with:
- GAP_ID: [identifier]
- STATUS: [RESOLVED/PARTIALLY_RESOLVED/UNRESOLVED]
- FINDINGS: [What was discovered]
- IMPLICATIONS: [What this means for the research]
- REMAINING_ISSUES: [What still needs attention]"""


# System prompt for overview synthesis
SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis agent that creates updated research overviews based on gap resolution findings.

Given:
1. The original RESEARCH_OVERVIEW.md
2. Gap resolution results with code execution findings
3. Project data and context

Create an UPDATED_RESEARCH_OVERVIEW.md that:
1. Preserves the original structure and completeness score format
2. Updates sections where gaps were resolved with new findings
3. Adjusts completeness scores based on resolved gaps
4. Marks resolved items as complete
5. Adds new data insights discovered through code execution
6. Updates the "Ready for Literature Review" assessment
7. Revises action items to remove completed tasks

CRITICAL RULES:
- Maintain the exact markdown format with YAML frontmatter
- Update completeness_score based on resolutions (each critical gap ~10-15 points)
- Include actual data findings from code execution
- Update "Ready for Literature Review" only if critical data gaps resolved
- Be specific about what was discovered, not generic
- Keep all sections that weren't addressed

OUTPUT: Complete markdown document ready to save as UPDATED_RESEARCH_OVERVIEW.md"""


class CodeExecutor:
    """
    Safely executes Python code in an isolated subprocess.
    """
    
    def __init__(self, timeout: int = 120, max_output_size: int = 100000):
        """
        Initialize code executor.
        
        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum output size in characters
        """
        self.timeout = timeout
        self.max_output_size = max_output_size
        # Prefer the current interpreter to ensure dependencies match the runtime.
        self.python_path = sys.executable

        # By default, do not leak the parent process environment (API keys, tokens)
        # into LLM-generated code execution.
        self.sanitize_env = True

    def _build_subprocess_env(self) -> Dict[str, str]:
        """Build a minimal environment for subprocess execution.

        Goal:
        - Reduce accidental secret leakage (API keys, tokens, passwords)
        - Keep enough environment for Python and common native deps to run
        """
        from src.utils.subprocess_env import build_minimal_subprocess_env

        return build_minimal_subprocess_env(sanitize_env=self.sanitize_env)
    
    def execute(self, code: str, working_dir: Optional[str] = None) -> CodeExecutionResult:
        """
        Execute Python code in a subprocess.
        
        Args:
            code: Python code to execute
            working_dir: Optional working directory for execution
            
        Returns:
            CodeExecutionResult with stdout, stderr, and status
        """
        start_time = time.time()

        from src.utils.subprocess_text import to_text

        # Always write the snippet into an isolated temp folder rather than the project.
        with tempfile.TemporaryDirectory(prefix="gia_code_exec_") as sandbox_dir:
            temp_path = str(Path(sandbox_dir) / "snippet.py")
            Path(temp_path).write_text(code, encoding="utf-8")

            try:
                env = self._build_subprocess_env()

                # Execute in subprocess with timeout.
                # -I reduces PYTHONPATH/user-site influence, while still allowing site-packages.
                result = subprocess.run(
                    [self.python_path, "-I", "-B", temp_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    cwd=working_dir or os.getcwd(),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                )

                stdout = (result.stdout or "")[: self.max_output_size]
                stderr = (result.stderr or "")[: self.max_output_size]

                execution_time = time.time() - start_time

                return CodeExecutionResult(
                    success=result.returncode == 0,
                    code=code,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=execution_time,
                    error=stderr if result.returncode != 0 else None,
                )

            except subprocess.TimeoutExpired as e:
                stdout = to_text(e.stdout)[: self.max_output_size]
                stderr = to_text(e.stderr)[: self.max_output_size]
                return CodeExecutionResult(
                    success=False,
                    code=code,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=time.time() - start_time,
                    error=f"Execution timed out after {self.timeout} seconds",
                )
            except Exception as e:
                return CodeExecutionResult(
                    success=False,
                    code=code,
                    stdout="",
                    stderr=f"Execution failed: {type(e).__name__}",
                    execution_time=time.time() - start_time,
                    error=f"Execution failed: {type(e).__name__}",
                )


class GapResolverAgent(BaseAgent):
    """
    Agent that resolves research gaps by generating and executing Python code.
    
    Uses Sonnet 4.5 for code generation and analysis.
    """
    
    def __init__(
        self,
        client: Optional[ClaudeClient] = None,
        execution_timeout: int = 120,
        max_code_attempts: int = 2,
    ):
        """
        Initialize gap resolver agent.
        
        Args:
            client: Optional shared ClaudeClient
            execution_timeout: Timeout for code execution in seconds
        """
        super().__init__(
            name="GapResolver",
            task_type=TaskType.CODING,  # Uses Sonnet for code generation
            system_prompt=GAP_RESOLVER_SYSTEM_PROMPT,
            client=client,
        )
        self.executor = CodeExecutor(timeout=execution_timeout)
        self.resolutions: List[GapResolution] = []
        self.max_code_attempts = max(1, int(max_code_attempts))
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Resolve gaps identified in the research overview.
        
        Args:
            context: Must contain 'project_folder', 'research_overview' content,
                     and 'project_data' with file paths.
                     
                     Optional:
                     - 'retry_gap_ids': List of gap IDs to retry (skip identification)
                     - 'previous_gaps': List of gap dicts from previous iteration
                     - 'iteration': Current iteration number
            
        Returns:
            AgentResult with gap resolutions and updated findings
        """
        start_time = time.time()
        total_tokens = 0
        
        project_folder = context.get("project_folder", "")
        research_overview = context.get("research_overview", "")
        project_data = context.get("project_data", {})
        retry_gap_ids = context.get("retry_gap_ids", [])
        previous_gaps = context.get("previous_gaps", [])
        iteration = context.get("iteration", 1)
        
        if not research_overview:
            # Try to load from file
            overview_path = Path(project_folder) / "RESEARCH_OVERVIEW.md"
            if overview_path.exists():
                research_overview = overview_path.read_text()
            else:
                return self._build_result(
                    success=False,
                    content="",
                    error="No research overview provided or found",
                    execution_time=time.time() - start_time,
                )
        
        # Get data file paths
        data_paths = self._extract_data_paths(project_folder, project_data)
        
        try:
            # Step 1: Identify actionable gaps (or reuse previous gaps for retry)
            if iteration > 1 and retry_gap_ids and previous_gaps:
                # Retry mode: filter previous gaps to only retry unresolved ones
                logger.info(f"Gap Resolver: Retrying {len(retry_gap_ids)} gaps from iteration {iteration-1}")
                gaps = [g for g in previous_gaps if g.get("id") in retry_gap_ids]
                gap_tokens = 0
            else:
                # Normal mode: identify gaps from overview
                logger.info("Gap Resolver: Identifying actionable gaps...")
                gaps, gap_tokens = await self._identify_gaps(research_overview, data_paths)
            total_tokens += gap_tokens
            
            if not gaps:
                logger.info("No actionable data gaps found")
                return self._build_result(
                    success=True,
                    content="No actionable data gaps found that require code execution.",
                    structured_data={"resolutions": [], "data_paths": data_paths},
                    tokens_used=total_tokens,
                    execution_time=time.time() - start_time,
                )
            
            # Step 2: Generate and execute code for each gap
            logger.info(f"Gap Resolver: Processing {len(gaps)} gaps...")
            for gap in gaps:
                logger.info(f"  Processing gap: {gap['id']} - {gap['description'][:50]}...")
                resolution, res_tokens = await self._resolve_gap(gap, data_paths, project_folder)
                total_tokens += res_tokens
                self.resolutions.append(resolution)
            
            # Step 3: Summarize findings
            logger.info("Gap Resolver: Synthesizing findings...")
            summary, sum_tokens = await self._synthesize_findings(research_overview)
            total_tokens += sum_tokens
            
            # Build structured output
            structured_data = {
                "resolutions": [r.to_dict() for r in self.resolutions],
                "data_paths": data_paths,
                "summary": summary,
                "resolved_count": sum(1 for r in self.resolutions if r.resolved),
                "total_gaps": len(self.resolutions),
            }
            
            return self._build_result(
                success=True,
                content=summary,
                structured_data=structured_data,
                tokens_used=total_tokens,
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            logger.error(f"Gap resolver error: {e}")
            return self._build_result(
                success=False,
                content="",
                error=str(e),
                tokens_used=total_tokens,
                execution_time=time.time() - start_time,
            )
    
    def _extract_data_paths(self, project_folder: str, project_data: dict) -> dict:
        """
        Extract and validate data file paths from project data.
        
        Returns dict with categorized file paths.
        """
        data_paths = {
            "parquet_files": [],
            "csv_files": [],
            "all_files": [],
            "project_folder": project_folder,
        }
        
        # Search for data files in project folder
        project_path = Path(project_folder)
        data_dir = project_path / "data"
        
        if data_dir.exists():
            # Find all parquet files
            for pq_file in data_dir.rglob("*.parquet"):
                # Skip macOS metadata files
                if "._" not in str(pq_file) and "__MACOSX" not in str(pq_file):
                    data_paths["parquet_files"].append(str(pq_file))
                    data_paths["all_files"].append(str(pq_file))
            
            # Find all CSV files
            for csv_file in data_dir.rglob("*.csv"):
                if "._" not in str(csv_file):
                    data_paths["csv_files"].append(str(csv_file))
                    data_paths["all_files"].append(str(csv_file))
        
        return data_paths
    
    def _get_schema_info(self, data_paths: dict) -> str:
        """
        Extract schema information from data files for prompt injection.
        
        Returns formatted string with column names, types, and sample values.
        """
        try:
            from src.utils.smart_data_loader import SmartDataLoader, schemas_to_prompt
            
            loader = SmartDataLoader()
            schemas = {}
            
            # Extract schemas from parquet files
            for path in data_paths.get("parquet_files", []):
                try:
                    name = Path(path).stem
                    schemas[name] = loader.extract_schema(path)
                except Exception as e:
                    logger.debug(f"Failed to extract schema from {path}: {e}")
            
            # Extract schemas from CSV files (limit to first few)
            for path in data_paths.get("csv_files", [])[:3]:
                try:
                    name = Path(path).stem
                    schemas[name] = loader.extract_schema(path)
                except Exception as e:
                    logger.debug(f"Failed to extract schema from {path}: {e}")
            
            if schemas:
                return schemas_to_prompt(schemas)
            else:
                return "DATA SCHEMAS: Unable to extract schema information."
                
        except ImportError:
            logger.debug("SmartDataLoader not available, skipping schema extraction")
            return "DATA SCHEMAS: Schema extraction not available."
        except Exception as e:
            logger.debug(f"Schema extraction failed: {e}")
            return f"DATA SCHEMAS: Extraction failed ({type(e).__name__})"
    
    async def _identify_gaps(self, overview: str, data_paths: dict) -> Tuple[List[dict], int]:
        """
        Parse overview and identify gaps that require code execution.
        
        Returns list of gap dictionaries and tokens used.
        """
        prompt = f"""Analyze this research overview and identify gaps that can be resolved through Python code execution.

RESEARCH OVERVIEW:
{overview}

AVAILABLE DATA FILES:
{json.dumps(data_paths, indent=2)}

Focus on gaps related to:
1. Data verification (symbol checks, dataset reconciliation)
2. Data quality assessment (missing values, outliers)
3. Sample construction (filtering, matching)
4. Descriptive statistics
5. Variable calculation verification

IMPORTANT: Return ONLY a valid JSON array. No markdown, no explanation, just the JSON.

Example response format:
[
  {{
    "id": "C1",
    "type": "critical",
    "description": "Verify data completeness for key variables",
    "code_approach": "Load data and check for missing values in required columns",
    "data_files_needed": ["{data_paths.get('parquet_files', ['data/sample.parquet'])[0] if data_paths.get('parquet_files') else 'data/sample.parquet'}"]
  }}
]

Rules:
- Use gap IDs from the overview (C1, I1, etc.) when available
- type must be one of: "critical", "important", "enhancement"
- Only include gaps where Python code can provide concrete resolution
- If no actionable gaps exist, return an empty array: []

Respond with ONLY the JSON array:"""
        
        response, tokens = await self._call_claude(prompt)
        
        # Parse JSON from response with multiple fallback strategies
        gaps = []
        parse_method = "none"
        
        # Strategy 1: Direct JSON parse (response is pure JSON)
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, list):
                gaps = parsed
                parse_method = "direct"
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON array using greedy regex
        if not gaps:
            try:
                # Look for array that spans multiple lines
                json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response)
                if json_match:
                    gaps = json.loads(json_match.group())
                    parse_method = "regex_greedy"
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Extract from code block
        if not gaps:
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = None
                if json_str:
                    gaps = json.loads(json_str)
                    parse_method = "code_block"
            except (json.JSONDecodeError, IndexError):
                pass
        
        # Strategy 4: Line-by-line object extraction (fallback)
        if not gaps:
            logger.warning("Could not parse gap list as JSON array, attempting object extraction")
            for obj_match in re.finditer(r'\{[^{}]+\}', response):
                try:
                    gap = json.loads(obj_match.group())
                    if "id" in gap and "description" in gap:
                        gaps.append(gap)
                        parse_method = "object_extraction"
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if gaps:
            logger.debug(f"Parsed {len(gaps)} gaps using method: {parse_method}")
        else:
            logger.warning(f"No gaps parsed from response (length={len(response)})")
        
        return gaps, tokens
    
    async def _resolve_gap(
        self,
        gap: dict,
        data_paths: dict,
        project_folder: str
    ) -> Tuple[GapResolution, int]:
        """
        Generate and execute code to resolve a specific gap.
        
        Returns GapResolution and tokens used.
        """
        total_tokens = 0
        
        resolution = GapResolution(
            gap_id=gap.get("id", "unknown"),
            gap_type=gap.get("type", "unknown"),
            gap_description=gap.get("description", ""),
            resolution_approach=gap.get("code_approach", ""),
        )
        
        # Extract schema information for data files
        schema_info = self._get_schema_info(data_paths)
        
        base_code_prompt = f"""Generate Python code to address this research gap:

GAP ID: {gap.get('id')}
DESCRIPTION: {gap.get('description')}
APPROACH: {gap.get('code_approach')}

AVAILABLE DATA FILES:
{json.dumps(data_paths, indent=2)}

{schema_info}

REQUIREMENTS:
1. Use pandas for data manipulation
2. Use pyarrow for parquet files: pd.read_parquet(path)
3. Print all findings using print() - be comprehensive
4. Handle errors with try/except
5. Use the exact file paths provided
6. Output summary statistics and key findings
7. If checking for symbol/ticker distinction, print unique values
8. If validating data, print sample sizes, date ranges, missing value counts
9. Keep output focused and informative

MANDATORY DEFENSIVE PATTERNS:
- Check if DataFrame is empty before any operation: if df.empty: print("WARNING: DataFrame empty")
- Verify columns exist before access: if 'col' not in df.columns: print(f"Missing column")
- Use dropna() before operations that fail on NaN
- For large datasets (>1M rows), sample first: df.sample(n=500000, random_state=42)
- Wrap pd.qcut() in try/except with pd.cut() fallback
- Use select_dtypes(include=['number']) before describe()

Return ONLY the complete Python code wrapped in a single ```python fenced block. Do not include any prose outside the code block."""

        last_code: Optional[str] = None
        last_exec: Optional[CodeExecutionResult] = None

        for attempt in range(1, self.max_code_attempts + 1):
            if attempt == 1:
                prompt = base_code_prompt
            else:
                stderr_snippet = (last_exec.stderr if last_exec else "")[:4000]
                error_message = (
                    last_exec.error
                    if last_exec
                    else "Previous attempt failed to generate valid code; execution did not run."
                )
                prompt = f"""The previously generated code failed to execute.

GAP ID: {gap.get('id')}
DESCRIPTION: {gap.get('description')}
APPROACH: {gap.get('code_approach')}

AVAILABLE DATA FILES:
{json.dumps(data_paths, indent=2)}

FAILED CODE:
```python
{(last_code or '')[:8000]}
```

EXECUTION ERROR:
{error_message}

STDERR (truncated):
{stderr_snippet}

Fix the code so it runs successfully and prints the requested findings.

Return ONLY the complete corrected Python code wrapped in a single ```python fenced block. Do not include any prose outside the code block."""

            code_response, code_tokens = await self._call_claude(prompt)
            total_tokens += code_tokens

            code = self._extract_code(code_response)
            if not code:
                resolution.execution_attempts.append(
                    {
                        "attempt": attempt,
                        "generated": False,
                        "execution_success": False,
                        "error": "Failed to extract code block from model response",
                    }
                )
                last_code = None
                last_exec = None
                continue

            resolution.code_generated = code
            last_code = code

            logger.debug(f"Executing code for gap {resolution.gap_id} (attempt {attempt}/{self.max_code_attempts})...")
            exec_result = self.executor.execute(code, working_dir=None)  # Use workspace root
            resolution.execution_result = exec_result
            last_exec = exec_result

            resolution.execution_attempts.append(
                {
                    "attempt": attempt,
                    "generated": True,
                    "execution_success": exec_result.success,
                    "execution_time": exec_result.execution_time,
                    "error": (exec_result.error or "")[:2000] if not exec_result.success else None,
                }
            )

            if exec_result.success:
                break

            logger.warning(f"Code execution failed for {resolution.gap_id} (attempt {attempt}): {exec_result.error}")

        if not resolution.execution_result or not resolution.execution_result.success:
            last_error = (resolution.execution_result.error if resolution.execution_result else "Unknown")
            resolution.findings = f"Code execution failed after {len(resolution.execution_attempts)} attempt(s): {last_error}"
            return resolution, total_tokens
        
        # Interpret results
        interpret_prompt = f"""Interpret these code execution results for research gap resolution:

GAP ID: {resolution.gap_id}
GAP DESCRIPTION: {resolution.gap_description}

CODE OUTPUT:
{exec_result.stdout[:8000]}

Based on this output:
1. What was discovered about the data?
2. Is the gap resolved, partially resolved, or unresolved?
3. What are the key findings?
4. What implications does this have for the research?
5. Are there any remaining issues?

Provide a structured interpretation."""
        
        interpretation, interp_tokens = await self._call_claude(interpret_prompt)
        total_tokens += interp_tokens
        
        resolution.findings = interpretation

        status = self._extract_status_from_interpretation(interpretation)
        if status == "RESOLVED":
            resolution.resolved = True
        elif status in {"PARTIALLY_RESOLVED", "UNRESOLVED"}:
            resolution.resolved = False
        else:
            resolution.resolved = "resolved" in interpretation.lower() and "unresolved" not in interpretation.lower()
        
        return resolution, total_tokens

    def _extract_status_from_interpretation(self, interpretation: str) -> Optional[str]:
        """Extract STATUS from the interpretation block.

        Expected format includes a line like:
        - STATUS: RESOLVED
        - STATUS: PARTIALLY_RESOLVED
        - STATUS: UNRESOLVED

        Returns:
            Optional[str]: One of "RESOLVED", "PARTIALLY_RESOLVED", "UNRESOLVED", or None.
        """
        for line in (interpretation or "").splitlines():
            if line.strip().upper().startswith("STATUS:"):
                value = line.split(":", 1)[1].strip().upper()
                value = value.replace(" ", "_")
                if value in {"RESOLVED", "PARTIALLY_RESOLVED", "UNRESOLVED"}:
                    return value
                return None
        return None
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from Claude's response."""
        # Look for code blocks
        code_pattern = r'```python\n([\s\S]*?)```'
        matches = re.findall(code_pattern, response)
        
        if matches:
            return matches[0].strip()
        
        # Try without language specifier
        code_pattern = r'```\n([\s\S]*?)```'
        matches = re.findall(code_pattern, response)
        
        if matches:
            return matches[0].strip()
        
        return None
    
    async def _synthesize_findings(self, original_overview: str) -> Tuple[str, int]:
        """
        Synthesize all gap resolution findings into a summary.
        
        Returns summary text and tokens used.
        """
        resolutions_text = "\n\n".join([
            f"### Gap {r.gap_id}: {r.gap_description[:100]}\n"
            f"**Status:** {'RESOLVED' if r.resolved else 'PARTIALLY RESOLVED' if r.execution_result and r.execution_result.success else 'UNRESOLVED'}\n"
            f"**Findings:**\n{r.findings[:2000]}"
            for r in self.resolutions
        ])
        
        prompt = f"""Synthesize the gap resolution findings into a comprehensive summary.

RESOLUTION RESULTS:
{resolutions_text}

Create a summary that:
1. Lists what was discovered for each gap
2. Identifies key data findings
3. Notes which gaps are now resolved
4. Highlights any new issues discovered
5. Assesses overall progress toward research readiness

Be specific about data characteristics discovered (date ranges, sample sizes, variable availability, etc.)."""
        
        summary, tokens = await self._call_claude(prompt)
        return summary, tokens


class OverviewUpdaterAgent(BaseAgent):
    """
    Agent that creates an updated research overview incorporating gap resolution findings.
    
    Uses Opus 4.5 for comprehensive synthesis.
    """
    
    def __init__(self, client: Optional[ClaudeClient] = None):
        """Initialize overview updater agent."""
        super().__init__(
            name="OverviewUpdater",
            task_type=TaskType.COMPLEX_REASONING,  # Uses Opus for synthesis
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            client=client,
        )
    
    async def execute(self, context: dict, max_retries: int = 5) -> AgentResult:
        """
        Create updated research overview with gap resolution findings.
        
        Args:
            context: Must contain 'research_overview', 'gap_resolutions',
                     and 'project_data'
            max_retries: Maximum retries on API overload errors
            
        Returns:
            AgentResult with updated overview content
        """
        import asyncio
        
        start_time = time.time()
        
        research_overview = context.get("research_overview", "")
        gap_resolutions = context.get("gap_resolutions", {})
        project_data = context.get("project_data", {})
        
        if not research_overview:
            return self._build_result(
                success=False,
                content="",
                error="No research overview provided",
                execution_time=time.time() - start_time,
            )
        
        # Build resolution summary for the prompt
        resolutions_data = gap_resolutions.get("resolutions", [])
        resolution_summary = self._build_resolution_summary(resolutions_data)
        
        prompt = f"""Create an updated RESEARCH_OVERVIEW.md incorporating the gap resolution findings.

ORIGINAL RESEARCH OVERVIEW (802 lines, you MUST preserve its full structure):
{research_overview}

GAP RESOLUTION FINDINGS:
{resolution_summary}

OVERALL SUMMARY:
{gap_resolutions.get('summary', 'No summary available')}

RESOLVED GAPS: {gap_resolutions.get('resolved_count', 0)} of {gap_resolutions.get('total_gaps', 0)}

PROJECT DATA:
{json.dumps(project_data, indent=2)[:2000]}

CRITICAL INSTRUCTIONS:
You MUST output the COMPLETE updated document - all ~800 lines with updates incorporated.
The original document has these sections that MUST ALL appear in your output:
- YAML frontmatter
- Executive Summary  
- Research Design (Research Question, Hypothesis, Scope)
- Data Status (with updated findings)
- Gap Analysis (with resolved status updates)
- Actionable Recommendations
- Success Metrics
- Appendices (Data Profile Summary, Key File Locations, Technical Notes)

DO NOT truncate or summarize. DO NOT skip sections. Output the ENTIRE document.

Updates to make:
1. Update YAML frontmatter with new completeness_score and generated_at timestamp
2. Update "Ready for Literature Review" status if appropriate
3. In Data Status section, incorporate actual findings from code execution (specific numbers)
4. Update Gap Analysis section to mark resolved gaps with ✓
5. Update Action Items to reflect completed tasks
6. Add a "Gap Resolution Findings" subsection after Executive Summary
7. Preserve ALL other content with minimal changes
8. Be specific about data findings (actual numbers, date ranges, etc.)

Begin output with the YAML frontmatter (---) and continue until the very end of the document."""
        
        # Retry loop for API overload errors
        last_error = None
        for attempt in range(max_retries):
            try:
                # Use higher max_tokens for long document synthesis
                # The original overview is ~10k tokens, we need room for additions
                response, tokens = await self._call_claude(
                    prompt, 
                    use_thinking=True,
                    max_tokens=48000,  # Increased for full document synthesis
                    budget_tokens=16000,
                )
                
                # Extract the markdown content (remove any preamble)
                updated_overview = self._extract_markdown(response)
                
                return self._build_result(
                    success=True,
                    content=updated_overview,
                    structured_data={
                        "resolved_count": gap_resolutions.get("resolved_count", 0),
                        "total_gaps": gap_resolutions.get("total_gaps", 0),
                    },
                    tokens_used=tokens,
                    execution_time=time.time() - start_time,
                )
            
            except Exception as e:
                error_str = str(e)
                last_error = e
                
                # Check for overload error
                if "overloaded" in error_str.lower() or "overloaded_error" in error_str.lower():
                    wait_time = 2 ** attempt * 10  # 10, 20, 40, 80, 160 seconds
                    logger.warning(f"API overloaded (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error
                    break
        
        # All retries exhausted or non-retryable error
        logger.error(f"Overview update error after {max_retries} attempts: {last_error}")
        return self._build_result(
            success=False,
            content="",
            error=str(last_error),
            execution_time=time.time() - start_time,
        )
    
    def _build_resolution_summary(self, resolutions: List[dict]) -> str:
        """Build a formatted summary of resolutions for the prompt."""
        if not resolutions:
            return "No gap resolutions available."
        
        summary_parts = []
        for res in resolutions:
            status = "RESOLVED" if res.get("resolved") else "UNRESOLVED"
            exec_result = res.get("execution_result", {})
            
            part = f"""
## Gap {res.get('gap_id', 'Unknown')}: {res.get('gap_description', '')[:200]}
**Type:** {res.get('gap_type', 'Unknown')}
**Status:** {status}
**Approach:** {res.get('resolution_approach', 'N/A')[:200]}

**Code Execution:**
- Success: {exec_result.get('success', False)}
- Execution Time: {exec_result.get('execution_time', 0):.2f}s

**Output/Findings:**
{res.get('findings', 'No findings')[:3000]}
"""
            summary_parts.append(part)
        
        return "\n---\n".join(summary_parts)
    
    def _extract_markdown(self, response: str) -> str:
        """Extract markdown content from response."""
        # If response starts with YAML frontmatter, return as-is
        if response.strip().startswith("---"):
            return response.strip()
        
        # Look for markdown block
        md_match = re.search(r'```markdown\n([\s\S]*?)```', response)
        if md_match:
            return md_match.group(1).strip()
        
        # Look for content starting with ---
        frontmatter_match = re.search(r'(---[\s\S]+)', response)
        if frontmatter_match:
            return frontmatter_match.group(1).strip()
        
        # Return the whole response if no pattern matched
        return response.strip()
