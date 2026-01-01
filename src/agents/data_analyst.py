"""
Data Analyst Agent
==================
Examines uploaded data files using Python to generate statistics,
schema information, and data quality assessments.

Uses Haiku 4.5 for fast data classification and extraction.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Any

from .base import BaseAgent, AgentResult
from src.llm.claude_client import TaskType
from loguru import logger

from src.config import INTAKE_SERVER


# System prompt for data analysis
DATA_ANALYST_PROMPT = """You are a data analyst agent for academic finance research.

Your role is to analyze data files and provide structured insights for research papers.

When given information about data files (structure, sample rows, statistics), you should:

1. DESCRIBE the data structure clearly
2. IDENTIFY key variables and their types
3. ASSESS data quality (missing values, outliers, coverage)
4. SUGGEST potential uses for the research question
5. FLAG any data issues that need attention

Output your analysis in a structured format with clear sections.

Be concise but thorough. Focus on what matters for empirical finance research.

IMPORTANT: 
- Do not make up data or statistics
- Only report what you can observe from the provided information
- Note any limitations in your analysis"""


class DataAnalystAgent(BaseAgent):
    """
    Agent that examines uploaded data files locally with Python.
    
    Uses Haiku 4.5 for fast processing of data classification tasks.
    """
    
    def __init__(self, client: Optional[Any] = None):
        super().__init__(
            name="DataAnalyst",
            task_type=TaskType.DATA_EXTRACTION,  # Uses Haiku
            system_prompt=DATA_ANALYST_PROMPT,
            client=client,
        )
    
    async def execute(self, context: dict) -> AgentResult:
        """
        Analyze data files in the project folder.
        
        Args:
            context: Must contain 'project_folder' path
            
        Returns:
            AgentResult with data analysis findings
        """
        start_time = time.time()
        project_folder = context.get("project_folder")
        
        if not project_folder:
            return self._build_result(
                success=False,
                content="",
                error="No project_folder provided in context",
            )
        
        data_folder = Path(project_folder) / "data"
        
        # Check if data folder exists and has files
        if not data_folder.exists():
            return self._build_result(
                success=True,
                content="No data folder found. Data will need to be acquired.",
                structured_data={"has_data": False, "files": []},
            )
        
        # Analyze all data files
        analysis_results = []
        file_summaries = []

        max_files = int(INTAKE_SERVER.MAX_ZIP_FILES)
        exclude_dirs = {"__pycache__", ".venv", ".git", "node_modules", "temp", "tmp"}
        visited_files = 0
        limit_reached = False

        # Use os.walk for efficient directory traversal, as it allows pruning.
        for root, dirs, files in os.walk(data_folder):
            # Prune directories to avoid traversing into them. This is a performance
            # optimization over Path.rglob(), which cannot be pruned.
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

            for filename in files:
                if filename.startswith('.'):
                    continue

                visited_files += 1
                if visited_files > max_files:
                    limit_reached = True
                    break

                file_path = Path(root) / filename
                file_analysis = await self._analyze_file(file_path)

                if file_analysis:
                    analysis_results.append(file_analysis)
                    file_summaries.append({
                        "file": str(file_path.relative_to(data_folder)),
                        "type": file_analysis.get("type"),
                        "rows": file_analysis.get("rows"),
                        "columns": file_analysis.get("columns"),
                    })

            if limit_reached:
                break
        
        if not analysis_results:
            return self._build_result(
                success=True,
                content="Data folder exists but contains no analyzable files.",
                structured_data={"has_data": False, "files": []},
            )
        
        # Use Claude to synthesize the analysis
        analysis_text = self._format_analysis_for_claude(analysis_results)
        project_data = context.get("project_data", {})
        research_question = project_data.get("research_question", "Not specified")
        
        user_message = f"""Analyze the following data files for a research project:

RESEARCH QUESTION:
{research_question}

DATA FILES ANALYZED:
{analysis_text}

Provide a comprehensive data assessment for this research project."""

        try:
            content, tokens = await self._call_claude(user_message)
            
            return self._build_result(
                success=True,
                content=content,
                structured_data={
                    "has_data": True,
                    "file_count": len(analysis_results),
                    "files": file_summaries,
                    "raw_analysis": analysis_results,
                },
                tokens_used=tokens,
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            return self._build_result(
                success=False,
                content="",
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    async def _analyze_file(self, file_path: Path) -> Optional[dict]:
        """
        Analyze a single data file using Python.
        
        Supports: CSV, Excel, Stata, JSON, Parquet
        """
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".csv":
                return self._analyze_csv(file_path)
            elif suffix in [".xlsx", ".xls"]:
                return self._analyze_excel(file_path)
            elif suffix == ".dta":
                return self._analyze_stata(file_path)
            elif suffix == ".json":
                return self._analyze_json(file_path)
            elif suffix == ".parquet":
                return self._analyze_parquet(file_path)
            else:
                return {
                    "file": file_path.name,
                    "type": "unknown",
                    "note": f"Unsupported file type: {suffix}",
                }
        except Exception as e:
            logger.warning(f"Could not analyze {file_path.name}: {e}")
            return {
                "file": file_path.name,
                "type": suffix,
                "error": str(e),
            }
    
    def _analyze_csv(self, file_path: Path) -> dict:
        """Analyze a CSV file."""
        import pandas as pd
        
        # Read with low memory to handle large files
        df = pd.read_csv(file_path, nrows=10000)  # Sample first 10k rows
        
        return self._analyze_dataframe(df, file_path.name, "csv")
    
    def _analyze_excel(self, file_path: Path) -> dict:
        """Analyze an Excel file."""
        import pandas as pd
        
        # Get sheet names
        xl = pd.ExcelFile(file_path)
        sheets = xl.sheet_names
        
        # Analyze first sheet
        df = pd.read_excel(file_path, sheet_name=0, nrows=10000)
        
        result = self._analyze_dataframe(df, file_path.name, "excel")
        result["sheets"] = sheets
        return result
    
    def _analyze_stata(self, file_path: Path) -> dict:
        """Analyze a Stata .dta file."""
        import pandas as pd
        
        df = pd.read_stata(file_path)
        return self._analyze_dataframe(df, file_path.name, "stata")
    
    def _analyze_json(self, file_path: Path) -> dict:
        """Analyze a JSON file."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            # Array of objects - treat as tabular
            import pandas as pd
            df = pd.DataFrame(data[:10000])
            return self._analyze_dataframe(df, file_path.name, "json")
        else:
            # Nested structure
            return {
                "file": file_path.name,
                "type": "json",
                "structure": "nested",
                "keys": list(data.keys()) if isinstance(data, dict) else None,
                "size_bytes": file_path.stat().st_size,
            }
    
    def _analyze_parquet(self, file_path: Path) -> dict:
        """Analyze a Parquet file."""
        import pandas as pd
        
        df = pd.read_parquet(file_path)
        return self._analyze_dataframe(df, file_path.name, "parquet")
    
    def _analyze_dataframe(self, df, filename: str, filetype: str) -> dict:
        """
        Analyze a pandas DataFrame and return structured statistics.
        """
        import pandas as pd
        
        # Basic info
        result = {
            "file": filename,
            "type": filetype,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
        }
        
        # Missing values
        missing = df.isnull().sum()
        result["missing_values"] = {
            col: int(missing[col]) 
            for col in df.columns 
            if missing[col] > 0
        }
        result["missing_pct"] = round(df.isnull().sum().sum() / df.size * 100, 2)
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().to_dict()
            result["numeric_stats"] = {
                col: {
                    "min": round(stats[col].get("min", 0), 4),
                    "max": round(stats[col].get("max", 0), 4),
                    "mean": round(stats[col].get("mean", 0), 4),
                    "std": round(stats[col].get("std", 0), 4),
                }
                for col in numeric_cols
            }
        
        # Date columns detection
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Suppress format inference warning by using infer_datetime_format
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        pd.to_datetime(df[col].head(100), format='mixed')
                    date_cols.append(col)
                except (ValueError, TypeError, pd.errors.ParserError):
                    pass
        result["potential_date_columns"] = date_cols
        
        # Sample rows
        result["sample_rows"] = df.head(3).to_dict(orient='records')
        
        return result
    
    def _format_analysis_for_claude(self, analyses: list) -> str:
        """Format file analyses for Claude consumption."""
        parts = []
        
        for analysis in analyses:
            if analysis.get("error"):
                parts.append(f"File: {analysis['file']} - Error: {analysis['error']}")
                continue
            
            part = f"""
FILE: {analysis['file']}
Type: {analysis['type']}
Rows: {analysis.get('rows', 'N/A')} | Columns: {analysis.get('columns', 'N/A')}
Missing Data: {analysis.get('missing_pct', 0)}%

Columns: {', '.join(analysis.get('column_names', [])[:20])}
{f"... and {len(analysis.get('column_names', [])) - 20} more" if len(analysis.get('column_names', [])) > 20 else ""}

Data Types:
{json.dumps(analysis.get('dtypes', {}), indent=2)}
"""
            if analysis.get('numeric_stats'):
                part += f"\nNumeric Statistics:\n{json.dumps(analysis.get('numeric_stats', {}), indent=2)}"
            
            if analysis.get('potential_date_columns'):
                part += f"\nPotential Date Columns: {', '.join(analysis['potential_date_columns'])}"
            
            parts.append(part)
        
        return "\n---\n".join(parts)
