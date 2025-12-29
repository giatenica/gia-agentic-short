"""
Smart Data Loader
=================
Defensive data loading utility with schema validation, sampling strategies,
and safe access patterns for large datasets.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Optional pandas import; the loader can be imported without pandas installed
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False

# Optional pyarrow import for efficient parquet operations
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    pq = None  # type: ignore
    HAS_PYARROW = False


@dataclass
class ColumnSchema:
    """Schema for a single column."""
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    sample_values: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "non_null_count": self.non_null_count,
            "null_count": self.null_count,
            "sample_values": self.sample_values,
        }


@dataclass
class DataFrameSchema:
    """Schema for a DataFrame."""
    path: str
    rows: int
    columns: int
    column_schemas: List[ColumnSchema] = field(default_factory=list)
    memory_mb: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "columns": self.columns,
            "column_schemas": [c.to_dict() for c in self.column_schemas],
            "memory_mb": round(self.memory_mb, 2),
            "error": self.error,
        }
    
    def to_prompt_string(self) -> str:
        """Generate a string suitable for injection into LLM prompts."""
        if self.error:
            return f"FILE: {self.path}\nERROR: {self.error}\n"
        
        lines = [
            f"FILE: {self.path}",
            f"ROWS: {self.rows:,}",
            f"COLUMNS: {self.columns}",
            f"MEMORY: {self.memory_mb:.1f} MB",
            "COLUMN DETAILS:",
        ]
        for col in self.column_schemas:
            sample_str = str(col.sample_values[:3]) if col.sample_values else "[]"
            lines.append(
                f"  - {col.name}: {col.dtype} "
                f"(non-null: {col.non_null_count:,}, null: {col.null_count:,}, "
                f"samples: {sample_str})"
            )
        return "\n".join(lines)


class SmartDataLoader:
    """
    Defensive data loader with schema extraction and safe sampling.
    
    Features:
    - Pre-load schema extraction without loading full data
    - Safe column access with existence checks
    - Automatic sampling for large datasets
    - Null-aware operations
    """
    
    DEFAULT_SAMPLE_THRESHOLD = 1_000_000  # Sample if rows > 1M
    DEFAULT_SAMPLE_SIZE = 100_000
    MAX_SAMPLE_VALUES = 5
    SAMPLE_BUFFER_MULTIPLIER = 1.5  # Buffer for row group sampling
    
    def __init__(
        self,
        sample_threshold: int = DEFAULT_SAMPLE_THRESHOLD,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
    ):
        """
        Initialize smart data loader.
        
        Args:
            sample_threshold: Row count above which to auto-sample
            sample_size: Number of rows to sample for large datasets
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for SmartDataLoader")
        
        self.sample_threshold = sample_threshold
        self.sample_size = sample_size
        self._schema_cache: Dict[str, DataFrameSchema] = {}
    
    def extract_schema(self, path: str, force: bool = False) -> DataFrameSchema:
        """
        Extract schema from a parquet or CSV file without loading all data.
        
        Args:
            path: Path to data file
            force: Force re-extraction even if cached
            
        Returns:
            DataFrameSchema with column information
        """
        if not force and path in self._schema_cache:
            return self._schema_cache[path]
        
        p = Path(path)
        if not p.exists():
            schema = DataFrameSchema(
                path=path,
                rows=0,
                columns=0,
                error=f"File not found: {path}",
            )
            self._schema_cache[path] = schema
            return schema
        
        try:
            # For parquet, use pyarrow metadata for fast schema extraction
            if p.suffix.lower() in (".parquet", ".pq"):
                import pyarrow.parquet as pq
                
                pf = pq.ParquetFile(path)
                metadata = pf.metadata
                schema_arrow = pf.schema_arrow
                
                rows = metadata.num_rows
                columns = len(schema_arrow)
                
                # Read a small sample for sample values
                sample_df = pf.read_row_groups([0]).to_pandas().head(self.MAX_SAMPLE_VALUES * 2)
                
                column_schemas = []
                for col_name in schema_arrow.names:
                    dtype = str(schema_arrow.field(col_name).type)
                    if col_name in sample_df.columns:
                        col_data = sample_df[col_name]
                        non_null = int(col_data.notna().sum())
                        null = len(col_data) - non_null
                        samples = col_data.dropna().head(self.MAX_SAMPLE_VALUES).tolist()
                    else:
                        non_null = 0
                        null = 0
                        samples = []
                    
                    column_schemas.append(ColumnSchema(
                        name=col_name,
                        dtype=dtype,
                        non_null_count=non_null,
                        null_count=null,
                        sample_values=samples,
                    ))
                
                memory_mb = p.stat().st_size / (1024 * 1024)
                
            else:
                # For CSV, read head and count lines
                df_head = pd.read_csv(path, nrows=1000)
                
                # Estimate row count
                with open(path, "rb") as f:
                    chunk = f.read(1024 * 1024)  # 1MB
                    newlines_per_mb = chunk.count(b"\n")
                    file_size = p.stat().st_size
                    rows = max(1, int(newlines_per_mb * file_size / (1024 * 1024)))
                
                columns = len(df_head.columns)
                
                column_schemas = []
                for col_name in df_head.columns:
                    col_data = df_head[col_name]
                    column_schemas.append(ColumnSchema(
                        name=col_name,
                        dtype=str(col_data.dtype),
                        non_null_count=int(col_data.notna().sum()),
                        null_count=int(col_data.isna().sum()),
                        sample_values=col_data.dropna().head(self.MAX_SAMPLE_VALUES).tolist(),
                    ))
                
                memory_mb = file_size / (1024 * 1024)
            
            schema = DataFrameSchema(
                path=path,
                rows=rows,
                columns=columns,
                column_schemas=column_schemas,
                memory_mb=memory_mb,
            )
            self._schema_cache[path] = schema
            return schema
            
        except Exception as e:
            logger.warning(f"Failed to extract schema from {path}: {e}")
            schema = DataFrameSchema(
                path=path,
                rows=0,
                columns=0,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
            self._schema_cache[path] = schema
            return schema
    
    def _load_parquet_sampled(
        self,
        path: str,
        sample_size: int,
        columns: Optional[List[str]] = None,
    ) -> "pd.DataFrame":
        """
        Load a sampled subset of a parquet file without loading the full dataset.
        
        Uses pyarrow to read row groups efficiently, avoiding memory spikes
        for large datasets. Falls back to full load + sample if pyarrow unavailable.
        
        Args:
            path: Path to parquet file
            sample_size: Number of rows to sample
            columns: Optional list of columns to load
            
        Returns:
            Sampled DataFrame (never None; raises exception on error)
        """
        if not HAS_PYARROW:
            # Fallback: load full dataset and sample
            if not HAS_PANDAS:
                raise ImportError("pandas is required for parquet loading")
            logger.debug(f"PyArrow unavailable, using full load for {path}")
            df = pd.read_parquet(path, columns=columns)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            return df
        
        try:
            # Use pyarrow for efficient row-group based sampling
            parquet_file = pq.ParquetFile(path)
            total_rows = parquet_file.metadata.num_rows
            
            if total_rows <= sample_size:
                # File is smaller than sample size, read all
                return pd.read_parquet(path, columns=columns)
            
            # Calculate sampling strategy
            # Read evenly spaced row groups to get diverse sample
            num_row_groups = parquet_file.metadata.num_row_groups
            
            # Determine which row groups to read for approximate sample size
            # Strategy: read row groups evenly spaced throughout the file
            # Use buffer multiplier to account for uneven row group sizes and ensure
            # we get enough rows. Reading slightly more row groups is better than
            # reading too few and having to re-read the file.
            sample_ratio = sample_size / total_rows
            groups_to_read = max(1, int(num_row_groups * sample_ratio * self.SAMPLE_BUFFER_MULTIPLIER))
            
            if groups_to_read >= num_row_groups:
                # Would read all groups anyway, just load and sample
                df = pd.read_parquet(path, columns=columns)
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                return df
            
            # Read evenly spaced row groups
            step = num_row_groups / groups_to_read
            group_indices = [int(i * step) for i in range(groups_to_read)]
            
            # Read selected row groups
            table = parquet_file.read_row_groups(
                group_indices,
                columns=columns,
            )
            df = table.to_pandas()
            
            # Final sampling to exact size if we got more rows than needed
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            logger.debug(
                f"Parquet sampling: read {len(group_indices)}/{num_row_groups} "
                f"row groups, got {len(df):,} rows from {total_rows:,} total"
            )
            
            return df
            
        except Exception as e:
            # Fallback to standard method on any error
            if not HAS_PANDAS:
                raise ImportError("pandas is required for parquet loading")
            logger.warning(
                f"PyArrow sampling failed for {path}, falling back to full load: {e}"
            )
            df = pd.read_parquet(path, columns=columns)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            return df
    
    def load_safe(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        sample: Optional[int] = None,
        auto_sample: bool = True,
    ) -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
        """
        Safely load a DataFrame with optional column selection and sampling.
        
        Args:
            path: Path to data file
            columns: List of columns to load (None = all)
            sample: Explicit sample size (overrides auto)
            auto_sample: Auto-sample if above threshold
            
        Returns:
            Tuple of (DataFrame or None, error message or None)
        """
        p = Path(path)
        if not p.exists():
            return None, f"File not found: {path}"
        
        try:
            # Get schema to determine row count
            schema = self.extract_schema(path)
            if schema.error:
                return None, schema.error
            
            # Validate requested columns exist
            available_cols = {c.name for c in schema.column_schemas}
            if columns:
                missing = set(columns) - available_cols
                if missing:
                    return None, f"Columns not found: {missing}. Available: {available_cols}"
            
            # Determine if sampling needed
            use_sample = sample
            if use_sample is None and auto_sample and schema.rows > self.sample_threshold:
                use_sample = self.sample_size
                logger.info(
                    f"Auto-sampling {path}: {schema.rows:,} rows -> {use_sample:,} rows"
                )
            
            # Load data
            if p.suffix.lower() in (".parquet", ".pq"):
                if use_sample:
                    # Use efficient row-group based sampling for parquet
                    df = self._load_parquet_sampled(path, use_sample, columns)
                else:
                    df = pd.read_parquet(path, columns=columns)
            else:
                if use_sample:
                    df = pd.read_csv(path, usecols=columns, nrows=use_sample)
                else:
                    df = pd.read_csv(path, usecols=columns)
            
            if df.empty:
                return df, "DataFrame is empty after loading"
            
            return df, None
            
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)[:500]}"
    
    def get_column_safe(
        self,
        df: "pd.DataFrame",
        column: str,
        default: Any = None,
    ) -> Tuple[Optional["pd.Series"], Optional[str]]:
        """
        Safely get a column from a DataFrame.
        
        Args:
            df: Source DataFrame
            column: Column name
            default: Default value if column missing
            
        Returns:
            Tuple of (Series or default, error message or None)
        """
        if df is None or df.empty:
            return default, "DataFrame is None or empty"
        
        if column not in df.columns:
            return default, f"Column '{column}' not found. Available: {list(df.columns)}"
        
        return df[column], None
    
    def describe_safe(
        self,
        df: "pd.DataFrame",
        columns: Optional[List[str]] = None,
    ) -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
        """
        Safely describe a DataFrame, handling empty DataFrames.
        
        Args:
            df: Source DataFrame
            columns: Optional list of columns to describe
            
        Returns:
            Tuple of (description DataFrame or None, error message or None)
        """
        if df is None:
            return None, "DataFrame is None"
        
        if df.empty:
            return None, "DataFrame is empty"
        
        try:
            if columns:
                available = [c for c in columns if c in df.columns]
                if not available:
                    return None, f"None of the requested columns found: {columns}"
                df_subset = df[available]
            else:
                df_subset = df
            
            # Select only numeric columns for describe
            numeric_df = df_subset.select_dtypes(include=["number"])
            if numeric_df.empty:
                return None, "No numeric columns to describe"
            
            return numeric_df.describe(), None
            
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)[:200]}"
    
    def qcut_safe(
        self,
        series: "pd.Series",
        q: int = 4,
        labels: Optional[List[str]] = None,
        duplicates: str = "drop",
    ) -> Tuple[Optional["pd.Series"], Optional[str]]:
        """
        Safely apply quantile cut, handling edge cases.
        
        Args:
            series: Series to bin
            q: Number of quantiles
            labels: Optional labels for bins
            duplicates: How to handle duplicate bin edges
            
        Returns:
            Tuple of (binned Series or None, error message or None)
        """
        if series is None or series.empty:
            return None, "Series is None or empty"
        
        # Filter to non-null values
        valid = series.dropna()
        if len(valid) < q:
            return None, f"Not enough valid values ({len(valid)}) for {q} quantiles"
        
        # Check for zero-heavy distribution
        zero_pct = (valid == 0).mean()
        if zero_pct > 0.5:
            logger.warning(
                f"Series has {zero_pct*100:.1f}% zeros; qcut may produce uneven bins"
            )
        
        try:
            result = pd.qcut(valid, q=q, labels=labels, duplicates=duplicates)
            return result, None
        except ValueError as e:
            if "Bin edges must be unique" in str(e):
                # Fallback: use cut with fixed bins
                try:
                    result = pd.cut(valid, bins=q, labels=labels, duplicates=duplicates)
                    return result, f"Warning: Used cut() instead of qcut() due to duplicate edges"
                except Exception as e2:
                    return None, f"Both qcut and cut failed: {e2}"
            return None, f"qcut failed: {e}"
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)[:200]}"


def extract_all_schemas(data_paths: Dict[str, str]) -> Dict[str, DataFrameSchema]:
    """
    Extract schemas for all data files in a paths dictionary.
    
    Args:
        data_paths: Dict mapping names to file paths
        
    Returns:
        Dict mapping names to DataFrameSchema objects
    """
    if not HAS_PANDAS:
        return {}
    
    loader = SmartDataLoader()
    schemas = {}
    
    for name, path in data_paths.items():
        if path and Path(path).suffix.lower() in (".parquet", ".pq", ".csv"):
            schemas[name] = loader.extract_schema(path)
    
    return schemas


def schemas_to_prompt(schemas: Dict[str, DataFrameSchema]) -> str:
    """
    Convert extracted schemas to a prompt-ready string.
    
    Args:
        schemas: Dict of name -> DataFrameSchema
        
    Returns:
        Formatted string for LLM prompt injection
    """
    if not schemas:
        return "No data file schemas available."
    
    lines = ["DATA FILE SCHEMAS:", "=" * 50]
    for name, schema in schemas.items():
        lines.append(f"\n{name}:")
        lines.append(schema.to_prompt_string())
        lines.append("-" * 40)
    
    return "\n".join(lines)


def save_schemas_json(schemas: Dict[str, DataFrameSchema], output_path: str) -> None:
    """Save schemas to a JSON file for debugging/caching."""
    data = {name: schema.to_dict() for name, schema in schemas.items()}
    Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_schemas_json(input_path: str) -> Dict[str, DataFrameSchema]:
    """Load schemas from a JSON file."""
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    schemas = {}
    for name, d in data.items():
        col_schemas = [
            ColumnSchema(**c) for c in d.get("column_schemas", [])
        ]
        schemas[name] = DataFrameSchema(
            path=d["path"],
            rows=d["rows"],
            columns=d["columns"],
            column_schemas=col_schemas,
            memory_mb=d.get("memory_mb", 0.0),
            error=d.get("error"),
        )
    return schemas
