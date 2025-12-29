"""
Tests for SmartDataLoader utility.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import tempfile
from pathlib import Path

import pytest

from src.utils.smart_data_loader import (
    SmartDataLoader,
    DataFrameSchema,
    ColumnSchema,
    extract_all_schemas,
    schemas_to_prompt,
    save_schemas_json,
    load_schemas_json,
)


@pytest.mark.unit
class TestColumnSchema:
    """Tests for ColumnSchema dataclass."""

    def test_column_schema_to_dict(self):
        cs = ColumnSchema(
            name="price",
            dtype="float64",
            non_null_count=100,
            null_count=5,
            sample_values=[1.0, 2.0, 3.0],
        )
        d = cs.to_dict()
        assert d["name"] == "price"
        assert d["dtype"] == "float64"
        assert d["non_null_count"] == 100
        assert d["null_count"] == 5
        assert d["sample_values"] == [1.0, 2.0, 3.0]


@pytest.mark.unit
class TestDataFrameSchema:
    """Tests for DataFrameSchema dataclass."""

    def test_dataframe_schema_to_dict(self):
        col = ColumnSchema(
            name="col1",
            dtype="int64",
            non_null_count=50,
            null_count=0,
            sample_values=[1, 2],
        )
        schema = DataFrameSchema(
            path="/data/test.parquet",
            rows=100,
            columns=1,
            column_schemas=[col],
            memory_mb=1.5,
        )
        d = schema.to_dict()
        assert d["path"] == "/data/test.parquet"
        assert d["rows"] == 100
        assert d["columns"] == 1
        assert d["memory_mb"] == 1.5
        assert len(d["column_schemas"]) == 1

    def test_dataframe_schema_to_prompt_string(self):
        col = ColumnSchema(
            name="volume",
            dtype="float64",
            non_null_count=1000,
            null_count=50,
            sample_values=[100.0, 200.0],
        )
        schema = DataFrameSchema(
            path="/data/test.parquet",
            rows=1050,
            columns=1,
            column_schemas=[col],
            memory_mb=2.5,
        )
        prompt_str = schema.to_prompt_string()
        assert "FILE: /data/test.parquet" in prompt_str
        assert "ROWS: 1,050" in prompt_str
        assert "volume" in prompt_str
        assert "float64" in prompt_str

    def test_dataframe_schema_error_to_prompt_string(self):
        schema = DataFrameSchema(
            path="/data/missing.parquet",
            rows=0,
            columns=0,
            error="File not found",
        )
        prompt_str = schema.to_prompt_string()
        assert "ERROR: File not found" in prompt_str


@pytest.mark.unit
class TestSmartDataLoader:
    """Tests for SmartDataLoader class."""

    def test_extract_schema_file_not_found(self):
        loader = SmartDataLoader()
        schema = loader.extract_schema("/nonexistent/path/file.parquet")
        assert schema.error is not None
        assert "not found" in schema.error.lower()

    def test_load_safe_file_not_found(self):
        loader = SmartDataLoader()
        df, error = loader.load_safe("/nonexistent/path/file.parquet")
        assert df is None
        assert "not found" in error.lower()

    def test_get_column_safe_none_df(self):
        loader = SmartDataLoader()
        result, error = loader.get_column_safe(None, "col1")
        assert result is None
        assert "None or empty" in error

    def test_describe_safe_none_df(self):
        loader = SmartDataLoader()
        result, error = loader.describe_safe(None)
        assert result is None
        assert "None" in error

    def test_qcut_safe_none_series(self):
        loader = SmartDataLoader()
        result, error = loader.qcut_safe(None)
        assert result is None
        assert "None or empty" in error


@pytest.mark.unit
class TestSchemaHelpers:
    """Tests for schema helper functions."""

    def test_schemas_to_prompt_empty(self):
        result = schemas_to_prompt({})
        assert "No data file schemas available" in result

    def test_schemas_to_prompt_with_schemas(self):
        col = ColumnSchema(
            name="price",
            dtype="float64",
            non_null_count=100,
            null_count=0,
            sample_values=[1.0],
        )
        schema = DataFrameSchema(
            path="/data/test.parquet",
            rows=100,
            columns=1,
            column_schemas=[col],
            memory_mb=1.0,
        )
        result = schemas_to_prompt({"test": schema})
        assert "DATA FILE SCHEMAS" in result
        assert "test:" in result
        assert "price" in result

    def test_save_and_load_schemas_json_roundtrip(self):
        col = ColumnSchema(
            name="ticker",
            dtype="object",
            non_null_count=50,
            null_count=0,
            sample_values=["AAPL", "GOOG"],
        )
        original = {
            "data1": DataFrameSchema(
                path="/data/data1.parquet",
                rows=100,
                columns=1,
                column_schemas=[col],
                memory_mb=1.5,
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "schemas.json")
            save_schemas_json(original, path)
            loaded = load_schemas_json(path)
        
        assert "data1" in loaded
        assert loaded["data1"].path == "/data/data1.parquet"
        assert loaded["data1"].rows == 100
        assert len(loaded["data1"].column_schemas) == 1
        assert loaded["data1"].column_schemas[0].name == "ticker"


@pytest.mark.unit
class TestExtractAllSchemas:
    """Tests for extract_all_schemas function."""

    def test_extract_all_schemas_empty_paths(self):
        result = extract_all_schemas({})
        assert result == {}

    def test_extract_all_schemas_skips_non_data_files(self):
        result = extract_all_schemas({
            "readme": "/path/to/readme.md",
            "config": "/path/to/config.json",
        })
        # Should skip non-data files
        assert len(result) == 0
