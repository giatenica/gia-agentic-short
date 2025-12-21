"""
Tests for Validation Utilities
==============================

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import sys
import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.validation import (
    is_safe_path,
    validate_path,
    validate_project_folder,
    sanitize_filename,
    safe_json_loads,
    validate_api_response,
)


@pytest.mark.unit
class TestIsSafePath:
    """Tests for is_safe_path function."""
    
    def test_safe_path_returns_true(self):
        """Normal paths should be considered safe."""
        assert is_safe_path("/tmp/test.txt") is True
        assert is_safe_path("./relative/path") is True
        assert is_safe_path("simple.txt") is True
    
    def test_path_traversal_returns_false(self):
        """Paths with .. should be detected as unsafe."""
        assert is_safe_path("../etc/passwd") is False
        assert is_safe_path("/var/../etc/shadow") is False
        assert is_safe_path("data/../../../etc/passwd") is False
    
    def test_dangerous_paths_return_false(self):
        """System paths should be detected as unsafe."""
        assert is_safe_path("/etc/passwd") is False
        assert is_safe_path("/root/.ssh/id_rsa") is False
    
    def test_path_with_base_dir(self, tmp_path):
        """Path under base directory should be safe."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        assert is_safe_path(subdir, base_dir=tmp_path) is True
        assert is_safe_path(Path("/etc"), base_dir=tmp_path) is False


@pytest.mark.unit
class TestValidatePath:
    """Tests for validate_path function."""
    
    def test_empty_path_raises(self):
        """Empty path should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_path("")
    
    def test_nonexistent_path_raises_when_must_exist(self, tmp_path):
        """Non-existent path should raise when must_exist=True."""
        with pytest.raises(FileNotFoundError):
            validate_path(tmp_path / "nonexistent.txt", must_exist=True)
    
    def test_existing_path_passes(self, tmp_path):
        """Existing path should pass validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        result = validate_path(test_file, must_exist=True)
        assert result == test_file
    
    def test_file_validation(self, tmp_path):
        """File type validation should work."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        result = validate_path(test_file, must_be_file=True)
        assert result == test_file
        
        with pytest.raises(ValueError, match="not a file"):
            validate_path(tmp_path, must_exist=True, must_be_file=True)
    
    def test_dir_validation(self, tmp_path):
        """Directory type validation should work."""
        result = validate_path(tmp_path, must_be_dir=True)
        assert result == tmp_path


@pytest.mark.unit
class TestValidateProjectFolder:
    """Tests for validate_project_folder function."""
    
    def test_valid_project_folder(self, tmp_path):
        """Valid project folder should pass."""
        project_json = tmp_path / "project.json"
        project_json.write_text('{"id": "test"}')
        
        result = validate_project_folder(tmp_path)
        assert result == tmp_path
    
    def test_missing_project_json_raises(self, tmp_path):
        """Missing project.json should raise ValueError."""
        with pytest.raises(ValueError, match="project.json not found"):
            validate_project_folder(tmp_path)


@pytest.mark.unit
class TestSanitizeFilename:
    """Tests for sanitize_filename function."""
    
    def test_basic_filename(self):
        """Basic filenames should pass unchanged."""
        assert sanitize_filename("document.pdf") == "document.pdf"
        assert sanitize_filename("my_file.txt") == "my_file.txt"
    
    def test_removes_dangerous_chars(self):
        """Dangerous characters should be removed."""
        assert sanitize_filename("file<>:name.txt") == "filename.txt"
        assert sanitize_filename("file\x00name.txt") == "filename.txt"
    
    def test_removes_path_separators(self):
        """Path separators should be removed."""
        assert sanitize_filename("/etc/passwd") == "passwd"
        # Backslashes are removed via os.path.basename
        assert sanitize_filename("..\\..\\file.txt") == "file.txt"
    
    def test_replaces_whitespace(self):
        """Whitespace should be replaced with underscores."""
        assert sanitize_filename("my file name.txt") == "my_file_name.txt"
        assert sanitize_filename("file  with   spaces.txt") == "file_with_spaces.txt"
    
    def test_empty_filename_returns_unnamed(self):
        """Empty filename should return 'unnamed'."""
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("...") == "unnamed"
    
    def test_truncates_long_filename(self):
        """Long filenames should be truncated."""
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name, max_length=50)
        assert len(result) <= 50
        assert result.endswith(".txt")


@pytest.mark.unit
class TestSafeJsonLoads:
    """Tests for safe_json_loads function."""
    
    def test_valid_json(self):
        """Valid JSON should be parsed correctly."""
        result = safe_json_loads('{"key": "value"}')
        assert result == {"key": "value"}
    
    def test_invalid_json_returns_default(self):
        """Invalid JSON should return default value."""
        result = safe_json_loads("not json", default={})
        assert result == {}
    
    def test_empty_input_returns_default(self):
        """Empty input should return default value."""
        assert safe_json_loads("", default=None) is None
        assert safe_json_loads(None, default=[]) == []
    
    def test_bytes_input(self):
        """Bytes input should be handled."""
        result = safe_json_loads(b'{"key": "value"}')
        assert result == {"key": "value"}


@pytest.mark.unit
class TestValidateApiResponse:
    """Tests for validate_api_response function."""
    
    def test_valid_response(self):
        """Valid response should return True."""
        response = {"status": "ok", "data": [1, 2, 3]}
        assert validate_api_response(response) is True
    
    def test_invalid_response_type(self):
        """Non-dict response should return False."""
        assert validate_api_response("string") is False
        assert validate_api_response([1, 2, 3]) is False
    
    def test_required_fields(self):
        """Missing required fields should return False."""
        response = {"status": "ok"}
        assert validate_api_response(response, required_fields=["status"]) is True
        assert validate_api_response(response, required_fields=["data"]) is False
