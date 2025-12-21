"""
Validation Utilities
====================
Security-focused validation functions for file paths, user input,
and data sanitization.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import re
import json
from pathlib import Path
from typing import Optional, Any, Union
from loguru import logger


# Allowed base directories for project operations
ALLOWED_BASE_DIRS = [
    "user-input",
    "tmp",
    "temp",
]


def is_safe_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> bool:
    """
    Check if a path is safe (no path traversal, no symlinks to outside).
    
    Args:
        path: Path to validate
        base_dir: Optional base directory that path must be under
        
    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Resolve the path to handle .. and symlinks
        resolved = Path(path).resolve()
        
        # Check for path traversal patterns in original path
        path_str = str(path)
        if ".." in path_str or path_str.startswith("/etc") or path_str.startswith("/root"):
            logger.warning(f"Potential path traversal detected: {path}")
            return False
        
        # If base_dir is provided, ensure path is under it
        if base_dir:
            base_resolved = Path(base_dir).resolve()
            try:
                resolved.relative_to(base_resolved)
            except ValueError:
                logger.warning(f"Path {resolved} is not under base directory {base_resolved}")
                return False
        
        return True
        
    except (OSError, ValueError) as e:
        logger.warning(f"Path validation error for {path}: {e}")
        return False


def validate_path(
    path: Union[str, Path],
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Validate a file path with security checks.
    
    Args:
        path: Path to validate
        must_exist: If True, path must exist
        must_be_file: If True, path must be a file
        must_be_dir: If True, path must be a directory
        base_dir: Optional base directory that path must be under
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path fails validation
        FileNotFoundError: If path must exist but doesn't
    """
    if not path:
        raise ValueError("Path cannot be empty")
    
    path_obj = Path(path)
    
    # Security check
    if not is_safe_path(path_obj, base_dir):
        raise ValueError(f"Path validation failed: {path}")
    
    # Existence check
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    # Type checks
    if must_be_file and path_obj.exists() and not path_obj.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    if must_be_dir and path_obj.exists() and not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    return path_obj


def validate_project_folder(project_folder: Union[str, Path]) -> Path:
    """
    Validate a project folder path.
    
    Ensures the path is safe and contains expected project structure.
    
    Args:
        project_folder: Path to project folder
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If validation fails
    """
    path = validate_path(project_folder, must_exist=True, must_be_dir=True)
    
    # Check for project.json (required file)
    project_json = path / "project.json"
    if not project_json.exists():
        raise ValueError(f"project.json not found in project folder: {path}")
    
    return path


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename for safe file system operations.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed"
    
    # Remove path separators
    filename = os.path.basename(filename)
    
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
    
    # Replace whitespace with underscores
    filename = re.sub(r'\s+', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Truncate to max length
    if len(filename) > max_length:
        # Preserve extension
        name, ext = os.path.splitext(filename)
        max_name_len = max_length - len(ext)
        filename = name[:max_name_len] + ext
    
    # Ensure non-empty
    if not filename:
        return "unnamed"
    
    return filename


def safe_json_loads(
    data: Union[str, bytes],
    default: Any = None,
    log_errors: bool = True,
) -> Any:
    """
    Safely parse JSON with error handling.
    
    Args:
        data: JSON string or bytes to parse
        default: Default value if parsing fails
        log_errors: Whether to log parsing errors
        
    Returns:
        Parsed JSON or default value
    """
    if not data:
        return default
    
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        if log_errors:
            logger.warning(f"JSON parsing error: {e}")
        return default


def validate_api_response(
    response: dict,
    required_fields: Optional[list] = None,
) -> bool:
    """
    Validate an API response structure.
    
    Args:
        response: Response dictionary to validate
        required_fields: List of required field names
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    if required_fields:
        for field in required_fields:
            if field not in response:
                logger.debug(f"Missing required field in response: {field}")
                return False
    
    return True
