"""
Centralized Configuration
=========================
Centralized configuration values and constants for the GIA Research Pipeline.

This module provides:
- Timeout configuration for various operations
- Environment variable defaults
- Path constants

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimeoutConfig:
    """Centralized timeout configuration in seconds."""
    
    # LLM API timeouts (Claude, etc.)
    LLM_API: int = 600  # 10 minutes for complex reasoning
    LLM_CONNECT: int = 30  # Connection timeout
    
    # External API timeouts (Edison, etc.)
    EXTERNAL_API: int = 1200  # 20 minutes for literature search
    
    # Code execution
    CODE_EXECUTION: int = 120  # 2 minutes for generated code
    
    # File operations
    FILE_LOCK: int = 30  # File lock acquisition
    
    # Cache operations
    CACHE_MAX_AGE_HOURS: int = 24

    # Evidence retrieval
    PDF_DOWNLOAD: int = int(os.getenv("GIA_PDF_DOWNLOAD_TIMEOUT", "120"))


@dataclass(frozen=True)
class FilenameConfig:
    """Filename constraints."""
    
    # Maximum filename length (Linux ext4/macOS HFS+ limit)
    MAX_LENGTH: int = 255


@dataclass(frozen=True)
class IntakeServerConfig:
    """Intake server configuration."""
    
    # Default port
    PORT: int = int(os.getenv("GIA_INTAKE_PORT", "8080"))
    
    # Upload limits
    MAX_UPLOAD_MB: int = int(os.getenv("GIA_MAX_UPLOAD_MB", "2048"))
    MAX_ZIP_FILES: int = int(os.getenv("GIA_MAX_ZIP_FILES", "20000"))
    MAX_ZIP_TOTAL_MB: int = int(os.getenv("GIA_MAX_ZIP_TOTAL_MB", "2048"))


@dataclass(frozen=True)
class TracingConfig:
    """Tracing configuration."""
    
    SERVICE_NAME: str = "gia-research-agents"
    OTLP_ENDPOINT: str = os.getenv("OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    ENABLED: bool = os.getenv("ENABLE_TRACING", "false").lower() == "true"


@dataclass(frozen=True)
class RetrievalConfig:
    """Network retrieval configuration."""

    MAX_PDF_BYTES: int = int(os.getenv("GIA_MAX_PDF_BYTES", str(100 * 1024 * 1024)))


# Global singleton instances
TIMEOUTS = TimeoutConfig()
FILENAMES = FilenameConfig()
INTAKE_SERVER = IntakeServerConfig()
TRACING = TracingConfig()
RETRIEVAL = RetrievalConfig()


def get_timeout(operation: str) -> int:
    """Get timeout for a specific operation type.
    
    Args:
        operation: One of 'llm', 'external', 'code', 'file_lock'
        
    Returns:
        Timeout in seconds
    """
    mapping = {
        "llm": TIMEOUTS.LLM_API,
        "llm_connect": TIMEOUTS.LLM_CONNECT,
        "external": TIMEOUTS.EXTERNAL_API,
        "code": TIMEOUTS.CODE_EXECUTION,
        "file_lock": TIMEOUTS.FILE_LOCK,
    }
    return mapping.get(operation, TIMEOUTS.LLM_API)
