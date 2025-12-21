"""
LLM Client Module
=================
Provides unified access to Claude and other LLM APIs.
"""

from .claude_client import ClaudeClient, get_claude_client, BatchRequest, BatchResult

__all__ = [
    "ClaudeClient",
    "get_claude_client",
    "BatchRequest",
    "BatchResult",
]
