"""
Claude API Client with Batch Processing and Prompt Caching
==========================================================
Provides optimized Claude API access with:
- Prompt caching for repeated context
- Batch processing for multiple requests
- Extended thinking support
- Token usage tracking

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import json
import asyncio
from typing import Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


@dataclass
class TokenUsage:
    """Track token usage across requests."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def cache_savings(self) -> float:
        """Calculate percentage of tokens served from cache."""
        total_input = self.input_tokens + self.cache_read_tokens
        if total_input == 0:
            return 0.0
        return (self.cache_read_tokens / total_input) * 100
    
    def add(self, usage: dict):
        """Add usage from API response."""
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
        self.cache_read_tokens += usage.get("cache_read_input_tokens", 0)


@dataclass 
class BatchRequest:
    """Represents a single request in a batch."""
    custom_id: str
    messages: list
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    system: Optional[str] = None
    temperature: float = 1.0


@dataclass
class BatchResult:
    """Result from a batch request."""
    custom_id: str
    content: str
    model: str
    usage: dict
    stop_reason: str
    error: Optional[str] = None


class ClaudeClient:
    """
    Claude API client with batch processing and prompt caching support.
    
    Features:
    - Prompt caching: Reuse expensive system prompts across requests
    - Batch processing: Submit up to 10,000 requests in a single batch
    - Extended thinking: Support for complex reasoning tasks
    - Token tracking: Monitor usage and cache efficiency
    """
    
    # Available models
    MODELS = {
        "opus": "claude-sonnet-4-20250514",  # Claude Opus 4.5
        "sonnet": "claude-sonnet-4-20250514",
        "haiku": "claude-3-5-haiku-20241022",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "opus",
        enable_caching: bool = True,
    ):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            default_model: Default model to use ('opus', 'sonnet', 'haiku')
            enable_caching: Whether to enable prompt caching
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        self.default_model = self.MODELS.get(default_model, default_model)
        self.enable_caching = enable_caching
        self.usage = TokenUsage()
        
        logger.info(f"Claude client initialized with model: {self.default_model}")
    
    def _prepare_cached_content(self, content: str, cache_type: str = "ephemeral") -> dict:
        """
        Prepare content block with cache control.
        
        Args:
            content: Text content to cache
            cache_type: Cache type ('ephemeral' for 5-minute TTL)
            
        Returns:
            Content block with cache_control
        """
        return {
            "type": "text",
            "text": content,
            "cache_control": {"type": cache_type}
        }
    
    def chat(
        self,
        messages: list,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        cache_system: bool = True,
    ) -> str:
        """
        Send a chat message to Claude.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system: System prompt (will be cached if cache_system=True)
            model: Model to use (defaults to self.default_model)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            cache_system: Whether to cache the system prompt
            
        Returns:
            Response text from Claude
        """
        model = model or self.default_model
        
        # Prepare system prompt with caching
        system_content = None
        if system:
            if self.enable_caching and cache_system:
                system_content = [self._prepare_cached_content(system)]
            else:
                system_content = system
        
        # Make request
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        
        if system_content:
            kwargs["system"] = system_content
        
        response = self.client.messages.create(**kwargs)
        
        # Track usage
        self.usage.add(response.usage.model_dump())
        
        logger.debug(f"Chat response: {response.usage}")
        
        return response.content[0].text
    
    async def chat_async(
        self,
        messages: list,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        cache_system: bool = True,
    ) -> str:
        """Async version of chat method."""
        model = model or self.default_model
        
        system_content = None
        if system:
            if self.enable_caching and cache_system:
                system_content = [self._prepare_cached_content(system)]
            else:
                system_content = system
        
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        
        if system_content:
            kwargs["system"] = system_content
        
        response = await self.async_client.messages.create(**kwargs)
        self.usage.add(response.usage.model_dump())
        
        return response.content[0].text
    
    def chat_with_thinking(
        self,
        messages: list,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 16000,
        budget_tokens: int = 10000,
    ) -> tuple[str, str]:
        """
        Chat with extended thinking enabled.
        
        Extended thinking allows Claude to reason through complex problems
        before responding, improving accuracy on difficult tasks.
        
        Args:
            messages: List of message dicts
            system: System prompt
            model: Model to use
            max_tokens: Maximum total tokens (thinking + response)
            budget_tokens: Token budget for thinking process
            
        Returns:
            Tuple of (thinking_text, response_text)
        """
        model = model or self.default_model
        
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "thinking": {
                "type": "enabled",
                "budget_tokens": budget_tokens
            },
            "messages": messages,
            "temperature": 1,  # Required for extended thinking
        }
        
        if system:
            kwargs["system"] = system
        
        response = self.client.messages.create(**kwargs)
        self.usage.add(response.usage.model_dump())
        
        thinking_text = ""
        response_text = ""
        
        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                response_text = block.text
        
        return thinking_text, response_text
    
    def create_batch(self, requests: list[BatchRequest]) -> str:
        """
        Create a batch of requests for asynchronous processing.
        
        Batch processing is ideal for:
        - Large-scale data processing
        - Bulk content generation
        - Non-time-sensitive tasks
        
        Args:
            requests: List of BatchRequest objects
            
        Returns:
            Batch ID for tracking
        """
        batch_requests = []
        
        for req in requests:
            batch_req = {
                "custom_id": req.custom_id,
                "params": {
                    "model": req.model,
                    "max_tokens": req.max_tokens,
                    "messages": req.messages,
                    "temperature": req.temperature,
                }
            }
            
            if req.system:
                batch_req["params"]["system"] = req.system
            
            batch_requests.append(batch_req)
        
        # Create batch
        batch = self.client.messages.batches.create(requests=batch_requests)
        
        logger.info(f"Created batch {batch.id} with {len(requests)} requests")
        
        return batch.id
    
    def get_batch_status(self, batch_id: str) -> dict:
        """
        Get the status of a batch.
        
        Returns:
            Dict with batch status information
        """
        batch = self.client.messages.batches.retrieve(batch_id)
        
        return {
            "id": batch.id,
            "status": batch.processing_status,
            "created_at": batch.created_at,
            "ended_at": batch.ended_at,
            "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
        }
    
    def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        """
        Retrieve results from a completed batch.
        
        Args:
            batch_id: ID of the completed batch
            
        Returns:
            List of BatchResult objects
        """
        results = []
        
        for result in self.client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                message = result.result.message
                results.append(BatchResult(
                    custom_id=result.custom_id,
                    content=message.content[0].text,
                    model=message.model,
                    usage=message.usage.model_dump(),
                    stop_reason=message.stop_reason,
                ))
            else:
                results.append(BatchResult(
                    custom_id=result.custom_id,
                    content="",
                    model="",
                    usage={},
                    stop_reason="error",
                    error=str(result.result.error) if hasattr(result.result, 'error') else "Unknown error",
                ))
        
        return results
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a pending batch."""
        try:
            self.client.messages.batches.cancel(batch_id)
            logger.info(f"Cancelled batch {batch_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel batch {batch_id}: {e}")
            return False
    
    def get_usage_summary(self) -> dict:
        """Get token usage summary."""
        return {
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "cache_creation_tokens": self.usage.cache_creation_tokens,
            "cache_read_tokens": self.usage.cache_read_tokens,
            "cache_savings_percent": f"{self.usage.cache_savings:.1f}%",
        }
    
    def reset_usage(self):
        """Reset token usage counters."""
        self.usage = TokenUsage()


# Convenience function for quick access
def get_claude_client(
    model: str = "opus",
    enable_caching: bool = True,
) -> ClaudeClient:
    """
    Get a configured Claude client.
    
    Args:
        model: Model to use ('opus', 'sonnet', 'haiku')
        enable_caching: Whether to enable prompt caching
        
    Returns:
        Configured ClaudeClient instance
    """
    return ClaudeClient(
        default_model=model,
        enable_caching=enable_caching,
    )


# Example usage
if __name__ == "__main__":
    from rich import print as rprint
    
    # Initialize client
    client = get_claude_client()
    
    # Test basic chat
    rprint("[bold]Testing basic chat...[/bold]")
    response = client.chat(
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        system="You are a helpful math tutor. Be concise.",
    )
    rprint(f"Response: {response}")
    
    # Test with caching (same system prompt)
    rprint("\n[bold]Testing prompt caching...[/bold]")
    response2 = client.chat(
        messages=[{"role": "user", "content": "What is 3 + 3?"}],
        system="You are a helpful math tutor. Be concise.",
    )
    rprint(f"Response: {response2}")
    
    # Show usage
    rprint("\n[bold]Token Usage:[/bold]")
    rprint(client.get_usage_summary())
