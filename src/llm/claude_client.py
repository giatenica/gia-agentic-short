"""
Claude API Client with Multi-Model Support, Batch Processing and Prompt Caching
================================================================================
Provides optimized Claude API access with:
- Multi-model support (Opus 4.5, Sonnet 4.5, Haiku 4.5)
- Task-based automatic model selection
- Prompt caching for repeated context (cache control when enabled)
- Batch processing for multiple requests
- Extended thinking with interleaved thinking support
- Token usage tracking and cost estimation

Model Selection Guide:
- Opus 4.5: Complex reasoning, scientific analysis, academic writing
- Sonnet 4.5: Coding, agents, data analysis, general tasks
- Haiku 4.5: Classification, summarization, high-volume tasks

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
import re
from pathlib import Path
from typing import Optional, Literal, Union
from dataclasses import dataclass
from enum import Enum

import anthropic
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def _load_env_file_lenient() -> None:
    """Load .env from repo root without raising or printing parse warnings."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        os.environ.setdefault(key, value)


_load_env_file_lenient()


class ModelTier(Enum):
    """Model tiers for task-based selection."""
    OPUS = "opus"       # Premium: Maximum intelligence, complex reasoning
    SONNET = "sonnet"   # Balanced: Complex agents, coding, general tasks
    HAIKU = "haiku"     # Fast: High-volume, low-latency tasks


class TaskType(Enum):
    """Task types for automatic model selection."""
    # Opus 4.5 tasks (maximum intelligence)
    COMPLEX_REASONING = "complex_reasoning"
    SCIENTIFIC_ANALYSIS = "scientific_analysis"
    ACADEMIC_WRITING = "academic_writing"
    MULTI_STEP_RESEARCH = "multi_step_research"
    
    # Sonnet 4.5 tasks (balanced performance)
    CODING = "coding"
    AGENTIC_WORKFLOW = "agentic_workflow"
    DATA_ANALYSIS = "data_analysis"
    GENERAL_CHAT = "general_chat"
    DOCUMENT_CREATION = "document_creation"
    
    # Haiku 4.5 tasks (speed-optimized)
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    DATA_EXTRACTION = "data_extraction"
    QUICK_RESPONSE = "quick_response"
    HIGH_VOLUME = "high_volume"


# Task to model mapping
TASK_MODEL_MAP = {
    # Opus 4.5: Premium intelligence tasks
    TaskType.COMPLEX_REASONING: ModelTier.OPUS,
    TaskType.SCIENTIFIC_ANALYSIS: ModelTier.OPUS,
    TaskType.ACADEMIC_WRITING: ModelTier.OPUS,
    TaskType.MULTI_STEP_RESEARCH: ModelTier.OPUS,
    
    # Sonnet 4.5: Balanced tasks
    TaskType.CODING: ModelTier.SONNET,
    TaskType.AGENTIC_WORKFLOW: ModelTier.SONNET,
    TaskType.DATA_ANALYSIS: ModelTier.SONNET,
    TaskType.GENERAL_CHAT: ModelTier.SONNET,
    TaskType.DOCUMENT_CREATION: ModelTier.SONNET,
    
    # Haiku 4.5: Speed-optimized tasks
    TaskType.CLASSIFICATION: ModelTier.HAIKU,
    TaskType.SUMMARIZATION: ModelTier.HAIKU,
    TaskType.DATA_EXTRACTION: ModelTier.HAIKU,
    TaskType.QUICK_RESPONSE: ModelTier.HAIKU,
    TaskType.HIGH_VOLUME: ModelTier.HAIKU,
}


@dataclass
class ModelInfo:
    """Information about a Claude model."""
    id: str
    alias: str
    tier: ModelTier
    description: str
    input_price_per_mtok: float
    output_price_per_mtok: float
    context_window: int
    max_output: int
    latency: str
    supports_thinking: bool = True
    supports_vision: bool = True


# Model registry with Claude 4.5 models
MODELS = {
    ModelTier.OPUS: ModelInfo(
        id="claude-opus-4-5-20251101",
        alias="claude-opus-4-5",
        tier=ModelTier.OPUS,
        description="Premium model with maximum intelligence for complex reasoning",
        input_price_per_mtok=5.0,
        output_price_per_mtok=25.0,
        context_window=200_000,
        max_output=64_000,
        latency="Moderate",
    ),
    ModelTier.SONNET: ModelInfo(
        id="claude-sonnet-4-5-20250929",
        alias="claude-sonnet-4-5",
        tier=ModelTier.SONNET,
        description="Smart model for complex agents and coding",
        input_price_per_mtok=3.0,
        output_price_per_mtok=15.0,
        context_window=200_000,
        max_output=64_000,
        latency="Fast",
    ),
    ModelTier.HAIKU: ModelInfo(
        id="claude-haiku-4-5-20251001",
        alias="claude-haiku-4-5",
        tier=ModelTier.HAIKU,
        description="Fastest model with near-frontier intelligence",
        input_price_per_mtok=1.0,
        output_price_per_mtok=5.0,
        context_window=200_000,
        max_output=64_000,
        latency="Fastest",
    ),
}


@dataclass
class TokenUsage:
    """Track token usage across requests."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    thinking_tokens: int = 0
    
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
    
    def estimate_cost(self, model: ModelTier) -> float:
        """Estimate cost in USD based on model pricing."""
        info = MODELS[model]
        input_cost = (self.input_tokens / 1_000_000) * info.input_price_per_mtok
        output_cost = (self.output_tokens / 1_000_000) * info.output_price_per_mtok
        return input_cost + output_cost


@dataclass 
class BatchRequest:
    """
    Represents a single request in a batch.
    
    Use batch processing for:
    - Large-scale data processing
    - Non-time-sensitive bulk tasks
    - Extended thinking with large budgets (32k+ tokens)
    - Processing multiple documents/queries
    """
    custom_id: str
    messages: list
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 16384  # No artificial limits
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
    Claude API client with multi-model support, batch processing and prompt caching.
    
    Design Philosophy:
    - No artificial token limits: Let Claude use full context for best results
    - Cache when possible: Cache stable system prompts when enabled
    - Batch when possible: Prefer batch API for non-urgent tasks
    - Think deeply: Extended thinking enabled for complex reasoning
    
    Features:
    - Multi-model: Opus 4.5, Sonnet 4.5, Haiku 4.5 with task-based selection
    - Prompt caching: Reuse stable system prompts (cache control)
    - Batch processing: Submit large numbers of requests asynchronously
    - Extended thinking: Complex reasoning with interleaved thinking support
    - Token tracking: Monitor usage, cache efficiency, and cost estimation
    
    Model Selection Guide:
    - Opus 4.5: Complex reasoning, scientific analysis, academic writing
    - Sonnet 4.5: Coding, agents, data analysis, general tasks (recommended default)
    - Haiku 4.5: Classification, summarization, high-volume tasks
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Union[ModelTier, str] = ModelTier.SONNET,
        enable_caching: bool = True,
    ):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            default_model: Default model tier (OPUS, SONNET, HAIKU) or string
            enable_caching: Whether to enable prompt caching
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        # Configure timeout: 600s (10 min) total, 30s connect for long-running LLM calls
        # Complex reasoning tasks with large contexts can take several minutes
        timeout_config = httpx.Timeout(600.0, connect=30.0)
        
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=timeout_config,
        )
        self.async_client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            timeout=timeout_config,
        )
        
        # Handle both string and ModelTier inputs for backward compatibility
        if isinstance(default_model, str):
            tier_map = {"opus": ModelTier.OPUS, "sonnet": ModelTier.SONNET, "haiku": ModelTier.HAIKU}
            self.default_model = tier_map.get(default_model.lower(), ModelTier.SONNET)
        else:
            self.default_model = default_model
        
        self.enable_caching = enable_caching
        self.usage = TokenUsage()
        
        model_info = MODELS[self.default_model]
        logger.info(f"Claude client initialized with model: {model_info.id}")
    
    def get_model_for_task(self, task: TaskType) -> ModelTier:
        """
        Get the recommended model tier for a specific task type.
        
        Args:
            task: The type of task to perform
            
        Returns:
            ModelTier enum for the task
        """
        return TASK_MODEL_MAP.get(task, self.default_model)
    
    def get_model_id(self, model: Optional[Union[ModelTier, str]] = None) -> str:
        """
        Get model ID string from tier or string.
        
        Args:
            model: Model tier enum or string name
            
        Returns:
            Model ID string for the API
        """
        if model is None:
            return MODELS[self.default_model].id
        
        if isinstance(model, str):
            tier_map = {"opus": ModelTier.OPUS, "sonnet": ModelTier.SONNET, "haiku": ModelTier.HAIKU}
            tier = tier_map.get(model.lower(), self.default_model)
        else:
            tier = model
        
        return MODELS[tier].id
    
    def _prepare_cached_content(self, content: str, cache_type: str = "ephemeral") -> dict:
        """
        Prepare content block with cache control.
        
        Args:
            content: Text content to cache
            cache_type: 'ephemeral' (5-min TTL)
            
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
        model: Optional[Union[ModelTier, str]] = None,
        task: Optional[TaskType] = None,
        max_tokens: int = 16384,
        temperature: float = 1.0,
        cache_system: bool = True,
        cache_ttl: Literal["ephemeral"] = "ephemeral",
    ) -> str:
        """
        Send a chat message to Claude.
        
        Best practices for quality results:
        - Don't artificially limit max_tokens; let Claude use what it needs
        - Always use system prompts with caching for repeated patterns
        - Use ephemeral cache (5-min TTL) for system prompts
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system: System prompt (will be cached if cache_system=True)
            model: Model tier to use (overrides task-based selection)
            task: Task type for automatic model selection
            max_tokens: Maximum tokens in response (default 16384, no artificial limits)
            temperature: Sampling temperature (0-1)
            cache_system: Whether to cache the system prompt
            cache_ttl: Cache duration ('ephemeral' = 5min)
            
        Returns:
            Response text from Claude
        """
        # Determine model: explicit > task-based > default
        if model:
            model_id = self.get_model_id(model)
        elif task:
            tier = self.get_model_for_task(task)
            model_id = self.get_model_id(tier)
        else:
            model_id = self.get_model_id()
        
        # Prepare system prompt with caching
        system_content = None
        if system:
            if self.enable_caching and cache_system:
                system_content = [self._prepare_cached_content(system, cache_ttl)]
            else:
                system_content = system
        
        # Make request
        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        
        if system_content:
            kwargs["system"] = system_content
        
        response = self.client.messages.create(**kwargs)
        
        # Track usage
        self.usage.add(response.usage.model_dump())
        
        logger.debug(f"Chat response [{model_id}]: {response.usage}")
        
        return response.content[0].text
    
    async def chat_async(
        self,
        messages: list,
        system: Optional[str] = None,
        model: Optional[Union[ModelTier, str]] = None,
        task: Optional[TaskType] = None,
        max_tokens: int = 16384,
        temperature: float = 1.0,
        cache_system: bool = True,
    ) -> str:
        """
        Async version of chat method with retry logic for resilience.
        
        Retries on transient API errors (rate limits, overload, network issues).
        Uses exponential backoff: 1s, 2s, 4s (3 attempts total).
        """
        # Determine model: explicit > task-based > default
        if model:
            model_id = self.get_model_id(model)
        elif task:
            tier = self.get_model_for_task(task)
            model_id = self.get_model_id(tier)
        else:
            model_id = self.get_model_id()
        
        system_content = None
        if system:
            if self.enable_caching and cache_system:
                system_content = [self._prepare_cached_content(system)]
            else:
                system_content = system
        
        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        
        if system_content:
            kwargs["system"] = system_content
        
        # Use retry decorator inline for async call
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)),
            reraise=True,
        )
        async def _make_request():
            return await self.async_client.messages.create(**kwargs)
        
        response = await _make_request()
        self.usage.add(response.usage.model_dump())
        
        return response.content[0].text
    
    def chat_with_thinking(
        self,
        messages: list,
        system: Optional[str] = None,
        model: Optional[Union[ModelTier, str]] = None,
        max_tokens: int = 32000,
        budget_tokens: int = 16000,
        interleaved: bool = False,
    ) -> tuple[str, str]:
        """
        Chat with extended thinking enabled for complex reasoning tasks.
        
        Extended thinking allows Claude to reason through complex problems
        before responding, significantly improving accuracy on difficult tasks.
        
        Best practices:
        - Use generous budget_tokens (16k+) for best reasoning quality
        - Don't artificially limit thinking; let Claude reason fully
        - Use batch processing for large workloads
        - Use interleaved thinking with tool use for multi-step reasoning
        - Consider Opus for maximum reasoning capability
        
        Args:
            messages: List of message dicts
            system: System prompt
            model: Model tier (all 4.5 models support thinking)
            max_tokens: Maximum total tokens (thinking + response), default 32k
            budget_tokens: Token budget for thinking (default 16k, min 1024)
            interleaved: Enable interleaved thinking for tool use
            
        Returns:
            Tuple of (thinking_text, response_text)
        """
        model_id = self.get_model_id(model)
        
        # Ensure minimum budget
        budget_tokens = max(1024, budget_tokens)
        
        kwargs = {
            "model": model_id,
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
        
        # Add interleaved thinking beta header if requested
        extra_headers = {}
        if interleaved:
            extra_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
        
        # Use streaming for extended thinking to handle long operations (>10 min)
        # Anthropic requires streaming for operations that may take longer
        # Use list append + join for O(n) performance instead of O(n^2) string concatenation
        thinking_parts = []
        response_parts = []
        
        with self.client.messages.stream(**kwargs, extra_headers=extra_headers if extra_headers else None) as stream:
            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'content_block_start':
                        # Handle start of thinking or text blocks
                        pass
                    elif event.type == 'content_block_delta':
                        if hasattr(event.delta, 'thinking'):
                            thinking_parts.append(event.delta.thinking)
                        elif hasattr(event.delta, 'text'):
                            response_parts.append(event.delta.text)
            
            # Get final message for usage stats
            final_message = stream.get_final_message()
            if final_message and final_message.usage:
                self.usage.add(final_message.usage.model_dump())
        
        return ''.join(thinking_parts), ''.join(response_parts)
    
    async def chat_with_thinking_async(
        self,
        messages: list,
        system: Optional[str] = None,
        model: Optional[Union[ModelTier, str]] = None,
        max_tokens: int = 32000,
        budget_tokens: int = 16000,
        interleaved: bool = False,
    ) -> tuple[str, str]:
        """
        Async version of chat_with_thinking for use in async contexts.
        
        Extended thinking allows Claude to reason through complex problems
        before responding, significantly improving accuracy on difficult tasks.
        
        Args:
            messages: List of message dicts
            system: System prompt
            model: Model tier (all 4.5 models support thinking)
            max_tokens: Maximum total tokens (thinking + response), default 32k
            budget_tokens: Token budget for thinking (default 16k, min 1024)
            interleaved: Enable interleaved thinking for tool use
            
        Returns:
            Tuple of (thinking_text, response_text)
        """
        model_id = self.get_model_id(model)
        budget_tokens = max(1024, budget_tokens)
        
        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "thinking": {
                "type": "enabled",
                "budget_tokens": budget_tokens
            },
            "messages": messages,
            "temperature": 1,
        }
        
        if system:
            kwargs["system"] = system
        
        extra_headers = {}
        if interleaved:
            extra_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
        
        # Use list append + join for O(n) performance instead of string concatenation
        thinking_parts = []
        response_parts = []
        
        async with self.async_client.messages.stream(**kwargs, extra_headers=extra_headers if extra_headers else None) as stream:
            async for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'content_block_delta':
                        if hasattr(event.delta, 'thinking'):
                            thinking_parts.append(event.delta.thinking)
                        elif hasattr(event.delta, 'text'):
                            response_parts.append(event.delta.text)
            
            final_message = await stream.get_final_message()
            if final_message and final_message.usage:
                self.usage.add(final_message.usage.model_dump())
        
        return ''.join(thinking_parts), ''.join(response_parts)
    
    def create_batch(self, requests: list[BatchRequest]) -> str:
        """
        Create a batch of requests for asynchronous processing.
        
        Batch processing is ideal for:
        - Large-scale data processing
        - Bulk content generation
        - Non-time-sensitive tasks
        - Extended thinking with large budgets (32k+ tokens)
        
        Batch APIs may have different pricing than interactive calls.
        
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
        """Get the status of a batch."""
        batch = self.client.messages.batches.retrieve(batch_id)
        
        return {
            "id": batch.id,
            "status": batch.processing_status,
            "created_at": batch.created_at,
            "ended_at": batch.ended_at,
            "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
        }
    
    def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        """Retrieve results from a completed batch."""
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
        """Get token usage summary with cost estimates."""
        return {
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "cache_creation_tokens": self.usage.cache_creation_tokens,
            "cache_read_tokens": self.usage.cache_read_tokens,
            "cache_savings_percent": f"{self.usage.cache_savings:.1f}%",
            "estimated_cost_usd": {
                "opus": f"${self.usage.estimate_cost(ModelTier.OPUS):.4f}",
                "sonnet": f"${self.usage.estimate_cost(ModelTier.SONNET):.4f}",
                "haiku": f"${self.usage.estimate_cost(ModelTier.HAIKU):.4f}",
            }
        }
    
    def reset_usage(self):
        """Reset token usage counters."""
        self.usage = TokenUsage()
    
    @staticmethod
    def list_models() -> dict:
        """List all available models with their specifications."""
        return {
            tier.value: {
                "id": info.id,
                "alias": info.alias,
                "description": info.description,
                "pricing": f"${info.input_price_per_mtok}/MTok in, ${info.output_price_per_mtok}/MTok out",
                "context_window": f"{info.context_window:,} tokens",
                "max_output": f"{info.max_output:,} tokens",
                "latency": info.latency,
                "supports_thinking": info.supports_thinking,
                "supports_vision": info.supports_vision,
            }
            for tier, info in MODELS.items()
        }


# Convenience functions
def get_claude_client(
    model: Union[ModelTier, str] = ModelTier.SONNET,
    enable_caching: bool = True,
) -> ClaudeClient:
    """
    Get a configured Claude client.
    
    Args:
        model: Default model tier (OPUS, SONNET, HAIKU) or string
        enable_caching: Whether to enable prompt caching
        
    Returns:
        Configured ClaudeClient instance
    """
    return ClaudeClient(
        default_model=model,
        enable_caching=enable_caching,
    )


def get_model_for_task(task: TaskType) -> str:
    """Get recommended model ID for a task type."""
    tier = TASK_MODEL_MAP.get(task, ModelTier.SONNET)
    return MODELS[tier].id


# Example usage
if __name__ == "__main__":
    from rich import print as rprint
    from rich.table import Table
    
    # List available models
    rprint("\n[bold]Available Claude 4.5 Models:[/bold]")
    table = Table(title="Claude 4.5 Model Comparison")
    table.add_column("Tier", style="cyan")
    table.add_column("Model ID", style="green")
    table.add_column("Best For", style="yellow")
    table.add_column("Latency", style="magenta")
    table.add_column("Pricing (In/Out)", style="dim")
    
    for tier, info in MODELS.items():
        table.add_row(
            tier.value.upper(),
            info.id,
            info.description[:45] + "..." if len(info.description) > 45 else info.description,
            info.latency,
            f"${info.input_price_per_mtok}/${info.output_price_per_mtok}",
        )
    rprint(table)
    
    # Show task mappings
    rprint("\n[bold]Task-Based Model Selection:[/bold]")
    task_table = Table(title="Task Type to Model Mapping")
    task_table.add_column("Task Type", style="cyan")
    task_table.add_column("Model Tier", style="green")
    task_table.add_column("Model ID", style="dim")
    
    for task, tier in TASK_MODEL_MAP.items():
        task_table.add_row(
            task.value,
            tier.value.upper(),
            MODELS[tier].id,
        )
    rprint(task_table)
    
    # Initialize client
    client = get_claude_client(model=ModelTier.SONNET)
    
    # Test basic chat with task-based model selection
    rprint("\n[bold]Testing task-based model selection...[/bold]")
    
    # Quick response (uses Haiku)
    response = client.chat(
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        task=TaskType.QUICK_RESPONSE,
    )
    rprint(f"Quick Response (Haiku): {response}")
    
    # Coding task (uses Sonnet)
    response = client.chat(
        messages=[{"role": "user", "content": "Write a Python one-liner to reverse a string."}],
        task=TaskType.CODING,
    )
    rprint(f"Coding (Sonnet): {response}")
    
    # Show usage
    rprint("\n[bold]Token Usage:[/bold]")
    rprint(client.get_usage_summary())
