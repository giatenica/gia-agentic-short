"""
Agent Best Practices and Standards
==================================
Core patterns and utilities for building agents that work efficiently with Claude API.

This module codifies best practices from Anthropic documentation and project experience:
- Date awareness for temporal context
- Web search awareness for latest information
- Optimal model selection by task
- Efficient prompt caching patterns
- Batch processing for bulk operations

All future agents should import from this module to ensure consistency.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from datetime import datetime, timezone
from typing import Optional, Literal
from dataclasses import dataclass
from enum import Enum

from src.llm.claude_client import TaskType, ModelTier, TASK_MODEL_MAP, MODELS


# =============================================================================
# DATE AND TEMPORAL AWARENESS
# =============================================================================

def get_current_date_context() -> str:
    """
    Get current date context for agent prompts.
    
    Returns formatted string to inject into system prompts so models
    know the current date for temporal reasoning.
    """
    now = datetime.now(timezone.utc)
    return f"""
CURRENT DATE: {now.strftime('%Y-%m-%d')}
CURRENT TIME (UTC): {now.strftime('%H:%M:%S')}
YEAR: {now.year}

Use this date as reference for:
- Temporal reasoning about events and deadlines
- Assessing recency of data and sources
- Understanding what information might be outdated
"""


def get_date_string() -> str:
    """Get simple ISO date string for timestamps."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


# =============================================================================
# WEB SEARCH AWARENESS
# =============================================================================

WEB_SEARCH_INSTRUCTIONS = """
WEB SEARCH AWARENESS:
You do NOT have direct internet access. However, you should:
1. Identify when current/latest information would improve your response
2. Clearly state when you need up-to-date information from web search
3. Flag outdated knowledge that should be verified
4. Request specific searches when needed (format: [SEARCH_NEEDED: "query"])

For academic research, flag when you need:
- Recent publications (after your knowledge cutoff)
- Current market data or statistics
- Latest regulatory or policy changes
- Recent news events affecting the research topic
"""


def should_suggest_web_search(topic: str) -> bool:
    """
    Determine if a topic likely needs current web information.
    
    Args:
        topic: The subject being researched
        
    Returns:
        True if web search would likely help
    """
    current_info_keywords = [
        'current', 'recent', 'latest', '2024', '2025', '2026',
        'today', 'now', 'price', 'rate', 'market', 'news',
        'regulation', 'policy', 'announcement', 'earnings',
        'stock', 'trading', 'volatility', 'breaking'
    ]
    topic_lower = topic.lower()
    return any(kw in topic_lower for kw in current_info_keywords)


# =============================================================================
# MODEL SELECTION GUIDELINES
# =============================================================================

@dataclass
class ModelGuidelines:
    """Guidelines for selecting the right model for each task."""
    
    @staticmethod
    def get_recommended_model(task_description: str) -> tuple[ModelTier, str]:
        """
        Get recommended model tier with explanation.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Tuple of (ModelTier, explanation)
        """
        task_lower = task_description.lower()
        
        # Opus 4.5: Complex reasoning tasks
        opus_keywords = [
            'reasoning', 'analysis', 'research', 'academic', 'scientific',
            'synthesis', 'complex', 'multi-step', 'evaluation', 'assessment',
            'hypothesis', 'theory', 'methodology', 'literature review'
        ]
        
        # Haiku 4.5: Simple, high-volume tasks
        haiku_keywords = [
            'classify', 'extract', 'summarize', 'format', 'parse',
            'validate', 'check', 'simple', 'quick', 'bulk', 'many'
        ]
        
        if any(kw in task_lower for kw in opus_keywords):
            return ModelTier.OPUS, "Complex reasoning requires maximum intelligence"
        
        if any(kw in task_lower for kw in haiku_keywords):
            return ModelTier.HAIKU, "High-volume/simple task benefits from speed"
        
        return ModelTier.SONNET, "Balanced performance for coding and agents"
    
    @staticmethod
    def explain_model_choice(task_type: TaskType) -> str:
        """Explain why a specific model was chosen for a task type."""
        tier = TASK_MODEL_MAP.get(task_type, ModelTier.SONNET)
        model_info = MODELS[tier]
        
        explanations = {
            ModelTier.OPUS: f"Using {model_info.id} for maximum intelligence. "
                           "Best for complex reasoning, scientific analysis, and academic writing.",
            ModelTier.SONNET: f"Using {model_info.id} for balanced performance. "
                             "Optimal for coding, agents, and general tasks.",
            ModelTier.HAIKU: f"Using {model_info.id} for speed efficiency. "
                            "Best for classification, summarization, and high-volume tasks.",
        }
        return explanations.get(tier, f"Using {model_info.id}")


# =============================================================================
# CACHING BEST PRACTICES
# =============================================================================

class CachingStrategy(Enum):
    """Prompt caching strategies based on Anthropic best practices."""
    
    # 5-minute cache: Use for frequently repeated prompts and system prompts
    # Note: Anthropic API only supports 'ephemeral' (5-min TTL)
    EPHEMERAL = "ephemeral"


@dataclass
class CachingGuidelines:
    """Guidelines for efficient prompt caching."""
    
    # Minimum tokens for caching by model
    MIN_CACHE_TOKENS = {
        ModelTier.OPUS: 4096,    # Claude Opus 4.5
        ModelTier.SONNET: 1024,  # Claude Sonnet 4.5
        ModelTier.HAIKU: 4096,   # Claude Haiku 4.5
    }
    
    @staticmethod
    def should_cache(text: str, model: ModelTier) -> bool:
        """
        Determine if content should be cached.
        
        Args:
            text: Content to potentially cache
            model: Model tier being used
            
        Returns:
            True if content should be cached
        """
        # Rough token estimate: ~4 chars per token
        estimated_tokens = len(text) // 4
        min_tokens = CachingGuidelines.MIN_CACHE_TOKENS.get(model, 1024)
        return estimated_tokens >= min_tokens
    
    @staticmethod
    def get_cache_ttl(use_case: str) -> CachingStrategy:
        """
        Get recommended cache TTL for use case.
        
        Args:
            use_case: Description of the caching use case
            
        Returns:
            Recommended CachingStrategy
        """
        # Current implementation only supports Anthropic's ephemeral cache control.
        # Keep the API stable so a longer TTL can be added later without breaking callers.
        _ = use_case
        return CachingStrategy.EPHEMERAL
    
    @staticmethod
    def get_caching_tips() -> str:
        """Get best practices for prompt caching."""
        return """
PROMPT CACHING BEST PRACTICES:
1. Place static content (tools, system, examples) at prompt beginning
2. Set cache breakpoints at end of reusable content
3. Use ephemeral cache control for frequently repeated prompts
4. Maximum 4 cache breakpoints per request
5. Cache read is discounted relative to base input cost
6. Cache write costs more than base input, but can be worth it for reuse
7. Structure: tools -> system -> messages (in this order)
"""


# =============================================================================
# BATCH PROCESSING GUIDELINES
# =============================================================================

@dataclass
class BatchingGuidelines:
    """Guidelines for efficient batch processing."""
    
    @staticmethod
    def should_use_batch(
        num_requests: int,
        time_sensitive: bool = False,
        cost_priority: bool = True,
    ) -> tuple[bool, str]:
        """
        Determine if batch processing should be used.
        
        Args:
            num_requests: Number of requests to process
            time_sensitive: Whether results are needed immediately
            cost_priority: Whether cost control is important
            
        Returns:
            Tuple of (should_batch, explanation)
        """
        if time_sensitive:
            return False, "Real-time results needed; use streaming API"
        
        if num_requests < 10:
            return False, "Small batch; parallel async calls may be faster"
        
        if cost_priority and num_requests >= 10:
            return True, f"Batch {num_requests} requests for cost control"
        
        if num_requests >= 100:
            return True, "Large batch benefits from async processing"
        
        return False, "Default to real-time API"
    
    @staticmethod
    def get_batching_tips() -> str:
        """Get best practices for batch processing."""
        return """
BATCH PROCESSING BEST PRACTICES:
1. Use batch processing for non-urgent bulk workloads
2. Use meaningful custom_id values (order not guaranteed)
3. Test request shape with the Messages API first
4. Monitor batch status and implement retry logic
"""


# =============================================================================
# AGENT SYSTEM PROMPT TEMPLATE
# =============================================================================

def build_enhanced_system_prompt(
    base_prompt: str,
    include_date: bool = True,
    include_web_awareness: bool = True,
    include_model_context: bool = False,
    model_tier: Optional[ModelTier] = None,
) -> str:
    """
    Build an enhanced system prompt with best practices.
    
    Args:
        base_prompt: The core agent-specific system prompt
        include_date: Whether to add current date context
        include_web_awareness: Whether to add web search awareness
        include_model_context: Whether to explain the model choice
        model_tier: Model tier being used (for context)
        
    Returns:
        Enhanced system prompt with all relevant context
    """
    sections = []
    
    # Add date context first (models should know current date)
    if include_date:
        sections.append(get_current_date_context())
    
    # Add the core agent prompt
    sections.append(base_prompt)
    
    # Add web search awareness
    if include_web_awareness:
        sections.append(WEB_SEARCH_INSTRUCTIONS)
    
    # Add model context if requested
    if include_model_context and model_tier:
        model_info = MODELS[model_tier]
        sections.append(f"""
MODEL CONTEXT:
Using {model_info.id} ({model_tier.value.upper()})
Optimized for: {model_info.description}
""")
    
    return "\n".join(sections)


# =============================================================================
# CRITICAL RULES FOR ALL AGENTS
# =============================================================================

CRITICAL_RULES = """
CRITICAL RULES FOR ALL RESPONSES:
1. NEVER make up data, statistics, numbers, or facts
2. NEVER use emojis
3. NEVER use em dashes (use semicolons, colons, or periods)
4. ALWAYS cite sources for quantitative claims
5. ALWAYS flag when information might be outdated
6. ALWAYS be specific and actionable
7. NEVER use banned words (delve, realm, harness, unlock, etc.)
"""

BANNED_WORDS = [
    'delve', 'realm', 'harness', 'unlock', 'tapestry', 'paradigm',
    'cutting-edge', 'revolutionize', 'landscape', 'potential', 'findings',
    'intricate', 'showcasing', 'crucial', 'pivotal', 'surpass',
    'meticulously', 'vibrant', 'unparalleled', 'underscore', 'leverage',
    'synergy', 'innovative', 'game-changer', 'testament', 'commendable',
    'meticulous', 'highlight', 'emphasize', 'boast', 'groundbreaking',
    'align', 'foster', 'showcase', 'enhance', 'holistic', 'garner',
    'accentuate', 'pioneering', 'trailblazing', 'unleash', 'versatile',
    'transformative', 'redefine', 'seamless', 'optimize', 'scalable',
    'robust', 'breakthrough', 'empower', 'streamline', 'intelligent',
    'smart', 'next-gen', 'frictionless', 'elevate', 'adaptive',
    'effortless', 'data-driven', 'insightful', 'proactive',
    'mission-critical', 'visionary', 'disruptive', 'reimagine', 'agile',
    'customizable', 'personalized', 'unprecedented', 'intuitive',
    'leading-edge', 'synergize', 'democratize', 'automate', 'accelerate',
    'state-of-the-art', 'dynamic', 'reliable', 'efficient', 'cloud-native',
    'immersive', 'predictive', 'transparent', 'proprietary', 'integrated',
    'plug-and-play', 'turnkey', 'future-proof', 'open-ended', 'AI-powered',
    'next-generation', 'always-on', 'hyper-personalized', 'results-driven',
    'machine-first', 'paradigm-shifting', 'novel', 'unique', 'utilize',
    'impactful'
]


def add_critical_rules(prompt: str) -> str:
    """Add critical rules to any system prompt."""
    return f"{prompt}\n\n{CRITICAL_RULES}"


# =============================================================================
# CONVENIENCE FUNCTIONS FOR AGENT BUILDERS
# =============================================================================

def get_agent_config(
    agent_name: str,
    base_prompt: str,
    task_type: TaskType,
    add_date: bool = True,
    add_web_awareness: bool = True,
    add_rules: bool = True,
) -> dict:
    """
    Get complete configuration for building an agent.
    
    This is the recommended way to set up new agents.
    
    Args:
        agent_name: Name of the agent
        base_prompt: Core system prompt
        task_type: Task type for model selection
        add_date: Include current date context
        add_web_awareness: Include web search awareness
        add_rules: Include critical rules
        
    Returns:
        Dict with all configuration needed for BaseAgent
    """
    model_tier = TASK_MODEL_MAP.get(task_type, ModelTier.SONNET)
    
    # Build enhanced system prompt
    system_prompt = build_enhanced_system_prompt(
        base_prompt=base_prompt,
        include_date=add_date,
        include_web_awareness=add_web_awareness,
        include_model_context=True,
        model_tier=model_tier,
    )
    
    if add_rules:
        system_prompt = add_critical_rules(system_prompt)
    
    return {
        "name": agent_name,
        "task_type": task_type,
        "system_prompt": system_prompt,
        "model_tier": model_tier,
        "model_explanation": ModelGuidelines.explain_model_choice(task_type),
    }


def log_agent_config(config: dict) -> None:
    """Log agent configuration for debugging."""
    from loguru import logger
    logger.info(f"Agent: {config['name']}")
    logger.info(f"Task Type: {config['task_type'].value}")
    logger.info(f"Model: {config['model_tier'].value}")
    logger.debug(f"Model Explanation: {config['model_explanation']}")
