"""
LLM Client Package

Provides integration with AWS Bedrock Claude models through:
- AWS Strands Agents SDK (preferred)
- Direct boto3 client (fallback)

The factory automatically selects the best available client.
"""

# Import from Strands client (primary)
from .strands_client import (
    Message,
    LLMResponse,
    ConversationContext,
    BaseLLMClient,
    StrandsAgentClient,
    BedrockClaudeClient,
    LLMClientFactory,
    get_llm_client,
    create_strands_agent,
)

__all__ = [
    # Data models
    "Message",
    "LLMResponse",
    "ConversationContext",
    
    # Base class
    "BaseLLMClient",
    
    # Client implementations
    "StrandsAgentClient",
    "BedrockClaudeClient",
    
    # Factory
    "LLMClientFactory",
    
    # Convenience functions
    "get_llm_client",
    "create_strands_agent",
]
