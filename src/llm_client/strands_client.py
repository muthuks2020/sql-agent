"""
LLM Client Module using AWS Strands Agents SDK
Provides agent-based interaction with Claude via AWS Bedrock
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union

from ..config import LLMConfig
from ..utils import get_logger, LLMError, SQLAgentMetrics, time_operation

logger = get_logger(__name__)


# ============================================================================
# Message and Response Models
# ============================================================================

@dataclass
class Message:
    """Chat message representation"""
    role: str  # "user", "assistant", or "system"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """LLM response representation"""
    content: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model_id": self.model_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stop_reason": self.stop_reason,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ConversationContext:
    """Maintains conversation history for multi-turn interactions"""
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    max_history: int = 20
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to history"""
        self.messages.append(Message(role="user", content=content))
        self._trim_history()
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history"""
        self.messages.append(Message(role="assistant", content=content))
        self._trim_history()
    
    def _trim_history(self) -> None:
        """Trim history to max_history messages"""
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for API call"""
        return [msg.to_dict() for msg in self.messages]
    
    def clear(self) -> None:
        """Clear conversation history"""
        self.messages.clear()


# ============================================================================
# Base LLM Client
# ============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[ConversationContext] = None,
        **kwargs
    ) -> LLMResponse:
        """Invoke the LLM with a prompt"""
        pass
    
    @abstractmethod
    def invoke_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[ConversationContext] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Invoke the LLM with automatic retry on failure"""
        pass


# ============================================================================
# Strands Agent Client
# ============================================================================

class StrandsAgentClient(BaseLLMClient):
    """
    AWS Strands Agents SDK Client
    
    Uses Strands Agents framework for structured agent-based interactions
    with AWS Bedrock Claude models.
    
    Features:
    - Tool-based agent interactions
    - Structured output parsing
    - Automatic retry with exponential backoff
    - Thread-safe operation
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._agent = None
        self._model = None
        self._lock = threading.Lock()
        self._initialized = False
    
    def _get_model(self):
        """Get or create the Strands model (lazy initialization)"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from strands import Agent
                        from strands.models import BedrockModel
                    except ImportError:
                        raise ImportError(
                            "strands-agents is required for this client. "
                            "Install it with: pip install strands-agents"
                        )
                    
                    # Create Bedrock model configuration
                    model_config = {
                        "model_id": self.config.model_id,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "top_p": self.config.top_p,
                    }
                    
                    # Add region if specified
                    if self.config.aws_region:
                        model_config["region_name"] = self.config.aws_region
                    
                    # Create the Bedrock model
                    self._model = BedrockModel(**model_config)
                    
                    self._initialized = True
                    logger.info(
                        f"Initialized Strands Bedrock model",
                        extra={"extra_fields": {
                            "region": self.config.aws_region,
                            "model_id": self.config.model_id
                        }}
                    )
        
        return self._model
    
    def _create_agent(self, system_prompt: Optional[str] = None, tools: Optional[List] = None):
        """Create a Strands agent with optional system prompt and tools"""
        try:
            from strands import Agent
        except ImportError:
            raise ImportError(
                "strands-agents is required. Install with: pip install strands-agents"
            )
        
        model = self._get_model()
        
        agent_kwargs = {
            "model": model,
        }
        
        if system_prompt:
            agent_kwargs["system_prompt"] = system_prompt
        
        if tools:
            agent_kwargs["tools"] = tools
        
        return Agent(**agent_kwargs)
    
    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[ConversationContext] = None,
        tools: Optional[List] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Invoke Claude via Strands Agent
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            context: Optional conversation context for multi-turn
            tools: Optional list of Strands tools
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated content
        """
        # Use context system prompt or provided one
        effective_system_prompt = system_prompt or (context.system_prompt if context else None)
        
        # Create agent
        agent = self._create_agent(system_prompt=effective_system_prompt, tools=tools)
        
        # Build conversation messages if context exists
        if context and context.messages:
            # For multi-turn, we need to include history in the prompt
            history_text = ""
            for msg in context.messages:
                if msg.role == "user":
                    history_text += f"User: {msg.content}\n"
                else:
                    history_text += f"Assistant: {msg.content}\n"
            
            if history_text:
                prompt = f"Previous conversation:\n{history_text}\nCurrent request:\n{prompt}"
        
        start_time = time.time()
        
        try:
            # Invoke the agent
            result = agent(prompt)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract content from result
            content = str(result)
            
            # Try to get token usage if available
            input_tokens = 0
            output_tokens = 0
            
            # Strands provides usage info in the result metadata
            if hasattr(result, 'usage'):
                input_tokens = getattr(result.usage, 'input_tokens', 0)
                output_tokens = getattr(result.usage, 'output_tokens', 0)
            
            # Record metrics
            SQLAgentMetrics.record_llm_call(
                duration=latency_ms / 1000,
                model_id=self.config.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            llm_response = LLMResponse(
                content=content,
                model_id=self.config.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason="end_turn",
                latency_ms=latency_ms,
            )
            
            # Update context if provided
            if context:
                context.add_user_message(prompt)
                context.add_assistant_message(content)
            
            logger.debug(
                f"Strands agent invocation successful",
                extra={"extra_fields": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms
                }}
            )
            
            return llm_response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Strands agent invocation failed: {str(e)}",
                extra={"extra_fields": {"latency_ms": latency_ms}}
            )
            raise LLMError(
                message=f"Strands agent invocation failed: {str(e)}",
                model_id=self.config.model_id,
                original_error=e
            )
    
    def invoke_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[ConversationContext] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Invoke Claude with automatic retry on failure
        
        Uses exponential backoff for retries
        """
        retries = max_retries if max_retries is not None else self.config.retry_attempts
        last_error = None
        
        for attempt in range(retries):
            try:
                return self.invoke(prompt, system_prompt, context, **kwargs)
            except LLMError as e:
                last_error = e
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    raise
                
                if attempt < retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Strands agent invocation failed, retrying in {delay}s",
                        extra={"extra_fields": {
                            "attempt": attempt + 1,
                            "max_retries": retries,
                            "error": str(e)
                        }}
                    )
                    time.sleep(delay)
        
        raise LLMError(
            message=f"Strands agent invocation failed after {retries} attempts",
            model_id=self.config.model_id,
            original_error=last_error
        )
    
    def _is_retryable_error(self, error: LLMError) -> bool:
        """Check if error is retryable"""
        error_str = str(error.original_error).lower() if error.original_error else ""
        
        retryable_patterns = [
            "throttling",
            "rate limit",
            "too many requests",
            "service unavailable",
            "timeout",
            "connection",
            "temporary",
        ]
        
        return any(pattern in error_str for pattern in retryable_patterns)
    
    def health_check(self) -> bool:
        """Check if the client can connect to Bedrock"""
        try:
            response = self.invoke(
                prompt="Say 'OK'",
                max_tokens=10
            )
            return bool(response.content)
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


# ============================================================================
# Bedrock Direct Client (Fallback without Strands)
# ============================================================================

class BedrockClaudeClient(BaseLLMClient):
    """
    AWS Bedrock Claude Client (Direct boto3)
    
    Fallback client using boto3 directly when Strands is not available.
    Thread-safe client for interacting with Claude via AWS Bedrock.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._lock = threading.Lock()
        self._initialized = False
    
    def _get_client(self):
        """Get or create Boto3 Bedrock client (lazy initialization)"""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        import boto3
                        from botocore.config import Config
                    except ImportError:
                        raise ImportError(
                            "boto3 is required for Bedrock integration. "
                            "Install it with: pip install boto3"
                        )
                    
                    boto_config = Config(
                        region_name=self.config.aws_region,
                        retries={
                            'max_attempts': 0,
                            'mode': 'standard'
                        },
                        connect_timeout=30,
                        read_timeout=self.config.request_timeout,
                    )
                    
                    session_kwargs = {}
                    
                    if self.config.aws_access_key_id:
                        session_kwargs['aws_access_key_id'] = self.config.aws_access_key_id.get_secret_value()
                    if self.config.aws_secret_access_key:
                        session_kwargs['aws_secret_access_key'] = self.config.aws_secret_access_key.get_secret_value()
                    if self.config.aws_session_token:
                        session_kwargs['aws_session_token'] = self.config.aws_session_token.get_secret_value()
                    
                    if session_kwargs:
                        session = boto3.Session(**session_kwargs)
                        self._client = session.client(
                            'bedrock-runtime',
                            config=boto_config
                        )
                    else:
                        self._client = boto3.client(
                            'bedrock-runtime',
                            config=boto_config,
                            region_name=self.config.aws_region
                        )
                    
                    self._initialized = True
                    logger.info(
                        f"Initialized Bedrock client",
                        extra={"extra_fields": {
                            "region": self.config.aws_region,
                            "model_id": self.config.model_id
                        }}
                    )
        
        return self._client
    
    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[ConversationContext] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Invoke Claude via Bedrock
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            context: Optional conversation context for multi-turn
            **kwargs: Additional parameters to override config
            
        Returns:
            LLMResponse with generated content
        """
        client = self._get_client()
        
        # Build messages
        messages = []
        
        if context and context.messages:
            messages.extend(context.get_messages_for_api())
        
        # Add current prompt as user message
        messages.append({"role": "user", "content": prompt})
        
        # Use context system prompt or provided one
        effective_system_prompt = system_prompt or (context.system_prompt if context else None)
        
        # Build request body
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "messages": messages,
        }
        
        if effective_system_prompt:
            request_body["system"] = effective_system_prompt
        
        start_time = time.time()
        
        try:
            response = client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            response_body = json.loads(response['body'].read())
            
            # Extract content
            content = ""
            if response_body.get("content"):
                for block in response_body["content"]:
                    if block.get("type") == "text":
                        content += block.get("text", "")
            
            # Extract token usage
            usage = response_body.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
            # Record metrics
            SQLAgentMetrics.record_llm_call(
                duration=latency_ms / 1000,
                model_id=self.config.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            llm_response = LLMResponse(
                content=content,
                model_id=self.config.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason=response_body.get("stop_reason"),
                latency_ms=latency_ms,
                raw_response=response_body,
            )
            
            # Update context if provided
            if context:
                context.add_user_message(prompt)
                context.add_assistant_message(content)
            
            logger.debug(
                f"LLM invocation successful",
                extra={"extra_fields": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms
                }}
            )
            
            return llm_response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                f"LLM invocation failed: {str(e)}",
                extra={"extra_fields": {"latency_ms": latency_ms}}
            )
            raise LLMError(
                message=f"Bedrock Claude invocation failed: {str(e)}",
                model_id=self.config.model_id,
                original_error=e
            )
    
    def invoke_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[ConversationContext] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Invoke Claude with automatic retry on failure"""
        retries = max_retries if max_retries is not None else self.config.retry_attempts
        last_error = None
        
        for attempt in range(retries):
            try:
                return self.invoke(prompt, system_prompt, context, **kwargs)
            except LLMError as e:
                last_error = e
                
                if not self._is_retryable_error(e):
                    raise
                
                if attempt < retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM invocation failed, retrying in {delay}s",
                        extra={"extra_fields": {
                            "attempt": attempt + 1,
                            "max_retries": retries,
                            "error": str(e)
                        }}
                    )
                    time.sleep(delay)
        
        raise LLMError(
            message=f"LLM invocation failed after {retries} attempts",
            model_id=self.config.model_id,
            original_error=last_error
        )
    
    def _is_retryable_error(self, error: LLMError) -> bool:
        """Check if error is retryable"""
        error_str = str(error.original_error).lower() if error.original_error else ""
        
        retryable_patterns = [
            "throttling",
            "rate limit",
            "too many requests",
            "service unavailable",
            "timeout",
            "connection",
            "temporary",
        ]
        
        return any(pattern in error_str for pattern in retryable_patterns)
    
    def health_check(self) -> bool:
        """Check if the client can connect to Bedrock"""
        try:
            response = self.invoke(
                prompt="Say 'OK'",
                max_tokens=10
            )
            return bool(response.content)
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


# ============================================================================
# Client Factory
# ============================================================================

class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    _clients: Dict[str, BaseLLMClient] = {}
    _lock = threading.Lock()
    _use_strands: Optional[bool] = None
    
    @classmethod
    def _check_strands_available(cls) -> bool:
        """Check if Strands SDK is available"""
        if cls._use_strands is None:
            try:
                from strands import Agent
                from strands.models import BedrockModel
                cls._use_strands = True
                logger.info("Strands Agents SDK available - using StrandsAgentClient")
            except ImportError:
                cls._use_strands = False
                logger.info("Strands Agents SDK not available - using BedrockClaudeClient")
        return cls._use_strands
    
    @classmethod
    def get_client(
        cls, 
        config: LLMConfig,
        force_strands: bool = False,
        force_boto3: bool = False
    ) -> BaseLLMClient:
        """
        Get or create LLM client instance
        
        Args:
            config: LLM configuration
            force_strands: Force use of Strands SDK (raises if not available)
            force_boto3: Force use of direct boto3 client
            
        Returns:
            LLM client instance
        """
        # Determine client type
        if force_boto3:
            client_type = "boto3"
        elif force_strands or cls._check_strands_available():
            client_type = "strands"
        else:
            client_type = "boto3"
        
        # Create unique key for this configuration
        key = f"{config.provider}_{config.model_id}_{config.aws_region}_{client_type}"
        
        if key not in cls._clients:
            with cls._lock:
                if key not in cls._clients:
                    if client_type == "strands":
                        cls._clients[key] = StrandsAgentClient(config)
                    else:
                        cls._clients[key] = BedrockClaudeClient(config)
        
        return cls._clients[key]
    
    @classmethod
    def clear_clients(cls) -> None:
        """Clear all cached clients"""
        with cls._lock:
            cls._clients.clear()


# ============================================================================
# Convenience Functions
# ============================================================================

def get_llm_client(config: LLMConfig, prefer_strands: bool = True) -> BaseLLMClient:
    """
    Get LLM client instance
    
    Args:
        config: LLM configuration
        prefer_strands: Whether to prefer Strands SDK if available
        
    Returns:
        LLM client instance
    """
    return LLMClientFactory.get_client(
        config, 
        force_strands=prefer_strands and LLMClientFactory._check_strands_available()
    )


def create_strands_agent(
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    system_prompt: Optional[str] = None,
    tools: Optional[List] = None,
    region: str = "us-east-1",
    temperature: float = 0.1
):
    """
    Create a Strands agent directly
    
    Args:
        model_id: Bedrock model ID
        system_prompt: System prompt for the agent
        tools: List of tools for the agent
        region: AWS region
        temperature: Temperature for generation
        
    Returns:
        Strands Agent instance
    """
    try:
        from strands import Agent
        from strands.models import BedrockModel
    except ImportError:
        raise ImportError(
            "strands-agents is required. Install with: pip install strands-agents"
        )
    
    model = BedrockModel(
        model_id=model_id,
        region_name=region,
        temperature=temperature,
    )
    
    agent_kwargs = {"model": model}
    
    if system_prompt:
        agent_kwargs["system_prompt"] = system_prompt
    
    if tools:
        agent_kwargs["tools"] = tools
    
    return Agent(**agent_kwargs)
