"""
Base Agent Module
Defines abstract base class for all agents in the system
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, Optional, TypeVar
import threading
import uuid

from ..config import AgentConfig, LLMConfig
from ..llm_client import BaseLLMClient, ConversationContext, get_llm_client
from ..utils import get_logger, set_agent_context, log_context

logger = get_logger(__name__)

# Type variables for generic agent input/output
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


@dataclass
class AgentMetadata:
    """Metadata for agent execution"""
    agent_name: str
    agent_version: str = "1.0.0"
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    iteration_count: int = 0
    token_usage: Dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "execution_id": self.execution_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "iteration_count": self.iteration_count,
            "token_usage": self.token_usage,
        }


class BaseAgent(ABC, Generic[InputType, OutputType]):
    """
    Abstract base class for agents
    
    Implements Template Method pattern for agent execution lifecycle
    """
    
    def __init__(
        self,
        name: str,
        llm_config: LLMConfig,
        agent_config: AgentConfig,
    ):
        self.name = name
        self.llm_config = llm_config
        self.agent_config = agent_config
        self._llm_client: Optional[BaseLLMClient] = None
        self._lock = threading.Lock()
        self._metadata = AgentMetadata(agent_name=name)
    
    @property
    def llm_client(self) -> BaseLLMClient:
        """Get or create LLM client (lazy initialization)"""
        if self._llm_client is None:
            with self._lock:
                if self._llm_client is None:
                    self._llm_client = get_llm_client(self.llm_config)
        return self._llm_client
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass
    
    @abstractmethod
    def _prepare_prompt(self, input_data: InputType, context: Dict[str, Any]) -> str:
        """
        Prepare the prompt for LLM invocation
        
        Args:
            input_data: Input data for the agent
            context: Additional context for prompt generation
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def _parse_response(self, response: str, input_data: InputType) -> OutputType:
        """
        Parse LLM response into output type
        
        Args:
            response: Raw LLM response
            input_data: Original input data for reference
            
        Returns:
            Parsed output
        """
        pass
    
    @abstractmethod
    def _validate_output(self, output: OutputType) -> bool:
        """
        Validate the agent output
        
        Args:
            output: Output to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def execute(
        self,
        input_data: InputType,
        context: Optional[Dict[str, Any]] = None,
        conversation: Optional[ConversationContext] = None,
    ) -> OutputType:
        """
        Execute the agent pipeline
        
        Template method that defines the execution lifecycle:
        1. Pre-process input
        2. Prepare prompt
        3. Invoke LLM
        4. Parse response
        5. Validate output
        6. Post-process output
        
        Args:
            input_data: Input data for the agent
            context: Additional context
            conversation: Optional conversation context for multi-turn
            
        Returns:
            Agent output
        """
        context = context or {}
        self._metadata.started_at = datetime.utcnow()
        self._metadata.iteration_count += 1
        
        with log_context(agent_name=self.name):
            logger.info(f"Starting agent execution", extra={"extra_fields": {
                "execution_id": self._metadata.execution_id,
                "iteration": self._metadata.iteration_count
            }})
            
            try:
                # Pre-process
                processed_input = self._pre_process(input_data, context)
                
                # Prepare prompt
                prompt = self._prepare_prompt(processed_input, context)
                
                # Create or use conversation context
                conv_context = conversation or ConversationContext(
                    system_prompt=self.system_prompt
                )
                
                # Invoke LLM
                llm_response = self.llm_client.invoke_with_retry(
                    prompt=prompt,
                    system_prompt=self.system_prompt if not conversation else None,
                    context=conv_context if conversation else None,
                )
                
                # Track token usage
                self._metadata.token_usage["input"] += llm_response.input_tokens
                self._metadata.token_usage["output"] += llm_response.output_tokens
                
                # Parse response
                output = self._parse_response(llm_response.content, processed_input)
                
                # Validate output
                if not self._validate_output(output):
                    logger.warning(f"Output validation failed")
                    output = self._handle_validation_failure(output, input_data, context)
                
                # Post-process
                final_output = self._post_process(output, context)
                
                self._metadata.completed_at = datetime.utcnow()
                
                logger.info(f"Agent execution completed", extra={"extra_fields": {
                    "execution_id": self._metadata.execution_id,
                    "duration_ms": (self._metadata.completed_at - self._metadata.started_at).total_seconds() * 1000
                }})
                
                return final_output
                
            except Exception as e:
                self._metadata.completed_at = datetime.utcnow()
                logger.error(f"Agent execution failed: {str(e)}")
                raise
    
    def _pre_process(self, input_data: InputType, context: Dict[str, Any]) -> InputType:
        """
        Pre-process input data (override for custom behavior)
        
        Default implementation returns input unchanged
        """
        return input_data
    
    def _post_process(self, output: OutputType, context: Dict[str, Any]) -> OutputType:
        """
        Post-process output data (override for custom behavior)
        
        Default implementation returns output unchanged
        """
        return output
    
    def _handle_validation_failure(
        self,
        output: OutputType,
        input_data: InputType,
        context: Dict[str, Any]
    ) -> OutputType:
        """
        Handle validation failure (override for custom behavior)
        
        Default implementation returns output unchanged
        """
        return output
    
    def get_metadata(self) -> AgentMetadata:
        """Get agent execution metadata"""
        return self._metadata
    
    def reset_metadata(self) -> None:
        """Reset metadata for new execution"""
        self._metadata = AgentMetadata(agent_name=self.name)
