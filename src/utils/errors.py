"""
Error Handling Module for SQL Agent System
Defines custom exceptions and error handling utilities
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification"""
    DATABASE = "database"
    SCHEMA = "schema"
    SQL_SYNTAX = "sql_syntax"
    SQL_SEMANTIC = "sql_semantic"
    LLM = "llm"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INTERNAL = "internal"


@dataclass
class ErrorContext:
    """Additional context for errors"""
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    agent_name: Optional[str] = None
    sql_query: Optional[str] = None
    original_query: Optional[str] = None
    database_type: Optional[str] = None
    iteration: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "agent_name": self.agent_name,
            "sql_query": self.sql_query,
            "original_query": self.original_query,
            "database_type": self.database_type,
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class SQLAgentError(Exception):
    """Base exception for SQL Agent System"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        recoverable: bool = True,
        suggestions: Optional[List[str]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.suggestions = suggestions or []
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "suggestions": self.suggestions,
            "context": self.context.to_dict(),
            "original_error": str(self.original_error) if self.original_error else None,
        }
    
    def __str__(self) -> str:
        return f"[{self.category.value}] {self.message}"


class DatabaseConnectionError(SQLAgentError):
    """Database connection failure"""
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            context=context,
            recoverable=True,
            suggestions=[
                "Check database host and port configuration",
                "Verify database credentials",
                "Ensure database server is running",
                "Check network connectivity",
                "Verify SSL configuration if enabled"
            ],
            original_error=original_error
        )


class SchemaError(SQLAgentError):
    """Schema-related errors"""
    
    def __init__(
        self,
        message: str,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        suggestions = ["Verify table/column names exist in the database"]
        if table_name:
            suggestions.append(f"Check if table '{table_name}' exists")
        if column_name:
            suggestions.append(f"Check if column '{column_name}' exists")
        
        super().__init__(
            message=message,
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recoverable=True,
            suggestions=suggestions,
            original_error=original_error
        )
        self.table_name = table_name
        self.column_name = column_name


class SQLSyntaxError(SQLAgentError):
    """SQL syntax errors"""
    
    def __init__(
        self,
        message: str,
        sql_query: Optional[str] = None,
        position: Optional[int] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        if context and sql_query:
            context.sql_query = sql_query
        
        suggestions = [
            "Review SQL syntax for the target database",
            "Check for missing or extra parentheses",
            "Verify keyword usage and spelling",
        ]
        if position:
            suggestions.append(f"Error detected near position {position}")
        
        super().__init__(
            message=message,
            category=ErrorCategory.SQL_SYNTAX,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recoverable=True,
            suggestions=suggestions,
            original_error=original_error
        )
        self.sql_query = sql_query
        self.position = position


class SQLSemanticError(SQLAgentError):
    """SQL semantic/logical errors"""
    
    def __init__(
        self,
        message: str,
        sql_query: Optional[str] = None,
        issue_type: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        if context and sql_query:
            context.sql_query = sql_query
        
        suggestions = [
            "Review the query logic against requirements",
            "Verify JOIN conditions",
            "Check WHERE clause conditions",
            "Validate GROUP BY and HAVING clauses",
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.SQL_SEMANTIC,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recoverable=True,
            suggestions=suggestions,
            original_error=original_error
        )
        self.sql_query = sql_query
        self.issue_type = issue_type


class LLMError(SQLAgentError):
    """LLM-related errors"""
    
    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.LLM,
            severity=ErrorSeverity.HIGH,
            context=context,
            recoverable=True,
            suggestions=[
                "Check AWS credentials and permissions",
                "Verify Bedrock model availability",
                "Review request payload format",
                "Check for rate limiting",
            ],
            original_error=original_error
        )
        self.model_id = model_id


class ValidationError(SQLAgentError):
    """Validation errors"""
    
    def __init__(
        self,
        message: str,
        validation_type: str = "general",
        failed_rules: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        suggestions = ["Review validation rules", "Check input data format"]
        if failed_rules:
            suggestions.extend([f"Fix validation: {rule}" for rule in failed_rules])
        
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recoverable=True,
            suggestions=suggestions,
            original_error=original_error
        )
        self.validation_type = validation_type
        self.failed_rules = failed_rules or []


class ConfigurationError(SQLAgentError):
    """Configuration errors"""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        suggestions = ["Review configuration settings"]
        if config_key:
            suggestions.append(f"Check configuration for key: {config_key}")
        
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recoverable=False,
            suggestions=suggestions,
            original_error=original_error
        )
        self.config_key = config_key


class TimeoutError(SQLAgentError):
    """Timeout errors"""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        suggestions = [
            "Increase timeout configuration",
            "Optimize query complexity",
            "Check system resources",
        ]
        if operation:
            suggestions.append(f"Review '{operation}' operation performance")
        
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recoverable=True,
            suggestions=suggestions,
            original_error=original_error
        )
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class MaxRetriesExceededError(SQLAgentError):
    """Maximum retries exceeded"""
    
    def __init__(
        self,
        message: str,
        max_retries: int,
        last_error: Optional[Exception] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.INTERNAL,
            severity=ErrorSeverity.HIGH,
            context=context,
            recoverable=False,
            suggestions=[
                f"Maximum retries ({max_retries}) exceeded",
                "Review underlying error cause",
                "Consider increasing retry limit",
            ],
            original_error=last_error
        )
        self.max_retries = max_retries


class SelfHealingFailedError(SQLAgentError):
    """Self-healing process failed"""
    
    def __init__(
        self,
        message: str,
        healing_attempts: int,
        original_sql: Optional[str] = None,
        final_sql: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recoverable=False,
            suggestions=[
                f"Self-healing failed after {healing_attempts} attempts",
                "Review original query requirements",
                "Check schema compatibility",
                "Consider manual SQL review",
            ],
            original_error=original_error
        )
        self.healing_attempts = healing_attempts
        self.original_sql = original_sql
        self.final_sql = final_sql


def format_error_for_llm(error: SQLAgentError) -> str:
    """Format error for LLM consumption during self-healing"""
    lines = [
        f"Error Type: {error.__class__.__name__}",
        f"Category: {error.category.value}",
        f"Message: {error.message}",
    ]
    
    if error.suggestions:
        lines.append("Suggestions:")
        for suggestion in error.suggestions:
            lines.append(f"  - {suggestion}")
    
    if error.context.sql_query:
        lines.append(f"SQL Query: {error.context.sql_query}")
    
    if error.original_error:
        lines.append(f"Original Error: {str(error.original_error)}")
    
    return "\n".join(lines)


def classify_database_error(error: Exception, db_type: str) -> SQLAgentError:
    """Classify a raw database error into appropriate SQLAgentError subclass"""
    error_str = str(error).lower()
    
    # Connection errors
    if any(term in error_str for term in ['connect', 'connection', 'refused', 'timeout', 'host']):
        return DatabaseConnectionError(
            message=str(error),
            original_error=error
        )
    
    # Syntax errors
    if any(term in error_str for term in ['syntax', 'parse', 'unexpected']):
        return SQLSyntaxError(
            message=str(error),
            original_error=error
        )
    
    # Schema errors
    if any(term in error_str for term in ['table', 'column', 'not found', 'does not exist', 'unknown']):
        return SchemaError(
            message=str(error),
            original_error=error
        )
    
    # Default to internal error
    return SQLAgentError(
        message=str(error),
        category=ErrorCategory.DATABASE,
        original_error=error
    )
