"""
Utilities Package for SQL Agent System
"""
from .logging import (
    setup_logging,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    set_request_id,
    get_request_id,
    set_agent_context,
    clear_context,
    log_context,
    log_operation,
)

from .errors import (
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    SQLAgentError,
    DatabaseConnectionError,
    SchemaError,
    SQLSyntaxError,
    SQLSemanticError,
    LLMError,
    ValidationError,
    ConfigurationError,
    TimeoutError,
    MaxRetriesExceededError,
    SelfHealingFailedError,
    format_error_for_llm,
    classify_database_error,
)

from .metrics import (
    MetricsCollector,
    get_metrics_collector,
    counter,
    gauge,
    histogram,
    timer,
    time_operation,
    SQLAgentMetrics,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "set_request_id",
    "get_request_id",
    "set_agent_context",
    "clear_context",
    "log_context",
    "log_operation",
    # Errors
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    "SQLAgentError",
    "DatabaseConnectionError",
    "SchemaError",
    "SQLSyntaxError",
    "SQLSemanticError",
    "LLMError",
    "ValidationError",
    "ConfigurationError",
    "TimeoutError",
    "MaxRetriesExceededError",
    "SelfHealingFailedError",
    "format_error_for_llm",
    "classify_database_error",
    # Metrics
    "MetricsCollector",
    "get_metrics_collector",
    "counter",
    "gauge",
    "histogram",
    "timer",
    "time_operation",
    "SQLAgentMetrics",
]
