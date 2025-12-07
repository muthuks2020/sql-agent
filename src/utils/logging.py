"""
Logging Utility Module for SQL Agent System
Provides structured logging with correlation IDs and context
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, Optional

# Thread-local storage for request context
_thread_local = threading.local()


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for production environments"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available
        correlation_id = getattr(_thread_local, 'correlation_id', None)
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        # Add request ID if available
        request_id = getattr(_thread_local, 'request_id', None)
        if request_id:
            log_entry["request_id"] = request_id
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter for development"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Build prefix with context
        prefix_parts = []
        correlation_id = getattr(_thread_local, 'correlation_id', None)
        if correlation_id:
            prefix_parts.append(f"[{correlation_id[:8]}]")
        
        request_id = getattr(_thread_local, 'request_id', None)
        if request_id:
            prefix_parts.append(f"[req:{request_id[:8]}]")
        
        prefix = " ".join(prefix_parts)
        if prefix:
            prefix = f"{prefix} "
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        formatted = (
            f"{color}{timestamp} | {record.levelname:8s}{self.RESET} | "
            f"{record.name:30s} | {prefix}{record.getMessage()}"
        )
        
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that includes context information"""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        extra = kwargs.get('extra', {})
        
        # Add thread-local context
        if hasattr(_thread_local, 'correlation_id'):
            extra['correlation_id'] = _thread_local.correlation_id
        if hasattr(_thread_local, 'request_id'):
            extra['request_id'] = _thread_local.request_id
        if hasattr(_thread_local, 'agent_name'):
            extra['agent_name'] = _thread_local.agent_name
        
        kwargs['extra'] = extra
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON structured logging (for production)
        log_file: Optional file path for logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING
    for logger_name in ['boto3', 'botocore', 'urllib3', 'asyncio']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> ContextLogger:
    """Get a context-aware logger"""
    base_logger = logging.getLogger(name)
    return ContextLogger(base_logger, {})


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for the current thread"""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    _thread_local.correlation_id = correlation_id
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get correlation ID for the current thread"""
    return getattr(_thread_local, 'correlation_id', None)


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID for the current thread"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    _thread_local.request_id = request_id
    return request_id


def get_request_id() -> Optional[str]:
    """Get request ID for the current thread"""
    return getattr(_thread_local, 'request_id', None)


def set_agent_context(agent_name: str) -> None:
    """Set agent name context for the current thread"""
    _thread_local.agent_name = agent_name


def clear_context() -> None:
    """Clear all thread-local context"""
    for attr in ['correlation_id', 'request_id', 'agent_name']:
        if hasattr(_thread_local, attr):
            delattr(_thread_local, attr)


@contextmanager
def log_context(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    agent_name: Optional[str] = None
) -> Generator[None, None, None]:
    """
    Context manager for setting logging context
    
    Usage:
        with log_context(correlation_id="abc123", agent_name="sql_generator"):
            logger.info("Processing request")
    """
    old_correlation_id = getattr(_thread_local, 'correlation_id', None)
    old_request_id = getattr(_thread_local, 'request_id', None)
    old_agent_name = getattr(_thread_local, 'agent_name', None)
    
    try:
        if correlation_id:
            _thread_local.correlation_id = correlation_id
        if request_id:
            _thread_local.request_id = request_id
        if agent_name:
            _thread_local.agent_name = agent_name
        yield
    finally:
        if old_correlation_id:
            _thread_local.correlation_id = old_correlation_id
        elif hasattr(_thread_local, 'correlation_id'):
            delattr(_thread_local, 'correlation_id')
        
        if old_request_id:
            _thread_local.request_id = old_request_id
        elif hasattr(_thread_local, 'request_id'):
            delattr(_thread_local, 'request_id')
        
        if old_agent_name:
            _thread_local.agent_name = old_agent_name
        elif hasattr(_thread_local, 'agent_name'):
            delattr(_thread_local, 'agent_name')


@contextmanager
def log_operation(
    logger: ContextLogger,
    operation: str,
    **extra_fields: Any
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for logging operation timing
    
    Usage:
        with log_operation(logger, "sql_generation", query_type="SELECT") as ctx:
            result = generate_sql()
            ctx['rows'] = len(result)
    """
    start_time = time.time()
    context: Dict[str, Any] = {"operation": operation, **extra_fields}
    
    logger.info(f"Starting {operation}", extra={"extra_fields": context})
    
    try:
        yield context
        elapsed = time.time() - start_time
        context['duration_ms'] = round(elapsed * 1000, 2)
        context['status'] = 'success'
        logger.info(f"Completed {operation}", extra={"extra_fields": context})
    except Exception as e:
        elapsed = time.time() - start_time
        context['duration_ms'] = round(elapsed * 1000, 2)
        context['status'] = 'error'
        context['error'] = str(e)
        context['error_type'] = type(e).__name__
        logger.error(f"Failed {operation}", extra={"extra_fields": context}, exc_info=True)
        raise
