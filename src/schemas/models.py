"""
Schema Definitions for SQL Agent System
Defines request/response models and agent state
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class RequestStatus(str, Enum):
    """Status of a SQL generation request"""
    PENDING = "pending"
    GENERATING = "generating"
    VALIDATING = "validating"
    HEALING = "healing"
    COMPLETED = "completed"
    FAILED = "failed"


class SQLQueryType(str, Enum):
    """Types of SQL queries"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    ALTER = "ALTER"
    DROP = "DROP"
    UNKNOWN = "UNKNOWN"


@dataclass
class SQLGenerationRequest:
    """Request for SQL generation"""
    user_query: str
    database_type: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    
    # Optional context
    table_hints: Optional[List[str]] = None
    column_hints: Optional[List[str]] = None
    additional_context: Optional[str] = None
    
    # Configuration overrides
    max_iterations: Optional[int] = None
    include_explanation: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "user_query": self.user_query,
            "database_type": self.database_type,
            "table_hints": self.table_hints,
            "column_hints": self.column_hints,
            "additional_context": self.additional_context,
            "max_iterations": self.max_iterations,
            "include_explanation": self.include_explanation,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SQLValidationIssue:
    """Represents a validation issue found in SQL"""
    issue_type: str  # "syntax", "schema", "semantic", "security"
    severity: str  # "error", "warning", "info"
    message: str
    location: Optional[str] = None  # e.g., "line 3, column 15"
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
        }


@dataclass
class SQLGenerationResult:
    """Result of SQL generation"""
    sql: str
    query_type: SQLQueryType
    explanation: Optional[str] = None
    confidence_score: float = 0.0
    tables_used: List[str] = field(default_factory=list)
    columns_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sql": self.sql,
            "query_type": self.query_type.value,
            "explanation": self.explanation,
            "confidence_score": self.confidence_score,
            "tables_used": self.tables_used,
            "columns_used": self.columns_used,
        }


@dataclass
class SQLValidationResult:
    """Result of SQL validation"""
    is_valid: bool
    issues: List[SQLValidationIssue] = field(default_factory=list)
    validated_sql: Optional[str] = None
    execution_plan: Optional[str] = None
    estimated_cost: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "validated_sql": self.validated_sql,
            "execution_plan": self.execution_plan,
            "estimated_cost": self.estimated_cost,
        }
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues"""
        return any(issue.severity == "error" for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues"""
        return any(issue.severity == "warning" for issue in self.issues)


@dataclass
class HealingAttempt:
    """Record of a self-healing attempt"""
    iteration: int
    original_sql: str
    healed_sql: str
    issues_addressed: List[str]
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "original_sql": self.original_sql,
            "healed_sql": self.healed_sql,
            "issues_addressed": self.issues_addressed,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SQLGenerationResponse:
    """Complete response from SQL generation pipeline"""
    request_id: str
    status: RequestStatus
    
    # Results
    final_sql: Optional[str] = None
    generation_result: Optional[SQLGenerationResult] = None
    validation_result: Optional[SQLValidationResult] = None
    
    # Healing history
    healing_attempts: List[HealingAttempt] = field(default_factory=list)
    total_iterations: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_ms: float = 0.0
    
    # Error information
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "final_sql": self.final_sql,
            "generation_result": self.generation_result.to_dict() if self.generation_result else None,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "healing_attempts": [attempt.to_dict() for attempt in self.healing_attempts],
            "total_iterations": self.total_iterations,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "error_message": self.error_message,
            "error_details": self.error_details,
        }
    
    @property
    def is_successful(self) -> bool:
        """Check if request completed successfully"""
        return self.status == RequestStatus.COMPLETED and self.final_sql is not None


@dataclass
class AgentState:
    """Thread-safe state management for agent processing"""
    request_id: str
    status: RequestStatus = RequestStatus.PENDING
    current_iteration: int = 0
    max_iterations: int = 5
    
    # Generation state
    user_query: str = ""
    schema_context: str = ""
    dialect_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Current SQL state
    current_sql: Optional[str] = None
    current_issues: List[SQLValidationIssue] = field(default_factory=list)
    
    # History
    generation_history: List[str] = field(default_factory=list)
    validation_history: List[SQLValidationResult] = field(default_factory=list)
    healing_history: List[HealingAttempt] = field(default_factory=list)
    
    # Timing
    started_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Error tracking
    last_error: Optional[str] = None
    error_count: int = 0
    
    def start(self) -> None:
        """Mark processing as started"""
        self.started_at = datetime.utcnow()
        self.last_updated = self.started_at
        self.status = RequestStatus.GENERATING
    
    def update_status(self, status: RequestStatus) -> None:
        """Update status and timestamp"""
        self.status = status
        self.last_updated = datetime.utcnow()
    
    def increment_iteration(self) -> bool:
        """
        Increment iteration counter
        
        Returns:
            True if within max iterations, False otherwise
        """
        self.current_iteration += 1
        self.last_updated = datetime.utcnow()
        return self.current_iteration <= self.max_iterations
    
    def record_generation(self, sql: str) -> None:
        """Record a SQL generation"""
        self.current_sql = sql
        self.generation_history.append(sql)
        self.last_updated = datetime.utcnow()
    
    def record_validation(self, result: SQLValidationResult) -> None:
        """Record a validation result"""
        self.current_issues = result.issues
        self.validation_history.append(result)
        self.last_updated = datetime.utcnow()
    
    def record_healing(self, attempt: HealingAttempt) -> None:
        """Record a healing attempt"""
        self.healing_history.append(attempt)
        if attempt.success:
            self.current_sql = attempt.healed_sql
        self.last_updated = datetime.utcnow()
    
    def record_error(self, error: str) -> None:
        """Record an error"""
        self.last_error = error
        self.error_count += 1
        self.last_updated = datetime.utcnow()
    
    def get_elapsed_time_ms(self) -> float:
        """Get elapsed time since start in milliseconds"""
        if not self.started_at:
            return 0.0
        end_time = self.last_updated or datetime.utcnow()
        return (end_time - self.started_at).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "current_sql": self.current_sql,
            "current_issues": [issue.to_dict() for issue in self.current_issues],
            "generation_count": len(self.generation_history),
            "validation_count": len(self.validation_history),
            "healing_count": len(self.healing_history),
            "elapsed_ms": self.get_elapsed_time_ms(),
            "error_count": self.error_count,
            "last_error": self.last_error,
        }


# SQL Query Type Detection
def detect_query_type(sql: str) -> SQLQueryType:
    """Detect the type of SQL query"""
    sql_upper = sql.strip().upper()
    
    if sql_upper.startswith("SELECT"):
        return SQLQueryType.SELECT
    elif sql_upper.startswith("INSERT"):
        return SQLQueryType.INSERT
    elif sql_upper.startswith("UPDATE"):
        return SQLQueryType.UPDATE
    elif sql_upper.startswith("DELETE"):
        return SQLQueryType.DELETE
    elif sql_upper.startswith("CREATE"):
        return SQLQueryType.CREATE
    elif sql_upper.startswith("ALTER"):
        return SQLQueryType.ALTER
    elif sql_upper.startswith("DROP"):
        return SQLQueryType.DROP
    else:
        return SQLQueryType.UNKNOWN
