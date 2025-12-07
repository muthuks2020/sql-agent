"""
Base Database Adapter Module
Defines abstract interface for database adapters using Template Method pattern
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type
import threading

from ..config import DatabaseConfig, DatabaseType


@dataclass
class TableSchema:
    """Schema information for a database table"""
    name: str
    columns: List["ColumnSchema"]
    primary_key: List[str] = field(default_factory=list)
    foreign_keys: List["ForeignKeySchema"] = field(default_factory=list)
    indexes: List["IndexSchema"] = field(default_factory=list)
    row_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "primary_key": self.primary_key,
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys],
            "indexes": [idx.to_dict() for idx in self.indexes],
            "row_count": self.row_count,
        }
    
    def to_ddl_string(self) -> str:
        """Generate DDL-like string representation for LLM context"""
        lines = [f"TABLE {self.name} ("]
        
        for col in self.columns:
            nullable = "" if col.nullable else " NOT NULL"
            default = f" DEFAULT {col.default_value}" if col.default_value else ""
            lines.append(f"    {col.name} {col.data_type}{nullable}{default},")
        
        if self.primary_key:
            lines.append(f"    PRIMARY KEY ({', '.join(self.primary_key)}),")
        
        for fk in self.foreign_keys:
            lines.append(
                f"    FOREIGN KEY ({', '.join(fk.columns)}) "
                f"REFERENCES {fk.referenced_table}({', '.join(fk.referenced_columns)}),"
            )
        
        # Remove trailing comma from last line
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]
        
        lines.append(");")
        
        return "\n".join(lines)


@dataclass
class ColumnSchema:
    """Schema information for a database column"""
    name: str
    data_type: str
    nullable: bool = True
    default_value: Optional[str] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_unique: bool = False
    is_indexed: bool = False
    comment: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "default_value": self.default_value,
            "max_length": self.max_length,
            "precision": self.precision,
            "scale": self.scale,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "is_unique": self.is_unique,
            "is_indexed": self.is_indexed,
            "comment": self.comment,
        }


@dataclass
class ForeignKeySchema:
    """Foreign key relationship information"""
    name: str
    columns: List[str]
    referenced_table: str
    referenced_columns: List[str]
    on_delete: Optional[str] = None
    on_update: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": self.columns,
            "referenced_table": self.referenced_table,
            "referenced_columns": self.referenced_columns,
            "on_delete": self.on_delete,
            "on_update": self.on_update,
        }


@dataclass
class IndexSchema:
    """Index information"""
    name: str
    columns: List[str]
    is_unique: bool = False
    is_primary: bool = False
    index_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": self.columns,
            "is_unique": self.is_unique,
            "is_primary": self.is_primary,
            "index_type": self.index_type,
        }


@dataclass
class DatabaseSchema:
    """Complete database schema"""
    database_name: str
    database_type: DatabaseType
    tables: Dict[str, TableSchema] = field(default_factory=dict)
    views: Dict[str, TableSchema] = field(default_factory=dict)
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "database_name": self.database_name,
            "database_type": self.database_type.value,
            "tables": {k: v.to_dict() for k, v in self.tables.items()},
            "views": {k: v.to_dict() for k, v in self.views.items()},
            "retrieved_at": self.retrieved_at.isoformat(),
        }
    
    def to_schema_string(self, include_views: bool = True) -> str:
        """Generate schema string for LLM context"""
        lines = [f"-- Database: {self.database_name} ({self.database_type.value})"]
        lines.append("")
        
        for table_name, table in sorted(self.tables.items()):
            lines.append(table.to_ddl_string())
            lines.append("")
        
        if include_views and self.views:
            lines.append("-- Views:")
            for view_name, view in sorted(self.views.items()):
                lines.append(view.to_ddl_string())
                lines.append("")
        
        return "\n".join(lines)
    
    def get_table_names(self) -> List[str]:
        """Get all table names"""
        return list(self.tables.keys())
    
    def get_table(self, name: str) -> Optional[TableSchema]:
        """Get table schema by name (case-insensitive)"""
        # Try exact match first
        if name in self.tables:
            return self.tables[name]
        
        # Try case-insensitive match
        name_lower = name.lower()
        for table_name, table in self.tables.items():
            if table_name.lower() == name_lower:
                return table
        
        return None


@dataclass
class QueryResult:
    """Result of a SQL query execution"""
    success: bool
    columns: List[str] = field(default_factory=list)
    rows: List[Tuple[Any, ...]] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "columns": self.columns,
            "rows": [list(row) for row in self.rows],
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "warnings": self.warnings,
        }


class BaseDatabaseAdapter(ABC):
    """
    Abstract base class for database adapters
    
    Implements Template Method pattern for database operations
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connection = None
        self._schema_cache: Optional[DatabaseSchema] = None
        self._schema_cache_time: Optional[datetime] = None
        self._lock = threading.Lock()
    
    @property
    @abstractmethod
    def database_type(self) -> DatabaseType:
        """Return the database type"""
        pass
    
    @property
    @abstractmethod
    def dialect_name(self) -> str:
        """Return SQL dialect name for this database"""
        pass
    
    @abstractmethod
    def connect(self) -> None:
        """Establish database connection"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active"""
        pass
    
    @abstractmethod
    def execute_query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a SQL query and return results"""
        pass
    
    @abstractmethod
    def _fetch_tables(self) -> List[TableSchema]:
        """Fetch table schemas from database"""
        pass
    
    @abstractmethod
    def _fetch_views(self) -> List[TableSchema]:
        """Fetch view schemas from database"""
        pass
    
    @abstractmethod
    def validate_sql_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL syntax without execution
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def get_sql_dialect_hints(self) -> Dict[str, Any]:
        """
        Get SQL dialect-specific hints for the LLM
        
        Returns:
            Dictionary with dialect information for SQL generation
        """
        pass
    
    def get_schema(self, force_refresh: bool = False) -> DatabaseSchema:
        """
        Get database schema with optional caching
        
        Args:
            force_refresh: Force schema refresh from database
            
        Returns:
            DatabaseSchema object
        """
        cache_ttl = self.config.query_timeout
        
        with self._lock:
            # Check cache validity
            if (
                not force_refresh
                and self._schema_cache is not None
                and self._schema_cache_time is not None
            ):
                age = (datetime.utcnow() - self._schema_cache_time).total_seconds()
                if age < cache_ttl:
                    return self._schema_cache
            
            # Fetch fresh schema
            tables = self._fetch_tables()
            views = self._fetch_views()
            
            self._schema_cache = DatabaseSchema(
                database_name=self.config.database,
                database_type=self.database_type,
                tables={t.name: t for t in tables},
                views={v.name: v for v in views},
            )
            self._schema_cache_time = datetime.utcnow()
            
            return self._schema_cache
    
    def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Test database connection
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            self.connect()
            return True, None
        except Exception as e:
            return False, str(e)
        finally:
            try:
                self.disconnect()
            except:
                pass
    
    def explain_query(self, sql: str) -> QueryResult:
        """
        Get query execution plan
        
        Default implementation uses EXPLAIN - override for database-specific behavior
        """
        return self.execute_query(f"EXPLAIN {sql}")
    
    def __enter__(self) -> "BaseDatabaseAdapter":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> QueryResult:
        """Get sample data from a table"""
        sql = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(sql)
    
    def get_column_statistics(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get basic statistics for a column"""
        stats = {}
        
        # Count distinct values
        result = self.execute_query(
            f"SELECT COUNT(DISTINCT {column_name}) as cnt FROM {table_name}"
        )
        if result.success and result.rows:
            stats["distinct_count"] = result.rows[0][0]
        
        # Count nulls
        result = self.execute_query(
            f"SELECT COUNT(*) as cnt FROM {table_name} WHERE {column_name} IS NULL"
        )
        if result.success and result.rows:
            stats["null_count"] = result.rows[0][0]
        
        return stats


# Type alias for adapter classes
AdapterClass = Type[BaseDatabaseAdapter]


class DatabaseAdapterRegistry:
    """Registry for database adapters using Factory pattern"""
    
    _adapters: Dict[DatabaseType, AdapterClass] = {}
    _lock = threading.Lock()
    
    @classmethod
    def register(cls, db_type: DatabaseType, adapter_class: AdapterClass) -> None:
        """Register a database adapter class"""
        with cls._lock:
            cls._adapters[db_type] = adapter_class
    
    @classmethod
    def get_adapter_class(cls, db_type: DatabaseType) -> AdapterClass:
        """Get adapter class for database type"""
        with cls._lock:
            if db_type not in cls._adapters:
                raise ValueError(f"No adapter registered for database type: {db_type}")
            return cls._adapters[db_type]
    
    @classmethod
    def create_adapter(cls, config: DatabaseConfig) -> BaseDatabaseAdapter:
        """Create adapter instance from configuration"""
        adapter_class = cls.get_adapter_class(config.db_type)
        return adapter_class(config)
    
    @classmethod
    def get_supported_types(cls) -> List[DatabaseType]:
        """Get list of supported database types"""
        with cls._lock:
            return list(cls._adapters.keys())
    
    @classmethod
    def is_supported(cls, db_type: DatabaseType) -> bool:
        """Check if database type is supported"""
        with cls._lock:
            return db_type in cls._adapters


def register_adapter(db_type: DatabaseType):
    """Decorator to register a database adapter class"""
    def decorator(cls: AdapterClass) -> AdapterClass:
        DatabaseAdapterRegistry.register(db_type, cls)
        return cls
    return decorator
