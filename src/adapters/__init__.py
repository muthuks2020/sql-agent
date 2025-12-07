"""
Database Adapters Package
Provides database-agnostic interface for SQL operations
"""
from .base import (
    BaseDatabaseAdapter,
    DatabaseAdapterRegistry,
    DatabaseSchema,
    TableSchema,
    ColumnSchema,
    ForeignKeySchema,
    IndexSchema,
    QueryResult,
    register_adapter,
)

# Import adapters to register them
from .mysql_adapter import MySQLAdapter
from .postgresql_adapter import PostgreSQLAdapter
from .oracle_adapter import OracleAdapter
from .sqlite_adapter import SQLiteAdapter

from ..config import DatabaseConfig, DatabaseType


def create_adapter(config: DatabaseConfig) -> BaseDatabaseAdapter:
    """
    Factory function to create database adapter from configuration
    
    Args:
        config: Database configuration
        
    Returns:
        Configured database adapter instance
        
    Raises:
        ValueError: If database type is not supported
    """
    return DatabaseAdapterRegistry.create_adapter(config)


def get_supported_databases() -> list:
    """Get list of supported database types"""
    return DatabaseAdapterRegistry.get_supported_types()


__all__ = [
    # Base classes
    "BaseDatabaseAdapter",
    "DatabaseAdapterRegistry",
    "DatabaseSchema",
    "TableSchema",
    "ColumnSchema",
    "ForeignKeySchema",
    "IndexSchema",
    "QueryResult",
    "register_adapter",
    # Concrete adapters
    "MySQLAdapter",
    "PostgreSQLAdapter",
    "OracleAdapter",
    "SQLiteAdapter",
    # Factory functions
    "create_adapter",
    "get_supported_databases",
]
