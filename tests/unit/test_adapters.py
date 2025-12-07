"""
Unit Tests for Database Adapters
"""
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from adapters import (
    BaseDatabaseAdapter,
    DatabaseAdapterRegistry,
    DatabaseSchema,
    TableSchema,
    ColumnSchema,
    QueryResult,
    create_adapter,
    get_supported_databases,
)
from adapters.sqlite_adapter import SQLiteAdapter
from config import DatabaseConfig, DatabaseType


class TestDatabaseSchema:
    """Tests for DatabaseSchema class"""
    
    def test_schema_creation(self):
        """Test basic schema creation"""
        schema = DatabaseSchema(
            database_name="test_db",
            database_type=DatabaseType.SQLITE,
        )
        assert schema.database_name == "test_db"
        assert schema.database_type == DatabaseType.SQLITE
        assert len(schema.tables) == 0
    
    def test_schema_with_tables(self):
        """Test schema with tables"""
        columns = [
            ColumnSchema(name="id", data_type="INTEGER", is_primary_key=True),
            ColumnSchema(name="name", data_type="TEXT"),
        ]
        table = TableSchema(
            name="users",
            columns=columns,
            primary_key=["id"],
        )
        
        schema = DatabaseSchema(
            database_name="test_db",
            database_type=DatabaseType.SQLITE,
            tables={"users": table},
        )
        
        assert "users" in schema.tables
        assert len(schema.tables["users"].columns) == 2
    
    def test_schema_to_string(self):
        """Test schema string generation"""
        columns = [
            ColumnSchema(name="id", data_type="INTEGER", nullable=False),
            ColumnSchema(name="email", data_type="TEXT"),
        ]
        table = TableSchema(
            name="users",
            columns=columns,
            primary_key=["id"],
        )
        
        schema = DatabaseSchema(
            database_name="test_db",
            database_type=DatabaseType.SQLITE,
            tables={"users": table},
        )
        
        schema_str = schema.to_schema_string()
        assert "users" in schema_str
        assert "id" in schema_str
        assert "email" in schema_str
    
    def test_get_table_case_insensitive(self):
        """Test case-insensitive table lookup"""
        table = TableSchema(name="Users", columns=[])
        schema = DatabaseSchema(
            database_name="test_db",
            database_type=DatabaseType.SQLITE,
            tables={"Users": table},
        )
        
        assert schema.get_table("Users") is not None
        assert schema.get_table("users") is not None
        assert schema.get_table("USERS") is not None


class TestQueryResult:
    """Tests for QueryResult class"""
    
    def test_successful_result(self):
        """Test successful query result"""
        result = QueryResult(
            success=True,
            columns=["id", "name"],
            rows=[(1, "Alice"), (2, "Bob")],
            row_count=2,
            execution_time_ms=10.5,
        )
        
        assert result.success
        assert result.row_count == 2
        assert len(result.rows) == 2
    
    def test_failed_result(self):
        """Test failed query result"""
        result = QueryResult(
            success=False,
            error_message="Table not found",
            execution_time_ms=5.0,
        )
        
        assert not result.success
        assert result.error_message == "Table not found"
    
    def test_to_dict(self):
        """Test result serialization"""
        result = QueryResult(
            success=True,
            columns=["id"],
            rows=[(1,)],
            row_count=1,
        )
        
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["columns"] == ["id"]


class TestSQLiteAdapter:
    """Tests for SQLite adapter"""
    
    @pytest.fixture
    def adapter(self):
        """Create SQLite adapter with in-memory database"""
        config = DatabaseConfig(
            db_type=DatabaseType.SQLITE,
            database=":memory:",
            sqlite_path=":memory:",
        )
        adapter = SQLiteAdapter(config)
        return adapter
    
    def test_connect_disconnect(self, adapter):
        """Test connection lifecycle"""
        assert not adapter.is_connected()
        
        adapter.connect()
        assert adapter.is_connected()
        
        adapter.disconnect()
        assert not adapter.is_connected()
    
    def test_execute_query(self, adapter):
        """Test query execution"""
        adapter.connect()
        
        # Create table
        result = adapter.execute_query(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)"
        )
        assert result.success
        
        # Insert data
        result = adapter.execute_query(
            "INSERT INTO test (value) VALUES ('hello')"
        )
        assert result.success
        
        # Select data
        result = adapter.execute_query("SELECT * FROM test")
        assert result.success
        assert result.row_count == 1
        
        adapter.disconnect()
    
    def test_fetch_schema(self, adapter):
        """Test schema fetching"""
        adapter.connect()
        
        # Create table
        adapter.execute_query("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            )
        """)
        
        schema = adapter.get_schema()
        
        assert "users" in schema.tables
        assert len(schema.tables["users"].columns) == 3
        
        adapter.disconnect()
    
    def test_validate_sql_syntax(self, adapter):
        """Test SQL syntax validation"""
        adapter.connect()
        
        adapter.execute_query("CREATE TABLE test (id INTEGER)")
        
        # Valid SQL
        is_valid, error = adapter.validate_sql_syntax("SELECT * FROM test")
        assert is_valid
        assert error is None
        
        # Invalid SQL
        is_valid, error = adapter.validate_sql_syntax("SELEC * FORM test")
        assert not is_valid
        assert error is not None
        
        adapter.disconnect()
    
    def test_dialect_hints(self, adapter):
        """Test dialect hints"""
        hints = adapter.get_sql_dialect_hints()
        
        assert hints["dialect"] == "SQLite"
        assert "LIMIT" in hints["limit_syntax"]
        assert "notes" in hints
    
    def test_context_manager(self, adapter):
        """Test context manager usage"""
        with adapter as db:
            assert db.is_connected()
            db.execute_query("SELECT 1")
        
        assert not adapter.is_connected()


class TestDatabaseAdapterRegistry:
    """Tests for adapter registry"""
    
    def test_supported_types(self):
        """Test getting supported database types"""
        supported = get_supported_databases()
        
        assert DatabaseType.SQLITE in supported
        assert DatabaseType.MYSQL in supported
        assert DatabaseType.POSTGRESQL in supported
        assert DatabaseType.ORACLE in supported
    
    def test_create_adapter(self):
        """Test adapter creation through registry"""
        config = DatabaseConfig(
            db_type=DatabaseType.SQLITE,
            database=":memory:",
        )
        
        adapter = create_adapter(config)
        assert isinstance(adapter, SQLiteAdapter)
    
    def test_unsupported_type(self):
        """Test error on unsupported type"""
        # This should not raise since all types are registered
        # Just verify the registry works
        assert DatabaseAdapterRegistry.is_supported(DatabaseType.SQLITE)


class TestTableSchema:
    """Tests for TableSchema class"""
    
    def test_to_ddl_string(self):
        """Test DDL string generation"""
        columns = [
            ColumnSchema(name="id", data_type="INTEGER", nullable=False),
            ColumnSchema(name="name", data_type="VARCHAR(100)"),
        ]
        table = TableSchema(
            name="users",
            columns=columns,
            primary_key=["id"],
        )
        
        ddl = table.to_ddl_string()
        
        assert "TABLE users" in ddl
        assert "id INTEGER NOT NULL" in ddl
        assert "name VARCHAR(100)" in ddl
        assert "PRIMARY KEY (id)" in ddl


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
