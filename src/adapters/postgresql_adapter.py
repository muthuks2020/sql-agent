"""
PostgreSQL Database Adapter
Implements database operations for PostgreSQL
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ..config import DatabaseConfig, DatabaseType
from .base import (
    BaseDatabaseAdapter,
    ColumnSchema,
    ForeignKeySchema,
    IndexSchema,
    QueryResult,
    TableSchema,
    register_adapter,
)


@register_adapter(DatabaseType.POSTGRESQL)
class PostgreSQLAdapter(BaseDatabaseAdapter):
    """PostgreSQL database adapter"""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._cursor = None
    
    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.POSTGRESQL
    
    @property
    def dialect_name(self) -> str:
        return "PostgreSQL"
    
    def connect(self) -> None:
        """Establish PostgreSQL connection"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. "
                "Install it with: pip install psycopg2-binary"
            )
        
        connection_params = {
            "host": self.config.host,
            "port": self.config.port,
            "dbname": self.config.database,
            "user": self.config.username,
            "password": self.config.password.get_secret_value() if self.config.password else None,
            "connect_timeout": self.config.connection_timeout,
        }
        
        # Add SSL configuration if enabled
        if self.config.ssl_enabled:
            connection_params["sslmode"] = "require"
            if self.config.ssl_ca_path:
                connection_params["sslrootcert"] = self.config.ssl_ca_path
        
        self._connection = psycopg2.connect(**connection_params)
        self._connection.autocommit = True
        self._cursor = self._connection.cursor(cursor_factory=RealDictCursor)
    
    def disconnect(self) -> None:
        """Close PostgreSQL connection"""
        if self._cursor:
            try:
                self._cursor.close()
            except:
                pass
            self._cursor = None
        
        if self._connection:
            try:
                self._connection.close()
            except:
                pass
            self._connection = None
    
    def is_connected(self) -> bool:
        """Check if connection is active"""
        if self._connection is None:
            return False
        try:
            # Check connection status
            return self._connection.closed == 0
        except:
            return False
    
    def execute_query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute SQL query"""
        if not self.is_connected():
            self.connect()
        
        start_time = time.time()
        
        try:
            if params:
                # Convert dict params to psycopg2 format
                self._cursor.execute(sql, params)
            else:
                self._cursor.execute(sql)
            
            # Check if query returns results
            if self._cursor.description:
                columns = [desc[0] for desc in self._cursor.description]
                rows = self._cursor.fetchall()
                # Convert RealDictRow to tuples
                tuple_rows = [tuple(dict(row).values()) for row in rows]
                row_count = len(tuple_rows)
            else:
                columns = []
                tuple_rows = []
                row_count = self._cursor.rowcount
            
            execution_time = (time.time() - start_time) * 1000
            
            return QueryResult(
                success=True,
                columns=columns,
                rows=tuple_rows,
                row_count=row_count,
                execution_time_ms=execution_time,
            )
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            # Rollback on error
            try:
                self._connection.rollback()
            except:
                pass
            return QueryResult(
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e),
            )
    
    def _fetch_tables(self) -> List[TableSchema]:
        """Fetch all table schemas"""
        tables = []
        
        # Get table names and row counts
        result = self.execute_query(
            """
            SELECT 
                t.table_name,
                (SELECT reltuples::bigint FROM pg_class WHERE relname = t.table_name) as row_count
            FROM information_schema.tables t
            WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
            ORDER BY t.table_name
            """
        )
        
        if not result.success:
            return tables
        
        for row in result.rows:
            table_name = row[0]
            row_count = row[1]
            
            columns = self._fetch_columns(table_name)
            primary_key = self._fetch_primary_key(table_name)
            foreign_keys = self._fetch_foreign_keys(table_name)
            indexes = self._fetch_indexes(table_name)
            
            tables.append(TableSchema(
                name=table_name,
                columns=columns,
                primary_key=primary_key,
                foreign_keys=foreign_keys,
                indexes=indexes,
                row_count=int(row_count) if row_count else None,
            ))
        
        return tables
    
    def _fetch_views(self) -> List[TableSchema]:
        """Fetch all view schemas"""
        views = []
        
        result = self.execute_query(
            """
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'public'
            ORDER BY table_name
            """
        )
        
        if not result.success:
            return views
        
        for row in result.rows:
            view_name = row[0]
            columns = self._fetch_columns(view_name)
            
            views.append(TableSchema(
                name=view_name,
                columns=columns,
            ))
        
        return views
    
    def _fetch_columns(self, table_name: str) -> List[ColumnSchema]:
        """Fetch columns for a table"""
        columns = []
        
        result = self.execute_query(
            """
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                col_description(
                    (SELECT oid FROM pg_class WHERE relname = c.table_name),
                    c.ordinal_position
                ) as column_comment
            FROM information_schema.columns c
            WHERE c.table_schema = 'public' AND c.table_name = %s
            ORDER BY c.ordinal_position
            """,
            (table_name,)
        )
        
        if not result.success:
            return columns
        
        # Get primary key columns
        pk_cols = set(self._fetch_primary_key(table_name))
        
        for row in result.rows:
            columns.append(ColumnSchema(
                name=row[0],
                data_type=row[1],
                nullable=row[2] == "YES",
                default_value=row[3],
                max_length=row[4],
                precision=row[5],
                scale=row[6],
                is_primary_key=row[0] in pk_cols,
                comment=row[7],
            ))
        
        return columns
    
    def _fetch_primary_key(self, table_name: str) -> List[str]:
        """Fetch primary key columns"""
        result = self.execute_query(
            """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary
            ORDER BY array_position(i.indkey, a.attnum)
            """,
            (table_name,)
        )
        
        if not result.success:
            return []
        
        return [row[0] for row in result.rows]
    
    def _fetch_foreign_keys(self, table_name: str) -> List[ForeignKeySchema]:
        """Fetch foreign key relationships"""
        foreign_keys = []
        
        result = self.execute_query(
            """
            SELECT
                tc.constraint_name,
                kcu.column_name,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column,
                rc.delete_rule,
                rc.update_rule
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            JOIN information_schema.referential_constraints rc
                ON tc.constraint_name = rc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY' 
                AND tc.table_schema = 'public'
                AND tc.table_name = %s
            ORDER BY tc.constraint_name, kcu.ordinal_position
            """,
            (table_name,)
        )
        
        if not result.success:
            return foreign_keys
        
        # Group by constraint name
        fk_dict: Dict[str, Dict] = {}
        for row in result.rows:
            constraint_name = row[0]
            if constraint_name not in fk_dict:
                fk_dict[constraint_name] = {
                    "columns": [],
                    "referenced_table": row[2],
                    "referenced_columns": [],
                    "on_delete": row[4],
                    "on_update": row[5],
                }
            fk_dict[constraint_name]["columns"].append(row[1])
            fk_dict[constraint_name]["referenced_columns"].append(row[3])
        
        for name, fk_data in fk_dict.items():
            foreign_keys.append(ForeignKeySchema(
                name=name,
                columns=fk_data["columns"],
                referenced_table=fk_data["referenced_table"],
                referenced_columns=fk_data["referenced_columns"],
                on_delete=fk_data["on_delete"],
                on_update=fk_data["on_update"],
            ))
        
        return foreign_keys
    
    def _fetch_indexes(self, table_name: str) -> List[IndexSchema]:
        """Fetch index information"""
        indexes = []
        
        result = self.execute_query(
            """
            SELECT
                i.relname AS index_name,
                a.attname AS column_name,
                ix.indisunique AS is_unique,
                ix.indisprimary AS is_primary,
                am.amname AS index_type
            FROM pg_class t
            JOIN pg_index ix ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_am am ON i.relam = am.oid
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            WHERE t.relkind = 'r' AND t.relname = %s
            ORDER BY i.relname, array_position(ix.indkey, a.attnum)
            """,
            (table_name,)
        )
        
        if not result.success:
            return indexes
        
        # Group by index name
        idx_dict: Dict[str, Dict] = {}
        for row in result.rows:
            index_name = row[0]
            if index_name not in idx_dict:
                idx_dict[index_name] = {
                    "columns": [],
                    "is_unique": row[2],
                    "is_primary": row[3],
                    "index_type": row[4],
                }
            idx_dict[index_name]["columns"].append(row[1])
        
        for name, idx_data in idx_dict.items():
            indexes.append(IndexSchema(
                name=name,
                columns=idx_data["columns"],
                is_unique=idx_data["is_unique"],
                is_primary=idx_data["is_primary"],
                index_type=idx_data["index_type"],
            ))
        
        return indexes
    
    def validate_sql_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL syntax using EXPLAIN"""
        if not self.is_connected():
            self.connect()
        
        try:
            self._cursor.execute(f"EXPLAIN {sql}")
            self._cursor.fetchall()
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_sql_dialect_hints(self) -> Dict[str, Any]:
        """Get PostgreSQL-specific dialect hints"""
        return {
            "dialect": "PostgreSQL",
            "string_concatenation": "|| operator or CONCAT() function",
            "limit_syntax": "LIMIT n",
            "offset_syntax": "LIMIT n OFFSET m",
            "current_timestamp": "NOW() or CURRENT_TIMESTAMP",
            "date_functions": ["DATE_TRUNC()", "EXTRACT()", "AGE()", "TO_CHAR()", "TO_DATE()"],
            "string_functions": ["CONCAT()", "SUBSTRING()", "LENGTH()", "UPPER()", "LOWER()", "TRIM()", "SPLIT_PART()"],
            "null_handling": "COALESCE() or NULLIF()",
            "case_sensitivity": "Case-sensitive by default",
            "identifier_quoting": 'Double quotes ("identifier")',
            "serial_type": "SERIAL, BIGSERIAL, or IDENTITY",
            "boolean_type": "BOOLEAN (true/false)",
            "json_support": "JSON and JSONB with ->, ->>, @>, etc.",
            "array_support": "Native ARRAY type with ANY(), ALL()",
            "window_functions": "Fully supported",
            "cte_support": "WITH clause and recursive CTEs",
            "upsert": "INSERT ... ON CONFLICT",
            "returning_clause": "RETURNING supported for INSERT/UPDATE/DELETE",
            "reserved_words": [
                "SELECT", "FROM", "WHERE", "JOIN", "ON", "GROUP", "BY",
                "HAVING", "ORDER", "LIMIT", "OFFSET", "UNION", "INSERT",
                "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "INDEX",
                "USER", "TABLE", "COLUMN"
            ],
            "notes": [
                "Use double quotes for case-sensitive identifiers",
                "Supports ILIKE for case-insensitive matching",
                "Has rich set of data types including arrays and ranges",
                "Supports partial indexes and expression indexes",
            ]
        }
    
    def explain_query(self, sql: str) -> QueryResult:
        """Get PostgreSQL query execution plan"""
        return self.execute_query(f"EXPLAIN (ANALYZE false, FORMAT JSON) {sql}")
