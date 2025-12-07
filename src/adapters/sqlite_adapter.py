"""
SQLite Database Adapter
Implements database operations for SQLite
"""
from __future__ import annotations

import re
import sqlite3
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


@register_adapter(DatabaseType.SQLITE)
class SQLiteAdapter(BaseDatabaseAdapter):
    """SQLite database adapter"""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._cursor = None
    
    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.SQLITE
    
    @property
    def dialect_name(self) -> str:
        return "SQLite"
    
    def connect(self) -> None:
        """Establish SQLite connection"""
        db_path = self.config.sqlite_path or self.config.database
        
        if not db_path:
            db_path = ":memory:"
        
        self._connection = sqlite3.connect(
            db_path,
            timeout=self.config.connection_timeout,
            check_same_thread=False,  # Allow multi-threaded access
        )
        
        # Enable foreign keys
        self._connection.execute("PRAGMA foreign_keys = ON")
        
        # Set row factory for named columns
        self._connection.row_factory = sqlite3.Row
        
        self._cursor = self._connection.cursor()
    
    def disconnect(self) -> None:
        """Close SQLite connection"""
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
            self._connection.execute("SELECT 1")
            return True
        except:
            return False
    
    def execute_query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute SQL query"""
        if not self.is_connected():
            self.connect()
        
        start_time = time.time()
        
        try:
            if params:
                # Convert dict params to named placeholders
                self._cursor.execute(sql, params)
            else:
                self._cursor.execute(sql)
            
            # Commit for write operations
            if sql.strip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")):
                self._connection.commit()
            
            # Check if query returns results
            if self._cursor.description:
                columns = [desc[0] for desc in self._cursor.description]
                rows = [tuple(row) for row in self._cursor.fetchall()]
                row_count = len(rows)
            else:
                columns = []
                rows = []
                row_count = self._cursor.rowcount
            
            execution_time = (time.time() - start_time) * 1000
            
            return QueryResult(
                success=True,
                columns=columns,
                rows=rows,
                row_count=row_count,
                execution_time_ms=execution_time,
            )
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
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
        
        result = self.execute_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        
        if not result.success:
            return tables
        
        for row in result.rows:
            table_name = row[0]
            
            columns = self._fetch_columns(table_name)
            primary_key = self._fetch_primary_key(table_name)
            foreign_keys = self._fetch_foreign_keys(table_name)
            indexes = self._fetch_indexes(table_name)
            row_count = self._get_row_count(table_name)
            
            tables.append(TableSchema(
                name=table_name,
                columns=columns,
                primary_key=primary_key,
                foreign_keys=foreign_keys,
                indexes=indexes,
                row_count=row_count,
            ))
        
        return tables
    
    def _fetch_views(self) -> List[TableSchema]:
        """Fetch all view schemas"""
        views = []
        
        result = self.execute_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'view'
            ORDER BY name
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
        
        result = self.execute_query(f"PRAGMA table_info('{table_name}')")
        
        if not result.success:
            return columns
        
        for row in result.rows:
            # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
            columns.append(ColumnSchema(
                name=row[1],
                data_type=row[2] or "TEXT",  # SQLite default
                nullable=not row[3],
                default_value=row[4],
                is_primary_key=bool(row[5]),
            ))
        
        return columns
    
    def _fetch_primary_key(self, table_name: str) -> List[str]:
        """Fetch primary key columns"""
        result = self.execute_query(f"PRAGMA table_info('{table_name}')")
        
        if not result.success:
            return []
        
        pk_cols = []
        for row in result.rows:
            if row[5]:  # pk column
                pk_cols.append((row[5], row[1]))  # (pk_order, column_name)
        
        # Sort by pk order and return column names
        pk_cols.sort(key=lambda x: x[0])
        return [col[1] for col in pk_cols]
    
    def _fetch_foreign_keys(self, table_name: str) -> List[ForeignKeySchema]:
        """Fetch foreign key relationships"""
        foreign_keys = []
        
        result = self.execute_query(f"PRAGMA foreign_key_list('{table_name}')")
        
        if not result.success:
            return foreign_keys
        
        # Group by id
        fk_dict: Dict[int, Dict] = {}
        for row in result.rows:
            # id, seq, table, from, to, on_update, on_delete, match
            fk_id = row[0]
            if fk_id not in fk_dict:
                fk_dict[fk_id] = {
                    "columns": [],
                    "referenced_table": row[2],
                    "referenced_columns": [],
                    "on_update": row[5],
                    "on_delete": row[6],
                }
            fk_dict[fk_id]["columns"].append(row[3])
            fk_dict[fk_id]["referenced_columns"].append(row[4])
        
        for fk_id, fk_data in fk_dict.items():
            foreign_keys.append(ForeignKeySchema(
                name=f"fk_{table_name}_{fk_id}",
                columns=fk_data["columns"],
                referenced_table=fk_data["referenced_table"],
                referenced_columns=fk_data["referenced_columns"],
                on_update=fk_data["on_update"],
                on_delete=fk_data["on_delete"],
            ))
        
        return foreign_keys
    
    def _fetch_indexes(self, table_name: str) -> List[IndexSchema]:
        """Fetch index information"""
        indexes = []
        
        result = self.execute_query(f"PRAGMA index_list('{table_name}')")
        
        if not result.success:
            return indexes
        
        for row in result.rows:
            # seq, name, unique, origin, partial
            index_name = row[1]
            is_unique = bool(row[2])
            origin = row[3]  # 'pk', 'c' (created), 'u' (unique constraint)
            
            # Get index columns
            col_result = self.execute_query(f"PRAGMA index_info('{index_name}')")
            
            if col_result.success:
                columns = [col_row[2] for col_row in col_result.rows]
                
                indexes.append(IndexSchema(
                    name=index_name,
                    columns=columns,
                    is_unique=is_unique,
                    is_primary=origin == 'pk',
                ))
        
        return indexes
    
    def _get_row_count(self, table_name: str) -> Optional[int]:
        """Get row count for a table"""
        result = self.execute_query(f"SELECT COUNT(*) FROM '{table_name}'")
        if result.success and result.rows:
            return result.rows[0][0]
        return None
    
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
        """Get SQLite-specific dialect hints"""
        return {
            "dialect": "SQLite",
            "string_concatenation": "|| operator",
            "limit_syntax": "LIMIT n",
            "offset_syntax": "LIMIT n OFFSET m",
            "current_timestamp": "datetime('now') or CURRENT_TIMESTAMP",
            "date_functions": ["date()", "time()", "datetime()", "julianday()", "strftime()"],
            "string_functions": ["substr()", "length()", "upper()", "lower()", "trim()", "instr()", "replace()"],
            "null_handling": "IFNULL() or COALESCE()",
            "case_sensitivity": "Case-insensitive for ASCII by default",
            "identifier_quoting": 'Double quotes ("identifier") or backticks or square brackets',
            "auto_increment": "INTEGER PRIMARY KEY (implicit) or AUTOINCREMENT",
            "boolean_type": "INTEGER (0/1) - no native BOOLEAN",
            "type_affinity": "Uses type affinity (INTEGER, TEXT, REAL, BLOB, NUMERIC)",
            "upsert": "INSERT OR REPLACE, or INSERT ... ON CONFLICT",
            "window_functions": "Supported (version 3.25+)",
            "cte_support": "WITH clause and recursive CTEs (version 3.8.3+)",
            "json_support": "JSON1 extension functions (json(), json_extract(), etc.)",
            "reserved_words": [
                "SELECT", "FROM", "WHERE", "JOIN", "ON", "GROUP", "BY",
                "HAVING", "ORDER", "LIMIT", "OFFSET", "UNION", "INSERT",
                "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "INDEX",
                "TABLE", "COLUMN", "ABORT", "CONFLICT", "REPLACE"
            ],
            "notes": [
                "Dynamically typed - any value can be stored in any column",
                "Single-file database - no server required",
                "Use LIKE for case-insensitive matching",
                "Foreign keys must be enabled with PRAGMA",
                "No native BOOLEAN, DATE, or DATETIME types",
            ]
        }
    
    def explain_query(self, sql: str) -> QueryResult:
        """Get SQLite query execution plan"""
        return self.execute_query(f"EXPLAIN QUERY PLAN {sql}")
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> QueryResult:
        """Get sample data from a table"""
        sql = f"SELECT * FROM \"{table_name}\" LIMIT {limit}"
        return self.execute_query(sql)
    
    def vacuum(self) -> QueryResult:
        """Run VACUUM to optimize database"""
        return self.execute_query("VACUUM")
    
    def get_database_size(self) -> Optional[int]:
        """Get database file size in bytes"""
        result = self.execute_query("PRAGMA page_count")
        if not result.success or not result.rows:
            return None
        page_count = result.rows[0][0]
        
        result = self.execute_query("PRAGMA page_size")
        if not result.success or not result.rows:
            return None
        page_size = result.rows[0][0]
        
        return page_count * page_size
