"""
MySQL Database Adapter
Implements database operations for MySQL/MariaDB
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


@register_adapter(DatabaseType.MYSQL)
class MySQLAdapter(BaseDatabaseAdapter):
    """MySQL/MariaDB database adapter"""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._cursor = None
    
    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.MYSQL
    
    @property
    def dialect_name(self) -> str:
        return "MySQL"
    
    def connect(self) -> None:
        """Establish MySQL connection"""
        try:
            import mysql.connector
            from mysql.connector import pooling
        except ImportError:
            raise ImportError(
                "mysql-connector-python is required for MySQL support. "
                "Install it with: pip install mysql-connector-python"
            )
        
        connection_config = {
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
            "user": self.config.username,
            "password": self.config.password.get_secret_value() if self.config.password else None,
            "connection_timeout": self.config.connection_timeout,
            "autocommit": True,
        }
        
        # Add SSL configuration if enabled
        if self.config.ssl_enabled:
            connection_config["ssl_ca"] = self.config.ssl_ca_path
        
        self._connection = mysql.connector.connect(**connection_config)
        self._cursor = self._connection.cursor(dictionary=True)
    
    def disconnect(self) -> None:
        """Close MySQL connection"""
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
            self._connection.ping(reconnect=False)
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
                self._cursor.execute(sql, params)
            else:
                self._cursor.execute(sql)
            
            # Check if query returns results
            if self._cursor.description:
                columns = [desc[0] for desc in self._cursor.description]
                rows = self._cursor.fetchall()
                # Convert dict rows to tuples
                tuple_rows = [tuple(row.values()) for row in rows]
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
            return QueryResult(
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e),
            )
    
    def _fetch_tables(self) -> List[TableSchema]:
        """Fetch all table schemas"""
        tables = []
        
        # Get table names
        result = self.execute_query(
            """
            SELECT TABLE_NAME, TABLE_ROWS
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'
            """,
            {"table_schema": self.config.database}
        )
        
        if not result.success:
            return tables
        
        for row in result.rows:
            table_name = row[0]
            row_count = row[1]
            
            # Get columns
            columns = self._fetch_columns(table_name)
            
            # Get primary key
            primary_key = self._fetch_primary_key(table_name)
            
            # Get foreign keys
            foreign_keys = self._fetch_foreign_keys(table_name)
            
            # Get indexes
            indexes = self._fetch_indexes(table_name)
            
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
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.VIEWS
            WHERE TABLE_SCHEMA = %s
            """,
            {"table_schema": self.config.database}
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
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                CHARACTER_MAXIMUM_LENGTH,
                NUMERIC_PRECISION,
                NUMERIC_SCALE,
                COLUMN_KEY,
                COLUMN_COMMENT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
            """,
            {"table_schema": self.config.database, "table_name": table_name}
        )
        
        if not result.success:
            return columns
        
        for row in result.rows:
            columns.append(ColumnSchema(
                name=row[0],
                data_type=row[1],
                nullable=row[2] == "YES",
                default_value=row[3],
                max_length=row[4],
                precision=row[5],
                scale=row[6],
                is_primary_key=row[7] == "PRI",
                is_unique=row[7] == "UNI",
                is_indexed=row[7] in ("PRI", "UNI", "MUL"),
                comment=row[8] if row[8] else None,
            ))
        
        return columns
    
    def _fetch_primary_key(self, table_name: str) -> List[str]:
        """Fetch primary key columns"""
        result = self.execute_query(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s 
              AND TABLE_NAME = %s 
              AND CONSTRAINT_NAME = 'PRIMARY'
            ORDER BY ORDINAL_POSITION
            """,
            {"table_schema": self.config.database, "table_name": table_name}
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
                CONSTRAINT_NAME,
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s 
              AND TABLE_NAME = %s 
              AND REFERENCED_TABLE_NAME IS NOT NULL
            ORDER BY CONSTRAINT_NAME, ORDINAL_POSITION
            """,
            {"table_schema": self.config.database, "table_name": table_name}
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
                }
            fk_dict[constraint_name]["columns"].append(row[1])
            fk_dict[constraint_name]["referenced_columns"].append(row[3])
        
        for name, fk_data in fk_dict.items():
            foreign_keys.append(ForeignKeySchema(
                name=name,
                columns=fk_data["columns"],
                referenced_table=fk_data["referenced_table"],
                referenced_columns=fk_data["referenced_columns"],
            ))
        
        return foreign_keys
    
    def _fetch_indexes(self, table_name: str) -> List[IndexSchema]:
        """Fetch index information"""
        indexes = []
        
        result = self.execute_query(
            """
            SELECT 
                INDEX_NAME,
                COLUMN_NAME,
                NON_UNIQUE,
                INDEX_TYPE
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY INDEX_NAME, SEQ_IN_INDEX
            """,
            {"table_schema": self.config.database, "table_name": table_name}
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
                    "is_unique": row[2] == 0,
                    "is_primary": index_name == "PRIMARY",
                    "index_type": row[3],
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
            # Use EXPLAIN to validate syntax
            self._cursor.execute(f"EXPLAIN {sql}")
            self._cursor.fetchall()
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_sql_dialect_hints(self) -> Dict[str, Any]:
        """Get MySQL-specific dialect hints"""
        return {
            "dialect": "MySQL",
            "string_concatenation": "CONCAT() function",
            "limit_syntax": "LIMIT n",
            "offset_syntax": "LIMIT n OFFSET m or LIMIT m, n",
            "current_timestamp": "NOW() or CURRENT_TIMESTAMP",
            "date_functions": ["DATE()", "YEAR()", "MONTH()", "DAY()", "DATE_FORMAT()"],
            "string_functions": ["CONCAT()", "SUBSTRING()", "LENGTH()", "UPPER()", "LOWER()", "TRIM()"],
            "null_handling": "IFNULL() or COALESCE()",
            "case_sensitivity": "Case-insensitive by default for string comparisons",
            "identifier_quoting": "Backticks (`identifier`)",
            "auto_increment": "AUTO_INCREMENT",
            "boolean_type": "BOOLEAN (stored as TINYINT(1))",
            "json_support": "JSON data type with JSON_EXTRACT(), JSON_OBJECT(), etc.",
            "window_functions": "Supported (MySQL 8.0+)",
            "cte_support": "WITH clause supported (MySQL 8.0+)",
            "reserved_words": [
                "SELECT", "FROM", "WHERE", "JOIN", "ON", "GROUP", "BY", 
                "HAVING", "ORDER", "LIMIT", "OFFSET", "UNION", "INSERT",
                "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "INDEX"
            ],
            "notes": [
                "Use backticks for reserved words as identifiers",
                "Default storage engine is InnoDB",
                "FULLTEXT indexes available for text search",
            ]
        }
    
    def explain_query(self, sql: str) -> QueryResult:
        """Get MySQL query execution plan"""
        return self.execute_query(f"EXPLAIN FORMAT=JSON {sql}")
