"""
Oracle Database Adapter
Implements database operations for Oracle Database
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


@register_adapter(DatabaseType.ORACLE)
class OracleAdapter(BaseDatabaseAdapter):
    """Oracle database adapter"""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._cursor = None
    
    @property
    def database_type(self) -> DatabaseType:
        return DatabaseType.ORACLE
    
    @property
    def dialect_name(self) -> str:
        return "Oracle"
    
    def connect(self) -> None:
        """Establish Oracle connection"""
        try:
            import oracledb
        except ImportError:
            raise ImportError(
                "oracledb is required for Oracle support. "
                "Install it with: pip install oracledb"
            )
        
        # Build DSN
        if self.config.oracle_service_name:
            dsn = f"{self.config.host}:{self.config.port}/{self.config.oracle_service_name}"
        elif self.config.oracle_sid:
            dsn = oracledb.makedsn(
                self.config.host,
                self.config.port,
                sid=self.config.oracle_sid
            )
        else:
            dsn = f"{self.config.host}:{self.config.port}/{self.config.database}"
        
        self._connection = oracledb.connect(
            user=self.config.username,
            password=self.config.password.get_secret_value() if self.config.password else None,
            dsn=dsn,
        )
        self._connection.autocommit = True
        self._cursor = self._connection.cursor()
    
    def disconnect(self) -> None:
        """Close Oracle connection"""
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
            self._connection.ping()
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
            SELECT table_name, num_rows
            FROM user_tables
            ORDER BY table_name
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
                row_count=row_count,
            ))
        
        return tables
    
    def _fetch_views(self) -> List[TableSchema]:
        """Fetch all view schemas"""
        views = []
        
        result = self.execute_query(
            """
            SELECT view_name
            FROM user_views
            ORDER BY view_name
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
                column_name,
                data_type,
                nullable,
                data_default,
                char_length,
                data_precision,
                data_scale
            FROM user_tab_columns
            WHERE table_name = :table_name
            ORDER BY column_id
            """,
            {"table_name": table_name.upper()}
        )
        
        if not result.success:
            return columns
        
        pk_cols = set(self._fetch_primary_key(table_name))
        
        for row in result.rows:
            columns.append(ColumnSchema(
                name=row[0],
                data_type=row[1],
                nullable=row[2] == "Y",
                default_value=row[3],
                max_length=row[4],
                precision=row[5],
                scale=row[6],
                is_primary_key=row[0] in pk_cols,
            ))
        
        return columns
    
    def _fetch_primary_key(self, table_name: str) -> List[str]:
        """Fetch primary key columns"""
        result = self.execute_query(
            """
            SELECT cols.column_name
            FROM user_constraints cons
            JOIN user_cons_columns cols 
                ON cons.constraint_name = cols.constraint_name
            WHERE cons.constraint_type = 'P'
                AND cons.table_name = :table_name
            ORDER BY cols.position
            """,
            {"table_name": table_name.upper()}
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
                a.constraint_name,
                a.column_name,
                c_pk.table_name AS referenced_table,
                b.column_name AS referenced_column,
                c.delete_rule
            FROM user_cons_columns a
            JOIN user_constraints c 
                ON a.constraint_name = c.constraint_name
            JOIN user_constraints c_pk 
                ON c.r_constraint_name = c_pk.constraint_name
            JOIN user_cons_columns b 
                ON c_pk.constraint_name = b.constraint_name
                AND a.position = b.position
            WHERE c.constraint_type = 'R'
                AND a.table_name = :table_name
            ORDER BY a.constraint_name, a.position
            """,
            {"table_name": table_name.upper()}
        )
        
        if not result.success:
            return foreign_keys
        
        fk_dict: Dict[str, Dict] = {}
        for row in result.rows:
            constraint_name = row[0]
            if constraint_name not in fk_dict:
                fk_dict[constraint_name] = {
                    "columns": [],
                    "referenced_table": row[2],
                    "referenced_columns": [],
                    "on_delete": row[4],
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
            ))
        
        return foreign_keys
    
    def _fetch_indexes(self, table_name: str) -> List[IndexSchema]:
        """Fetch index information"""
        indexes = []
        
        result = self.execute_query(
            """
            SELECT 
                i.index_name,
                ic.column_name,
                i.uniqueness,
                i.index_type
            FROM user_indexes i
            JOIN user_ind_columns ic 
                ON i.index_name = ic.index_name
            WHERE i.table_name = :table_name
            ORDER BY i.index_name, ic.column_position
            """,
            {"table_name": table_name.upper()}
        )
        
        if not result.success:
            return indexes
        
        idx_dict: Dict[str, Dict] = {}
        for row in result.rows:
            index_name = row[0]
            if index_name not in idx_dict:
                idx_dict[index_name] = {
                    "columns": [],
                    "is_unique": row[2] == "UNIQUE",
                    "index_type": row[3],
                }
            idx_dict[index_name]["columns"].append(row[1])
        
        for name, idx_data in idx_dict.items():
            indexes.append(IndexSchema(
                name=name,
                columns=idx_data["columns"],
                is_unique=idx_data["is_unique"],
                index_type=idx_data["index_type"],
            ))
        
        return indexes
    
    def validate_sql_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL syntax using EXPLAIN PLAN"""
        if not self.is_connected():
            self.connect()
        
        try:
            self._cursor.execute(f"EXPLAIN PLAN FOR {sql}")
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_sql_dialect_hints(self) -> Dict[str, Any]:
        """Get Oracle-specific dialect hints"""
        return {
            "dialect": "Oracle",
            "string_concatenation": "|| operator",
            "limit_syntax": "FETCH FIRST n ROWS ONLY (12c+) or ROWNUM <= n",
            "offset_syntax": "OFFSET n ROWS FETCH NEXT m ROWS ONLY",
            "current_timestamp": "SYSDATE or SYSTIMESTAMP",
            "date_functions": ["TO_DATE()", "TO_CHAR()", "EXTRACT()", "ADD_MONTHS()", "TRUNC()"],
            "string_functions": ["CONCAT()", "SUBSTR()", "LENGTH()", "UPPER()", "LOWER()", "TRIM()", "INSTR()"],
            "null_handling": "NVL() or COALESCE()",
            "case_sensitivity": "Case-insensitive for keywords, case-sensitive for identifiers",
            "identifier_quoting": 'Double quotes ("identifier")',
            "sequence_type": "SEQUENCE objects",
            "boolean_type": "No native BOOLEAN in SQL (use NUMBER(1) or VARCHAR2(1))",
            "hierarchical": "CONNECT BY for hierarchical queries",
            "analytical": "Powerful analytical functions (OVER, PARTITION BY)",
            "dual_table": "Use FROM DUAL for single-row queries",
            "outer_join": "Both ANSI and Oracle (+) syntax supported",
            "reserved_words": [
                "SELECT", "FROM", "WHERE", "JOIN", "ON", "GROUP", "BY",
                "HAVING", "ORDER", "UNION", "INSERT", "UPDATE", "DELETE",
                "CREATE", "DROP", "ALTER", "INDEX", "TABLE", "VIEW",
                "USER", "LEVEL", "ROWNUM", "ROWID"
            ],
            "notes": [
                "Empty strings are treated as NULL",
                "Use TO_DATE for date literals",
                "VARCHAR2 preferred over VARCHAR",
                "Case-sensitive string comparisons by default",
                "MERGE statement for upsert operations",
            ]
        }
    
    def explain_query(self, sql: str) -> QueryResult:
        """Get Oracle query execution plan"""
        # Execute EXPLAIN PLAN
        self.execute_query(f"EXPLAIN PLAN FOR {sql}")
        # Retrieve the plan
        return self.execute_query(
            """
            SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY())
            """
        )
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> QueryResult:
        """Get sample data from a table (Oracle syntax)"""
        sql = f"SELECT * FROM {table_name} WHERE ROWNUM <= {limit}"
        return self.execute_query(sql)
