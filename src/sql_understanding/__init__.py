"""
SQL Understanding Module

Provides utilities for parsing, analyzing, and understanding SQL queries.
Uses sqlparse for robust SQL parsing.

Features:
- Extract SQL from text/markdown
- Parse SQL structure (tables, columns, joins)
- Format and beautify SQL
- Safety checks
- Query type detection
- Complexity analysis
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set

try:
    import sqlparse
    from sqlparse.sql import (
        IdentifierList, 
        Identifier, 
        Where, 
        Parenthesis,
        Function,
        Token,
    )
    from sqlparse.tokens import Keyword, DML, DDL, Name, Punctuation
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False


# SQL Reserved Words (common across dialects)
SQL_RESERVED_WORDS = {
    'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'IS', 'NULL',
    'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'FULL', 'CROSS', 'ON',
    'GROUP', 'BY', 'HAVING', 'ORDER', 'ASC', 'DESC', 'LIMIT', 'OFFSET',
    'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'TRUNCATE',
    'CREATE', 'ALTER', 'DROP', 'TABLE', 'INDEX', 'VIEW', 'DATABASE',
    'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'UNIQUE', 'CONSTRAINT',
    'DEFAULT', 'AUTO_INCREMENT', 'SERIAL', 'IDENTITY',
    'UNION', 'ALL', 'INTERSECT', 'EXCEPT', 'DISTINCT',
    'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS',
    'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COALESCE', 'NULLIF',
    'LIKE', 'BETWEEN', 'EXISTS', 'ANY', 'SOME',
    'TRUE', 'FALSE', 'BOOLEAN', 'INT', 'INTEGER', 'VARCHAR', 'TEXT',
    'DATE', 'TIME', 'TIMESTAMP', 'DATETIME', 'DECIMAL', 'NUMERIC', 'FLOAT',
    'WITH', 'RECURSIVE', 'CTE',
}


class SQLQueryType(Enum):
    """Types of SQL queries"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    ALTER = "ALTER"
    DROP = "DROP"
    TRUNCATE = "TRUNCATE"
    MERGE = "MERGE"
    WITH = "WITH"  # CTE
    UNKNOWN = "UNKNOWN"


class JoinType(Enum):
    """Types of SQL joins"""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"
    NATURAL = "NATURAL"


@dataclass
class TableReference:
    """Represents a table reference in SQL"""
    name: str
    alias: Optional[str] = None
    schema: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        if self.schema:
            return f"{self.schema}.{self.name}"
        return self.name
    
    @property
    def effective_name(self) -> str:
        """Name to use in query (alias if exists, else name)"""
        return self.alias or self.name


@dataclass
class ColumnReference:
    """Represents a column reference in SQL"""
    name: str
    table: Optional[str] = None  # Table name or alias
    alias: Optional[str] = None
    is_aggregate: bool = False
    aggregate_function: Optional[str] = None


@dataclass
class JoinClause:
    """Represents a JOIN clause"""
    join_type: JoinType
    table: TableReference
    condition: Optional[str] = None


@dataclass
class SQLStructure:
    """Parsed structure of a SQL query"""
    query_type: SQLQueryType
    original_sql: str
    formatted_sql: str
    
    # Tables
    tables: List[TableReference] = field(default_factory=list)
    
    # Columns
    select_columns: List[ColumnReference] = field(default_factory=list)
    where_columns: List[ColumnReference] = field(default_factory=list)
    group_by_columns: List[ColumnReference] = field(default_factory=list)
    order_by_columns: List[ColumnReference] = field(default_factory=list)
    
    # Joins
    joins: List[JoinClause] = field(default_factory=list)
    
    # Clauses present
    has_where: bool = False
    has_group_by: bool = False
    has_having: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    has_distinct: bool = False
    has_subquery: bool = False
    has_cte: bool = False
    
    # Aggregations
    has_aggregation: bool = False
    aggregate_functions: List[str] = field(default_factory=list)
    
    # Issues/warnings
    issues: List[Dict[str, str]] = field(default_factory=list)
    
    # Complexity metrics
    table_count: int = 0
    join_count: int = 0
    condition_count: int = 0
    subquery_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query_type": self.query_type.value,
            "tables": [{"name": t.name, "alias": t.alias} for t in self.tables],
            "select_columns": [{"name": c.name, "table": c.table} for c in self.select_columns],
            "joins": [{"type": j.join_type.value, "table": j.table.name} for j in self.joins],
            "has_where": self.has_where,
            "has_group_by": self.has_group_by,
            "has_order_by": self.has_order_by,
            "has_aggregation": self.has_aggregation,
            "aggregate_functions": self.aggregate_functions,
            "table_count": self.table_count,
            "join_count": self.join_count,
            "issues": self.issues,
        }


class SQLParser:
    """
    SQL Parser using sqlparse library.
    
    Extracts structural information from SQL queries including:
    - Tables and their aliases
    - Columns referenced
    - Join clauses
    - Aggregations
    - Query complexity metrics
    """
    
    AGGREGATE_FUNCTIONS = {'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT', 'STRING_AGG'}
    
    def __init__(self):
        if not SQLPARSE_AVAILABLE:
            raise ImportError("sqlparse is required for SQL parsing. Install with: pip install sqlparse")
    
    def parse(self, sql: str) -> SQLStructure:
        """
        Parse SQL query and extract structure.
        
        Args:
            sql: SQL query string
            
        Returns:
            SQLStructure with parsed information
        """
        # Clean and format SQL
        sql = sql.strip()
        formatted = self.format_sql(sql)
        
        # Parse with sqlparse
        parsed = sqlparse.parse(sql)
        if not parsed:
            return SQLStructure(
                query_type=SQLQueryType.UNKNOWN,
                original_sql=sql,
                formatted_sql=formatted,
                issues=[{"type": "ERROR", "message": "Could not parse SQL"}]
            )
        
        statement = parsed[0]
        
        # Create structure
        structure = SQLStructure(
            query_type=self._detect_query_type(statement),
            original_sql=sql,
            formatted_sql=formatted,
        )
        
        # Extract components
        structure.tables = self._extract_tables(statement)
        structure.select_columns = self._extract_select_columns(statement)
        structure.joins = self._extract_joins(sql)
        
        # Detect clauses
        sql_upper = sql.upper()
        structure.has_where = ' WHERE ' in sql_upper
        structure.has_group_by = ' GROUP BY ' in sql_upper
        structure.has_having = ' HAVING ' in sql_upper
        structure.has_order_by = ' ORDER BY ' in sql_upper
        structure.has_limit = ' LIMIT ' in sql_upper or ' FETCH ' in sql_upper or ' ROWNUM' in sql_upper
        structure.has_distinct = 'SELECT DISTINCT' in sql_upper or 'SELECT ALL DISTINCT' in sql_upper
        structure.has_subquery = sql_upper.count('SELECT') > 1
        structure.has_cte = sql_upper.strip().startswith('WITH ')
        
        # Detect aggregations
        structure.aggregate_functions = self._extract_aggregate_functions(sql)
        structure.has_aggregation = len(structure.aggregate_functions) > 0
        
        # Calculate metrics
        structure.table_count = len(structure.tables)
        structure.join_count = len(structure.joins)
        structure.subquery_count = max(0, sql_upper.count('SELECT') - 1)
        structure.condition_count = sql_upper.count(' AND ') + sql_upper.count(' OR ') + 1 if structure.has_where else 0
        
        # Check for issues
        structure.issues = self._check_issues(sql, structure)
        
        return structure
    
    def _detect_query_type(self, statement) -> SQLQueryType:
        """Detect the type of SQL query"""
        query_type = statement.get_type()
        
        type_map = {
            'SELECT': SQLQueryType.SELECT,
            'INSERT': SQLQueryType.INSERT,
            'UPDATE': SQLQueryType.UPDATE,
            'DELETE': SQLQueryType.DELETE,
            'CREATE': SQLQueryType.CREATE,
            'ALTER': SQLQueryType.ALTER,
            'DROP': SQLQueryType.DROP,
            'TRUNCATE': SQLQueryType.TRUNCATE,
            'MERGE': SQLQueryType.MERGE,
        }
        
        # Check for CTE (WITH clause)
        sql_upper = str(statement).upper().strip()
        if sql_upper.startswith('WITH '):
            return SQLQueryType.WITH
        
        return type_map.get(query_type, SQLQueryType.UNKNOWN)
    
    def _extract_tables(self, statement) -> List[TableReference]:
        """Extract table references from SQL"""
        tables = []
        
        # Get FROM clause tokens
        from_seen = False
        for token in statement.tokens:
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
                continue
            
            if from_seen:
                if token.ttype is Keyword and token.value.upper() in ('WHERE', 'GROUP', 'ORDER', 'LIMIT', 'HAVING'):
                    break
                
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        table = self._parse_table_identifier(identifier)
                        if table:
                            tables.append(table)
                elif isinstance(token, Identifier):
                    table = self._parse_table_identifier(token)
                    if table:
                        tables.append(table)
                elif token.ttype is Name or (token.ttype is None and not token.is_whitespace):
                    name = token.value.strip()
                    if name and name.upper() not in ('JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'ON'):
                        tables.append(TableReference(name=name))
        
        return tables
    
    def _parse_table_identifier(self, identifier) -> Optional[TableReference]:
        """Parse a table identifier (possibly with alias)"""
        if isinstance(identifier, Identifier):
            name = identifier.get_real_name()
            alias = identifier.get_alias()
            
            # Handle schema.table notation
            parts = name.split('.') if name else []
            if len(parts) == 2:
                return TableReference(name=parts[1], alias=alias, schema=parts[0])
            elif name:
                return TableReference(name=name, alias=alias)
        
        return None
    
    def _extract_select_columns(self, statement) -> List[ColumnReference]:
        """Extract columns from SELECT clause"""
        columns = []
        
        select_seen = False
        for token in statement.tokens:
            if token.ttype is DML and token.value.upper() == 'SELECT':
                select_seen = True
                continue
            
            if select_seen:
                if token.ttype is Keyword and token.value.upper() in ('FROM', 'INTO'):
                    break
                
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        col = self._parse_column_identifier(identifier)
                        if col:
                            columns.append(col)
                elif isinstance(token, Identifier):
                    col = self._parse_column_identifier(token)
                    if col:
                        columns.append(col)
                elif isinstance(token, Function):
                    columns.append(ColumnReference(
                        name=str(token),
                        is_aggregate=self._is_aggregate_function(token)
                    ))
        
        return columns
    
    def _parse_column_identifier(self, identifier) -> Optional[ColumnReference]:
        """Parse a column identifier"""
        if isinstance(identifier, Identifier):
            name = identifier.get_real_name()
            alias = identifier.get_alias()
            
            # Check for table.column notation
            parts = name.split('.') if name else []
            if len(parts) == 2:
                return ColumnReference(name=parts[1], table=parts[0], alias=alias)
            elif name:
                # Check if it's an aggregate function
                is_agg = any(func in str(identifier).upper() for func in self.AGGREGATE_FUNCTIONS)
                return ColumnReference(name=name, alias=alias, is_aggregate=is_agg)
        
        return None
    
    def _is_aggregate_function(self, token) -> bool:
        """Check if token is an aggregate function"""
        func_name = str(token).split('(')[0].upper().strip()
        return func_name in self.AGGREGATE_FUNCTIONS
    
    def _extract_joins(self, sql: str) -> List[JoinClause]:
        """Extract JOIN clauses from SQL"""
        joins = []
        
        # Patterns for different join types
        join_patterns = [
            (r'INNER\s+JOIN\s+([`"\[\]]?\w+[`"\]\]]?)(?:\s+(?:AS\s+)?([`"\[\]]?\w+[`"\]\]]?))?', JoinType.INNER),
            (r'LEFT\s+(?:OUTER\s+)?JOIN\s+([`"\[\]]?\w+[`"\]\]]?)(?:\s+(?:AS\s+)?([`"\[\]]?\w+[`"\]\]]?))?', JoinType.LEFT),
            (r'RIGHT\s+(?:OUTER\s+)?JOIN\s+([`"\[\]]?\w+[`"\]\]]?)(?:\s+(?:AS\s+)?([`"\[\]]?\w+[`"\]\]]?))?', JoinType.RIGHT),
            (r'FULL\s+(?:OUTER\s+)?JOIN\s+([`"\[\]]?\w+[`"\]\]]?)(?:\s+(?:AS\s+)?([`"\[\]]?\w+[`"\]\]]?))?', JoinType.FULL),
            (r'CROSS\s+JOIN\s+([`"\[\]]?\w+[`"\]\]]?)(?:\s+(?:AS\s+)?([`"\[\]]?\w+[`"\]\]]?))?', JoinType.CROSS),
            (r'NATURAL\s+JOIN\s+([`"\[\]]?\w+[`"\]\]]?)(?:\s+(?:AS\s+)?([`"\[\]]?\w+[`"\]\]]?))?', JoinType.NATURAL),
            (r'(?<!INNER\s)(?<!LEFT\s)(?<!RIGHT\s)(?<!FULL\s)(?<!CROSS\s)(?<!NATURAL\s)JOIN\s+([`"\[\]]?\w+[`"\]\]]?)(?:\s+(?:AS\s+)?([`"\[\]]?\w+[`"\]\]]?))?', JoinType.INNER),
        ]
        
        for pattern, join_type in join_patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            for match in matches:
                table_name = re.sub(r'[`"\[\]]', '', match[0])
                alias = re.sub(r'[`"\[\]]', '', match[1]) if len(match) > 1 and match[1] else None
                
                joins.append(JoinClause(
                    join_type=join_type,
                    table=TableReference(name=table_name, alias=alias)
                ))
        
        return joins
    
    def _extract_aggregate_functions(self, sql: str) -> List[str]:
        """Extract aggregate function calls from SQL"""
        functions = []
        
        for func in self.AGGREGATE_FUNCTIONS:
            pattern = rf'{func}\s*\([^)]*\)'
            matches = re.findall(pattern, sql, re.IGNORECASE)
            functions.extend(matches)
        
        return functions
    
    def _check_issues(self, sql: str, structure: SQLStructure) -> List[Dict[str, str]]:
        """Check for common SQL issues"""
        issues = []
        sql_upper = sql.upper()
        
        # SELECT * warning
        if 'SELECT *' in sql_upper or 'SELECT  *' in sql_upper:
            issues.append({
                "type": "WARNING",
                "message": "SELECT * is not recommended for production - specify columns explicitly"
            })
        
        # UPDATE/DELETE without WHERE
        if structure.query_type == SQLQueryType.UPDATE and not structure.has_where:
            issues.append({
                "type": "WARNING",
                "message": "UPDATE without WHERE clause will affect all rows"
            })
        
        if structure.query_type == SQLQueryType.DELETE and not structure.has_where:
            issues.append({
                "type": "WARNING",
                "message": "DELETE without WHERE clause will delete all rows"
            })
        
        # GROUP BY without aggregation
        if structure.has_group_by and not structure.has_aggregation:
            issues.append({
                "type": "WARNING",
                "message": "GROUP BY without aggregate functions may not produce expected results"
            })
        
        # Aggregation without GROUP BY (when multiple non-aggregated columns)
        if structure.has_aggregation and not structure.has_group_by:
            non_agg_cols = [c for c in structure.select_columns if not c.is_aggregate]
            if len(non_agg_cols) > 0:
                issues.append({
                    "type": "WARNING",
                    "message": "Mixing aggregated and non-aggregated columns without GROUP BY"
                })
        
        # Missing LIMIT on large SELECT
        if structure.query_type == SQLQueryType.SELECT and not structure.has_where and not structure.has_limit:
            issues.append({
                "type": "INFO",
                "message": "Consider adding LIMIT for large tables without WHERE clause"
            })
        
        # Potential Cartesian product
        if structure.table_count > 1 and structure.join_count == 0 and structure.query_type == SQLQueryType.SELECT:
            issues.append({
                "type": "WARNING",
                "message": "Multiple tables without explicit JOIN - possible Cartesian product"
            })
        
        return issues
    
    def format_sql(
        self,
        sql: str,
        reindent: bool = True,
        keyword_case: str = 'upper',
        identifier_case: Optional[str] = None,
        strip_comments: bool = False,
    ) -> str:
        """
        Format SQL query for readability.
        
        Args:
            sql: SQL query to format
            reindent: Whether to reindent
            keyword_case: Case for keywords ('upper', 'lower', 'capitalize')
            identifier_case: Case for identifiers
            strip_comments: Whether to remove comments
            
        Returns:
            Formatted SQL
        """
        try:
            return sqlparse.format(
                sql,
                reindent=reindent,
                keyword_case=keyword_case,
                identifier_case=identifier_case,
                strip_comments=strip_comments,
                indent_tabs=False,
                indent_width=2,
            )
        except Exception:
            return sql


def extract_sql_from_text(text: str) -> str:
    """
    Extract SQL query from text that may contain markdown or explanations.
    
    Args:
        text: Text potentially containing SQL
        
    Returns:
        Extracted SQL query
    """
    # Try SQL code block first
    sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try generic code block
    code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # Try to find SQL statement patterns
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']
    
    for keyword in sql_keywords:
        pattern = rf'({keyword}\s+.*?)(?:;|\Z)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Return cleaned text as fallback
    return text.strip()


def format_sql(
    sql: str,
    reindent: bool = True,
    keyword_case: str = 'upper',
) -> str:
    """
    Format SQL query for readability.
    
    Args:
        sql: SQL query to format
        reindent: Whether to reindent
        keyword_case: Case for keywords
        
    Returns:
        Formatted SQL
    """
    if not SQLPARSE_AVAILABLE:
        return sql
    
    try:
        return sqlparse.format(
            sql,
            reindent=reindent,
            keyword_case=keyword_case,
            indent_tabs=False,
            indent_width=2,
        )
    except Exception:
        return sql


def extract_tables_from_sql(sql: str) -> List[str]:
    """
    Extract table names from SQL query.
    
    Args:
        sql: SQL query
        
    Returns:
        List of table names
    """
    tables = []
    
    # Patterns for table references
    patterns = [
        r'FROM\s+([`"\[\]]?\w+[`"\]\]]?)',
        r'JOIN\s+([`"\[\]]?\w+[`"\]\]]?)',
        r'INTO\s+([`"\[\]]?\w+[`"\]\]]?)',
        r'UPDATE\s+([`"\[\]]?\w+[`"\]\]]?)',
        r'TABLE\s+([`"\[\]]?\w+[`"\]\]]?)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, sql, re.IGNORECASE)
        for match in matches:
            # Remove quotes
            clean = re.sub(r'[`"\[\]]', '', match)
            if clean and clean.upper() not in SQL_RESERVED_WORDS:
                tables.append(clean)
    
    return list(set(tables))


def extract_columns_from_sql(sql: str) -> List[Tuple[Optional[str], str]]:
    """
    Extract column references from SQL query.
    
    Args:
        sql: SQL query
        
    Returns:
        List of (table, column) tuples
    """
    columns = []
    
    # Pattern for table.column
    pattern = r'([`"\[\]]?\w+[`"\]\]]?)\.([`"\[\]]?\w+[`"\]\]]?)'
    matches = re.findall(pattern, sql)
    
    for table, column in matches:
        table_clean = re.sub(r'[`"\[\]]', '', table)
        column_clean = re.sub(r'[`"\[\]]', '', column)
        
        # Skip if it looks like a function or keyword
        if column_clean.upper() not in SQL_RESERVED_WORDS:
            columns.append((table_clean, column_clean))
    
    return columns


def sanitize_identifier(identifier: str, dialect: str = 'ansi') -> str:
    """
    Sanitize SQL identifier (table/column name).
    
    Args:
        identifier: Identifier to sanitize
        dialect: SQL dialect for quoting style
        
    Returns:
        Sanitized and properly quoted identifier
    """
    # Remove any existing quotes
    identifier = re.sub(r'[`"\[\]]', '', identifier)
    
    # Check if quoting is needed
    needs_quoting = (
        not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier) or
        identifier.upper() in SQL_RESERVED_WORDS
    )
    
    if not needs_quoting:
        return identifier
    
    # Quote based on dialect
    quotes = {
        'mysql': ('`', '`'),
        'postgresql': ('"', '"'),
        'postgres': ('"', '"'),
        'oracle': ('"', '"'),
        'sqlserver': ('[', ']'),
        'mssql': ('[', ']'),
        'sqlite': ('"', '"'),
        'ansi': ('"', '"'),
    }
    
    left, right = quotes.get(dialect.lower(), ('"', '"'))
    return f"{left}{identifier}{right}"


def is_safe_query(sql: str) -> Tuple[bool, List[str]]:
    """
    Check if SQL query is safe to execute.
    
    Checks for:
    - Dangerous statements (DROP, TRUNCATE)
    - SQL injection patterns
    - Multiple statements
    
    Args:
        sql: SQL query to check
        
    Returns:
        Tuple of (is_safe, list_of_issues)
    """
    issues = []
    sql_upper = sql.upper()
    
    # Check for dangerous statements
    dangerous_patterns = [
        (r'\bDROP\s+TABLE\b', "DROP TABLE statement detected"),
        (r'\bDROP\s+DATABASE\b', "DROP DATABASE statement detected"),
        (r'\bTRUNCATE\s+TABLE\b', "TRUNCATE TABLE statement detected"),
        (r'\bDELETE\s+FROM\s+\w+\s*(?:;|$)', "DELETE without WHERE clause"),
        (r'\bUPDATE\s+\w+\s+SET\s+.*(?:;|$)(?!.*WHERE)', "UPDATE without WHERE clause"),
    ]
    
    for pattern, message in dangerous_patterns:
        if re.search(pattern, sql_upper):
            issues.append(message)
    
    # Check for SQL injection patterns
    injection_patterns = [
        (r";\s*--", "Comment after semicolon (potential injection)"),
        (r"'\s*OR\s+'?\d+'?\s*=\s*'?\d+", "OR always-true condition (potential injection)"),
        (r"'\s*OR\s+'[^']*'\s*=\s*'[^']*'", "OR string comparison (potential injection)"),
        (r"UNION\s+SELECT", "UNION SELECT (potential injection)"),
        (r";\s*(?:DROP|DELETE|UPDATE|INSERT)", "Multiple statements (potential injection)"),
    ]
    
    for pattern, message in injection_patterns:
        if re.search(pattern, sql_upper):
            issues.append(message)
    
    # Check for multiple statements
    statements = [s.strip() for s in sql.split(';') if s.strip()]
    if len(statements) > 1:
        issues.append(f"Multiple SQL statements detected ({len(statements)})")
    
    return (len(issues) == 0, issues)


def get_query_complexity(sql: str) -> Dict[str, Any]:
    """
    Analyze SQL query complexity.
    
    Args:
        sql: SQL query
        
    Returns:
        Dict with complexity metrics
    """
    sql_upper = sql.upper()
    
    return {
        "table_count": len(extract_tables_from_sql(sql)),
        "join_count": sql_upper.count(' JOIN '),
        "subquery_count": max(0, sql_upper.count('SELECT') - 1),
        "condition_count": sql_upper.count(' AND ') + sql_upper.count(' OR '),
        "has_aggregation": any(f in sql_upper for f in ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(']),
        "has_grouping": ' GROUP BY ' in sql_upper,
        "has_ordering": ' ORDER BY ' in sql_upper,
        "has_limit": ' LIMIT ' in sql_upper or ' FETCH ' in sql_upper,
        "has_distinct": 'DISTINCT' in sql_upper,
        "has_union": ' UNION ' in sql_upper,
        "has_cte": sql_upper.strip().startswith('WITH '),
        "character_count": len(sql),
    }


# Convenience function for parsing
def parse_sql(sql: str) -> SQLStructure:
    """
    Parse SQL query and return structure.
    
    Args:
        sql: SQL query string
        
    Returns:
        SQLStructure with parsed information
    """
    parser = SQLParser()
    return parser.parse(sql)
