"""
SQL Validator Agent
Validates SQL queries and provides self-healing capabilities
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..config import AgentConfig, LLMConfig
from ..adapters import BaseDatabaseAdapter, DatabaseSchema
from ..schemas import (
    SQLValidationResult,
    SQLValidationIssue,
    SQLGenerationResult,
    HealingAttempt,
)
from ..utils import get_logger, SQLAgentMetrics, format_error_for_llm
from .base_agent import BaseAgent

logger = get_logger(__name__)


@dataclass
class SQLValidatorInput:
    """Input for SQL Validator Agent"""
    sql: str
    schema: DatabaseSchema
    dialect_hints: Dict[str, Any]
    original_query: Optional[str] = None
    syntax_error: Optional[str] = None
    
    def to_prompt_context(self) -> str:
        """Convert to context string for prompt"""
        parts = [
            "SQL TO VALIDATE:",
            "```sql",
            self.sql,
            "```",
            "\nDATABASE SCHEMA:",
            self.schema.to_schema_string(),
            f"\nSQL DIALECT: {self.dialect_hints.get('dialect', 'Unknown')}",
        ]
        
        if self.original_query:
            parts.append(f"\nORIGINAL USER REQUEST: {self.original_query}")
        
        if self.syntax_error:
            parts.append(f"\nSYNTAX ERROR FROM DATABASE: {self.syntax_error}")
        
        return "\n".join(parts)


@dataclass
class SQLHealerInput:
    """Input for SQL self-healing"""
    sql: str
    schema: DatabaseSchema
    dialect_hints: Dict[str, Any]
    issues: List[SQLValidationIssue]
    original_query: Optional[str] = None
    
    def to_prompt_context(self) -> str:
        """Convert to context string for prompt"""
        parts = [
            "SQL WITH ISSUES:",
            "```sql",
            self.sql,
            "```",
            "\nIDENTIFIED ISSUES:"
        ]
        
        for i, issue in enumerate(self.issues, 1):
            parts.append(f"{i}. [{issue.severity.upper()}] {issue.issue_type}: {issue.message}")
            if issue.suggestion:
                parts.append(f"   Suggestion: {issue.suggestion}")
        
        parts.append("\nDATABASE SCHEMA:")
        parts.append(self.schema.to_schema_string())
        parts.append(f"\nSQL DIALECT: {self.dialect_hints.get('dialect', 'Unknown')}")
        
        if self.original_query:
            parts.append(f"\nORIGINAL USER REQUEST: {self.original_query}")
        
        return "\n".join(parts)


class SQLValidatorAgent(BaseAgent[SQLValidatorInput, SQLValidationResult]):
    """
    SQL Validator Agent
    
    Responsible for validating SQL queries against schema and syntax rules.
    Identifies issues and provides detailed feedback.
    """
    
    SYSTEM_PROMPT = """You are an expert SQL validator. Your task is to thoroughly validate SQL queries for correctness, efficiency, and safety.

VALIDATION CHECKS:
1. SYNTAX: Check SQL syntax is correct for the specified dialect
2. SCHEMA: Verify all tables, columns, and relationships exist
3. SEMANTIC: Check query logic makes sense (JOINs, WHERE conditions, etc.)
4. SECURITY: Identify potential SQL injection risks or dangerous operations
5. PERFORMANCE: Note potential performance issues (missing indexes, full table scans)
6. BEST PRACTICES: Check adherence to SQL best practices

OUTPUT FORMAT:
Respond with validation results in this exact format:

VALID: true/false
ISSUES:
- [SEVERITY:TYPE] Description of issue | SUGGESTION: How to fix
- [SEVERITY:TYPE] Description of issue | SUGGESTION: How to fix

Where SEVERITY is one of: error, warning, info
Where TYPE is one of: syntax, schema, semantic, security, performance

EXECUTION_PLAN: Brief description of how the query would execute

Example:
VALID: false
ISSUES:
- [error:schema] Table 'users' does not exist in the schema | SUGGESTION: Use 'user_accounts' table instead
- [warning:performance] SELECT * retrieves all columns | SUGGESTION: Specify only needed columns

EXECUTION_PLAN: Full table scan on user_accounts, filter by status

If the SQL is valid with no issues, respond:
VALID: true
ISSUES: none
EXECUTION_PLAN: Description of query execution"""
    
    def __init__(
        self,
        llm_config: LLMConfig,
        agent_config: AgentConfig,
    ):
        super().__init__(
            name="sql_validator",
            llm_config=llm_config,
            agent_config=agent_config,
        )
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def _prepare_prompt(self, input_data: SQLValidatorInput, context: Dict[str, Any]) -> str:
        """Prepare the prompt for validation"""
        return input_data.to_prompt_context() + "\n\nValidate the SQL query above."
    
    def _parse_response(self, response: str, input_data: SQLValidatorInput) -> SQLValidationResult:
        """Parse LLM response into SQLValidationResult"""
        # Extract validity
        is_valid = self._extract_validity(response)
        
        # Extract issues
        issues = self._extract_issues(response)
        
        # Extract execution plan
        execution_plan = self._extract_section(response, "EXECUTION_PLAN:")
        
        return SQLValidationResult(
            is_valid=is_valid,
            issues=issues,
            validated_sql=input_data.sql if is_valid else None,
            execution_plan=execution_plan,
        )
    
    def _extract_validity(self, response: str) -> bool:
        """Extract validity from response"""
        lines = response.upper().split("\n")
        for line in lines:
            if "VALID:" in line:
                return "TRUE" in line.split("VALID:")[-1]
        return False
    
    def _extract_issues(self, response: str) -> List[SQLValidationIssue]:
        """Extract issues from response"""
        issues = []
        
        # Pattern: - [severity:type] message | SUGGESTION: suggestion
        pattern = r"-\s*\[(\w+):(\w+)\]\s*([^|]+)(?:\|\s*SUGGESTION:\s*(.+))?"
        
        for match in re.finditer(pattern, response, re.IGNORECASE):
            severity = match.group(1).lower()
            issue_type = match.group(2).lower()
            message = match.group(3).strip()
            suggestion = match.group(4).strip() if match.group(4) else None
            
            issues.append(SQLValidationIssue(
                issue_type=issue_type,
                severity=severity,
                message=message,
                suggestion=suggestion,
            ))
        
        return issues
    
    def _extract_section(self, response: str, marker: str) -> str:
        """Extract a section from the response"""
        lines = response.split("\n")
        for i, line in enumerate(lines):
            if marker.upper() in line.upper():
                content = line.split(":", 1)[-1].strip()
                if content:
                    return content
                if i + 1 < len(lines):
                    return lines[i + 1].strip()
        return ""
    
    def _validate_output(self, output: SQLValidationResult) -> bool:
        """Validate the validation result"""
        # Result is always valid as long as we got a response
        return True
    
    def validate(
        self,
        sql: str,
        adapter: BaseDatabaseAdapter,
        original_query: Optional[str] = None,
    ) -> SQLValidationResult:
        """
        Validate SQL query
        
        Args:
            sql: SQL query to validate
            adapter: Database adapter for schema and syntax validation
            original_query: Original user query for context
            
        Returns:
            SQLValidationResult with validation details
        """
        import time
        start_time = time.time()
        
        # First, try database-level syntax validation
        syntax_valid, syntax_error = adapter.validate_sql_syntax(sql)
        
        # Get schema and dialect hints
        schema = adapter.get_schema()
        dialect_hints = adapter.get_sql_dialect_hints()
        
        # Create input
        input_data = SQLValidatorInput(
            sql=sql,
            schema=schema,
            dialect_hints=dialect_hints,
            original_query=original_query,
            syntax_error=syntax_error if not syntax_valid else None,
        )
        
        # Execute LLM validation
        result = self.execute(input_data)
        
        # Add syntax error if database validation failed
        if not syntax_valid and syntax_error:
            syntax_issue = SQLValidationIssue(
                issue_type="syntax",
                severity="error",
                message=f"Database syntax error: {syntax_error}",
                suggestion="Fix the SQL syntax according to the error message",
            )
            # Insert at beginning
            result.issues.insert(0, syntax_issue)
            result.is_valid = False
        
        # Record metrics
        duration = time.time() - start_time
        SQLAgentMetrics.record_sql_validation(
            duration=duration,
            success=result.is_valid,
            db_type=adapter.database_type.value
        )
        
        return result


class SQLHealerAgent(BaseAgent[SQLHealerInput, SQLGenerationResult]):
    """
    SQL Self-Healer Agent
    
    Responsible for fixing SQL queries based on identified issues.
    Takes broken SQL and validation issues, produces corrected SQL.
    """
    
    SYSTEM_PROMPT = """You are an expert SQL fixer. Your task is to correct SQL queries based on identified issues.

GUIDELINES:
1. Fix all identified issues while preserving the original query intent
2. Ensure the corrected SQL is syntactically valid for the specified dialect
3. Use correct table and column names from the provided schema
4. Maintain query efficiency and follow best practices
5. Add comments explaining significant changes

OUTPUT FORMAT:
Respond with the corrected SQL in a code block and an explanation:

```sql
-- Corrected SQL query
SELECT corrected_columns
FROM corrected_table
WHERE corrected_conditions;
```

CHANGES_MADE:
1. Description of first fix
2. Description of second fix

CONFIDENCE: 0.95

IMPORTANT:
- Preserve the original query intent
- Only make changes necessary to fix the identified issues
- If an issue cannot be fixed, explain why
- Validate all table/column names against the schema"""
    
    def __init__(
        self,
        llm_config: LLMConfig,
        agent_config: AgentConfig,
    ):
        super().__init__(
            name="sql_healer",
            llm_config=llm_config,
            agent_config=agent_config,
        )
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def _prepare_prompt(self, input_data: SQLHealerInput, context: Dict[str, Any]) -> str:
        """Prepare the prompt for healing"""
        return input_data.to_prompt_context() + "\n\nFix the issues in the SQL query above."
    
    def _parse_response(self, response: str, input_data: SQLHealerInput) -> SQLGenerationResult:
        """Parse LLM response into SQLGenerationResult"""
        from .sql_generator import SQLGeneratorAgent
        
        # Reuse SQL extraction from generator
        sql = self._extract_sql(response)
        
        # Extract changes made
        changes = self._extract_section(response, "CHANGES_MADE:")
        
        # Extract confidence
        confidence_str = self._extract_section(response, "CONFIDENCE:")
        try:
            confidence = float(confidence_str) if confidence_str else 0.8
        except ValueError:
            confidence = 0.8
        
        from ..schemas import detect_query_type
        query_type = detect_query_type(sql) if sql else SQLQueryType.UNKNOWN
        
        return SQLGenerationResult(
            sql=sql,
            query_type=query_type,
            explanation=changes,
            confidence_score=confidence,
        )
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL from response"""
        sql_pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(sql_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        code_pattern = r"```\s*(.*?)\s*```"
        match = re.search(code_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _extract_section(self, response: str, marker: str) -> str:
        """Extract a section from the response"""
        lines = response.split("\n")
        result_lines = []
        capturing = False
        
        for line in lines:
            if marker.upper() in line.upper():
                capturing = True
                content = line.split(":", 1)[-1].strip()
                if content:
                    result_lines.append(content)
                continue
            
            if capturing:
                if any(m in line.upper() for m in ["CONFIDENCE:", "```"]):
                    break
                if line.strip():
                    result_lines.append(line.strip())
        
        return "\n".join(result_lines)
    
    def _validate_output(self, output: SQLGenerationResult) -> bool:
        """Validate the healed SQL"""
        return bool(output.sql)
    
    def heal(
        self,
        sql: str,
        issues: List[SQLValidationIssue],
        adapter: BaseDatabaseAdapter,
        original_query: Optional[str] = None,
    ) -> Tuple[SQLGenerationResult, HealingAttempt]:
        """
        Heal SQL query
        
        Args:
            sql: SQL query with issues
            issues: List of identified issues
            adapter: Database adapter for schema access
            original_query: Original user query for context
            
        Returns:
            Tuple of (healed result, healing attempt record)
        """
        import time
        start_time = time.time()
        
        # Get schema and dialect hints
        schema = adapter.get_schema()
        dialect_hints = adapter.get_sql_dialect_hints()
        
        # Create input
        input_data = SQLHealerInput(
            sql=sql,
            schema=schema,
            dialect_hints=dialect_hints,
            issues=issues,
            original_query=original_query,
        )
        
        # Execute healing
        result = self.execute(input_data)
        
        # Create healing attempt record
        healing_attempt = HealingAttempt(
            iteration=self._metadata.iteration_count,
            original_sql=sql,
            healed_sql=result.sql,
            issues_addressed=[issue.message for issue in issues],
            success=bool(result.sql and result.sql != sql),
        )
        
        # Record metrics
        duration = time.time() - start_time
        SQLAgentMetrics.record_self_healing(
            duration=duration,
            success=healing_attempt.success,
            iterations=1,
            db_type=adapter.database_type.value
        )
        
        return result, healing_attempt


# Import for type hints
from ..schemas import SQLQueryType
