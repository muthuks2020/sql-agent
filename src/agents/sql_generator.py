"""
SQL Generator Agent
Generates SQL queries from natural language using LLM
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import AgentConfig, LLMConfig
from ..adapters import BaseDatabaseAdapter, DatabaseSchema
from ..schemas import SQLGenerationResult, SQLQueryType, detect_query_type
from ..utils import get_logger, SQLAgentMetrics
from ..query_understanding import QueryUnderstanding, QueryAnalysis, analyze_query
from .base_agent import BaseAgent

logger = get_logger(__name__)


@dataclass
class SQLGeneratorInput:
    """Input for SQL Generator Agent"""
    user_query: str
    schema: DatabaseSchema
    dialect_hints: Dict[str, Any]
    table_hints: Optional[List[str]] = None
    column_hints: Optional[List[str]] = None
    additional_context: Optional[str] = None
    query_analysis: Optional[QueryAnalysis] = None
    
    def to_prompt_context(self) -> str:
        """Convert to context string for prompt"""
        parts = []
        
        # Add query analysis if available (this helps the LLM understand intent)
        if self.query_analysis:
            parts.append(self.query_analysis.to_prompt_context())
            parts.append("")
        
        # Add schema
        parts.append("DATABASE SCHEMA:")
        parts.append(self.schema.to_schema_string())
        
        # Add dialect hints
        parts.append("\nSQL DIALECT INFORMATION:")
        parts.append(f"Database Type: {self.dialect_hints.get('dialect', 'Unknown')}")
        
        if self.dialect_hints.get('notes'):
            parts.append("Important Notes:")
            for note in self.dialect_hints['notes']:
                parts.append(f"  - {note}")
        
        # Add table hints if provided (or from query analysis)
        effective_table_hints = self.table_hints or []
        if self.query_analysis and self.query_analysis.potential_tables:
            effective_table_hints = list(set(effective_table_hints + self.query_analysis.potential_tables))
        
        if effective_table_hints:
            parts.append(f"\nRELEVANT TABLES (focus on these): {', '.join(effective_table_hints)}")
        
        # Add column hints if provided (or from query analysis)
        effective_column_hints = self.column_hints or []
        if self.query_analysis and self.query_analysis.potential_columns:
            effective_column_hints = list(set(effective_column_hints + self.query_analysis.potential_columns))
        
        if effective_column_hints:
            parts.append(f"RELEVANT COLUMNS: {', '.join(effective_column_hints)}")
        
        # Add additional context
        if self.additional_context:
            parts.append(f"\nADDITIONAL CONTEXT:\n{self.additional_context}")
        
        return "\n".join(parts)


class SQLGeneratorAgent(BaseAgent[SQLGeneratorInput, SQLGenerationResult]):
    """
    SQL Generator Agent
    
    Responsible for generating SQL queries from natural language descriptions.
    Uses database schema and dialect hints to produce accurate SQL.
    """
    
    SYSTEM_PROMPT = """You are an expert SQL query generator. Your task is to convert natural language queries into accurate, efficient SQL statements.

GUIDELINES:
1. Generate syntactically correct SQL for the specified database dialect
2. Use proper table and column names from the provided schema
3. Include appropriate JOINs when querying multiple tables
4. Use aliases for readability when beneficial
5. Add comments only when the query logic is complex
6. Optimize for performance when possible (use indexes, avoid SELECT *)
7. Handle NULL values appropriately
8. Use parameterized query placeholders where user input would be needed

OUTPUT FORMAT:
You must respond with the SQL query wrapped in ```sql``` code blocks.
After the SQL, provide a brief explanation of the query logic.
Also list the tables and columns used.

Example response format:
```sql
SELECT column1, column2
FROM table1
WHERE condition;
```

EXPLANATION: Brief description of what the query does.
TABLES_USED: table1, table2
COLUMNS_USED: column1, column2, column3
CONFIDENCE: 0.95

IMPORTANT:
- Only generate SELECT, INSERT, UPDATE, DELETE statements unless specifically asked for DDL
- If the request is ambiguous, make reasonable assumptions and note them
- If the request cannot be fulfilled with the given schema, explain why
- Never generate SQL that could cause data loss without explicit confirmation words in the request"""
    
    def __init__(
        self,
        llm_config: LLMConfig,
        agent_config: AgentConfig,
    ):
        super().__init__(
            name="sql_generator",
            llm_config=llm_config,
            agent_config=agent_config,
        )
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def _prepare_prompt(self, input_data: SQLGeneratorInput, context: Dict[str, Any]) -> str:
        """Prepare the prompt for SQL generation"""
        prompt_parts = [
            input_data.to_prompt_context(),
            "\n" + "=" * 50,
            "\nUSER REQUEST:",
            input_data.user_query,
            "\nGenerate the SQL query for the above request."
        ]
        
        # Add any previous errors for self-healing context
        if context.get("previous_errors"):
            prompt_parts.append("\nPREVIOUS ATTEMPT FAILED WITH ERRORS:")
            for error in context["previous_errors"]:
                prompt_parts.append(f"  - {error}")
            prompt_parts.append("\nPlease fix these issues in your new SQL query.")
        
        return "\n".join(prompt_parts)
    
    def _parse_response(self, response: str, input_data: SQLGeneratorInput) -> SQLGenerationResult:
        """Parse LLM response into SQLGenerationResult"""
        # Extract SQL from code block
        sql = self._extract_sql(response)
        
        # Extract explanation
        explanation = self._extract_section(response, "EXPLANATION:")
        
        # Extract tables used
        tables_str = self._extract_section(response, "TABLES_USED:")
        tables_used = [t.strip() for t in tables_str.split(",")] if tables_str else []
        
        # Extract columns used
        columns_str = self._extract_section(response, "COLUMNS_USED:")
        columns_used = [c.strip() for c in columns_str.split(",")] if columns_str else []
        
        # Extract confidence
        confidence_str = self._extract_section(response, "CONFIDENCE:")
        try:
            confidence = float(confidence_str) if confidence_str else 0.8
        except ValueError:
            confidence = 0.8
        
        # Detect query type
        query_type = detect_query_type(sql) if sql else SQLQueryType.UNKNOWN
        
        return SQLGenerationResult(
            sql=sql,
            query_type=query_type,
            explanation=explanation,
            confidence_score=confidence,
            tables_used=tables_used,
            columns_used=columns_used,
        )
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL from response"""
        # Try to extract from code block
        sql_pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(sql_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Try generic code block
        code_pattern = r"```\s*(.*?)\s*```"
        match = re.search(code_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If no code block, try to extract SQL-like content
        lines = response.split("\n")
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line_upper = line.strip().upper()
            if any(line_upper.startswith(kw) for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "WITH"]):
                in_sql = True
            
            if in_sql:
                # Stop at explanation markers
                if any(marker in line.upper() for marker in ["EXPLANATION:", "TABLES_USED:", "COLUMNS_USED:", "CONFIDENCE:"]):
                    break
                sql_lines.append(line)
        
        return "\n".join(sql_lines).strip()
    
    def _extract_section(self, response: str, marker: str) -> str:
        """Extract a section from the response"""
        lines = response.split("\n")
        for i, line in enumerate(lines):
            if marker in line.upper():
                # Get content after marker
                content = line.split(":", 1)[-1].strip()
                if content:
                    return content
                # Check next line
                if i + 1 < len(lines):
                    return lines[i + 1].strip()
        return ""
    
    def _validate_output(self, output: SQLGenerationResult) -> bool:
        """Validate the generated SQL"""
        if not output.sql:
            logger.warning("Generated SQL is empty")
            return False
        
        # Basic validation
        sql_upper = output.sql.upper()
        
        # Check for basic SQL structure
        valid_starts = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "WITH"]
        if not any(sql_upper.strip().startswith(kw) for kw in valid_starts):
            logger.warning("SQL does not start with a valid keyword")
            return False
        
        return True
    
    def _handle_validation_failure(
        self,
        output: SQLGenerationResult,
        input_data: SQLGeneratorInput,
        context: Dict[str, Any]
    ) -> SQLGenerationResult:
        """Handle validation failure by returning result with low confidence"""
        output.confidence_score = 0.0
        return output
    
    def generate(
        self,
        user_query: str,
        adapter: BaseDatabaseAdapter,
        table_hints: Optional[List[str]] = None,
        column_hints: Optional[List[str]] = None,
        additional_context: Optional[str] = None,
        previous_errors: Optional[List[str]] = None,
        query_analysis: Optional[QueryAnalysis] = None,
    ) -> SQLGenerationResult:
        """
        Convenience method to generate SQL
        
        Args:
            user_query: Natural language query
            adapter: Database adapter for schema access
            table_hints: Optional list of relevant tables
            column_hints: Optional list of relevant columns
            additional_context: Optional additional context
            previous_errors: Optional list of previous errors for self-healing
            query_analysis: Optional pre-computed query analysis
            
        Returns:
            SQLGenerationResult with generated SQL
        """
        import time
        start_time = time.time()
        
        # Get schema and dialect hints
        schema = adapter.get_schema()
        dialect_hints = adapter.get_sql_dialect_hints()
        
        # Create input with query analysis
        input_data = SQLGeneratorInput(
            user_query=user_query,
            schema=schema,
            dialect_hints=dialect_hints,
            table_hints=table_hints,
            column_hints=column_hints,
            additional_context=additional_context,
            query_analysis=query_analysis,
        )
        
        # Create context
        context = {}
        if previous_errors:
            context["previous_errors"] = previous_errors
        
        # Execute agent
        result = self.execute(input_data, context)
        
        # Record metrics
        duration = time.time() - start_time
        SQLAgentMetrics.record_sql_generation(
            duration=duration,
            success=bool(result.sql),
            db_type=adapter.database_type.value
        )
        
        return result
