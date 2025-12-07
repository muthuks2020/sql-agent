"""
SQL Agent System
================

A production-ready, modular multi-agent system for SQL generation and validation
using AWS Strands Agents SDK and AWS Bedrock Claude.

Features:
- Natural language to SQL conversion using Claude AI
- Multi-database support (MySQL, PostgreSQL, Oracle, SQLite)
- Self-healing SQL validation and correction
- Intelligent schema understanding from files or auto-discovery
- AWS Strands Agents SDK integration (with boto3 fallback)
- Thread-safe operation for load testing
- Comprehensive metrics and logging

Quick Start:
------------

    from sql_agent_system import create_pipeline, SQLGenerationRequest
    
    # Create pipeline
    pipeline = create_pipeline(
        db_type="mysql",
        database="mydb",
        host="localhost",
        username="user",
        password="password"
    )
    
    # Generate SQL
    request = SQLGenerationRequest(
        user_query="Show me all users who registered last month",
        database_type="mysql"
    )
    
    response = pipeline.process(request)
    print(response.final_sql)

With Schema Files:
------------------

    # Use schema files for better understanding
    pipeline = create_pipeline(
        db_type="postgresql",
        database="myapp",
        schema_file="schema.yaml",      # Table/column descriptions
        description_file="docs.txt",     # Natural language docs
    )

Using Strands Agents Directly:
------------------------------

    from sql_agent_system import create_strands_agent
    
    # Create a Strands agent with custom system prompt
    agent = create_strands_agent(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        system_prompt="You are a SQL expert.",
        region="us-east-1"
    )
    
    result = agent("Generate SQL for listing all users")
"""

__version__ = "1.0.0"
__author__ = "SQL Agent Team"

# Configuration
from .config import (
    DatabaseType,
    LLMProvider,
    LogLevel,
    DatabaseConfig,
    LLMConfig,
    AgentConfig,
    MetricsConfig,
    LoadTestConfig,
    SystemConfig,
    get_config,
    set_config,
)

# Database Adapters
from .adapters import (
    BaseDatabaseAdapter,
    DatabaseAdapterRegistry,
    DatabaseSchema,
    TableSchema,
    ColumnSchema,
    QueryResult,
    create_adapter,
    get_supported_databases,
    MySQLAdapter,
    PostgreSQLAdapter,
    OracleAdapter,
    SQLiteAdapter,
)

# LLM Client
from .llm_client import (
    BaseLLMClient,
    BedrockClaudeClient,
    StrandsAgentClient,
    LLMClientFactory,
    LLMResponse,
    ConversationContext,
    get_llm_client,
    create_strands_agent,
)

# Agents
from .agents import (
    BaseAgent,
    SQLGeneratorAgent,
    SQLValidatorAgent,
    SQLHealerAgent,
)

# Schemas
from .schemas import (
    RequestStatus,
    SQLQueryType,
    SQLGenerationRequest,
    SQLGenerationResponse,
    SQLGenerationResult,
    SQLValidationResult,
    SQLValidationIssue,
    HealingAttempt,
    AgentState,
)

# Orchestration
from .orchestration import (
    SQLAgentPipeline,
    PipelineConfig,
    PipelineBuilder,
    create_pipeline,
)

# Utilities
from .utils import (
    setup_logging,
    get_logger,
    SQLAgentError,
    DatabaseConnectionError,
    SchemaError,
    SQLSyntaxError,
    SQLSemanticError,
    LLMError,
    ValidationError,
    get_metrics_collector,
    SQLAgentMetrics,
)

# Schema Intelligence (optional but recommended)
try:
    from .schema_intelligence import (
        # Models
        SemanticType,
        RelationshipType,
        ColumnSemantics,
        TableSemantics,
        Relationship,
        BusinessContext,
        SemanticModel,
        # Providers
        FileSchemaProvider,
        TextDescriptionProvider,
        DatabaseSchemaProvider,
        HybridSchemaProvider,
        # Builder
        SemanticModelBuilder,
        QuickModelBuilder,
        SchemaContextGenerator,
        ModelBuilderConfig,
        # Discovery
        RelationshipDiscoveryEngine,
        # Manager (main entry point)
        SchemaIntelligenceManager,
        SchemaIntelligenceConfig,
        create_schema_model,
        discover_schema_from_database,
        # Smart Loader
        SmartSchemaLoader,
        LoaderConfig,
        load_schema,
        analyze_database,
    )
    SCHEMA_INTELLIGENCE_AVAILABLE = True
except ImportError:
    SCHEMA_INTELLIGENCE_AVAILABLE = False

# Query Understanding (NLP analysis of user queries)
try:
    from .query_understanding import (
        QueryUnderstanding,
        QueryAnalysis,
        SQLOperationType,
        AggregationType,
        FilterType,
        TimeFrame,
        SortDirection,
        TimeReference,
        SortRequirement,
        FilterRequirement,
        JoinRequirement,
        analyze_query,
    )
    QUERY_UNDERSTANDING_AVAILABLE = True
except ImportError:
    QUERY_UNDERSTANDING_AVAILABLE = False

# SQL Understanding (SQL parsing and analysis)
try:
    from .sql_understanding import (
        SQLParser,
        SQLStructure,
        SQLQueryType as ParsedSQLQueryType,
        JoinType,
        TableReference,
        ColumnReference,
        JoinClause,
        extract_sql_from_text,
        format_sql,
        extract_tables_from_sql,
        extract_columns_from_sql,
        sanitize_identifier,
        is_safe_query,
        get_query_complexity,
        parse_sql,
        SQLPARSE_AVAILABLE,
    )
    SQL_UNDERSTANDING_AVAILABLE = True
except ImportError:
    SQL_UNDERSTANDING_AVAILABLE = False

__all__ = [
    # Version
    "__version__",
    # Configuration
    "DatabaseType",
    "LLMProvider",
    "LogLevel",
    "DatabaseConfig",
    "LLMConfig",
    "AgentConfig",
    "MetricsConfig",
    "LoadTestConfig",
    "SystemConfig",
    "get_config",
    "set_config",
    # Adapters
    "BaseDatabaseAdapter",
    "DatabaseAdapterRegistry",
    "DatabaseSchema",
    "TableSchema",
    "ColumnSchema",
    "QueryResult",
    "create_adapter",
    "get_supported_databases",
    "MySQLAdapter",
    "PostgreSQLAdapter",
    "OracleAdapter",
    "SQLiteAdapter",
    # LLM
    "BaseLLMClient",
    "BedrockClaudeClient",
    "StrandsAgentClient",
    "LLMClientFactory",
    "LLMResponse",
    "ConversationContext",
    "get_llm_client",
    "create_strands_agent",
    # Agents
    "BaseAgent",
    "SQLGeneratorAgent",
    "SQLValidatorAgent",
    "SQLHealerAgent",
    # Schemas
    "RequestStatus",
    "SQLQueryType",
    "SQLGenerationRequest",
    "SQLGenerationResponse",
    "SQLGenerationResult",
    "SQLValidationResult",
    "SQLValidationIssue",
    "HealingAttempt",
    "AgentState",
    # Orchestration
    "SQLAgentPipeline",
    "PipelineConfig",
    "PipelineBuilder",
    "create_pipeline",
    # Utilities
    "setup_logging",
    "get_logger",
    "SQLAgentError",
    "DatabaseConnectionError",
    "SchemaError",
    "SQLSyntaxError",
    "SQLSemanticError",
    "LLMError",
    "ValidationError",
    "get_metrics_collector",
    "SQLAgentMetrics",
    # Schema Intelligence
    "SCHEMA_INTELLIGENCE_AVAILABLE",
    "SemanticType",
    "RelationshipType",
    "ColumnSemantics",
    "TableSemantics",
    "Relationship",
    "BusinessContext",
    "SemanticModel",
    "FileSchemaProvider",
    "TextDescriptionProvider",
    "DatabaseSchemaProvider",
    "HybridSchemaProvider",
    "SemanticModelBuilder",
    "QuickModelBuilder",
    "SchemaContextGenerator",
    "ModelBuilderConfig",
    "RelationshipDiscoveryEngine",
    "SchemaIntelligenceManager",
    "SchemaIntelligenceConfig",
    "create_schema_model",
    "discover_schema_from_database",
    "SmartSchemaLoader",
    "LoaderConfig",
    "load_schema",
    "analyze_database",
    # Query Understanding
    "QUERY_UNDERSTANDING_AVAILABLE",
    "QueryUnderstanding",
    "QueryAnalysis",
    "SQLOperationType",
    "AggregationType",
    "FilterType",
    "TimeFrame",
    "SortDirection",
    "TimeReference",
    "SortRequirement",
    "FilterRequirement",
    "JoinRequirement",
    "analyze_query",
    # SQL Understanding (Parsing)
    "SQL_UNDERSTANDING_AVAILABLE",
    "SQLPARSE_AVAILABLE",
    "SQLParser",
    "SQLStructure",
    "ParsedSQLQueryType",
    "JoinType",
    "TableReference",
    "ColumnReference",
    "JoinClause",
    "extract_sql_from_text",
    "format_sql",
    "extract_tables_from_sql",
    "extract_columns_from_sql",
    "sanitize_identifier",
    "is_safe_query",
    "get_query_complexity",
    "parse_sql",
]
