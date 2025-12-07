"""
SQL Agent Orchestration Pipeline
Coordinates SQL Generator and Validator agents in a self-healing loop

Supports intelligent schema loading from:
- Schema files (YAML/JSON)
- Text description files
- Database auto-discovery (with relationship inference)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import uuid

from ..config import SystemConfig, AgentConfig, LLMConfig, DatabaseConfig
from ..adapters import BaseDatabaseAdapter, create_adapter
from ..agents import SQLGeneratorAgent, SQLValidatorAgent, SQLHealerAgent
from ..schemas import (
    SQLGenerationRequest,
    SQLGenerationResponse,
    SQLGenerationResult,
    SQLValidationResult,
    HealingAttempt,
    RequestStatus,
    AgentState,
)
from ..utils import (
    get_logger,
    set_correlation_id,
    set_request_id,
    log_context,
    log_operation,
    SQLAgentMetrics,
    MaxRetriesExceededError,
    SelfHealingFailedError,
)

# Schema intelligence imports
try:
    from ..schema_intelligence import (
        SemanticModel,
        SemanticModelBuilder,
        QuickModelBuilder,
        SchemaContextGenerator,
        ModelBuilderConfig,
    )
    SCHEMA_INTELLIGENCE_AVAILABLE = True
except ImportError:
    SCHEMA_INTELLIGENCE_AVAILABLE = False

# Query understanding imports
try:
    from ..query_understanding import QueryUnderstanding, QueryAnalysis, analyze_query
    QUERY_UNDERSTANDING_AVAILABLE = True
except ImportError:
    QUERY_UNDERSTANDING_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the orchestration pipeline"""
    max_healing_iterations: int = 5
    enable_self_healing: bool = True
    validation_strictness: str = "medium"  # low, medium, high
    parallel_validation: bool = False
    timeout_seconds: float = 300.0
    fail_fast: bool = False
    
    # Schema Intelligence options
    schema_file: Optional[str] = None  # Path to YAML/JSON schema file
    description_file: Optional[str] = None  # Path to text description file
    auto_discover_relationships: bool = True
    use_llm_for_relationships: bool = False  # Expensive but more accurate
    use_semantic_context: bool = True  # Use enriched context for better SQL


class SQLAgentPipeline:
    """
    Main orchestration pipeline for SQL generation and validation
    
    Coordinates the flow between SQL Generator, Validator, and Healer agents
    in a self-healing loop until valid SQL is produced or max iterations reached.
    
    Supports intelligent schema loading from:
    - Schema files (YAML/JSON) with column descriptions and relationships
    - Text description files (natural language docs)
    - Database auto-discovery with relationship inference
    
    Usage:
        # Basic usage
        pipeline = SQLAgentPipeline(config)
        response = pipeline.process(request)
        
        # With schema files
        pipeline = SQLAgentPipeline(
            config,
            pipeline_config=PipelineConfig(
                schema_file="schema.yaml",
                description_file="docs.txt",
            )
        )
    """
    
    def __init__(
        self,
        config: SystemConfig,
        pipeline_config: Optional[PipelineConfig] = None,
    ):
        self.config = config
        self.pipeline_config = pipeline_config or PipelineConfig(
            max_healing_iterations=config.agent.max_iterations,
            enable_self_healing=config.agent.enable_self_healing,
            validation_strictness=config.agent.validation_strictness,
        )
        
        # Initialize agents
        self._generator: Optional[SQLGeneratorAgent] = None
        self._validator: Optional[SQLValidatorAgent] = None
        self._healer: Optional[SQLHealerAgent] = None
        
        # Initialize database adapter
        self._adapter: Optional[BaseDatabaseAdapter] = None
        
        # Schema intelligence
        self._semantic_model: Optional[SemanticModel] = None if SCHEMA_INTELLIGENCE_AVAILABLE else None
        self._context_generator: Optional[SchemaContextGenerator] = None if SCHEMA_INTELLIGENCE_AVAILABLE else None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # State tracking
        self._active_requests: Dict[str, AgentState] = {}
    
    @property
    def generator(self) -> SQLGeneratorAgent:
        """Get or create SQL Generator agent"""
        if self._generator is None:
            self._generator = SQLGeneratorAgent(
                llm_config=self.config.llm,
                agent_config=self.config.agent,
            )
        return self._generator
    
    @property
    def validator(self) -> SQLValidatorAgent:
        """Get or create SQL Validator agent"""
        if self._validator is None:
            self._validator = SQLValidatorAgent(
                llm_config=self.config.llm,
                agent_config=self.config.agent,
            )
        return self._validator
    
    @property
    def healer(self) -> SQLHealerAgent:
        """Get or create SQL Healer agent"""
        if self._healer is None:
            self._healer = SQLHealerAgent(
                llm_config=self.config.llm,
                agent_config=self.config.agent,
            )
        return self._healer
    
    @property
    def adapter(self) -> BaseDatabaseAdapter:
        """Get or create database adapter"""
        if self._adapter is None:
            self._adapter = create_adapter(self.config.database)
        return self._adapter
    
    @property
    def semantic_model(self) -> Optional[SemanticModel]:
        """Get or build semantic model"""
        if not SCHEMA_INTELLIGENCE_AVAILABLE:
            return None
        
        if self._semantic_model is None:
            self._build_semantic_model()
        return self._semantic_model
    
    def set_adapter(self, adapter: BaseDatabaseAdapter) -> None:
        """Set a custom database adapter"""
        self._adapter = adapter
        # Reset semantic model when adapter changes
        self._semantic_model = None
        self._context_generator = None
    
    def set_semantic_model(self, model: SemanticModel) -> None:
        """Set a pre-built semantic model"""
        if not SCHEMA_INTELLIGENCE_AVAILABLE:
            logger.warning("Schema intelligence not available")
            return
        self._semantic_model = model
        self._context_generator = SchemaContextGenerator(model)
    
    def _build_semantic_model(self) -> None:
        """Build semantic model from available sources"""
        if not SCHEMA_INTELLIGENCE_AVAILABLE:
            return
        
        try:
            # Ensure adapter is connected
            if not self.adapter.is_connected():
                self.adapter.connect()
            
            # Build model using QuickModelBuilder
            if self.pipeline_config.schema_file or self.pipeline_config.description_file:
                # Hybrid mode: files + database
                self._semantic_model = QuickModelBuilder.from_hybrid(
                    adapter=self.adapter,
                    schema_file=self.pipeline_config.schema_file,
                    description_file=self.pipeline_config.description_file,
                    llm_client=None,  # Don't use LLM for relationships by default
                    use_llm_inference=self.pipeline_config.use_llm_for_relationships,
                )
            else:
                # Database only
                self._semantic_model = QuickModelBuilder.from_database(
                    adapter=self.adapter,
                    discover_relationships=self.pipeline_config.auto_discover_relationships,
                    include_samples=True,
                )
            
            self._context_generator = SchemaContextGenerator(self._semantic_model)
            logger.info(
                f"Built semantic model: {len(self._semantic_model.tables)} tables, "
                f"{len(self._semantic_model.relationships)} relationships"
            )
        
        except Exception as e:
            logger.warning(f"Could not build semantic model: {e}. Falling back to basic schema.")
            self._semantic_model = None
            self._context_generator = None
    
    def _get_schema_context(self, user_query: str) -> str:
        """Get schema context for SQL generation"""
        # Use semantic model if available
        if self._context_generator and self.pipeline_config.use_semantic_context:
            return self._context_generator.generate_query_focused_context(user_query)
        
        # Fallback to basic schema
        schema = self.adapter.get_schema()
        return schema.to_schema_string()
    
    def process(self, request: SQLGenerationRequest) -> SQLGenerationResponse:
        """
        Process a SQL generation request through the full pipeline
        
        Pipeline stages:
        1. Generate initial SQL
        2. Validate SQL
        3. If invalid and self-healing enabled, heal and re-validate
        4. Repeat until valid or max iterations reached
        
        Args:
            request: SQL generation request
            
        Returns:
            SQLGenerationResponse with final SQL or error details
        """
        # Set up context
        correlation_id = request.correlation_id or str(uuid.uuid4())
        set_correlation_id(correlation_id)
        set_request_id(request.request_id)
        
        # Initialize state
        state = AgentState(
            request_id=request.request_id,
            max_iterations=request.max_iterations or self.pipeline_config.max_healing_iterations,
            user_query=request.user_query,
        )
        state.start()
        
        # Track active request
        with self._lock:
            self._active_requests[request.request_id] = state
        
        # Initialize response
        response = SQLGenerationResponse(
            request_id=request.request_id,
            status=RequestStatus.GENERATING,
            started_at=state.started_at,
        )
        
        try:
            with log_context(correlation_id=correlation_id, request_id=request.request_id):
                with log_operation(logger, "sql_pipeline", user_query=request.user_query[:100]):
                    
                    # Ensure database connection
                    if not self.adapter.is_connected():
                        self.adapter.connect()
                    
                    # Get schema context (enhanced with semantic model if available)
                    state.schema_context = self._get_schema_context(request.user_query)
                    state.dialect_hints = self.adapter.get_sql_dialect_hints()
                    
                    # Stage 1: Generate initial SQL
                    logger.info("Stage 1: Generating SQL")
                    generation_result = self._generate_sql(request, state)
                    response.generation_result = generation_result
                    
                    if not generation_result.sql:
                        raise ValueError("SQL generation produced empty result")
                    
                    state.record_generation(generation_result.sql)
                    
                    # Stage 2: Validate SQL
                    logger.info("Stage 2: Validating SQL")
                    state.update_status(RequestStatus.VALIDATING)
                    validation_result = self._validate_sql(
                        generation_result.sql,
                        request.user_query,
                        state
                    )
                    response.validation_result = validation_result
                    state.record_validation(validation_result)
                    
                    # Stage 3: Self-healing loop if needed
                    if not validation_result.is_valid and self.pipeline_config.enable_self_healing:
                        logger.info("Stage 3: Starting self-healing loop")
                        state.update_status(RequestStatus.HEALING)
                        
                        final_sql, healing_attempts = self._healing_loop(
                            generation_result.sql,
                            validation_result,
                            request.user_query,
                            state
                        )
                        
                        response.healing_attempts = healing_attempts
                        response.final_sql = final_sql
                        
                        # Final validation
                        if final_sql:
                            final_validation = self._validate_sql(
                                final_sql,
                                request.user_query,
                                state
                            )
                            response.validation_result = final_validation
                            
                            if not final_validation.is_valid:
                                raise SelfHealingFailedError(
                                    message="Self-healing could not produce valid SQL",
                                    healing_attempts=len(healing_attempts),
                                    original_sql=generation_result.sql,
                                    final_sql=final_sql,
                                )
                    else:
                        response.final_sql = generation_result.sql
                    
                    # Mark as completed
                    response.status = RequestStatus.COMPLETED
                    state.update_status(RequestStatus.COMPLETED)
                    
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            response.status = RequestStatus.FAILED
            response.error_message = str(e)
            response.error_details = {"type": type(e).__name__}
            state.update_status(RequestStatus.FAILED)
            state.record_error(str(e))
            
            SQLAgentMetrics.record_error(
                error_type=type(e).__name__,
                category="pipeline",
                db_type=self.config.database.db_type.value
            )
        
        finally:
            # Clean up
            response.completed_at = datetime.utcnow()
            response.total_iterations = state.current_iteration
            response.total_duration_ms = state.get_elapsed_time_ms()
            
            # Remove from active requests
            with self._lock:
                self._active_requests.pop(request.request_id, None)
        
        return response
    
    def _generate_sql(
        self,
        request: SQLGenerationRequest,
        state: AgentState,
        previous_errors: Optional[List[str]] = None
    ) -> SQLGenerationResult:
        """Generate SQL using the generator agent with query understanding"""
        # Analyze the query to understand user intent
        query_analysis = None
        if QUERY_UNDERSTANDING_AVAILABLE:
            try:
                # Get schema info for better analysis
                schema = self.adapter.get_schema()
                schema_tables = [t.name for t in schema.tables]
                schema_columns = {t.name: [c.name for c in t.columns] for t in schema.tables}
                
                # Get business terms from semantic model if available
                business_terms = {}
                if self._semantic_model and self._semantic_model.business_context:
                    business_terms = self._semantic_model.business_context.terminology or {}
                
                # Analyze the query
                query_analysis = analyze_query(
                    query=request.user_query,
                    schema_tables=schema_tables,
                    schema_columns=schema_columns,
                    business_terms=business_terms,
                )
                
                logger.info(
                    f"Query analysis: operation={query_analysis.operation.value}, "
                    f"tables={query_analysis.potential_tables}, "
                    f"has_aggregation={query_analysis.has_aggregation}, "
                    f"has_filtering={query_analysis.has_filtering}, "
                    f"confidence={query_analysis.confidence:.2f}"
                )
            except Exception as e:
                logger.warning(f"Query analysis failed: {e}")
        
        return self.generator.generate(
            user_query=request.user_query,
            adapter=self.adapter,
            table_hints=request.table_hints,
            column_hints=request.column_hints,
            additional_context=request.additional_context,
            previous_errors=previous_errors,
            query_analysis=query_analysis,
        )
    
    def _validate_sql(
        self,
        sql: str,
        original_query: str,
        state: AgentState
    ) -> SQLValidationResult:
        """Validate SQL using the validator agent"""
        return self.validator.validate(
            sql=sql,
            adapter=self.adapter,
            original_query=original_query,
        )
    
    def _healing_loop(
        self,
        sql: str,
        validation_result: SQLValidationResult,
        original_query: str,
        state: AgentState
    ) -> tuple:
        """
        Self-healing loop that attempts to fix SQL issues
        
        Returns:
            Tuple of (final_sql, list of healing attempts)
        """
        current_sql = sql
        current_issues = validation_result.issues
        healing_attempts = []
        
        while state.increment_iteration():
            if not current_issues:
                break
            
            logger.info(f"Healing iteration {state.current_iteration}")
            
            # Attempt healing
            healed_result, attempt = self.healer.heal(
                sql=current_sql,
                issues=current_issues,
                adapter=self.adapter,
                original_query=original_query,
            )
            
            healing_attempts.append(attempt)
            state.record_healing(attempt)
            
            if not healed_result.sql or healed_result.sql == current_sql:
                logger.warning("Healing produced no changes")
                break
            
            current_sql = healed_result.sql
            
            # Re-validate
            new_validation = self._validate_sql(current_sql, original_query, state)
            state.record_validation(new_validation)
            
            if new_validation.is_valid:
                logger.info("Self-healing successful")
                return current_sql, healing_attempts
            
            # Check if we're making progress (fewer/different issues)
            new_error_count = sum(1 for i in new_validation.issues if i.severity == "error")
            old_error_count = sum(1 for i in current_issues if i.severity == "error")
            
            if new_error_count >= old_error_count:
                # Not making progress
                logger.warning("Self-healing not making progress")
                if self.pipeline_config.fail_fast:
                    break
            
            current_issues = new_validation.issues
        
        return current_sql, healing_attempts
    
    def get_request_state(self, request_id: str) -> Optional[AgentState]:
        """Get the current state of a request"""
        with self._lock:
            return self._active_requests.get(request_id)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request"""
        with self._lock:
            if request_id in self._active_requests:
                self._active_requests[request_id].update_status(RequestStatus.FAILED)
                self._active_requests[request_id].record_error("Cancelled by user")
                return True
        return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Check database connection
        try:
            db_ok, db_error = self.adapter.test_connection()
            health["components"]["database"] = {
                "status": "healthy" if db_ok else "unhealthy",
                "error": db_error,
            }
        except Exception as e:
            health["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "unhealthy"
        
        # Check LLM connectivity (via generator)
        try:
            from ..llm_client import get_llm_client
            llm_client = get_llm_client(self.config.llm)
            llm_ok = llm_client.health_check()
            health["components"]["llm"] = {
                "status": "healthy" if llm_ok else "unhealthy",
            }
        except Exception as e:
            health["components"]["llm"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "unhealthy"
        
        return health
    
    def close(self) -> None:
        """Close all connections and clean up resources"""
        if self._adapter:
            try:
                self._adapter.disconnect()
            except:
                pass
        
        with self._lock:
            self._active_requests.clear()


class PipelineBuilder:
    """Builder pattern for constructing SQLAgentPipeline"""
    
    def __init__(self):
        self._db_config: Optional[DatabaseConfig] = None
        self._llm_config: Optional[LLMConfig] = None
        self._agent_config: Optional[AgentConfig] = None
        self._pipeline_config: Optional[PipelineConfig] = None
        self._custom_adapter: Optional[BaseDatabaseAdapter] = None
        self._semantic_model: Optional[SemanticModel] = None if SCHEMA_INTELLIGENCE_AVAILABLE else None
    
    def with_database(self, config: DatabaseConfig) -> "PipelineBuilder":
        """Set database configuration"""
        self._db_config = config
        return self
    
    def with_llm(self, config: LLMConfig) -> "PipelineBuilder":
        """Set LLM configuration"""
        self._llm_config = config
        return self
    
    def with_agent_config(self, config: AgentConfig) -> "PipelineBuilder":
        """Set agent configuration"""
        self._agent_config = config
        return self
    
    def with_pipeline_config(self, config: PipelineConfig) -> "PipelineBuilder":
        """Set pipeline configuration"""
        self._pipeline_config = config
        return self
    
    def with_adapter(self, adapter: BaseDatabaseAdapter) -> "PipelineBuilder":
        """Set custom database adapter"""
        self._custom_adapter = adapter
        return self
    
    def with_schema_file(self, path: str) -> "PipelineBuilder":
        """
        Add schema file (YAML/JSON) with table/column descriptions
        
        Example schema.yaml:
            tables:
              users:
                description: User accounts
                columns:
                  id: {type: integer, primary_key: true}
                  email: {type: varchar, semantic_type: email}
        """
        if self._pipeline_config is None:
            self._pipeline_config = PipelineConfig()
        self._pipeline_config.schema_file = path
        return self
    
    def with_description_file(self, path: str) -> "PipelineBuilder":
        """
        Add text description file with natural language documentation
        
        Example docs.txt:
            ### users
            The users table stores customer account information.
            - id: Unique identifier
            - email: Customer's email address
        """
        if self._pipeline_config is None:
            self._pipeline_config = PipelineConfig()
        self._pipeline_config.description_file = path
        return self
    
    def with_semantic_model(self, model: SemanticModel) -> "PipelineBuilder":
        """Set a pre-built semantic model"""
        if not SCHEMA_INTELLIGENCE_AVAILABLE:
            logger.warning("Schema intelligence not available")
            return self
        self._semantic_model = model
        return self
    
    def with_max_iterations(self, max_iterations: int) -> "PipelineBuilder":
        """Set maximum healing iterations"""
        if self._pipeline_config is None:
            self._pipeline_config = PipelineConfig()
        self._pipeline_config.max_healing_iterations = max_iterations
        return self
    
    def with_self_healing(self, enabled: bool = True) -> "PipelineBuilder":
        """Enable or disable self-healing"""
        if self._pipeline_config is None:
            self._pipeline_config = PipelineConfig()
        self._pipeline_config.enable_self_healing = enabled
        return self
    
    def with_relationship_discovery(self, enabled: bool = True, use_llm: bool = False) -> "PipelineBuilder":
        """Configure automatic relationship discovery"""
        if self._pipeline_config is None:
            self._pipeline_config = PipelineConfig()
        self._pipeline_config.auto_discover_relationships = enabled
        self._pipeline_config.use_llm_for_relationships = use_llm
        return self
    
    def build(self) -> SQLAgentPipeline:
        """Build the pipeline"""
        if self._db_config is None:
            raise ValueError("Database configuration is required")
        
        system_config = SystemConfig(
            database=self._db_config,
            llm=self._llm_config or LLMConfig(),
            agent=self._agent_config or AgentConfig(),
        )
        
        pipeline = SQLAgentPipeline(
            config=system_config,
            pipeline_config=self._pipeline_config,
        )
        
        if self._custom_adapter:
            pipeline.set_adapter(self._custom_adapter)
        
        if self._semantic_model:
            pipeline.set_semantic_model(self._semantic_model)
        
        return pipeline


# Convenience function for quick pipeline creation
def create_pipeline(
    db_type: str,
    database: str,
    host: str = "localhost",
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    aws_region: str = "us-east-1",
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    schema_file: Optional[str] = None,
    description_file: Optional[str] = None,
    **kwargs
) -> SQLAgentPipeline:
    """
    Create a pipeline with minimal configuration
    
    Args:
        db_type: Database type (mysql, postgresql, oracle, sqlite)
        database: Database name
        host: Database host
        port: Database port (uses default if not specified)
        username: Database username
        password: Database password
        aws_region: AWS region for Bedrock
        model_id: Claude model ID
        schema_file: Path to YAML/JSON schema file with descriptions
        description_file: Path to text file with natural language docs
        **kwargs: Additional configuration options
        
    Returns:
        Configured SQLAgentPipeline
        
    Example:
        # Simple usage
        pipeline = create_pipeline(db_type="sqlite", database=":memory:")
        
        # With schema files
        pipeline = create_pipeline(
            db_type="postgresql",
            database="myapp",
            host="localhost",
            username="postgres",
            password="secret",
            schema_file="schema.yaml",
            description_file="docs.txt",
        )
    """
    from ..config import DatabaseType
    from pydantic import SecretStr
    
    db_type_enum = DatabaseType(db_type.lower())
    
    # Set default port based on database type
    if port is None:
        default_ports = {
            DatabaseType.MYSQL: 3306,
            DatabaseType.POSTGRESQL: 5432,
            DatabaseType.ORACLE: 1521,
            DatabaseType.SQLITE: 0,
        }
        port = default_ports.get(db_type_enum, 3306)
    
    db_config = DatabaseConfig(
        db_type=db_type_enum,
        host=host,
        port=port,
        database=database,
        username=username,
        password=SecretStr(password) if password else None,
        sqlite_path=kwargs.get("sqlite_path"),
    )
    
    llm_config = LLMConfig(
        aws_region=aws_region,
        model_id=model_id,
        max_tokens=kwargs.get("max_tokens", 4096),
        temperature=kwargs.get("temperature", 0.1),
    )
    
    # Create pipeline config with schema files
    pipeline_config = PipelineConfig(
        schema_file=schema_file,
        description_file=description_file,
        auto_discover_relationships=kwargs.get("auto_discover_relationships", True),
        use_semantic_context=kwargs.get("use_semantic_context", True),
    )
    
    system_config = SystemConfig(
        database=db_config,
        llm=llm_config,
    )
    
    return SQLAgentPipeline(config=system_config, pipeline_config=pipeline_config)
