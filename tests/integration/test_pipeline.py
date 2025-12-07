"""
Integration Tests for SQL Agent Pipeline
Tests end-to-end functionality with mocked LLM responses
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config import DatabaseConfig, LLMConfig, AgentConfig, SystemConfig, DatabaseType
from adapters import SQLiteAdapter, create_adapter
from schemas import SQLGenerationRequest, RequestStatus
from orchestration import SQLAgentPipeline, PipelineConfig, PipelineBuilder


class MockLLMClient:
    """Mock LLM client for testing"""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
    
    def invoke(self, prompt, system_prompt=None, context=None, **kwargs):
        """Return mock response"""
        from llm_client import LLMResponse
        
        if self.call_count < len(self.responses):
            content = self.responses[self.call_count]
        else:
            content = self.responses[-1] if self.responses else ""
        
        self.call_count += 1
        
        return LLMResponse(
            content=content,
            model_id="mock-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=100.0,
        )
    
    def invoke_with_retry(self, prompt, system_prompt=None, context=None, max_retries=None, **kwargs):
        """Return mock response with retry"""
        return self.invoke(prompt, system_prompt, context, **kwargs)
    
    def health_check(self):
        return True


class TestPipelineWithSQLite:
    """Integration tests with SQLite database"""
    
    @pytest.fixture
    def sqlite_adapter(self):
        """Create SQLite adapter with test schema"""
        config = DatabaseConfig(
            db_type=DatabaseType.SQLITE,
            database=":memory:",
            sqlite_path=":memory:",
        )
        adapter = SQLiteAdapter(config)
        adapter.connect()
        
        # Create test schema
        adapter.execute_query("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        adapter.execute_query("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                total DECIMAL(10,2),
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        adapter.execute_query("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price DECIMAL(10,2),
                stock INTEGER DEFAULT 0
            )
        """)
        
        yield adapter
        adapter.disconnect()
    
    @pytest.fixture
    def system_config(self):
        """Create system configuration"""
        return SystemConfig(
            database=DatabaseConfig(
                db_type=DatabaseType.SQLITE,
                database=":memory:",
            ),
            llm=LLMConfig(),
            agent=AgentConfig(max_iterations=3),
        )
    
    @pytest.fixture
    def mock_llm_responses(self):
        """Standard mock LLM responses"""
        return {
            "simple_select": """
```sql
SELECT * FROM users WHERE id = 1;
```

EXPLANATION: Selects user with ID 1
TABLES_USED: users
COLUMNS_USED: id, name, email, created_at
CONFIDENCE: 0.95
""",
            "valid_validation": """
VALID: true
ISSUES: none
EXECUTION_PLAN: Index scan on users.id
""",
            "invalid_validation": """
VALID: false
ISSUES:
- [error:schema] Table 'user' does not exist | SUGGESTION: Use 'users' table
EXECUTION_PLAN: N/A
""",
            "healed_sql": """
```sql
SELECT * FROM users WHERE id = 1;
```

CHANGES_MADE:
1. Fixed table name from 'user' to 'users'

CONFIDENCE: 0.9
""",
        }
    
    def test_pipeline_health_check(self, sqlite_adapter, system_config):
        """Test pipeline health check"""
        pipeline = SQLAgentPipeline(system_config)
        pipeline.set_adapter(sqlite_adapter)
        
        # Mock the LLM client
        with patch.object(pipeline, '_generator') as mock_gen:
            mock_gen._llm_client = MockLLMClient(["OK"])
            mock_gen._llm_client.health_check = lambda: True
            
            health = pipeline.health_check()
            
            assert health["status"] == "healthy" or health["components"]["database"]["status"] == "healthy"
    
    def test_schema_retrieval(self, sqlite_adapter):
        """Test schema retrieval from database"""
        schema = sqlite_adapter.get_schema()
        
        assert "users" in schema.tables
        assert "orders" in schema.tables
        assert "products" in schema.tables
        
        users_table = schema.tables["users"]
        assert len(users_table.columns) == 4
        
        column_names = [c.name for c in users_table.columns]
        assert "id" in column_names
        assert "name" in column_names
        assert "email" in column_names
    
    @patch('llm_client.LLMClientFactory.get_client')
    def test_successful_generation(self, mock_get_client, sqlite_adapter, system_config, mock_llm_responses):
        """Test successful SQL generation"""
        # Setup mock
        mock_client = MockLLMClient([
            mock_llm_responses["simple_select"],
            mock_llm_responses["valid_validation"],
        ])
        mock_get_client.return_value = mock_client
        
        pipeline = SQLAgentPipeline(system_config)
        pipeline.set_adapter(sqlite_adapter)
        
        request = SQLGenerationRequest(
            user_query="Get user with ID 1",
            database_type="sqlite",
        )
        
        response = pipeline.process(request)
        
        assert response.status == RequestStatus.COMPLETED
        assert response.final_sql is not None
        assert "SELECT" in response.final_sql.upper()
    
    @patch('llm_client.LLMClientFactory.get_client')
    def test_generation_with_healing(self, mock_get_client, sqlite_adapter, system_config, mock_llm_responses):
        """Test SQL generation with self-healing"""
        # Setup mock - first generate bad SQL, then validate as invalid, then heal
        mock_client = MockLLMClient([
            """
```sql
SELECT * FROM user WHERE id = 1;
```
EXPLANATION: Bad SQL
TABLES_USED: user
COLUMNS_USED: id
CONFIDENCE: 0.8
""",
            mock_llm_responses["invalid_validation"],
            mock_llm_responses["healed_sql"],
            mock_llm_responses["valid_validation"],
        ])
        mock_get_client.return_value = mock_client
        
        pipeline_config = PipelineConfig(
            max_healing_iterations=3,
            enable_self_healing=True,
        )
        pipeline = SQLAgentPipeline(system_config, pipeline_config)
        pipeline.set_adapter(sqlite_adapter)
        
        request = SQLGenerationRequest(
            user_query="Get user with ID 1",
            database_type="sqlite",
        )
        
        response = pipeline.process(request)
        
        # Should complete even if healing was needed
        assert response.status == RequestStatus.COMPLETED or response.total_iterations > 0
    
    def test_request_state_tracking(self, sqlite_adapter, system_config):
        """Test request state tracking"""
        pipeline = SQLAgentPipeline(system_config)
        pipeline.set_adapter(sqlite_adapter)
        
        # Initially no active requests
        state = pipeline.get_request_state("nonexistent")
        assert state is None


class TestPipelineBuilder:
    """Tests for PipelineBuilder"""
    
    def test_builder_basic(self):
        """Test basic builder usage"""
        db_config = DatabaseConfig(
            db_type=DatabaseType.SQLITE,
            database=":memory:",
        )
        
        pipeline = (
            PipelineBuilder()
            .with_database(db_config)
            .with_max_iterations(5)
            .with_self_healing(True)
            .build()
        )
        
        assert pipeline is not None
        assert pipeline.pipeline_config.max_healing_iterations == 5
        assert pipeline.pipeline_config.enable_self_healing is True
    
    def test_builder_with_custom_config(self):
        """Test builder with custom configurations"""
        db_config = DatabaseConfig(
            db_type=DatabaseType.SQLITE,
            database=":memory:",
        )
        
        llm_config = LLMConfig(
            model_id="anthropic.claude-3-sonnet",
            temperature=0.2,
        )
        
        agent_config = AgentConfig(
            max_iterations=10,
            validation_strictness="high",
        )
        
        pipeline = (
            PipelineBuilder()
            .with_database(db_config)
            .with_llm(llm_config)
            .with_agent_config(agent_config)
            .build()
        )
        
        assert pipeline.config.llm.temperature == 0.2
        assert pipeline.config.agent.validation_strictness == "high"
    
    def test_builder_requires_database(self):
        """Test that builder requires database config"""
        with pytest.raises(ValueError, match="Database configuration"):
            PipelineBuilder().build()


class TestPipelineEdgeCases:
    """Edge case tests for pipeline"""
    
    @pytest.fixture
    def basic_config(self):
        return SystemConfig(
            database=DatabaseConfig(
                db_type=DatabaseType.SQLITE,
                database=":memory:",
            ),
        )
    
    def test_empty_user_query(self, basic_config):
        """Test handling of empty user query"""
        pipeline = SQLAgentPipeline(basic_config)
        
        request = SQLGenerationRequest(
            user_query="",
            database_type="sqlite",
        )
        
        # Should handle gracefully
        response = pipeline.process(request)
        # Will likely fail but shouldn't crash
        assert response.status in [RequestStatus.COMPLETED, RequestStatus.FAILED]
    
    def test_pipeline_close(self, basic_config):
        """Test pipeline cleanup"""
        pipeline = SQLAgentPipeline(basic_config)
        
        # Should not raise
        pipeline.close()
    
    def test_cancel_nonexistent_request(self, basic_config):
        """Test canceling nonexistent request"""
        pipeline = SQLAgentPipeline(basic_config)
        
        result = pipeline.cancel_request("nonexistent-id")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
