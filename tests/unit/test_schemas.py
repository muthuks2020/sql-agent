"""
Unit Tests for Schemas and Models
"""
import pytest
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from schemas import (
    RequestStatus,
    SQLQueryType,
    SQLGenerationRequest,
    SQLGenerationResponse,
    SQLGenerationResult,
    SQLValidationResult,
    SQLValidationIssue,
    HealingAttempt,
    AgentState,
    detect_query_type,
)


class TestSQLQueryType:
    """Tests for SQL query type detection"""
    
    def test_detect_select(self):
        """Test SELECT query detection"""
        assert detect_query_type("SELECT * FROM users") == SQLQueryType.SELECT
        assert detect_query_type("  SELECT id FROM orders") == SQLQueryType.SELECT
        assert detect_query_type("select name from products") == SQLQueryType.SELECT
    
    def test_detect_insert(self):
        """Test INSERT query detection"""
        assert detect_query_type("INSERT INTO users VALUES (1)") == SQLQueryType.INSERT
        assert detect_query_type("insert into orders (id) values (1)") == SQLQueryType.INSERT
    
    def test_detect_update(self):
        """Test UPDATE query detection"""
        assert detect_query_type("UPDATE users SET name = 'test'") == SQLQueryType.UPDATE
    
    def test_detect_delete(self):
        """Test DELETE query detection"""
        assert detect_query_type("DELETE FROM users WHERE id = 1") == SQLQueryType.DELETE
    
    def test_detect_ddl(self):
        """Test DDL query detection"""
        assert detect_query_type("CREATE TABLE users (id INT)") == SQLQueryType.CREATE
        assert detect_query_type("ALTER TABLE users ADD COLUMN name TEXT") == SQLQueryType.ALTER
        assert detect_query_type("DROP TABLE users") == SQLQueryType.DROP
    
    def test_detect_unknown(self):
        """Test unknown query detection"""
        assert detect_query_type("EXPLAIN SELECT * FROM users") == SQLQueryType.UNKNOWN
        assert detect_query_type("INVALID QUERY") == SQLQueryType.UNKNOWN


class TestSQLGenerationRequest:
    """Tests for SQLGenerationRequest"""
    
    def test_request_creation(self):
        """Test basic request creation"""
        request = SQLGenerationRequest(
            user_query="Show all users",
            database_type="mysql",
        )
        
        assert request.user_query == "Show all users"
        assert request.database_type == "mysql"
        assert request.request_id is not None
        assert request.created_at is not None
    
    def test_request_with_hints(self):
        """Test request with table/column hints"""
        request = SQLGenerationRequest(
            user_query="Show user names",
            database_type="postgresql",
            table_hints=["users", "profiles"],
            column_hints=["name", "email"],
        )
        
        assert request.table_hints == ["users", "profiles"]
        assert request.column_hints == ["name", "email"]
    
    def test_request_to_dict(self):
        """Test request serialization"""
        request = SQLGenerationRequest(
            user_query="Test query",
            database_type="sqlite",
        )
        
        request_dict = request.to_dict()
        
        assert request_dict["user_query"] == "Test query"
        assert request_dict["database_type"] == "sqlite"
        assert "request_id" in request_dict


class TestSQLValidationIssue:
    """Tests for SQLValidationIssue"""
    
    def test_issue_creation(self):
        """Test issue creation"""
        issue = SQLValidationIssue(
            issue_type="syntax",
            severity="error",
            message="Missing semicolon",
            suggestion="Add semicolon at end of query",
        )
        
        assert issue.issue_type == "syntax"
        assert issue.severity == "error"
        assert issue.message == "Missing semicolon"
    
    def test_issue_to_dict(self):
        """Test issue serialization"""
        issue = SQLValidationIssue(
            issue_type="schema",
            severity="warning",
            message="Column may not exist",
        )
        
        issue_dict = issue.to_dict()
        
        assert issue_dict["issue_type"] == "schema"
        assert issue_dict["severity"] == "warning"


class TestSQLValidationResult:
    """Tests for SQLValidationResult"""
    
    def test_valid_result(self):
        """Test valid validation result"""
        result = SQLValidationResult(
            is_valid=True,
            validated_sql="SELECT * FROM users",
        )
        
        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings
    
    def test_invalid_result_with_errors(self):
        """Test invalid result with errors"""
        issues = [
            SQLValidationIssue(
                issue_type="syntax",
                severity="error",
                message="Syntax error",
            ),
            SQLValidationIssue(
                issue_type="schema",
                severity="warning",
                message="Schema warning",
            ),
        ]
        
        result = SQLValidationResult(
            is_valid=False,
            issues=issues,
        )
        
        assert not result.is_valid
        assert result.has_errors
        assert result.has_warnings
    
    def test_result_to_dict(self):
        """Test result serialization"""
        result = SQLValidationResult(
            is_valid=True,
            issues=[],
            execution_plan="Table scan",
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["is_valid"] is True
        assert result_dict["execution_plan"] == "Table scan"


class TestHealingAttempt:
    """Tests for HealingAttempt"""
    
    def test_healing_attempt_creation(self):
        """Test healing attempt creation"""
        attempt = HealingAttempt(
            iteration=1,
            original_sql="SELEC * FROM users",
            healed_sql="SELECT * FROM users",
            issues_addressed=["Fixed syntax error"],
            success=True,
        )
        
        assert attempt.iteration == 1
        assert attempt.success
        assert len(attempt.issues_addressed) == 1
    
    def test_healing_attempt_to_dict(self):
        """Test attempt serialization"""
        attempt = HealingAttempt(
            iteration=2,
            original_sql="bad sql",
            healed_sql="good sql",
            issues_addressed=["fix1", "fix2"],
            success=True,
        )
        
        attempt_dict = attempt.to_dict()
        
        assert attempt_dict["iteration"] == 2
        assert attempt_dict["success"] is True


class TestSQLGenerationResponse:
    """Tests for SQLGenerationResponse"""
    
    def test_successful_response(self):
        """Test successful response"""
        response = SQLGenerationResponse(
            request_id="test-123",
            status=RequestStatus.COMPLETED,
            final_sql="SELECT * FROM users",
        )
        
        assert response.is_successful
        assert response.status == RequestStatus.COMPLETED
    
    def test_failed_response(self):
        """Test failed response"""
        response = SQLGenerationResponse(
            request_id="test-456",
            status=RequestStatus.FAILED,
            error_message="Generation failed",
        )
        
        assert not response.is_successful
        assert response.error_message is not None
    
    def test_response_with_healing(self):
        """Test response with healing attempts"""
        healing_attempts = [
            HealingAttempt(
                iteration=1,
                original_sql="bad",
                healed_sql="good",
                issues_addressed=["fix"],
                success=True,
            )
        ]
        
        response = SQLGenerationResponse(
            request_id="test-789",
            status=RequestStatus.COMPLETED,
            final_sql="good",
            healing_attempts=healing_attempts,
            total_iterations=1,
        )
        
        assert len(response.healing_attempts) == 1
        assert response.total_iterations == 1


class TestAgentState:
    """Tests for AgentState"""
    
    def test_state_initialization(self):
        """Test state initialization"""
        state = AgentState(
            request_id="test-state",
            max_iterations=5,
        )
        
        assert state.status == RequestStatus.PENDING
        assert state.current_iteration == 0
        assert state.max_iterations == 5
    
    def test_state_start(self):
        """Test starting state"""
        state = AgentState(request_id="test")
        state.start()
        
        assert state.status == RequestStatus.GENERATING
        assert state.started_at is not None
    
    def test_increment_iteration(self):
        """Test iteration increment"""
        state = AgentState(request_id="test", max_iterations=3)
        
        assert state.increment_iteration()  # 1
        assert state.increment_iteration()  # 2
        assert state.increment_iteration()  # 3
        assert not state.increment_iteration()  # 4 > 3
    
    def test_record_generation(self):
        """Test recording generation"""
        state = AgentState(request_id="test")
        state.record_generation("SELECT 1")
        
        assert state.current_sql == "SELECT 1"
        assert len(state.generation_history) == 1
    
    def test_record_validation(self):
        """Test recording validation"""
        state = AgentState(request_id="test")
        result = SQLValidationResult(is_valid=True)
        state.record_validation(result)
        
        assert len(state.validation_history) == 1
    
    def test_record_error(self):
        """Test recording error"""
        state = AgentState(request_id="test")
        state.record_error("Test error")
        
        assert state.last_error == "Test error"
        assert state.error_count == 1
    
    def test_elapsed_time(self):
        """Test elapsed time calculation"""
        state = AgentState(request_id="test")
        state.start()
        
        # Should be non-zero after start
        assert state.get_elapsed_time_ms() >= 0
    
    def test_state_to_dict(self):
        """Test state serialization"""
        state = AgentState(
            request_id="test",
            user_query="test query",
        )
        state.start()
        state.record_generation("SELECT 1")
        
        state_dict = state.to_dict()
        
        assert state_dict["request_id"] == "test"
        assert state_dict["current_sql"] == "SELECT 1"


class TestSQLGenerationResult:
    """Tests for SQLGenerationResult"""
    
    def test_result_creation(self):
        """Test result creation"""
        result = SQLGenerationResult(
            sql="SELECT * FROM users",
            query_type=SQLQueryType.SELECT,
            explanation="Selects all users",
            confidence_score=0.95,
            tables_used=["users"],
            columns_used=["*"],
        )
        
        assert result.sql == "SELECT * FROM users"
        assert result.query_type == SQLQueryType.SELECT
        assert result.confidence_score == 0.95
    
    def test_result_to_dict(self):
        """Test result serialization"""
        result = SQLGenerationResult(
            sql="SELECT 1",
            query_type=SQLQueryType.SELECT,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["sql"] == "SELECT 1"
        assert result_dict["query_type"] == "SELECT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
