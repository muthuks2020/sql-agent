"""
Load Tests for SQL Agent System using Locust
Simulates concurrent users making SQL generation requests
"""
import os
import sys
import time
import random
import json
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from locust import User, task, between, events, tag
    from locust.runners import MasterRunner
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    print("Locust not installed. Install with: pip install locust")


# Sample queries for load testing
SAMPLE_QUERIES = [
    "Show me all users",
    "Get orders from last month",
    "Find products with low stock",
    "List users who made orders over $100",
    "Show total revenue by month",
    "Get top 10 customers by order count",
    "Find products never ordered",
    "Show average order value per user",
    "List orders with their product details",
    "Get users who registered this year",
    "Show products sorted by price",
    "Find duplicate email addresses",
    "Get orders pending for more than 7 days",
    "List users without any orders",
    "Show sales summary by product category",
]


if LOCUST_AVAILABLE:
    
    class SQLAgentUser(User):
        """
        Locust user that simulates SQL generation requests
        """
        
        # Wait between 1 and 3 seconds between tasks
        wait_time = between(1, 3)
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pipeline = None
            self.request_count = 0
            self.success_count = 0
            self.failure_count = 0
        
        def on_start(self):
            """Called when a simulated user starts"""
            # Import here to avoid issues if dependencies aren't installed
            from config import DatabaseConfig, LLMConfig, SystemConfig, DatabaseType
            from orchestration import SQLAgentPipeline
            from adapters import SQLiteAdapter
            
            # Create in-memory SQLite for testing
            db_config = DatabaseConfig(
                db_type=DatabaseType.SQLITE,
                database=":memory:",
                sqlite_path=":memory:",
            )
            
            # Use mock LLM config (in real tests, use actual Bedrock)
            llm_config = LLMConfig(
                model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
                aws_region=os.getenv("AWS_REGION", "us-east-1"),
            )
            
            system_config = SystemConfig(
                database=db_config,
                llm=llm_config,
            )
            
            self.pipeline = SQLAgentPipeline(system_config)
            
            # Setup test database
            self._setup_test_database()
        
        def _setup_test_database(self):
            """Setup test database schema"""
            adapter = self.pipeline.adapter
            adapter.connect()
            
            # Create test tables
            adapter.execute_query("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            adapter.execute_query("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    total DECIMAL(10,2),
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            adapter.execute_query("""
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    price DECIMAL(10,2),
                    stock INTEGER DEFAULT 0,
                    category TEXT
                )
            """)
            
            # Insert sample data
            for i in range(10):
                adapter.execute_query(
                    f"INSERT OR IGNORE INTO users (id, name, email) VALUES ({i}, 'User {i}', 'user{i}@example.com')"
                )
        
        def on_stop(self):
            """Called when a simulated user stops"""
            if self.pipeline:
                self.pipeline.close()
        
        @task(10)
        @tag('simple')
        def simple_query(self):
            """Simple SELECT query"""
            self._run_query(random.choice([
                "Show me all users",
                "List all products",
                "Get all orders",
            ]))
        
        @task(5)
        @tag('medium')
        def medium_query(self):
            """Medium complexity query with joins"""
            self._run_query(random.choice([
                "Show users with their orders",
                "Get products and their total sales",
                "Find users who made orders",
            ]))
        
        @task(2)
        @tag('complex')
        def complex_query(self):
            """Complex query with aggregations"""
            self._run_query(random.choice([
                "Show total revenue by user sorted by amount",
                "Get average order value per month",
                "Find top customers by total spending",
            ]))
        
        @task(1)
        @tag('random')
        def random_query(self):
            """Random query from sample list"""
            self._run_query(random.choice(SAMPLE_QUERIES))
        
        def _run_query(self, query: str):
            """Run a query through the pipeline and record metrics"""
            from schemas import SQLGenerationRequest, RequestStatus
            
            request = SQLGenerationRequest(
                user_query=query,
                database_type="sqlite",
            )
            
            start_time = time.time()
            
            try:
                # Note: In real load tests with actual LLM, this would make API calls
                # For demo purposes, we'll simulate the response
                response = self._simulate_response(request)
                
                response_time = (time.time() - start_time) * 1000  # ms
                
                if response["status"] == "completed":
                    events.request.fire(
                        request_type="SQL_GENERATION",
                        name=self._categorize_query(query),
                        response_time=response_time,
                        response_length=len(response.get("sql", "")),
                        exception=None,
                        context={},
                    )
                    self.success_count += 1
                else:
                    events.request.fire(
                        request_type="SQL_GENERATION",
                        name=self._categorize_query(query),
                        response_time=response_time,
                        response_length=0,
                        exception=Exception(response.get("error", "Unknown error")),
                        context={},
                    )
                    self.failure_count += 1
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                events.request.fire(
                    request_type="SQL_GENERATION",
                    name=self._categorize_query(query),
                    response_time=response_time,
                    response_length=0,
                    exception=e,
                    context={},
                )
                self.failure_count += 1
            
            self.request_count += 1
        
        def _simulate_response(self, request) -> Dict[str, Any]:
            """
            Simulate pipeline response for load testing without LLM calls
            In production tests, replace this with actual pipeline.process() call
            """
            # Simulate some processing time
            time.sleep(random.uniform(0.1, 0.5))
            
            # Simulate success rate
            if random.random() < 0.95:  # 95% success rate
                return {
                    "status": "completed",
                    "sql": f"SELECT * FROM users -- Generated for: {request.user_query[:50]}",
                    "iterations": random.randint(1, 3),
                }
            else:
                return {
                    "status": "failed",
                    "error": "Simulated failure",
                }
        
        def _categorize_query(self, query: str) -> str:
            """Categorize query for metrics grouping"""
            query_lower = query.lower()
            
            if "join" in query_lower or "with" in query_lower:
                return "complex_query"
            elif any(word in query_lower for word in ["sum", "avg", "count", "group"]):
                return "aggregate_query"
            elif "user" in query_lower:
                return "user_query"
            elif "order" in query_lower:
                return "order_query"
            elif "product" in query_lower:
                return "product_query"
            else:
                return "other_query"
    
    
    class SQLAgentLoadTest(User):
        """
        Load test with actual LLM calls (use with caution - costs money!)
        """
        
        wait_time = between(2, 5)  # Longer wait to avoid rate limits
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pipeline = None
        
        def on_start(self):
            """Initialize pipeline with real LLM"""
            from config import DatabaseConfig, LLMConfig, SystemConfig, DatabaseType
            from orchestration import SQLAgentPipeline
            
            # Check for required environment variables
            if not os.getenv("AWS_REGION"):
                raise EnvironmentError("AWS_REGION environment variable required")
            
            db_config = DatabaseConfig(
                db_type=DatabaseType.SQLITE,
                database=":memory:",
                sqlite_path=":memory:",
            )
            
            llm_config = LLMConfig(
                model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                aws_region=os.getenv("AWS_REGION", "us-east-1"),
            )
            
            system_config = SystemConfig(
                database=db_config,
                llm=llm_config,
            )
            
            self.pipeline = SQLAgentPipeline(system_config)
            self._setup_test_database()
        
        def _setup_test_database(self):
            """Setup test schema"""
            adapter = self.pipeline.adapter
            adapter.connect()
            
            adapter.execute_query("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    email TEXT
                )
            """)
            
            adapter.execute_query("""
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    amount DECIMAL
                )
            """)
        
        @task
        @tag('real_llm')
        def real_llm_query(self):
            """Make actual LLM call"""
            from schemas import SQLGenerationRequest
            
            query = random.choice(SAMPLE_QUERIES[:5])  # Use simpler queries
            
            request = SQLGenerationRequest(
                user_query=query,
                database_type="sqlite",
            )
            
            start_time = time.time()
            
            try:
                response = self.pipeline.process(request)
                response_time = (time.time() - start_time) * 1000
                
                events.request.fire(
                    request_type="REAL_LLM_CALL",
                    name="sql_generation",
                    response_time=response_time,
                    response_length=len(response.final_sql or ""),
                    exception=None if response.final_sql else Exception(response.error_message),
                    context={},
                )
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                events.request.fire(
                    request_type="REAL_LLM_CALL",
                    name="sql_generation",
                    response_time=response_time,
                    response_length=0,
                    exception=e,
                    context={},
                )


# Standalone test runner
def run_simple_load_test(num_requests: int = 100, concurrency: int = 5):
    """
    Run a simple load test without Locust
    Useful for quick validation
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print(f"Running simple load test: {num_requests} requests, {concurrency} concurrent")
    
    results = {
        "total": 0,
        "success": 0,
        "failure": 0,
        "total_time": 0,
        "times": [],
    }
    
    lock = threading.Lock()
    
    def run_request(query: str) -> Dict[str, Any]:
        """Run a single request"""
        start = time.time()
        
        # Simulate processing
        time.sleep(random.uniform(0.1, 0.3))
        success = random.random() < 0.95
        
        elapsed = time.time() - start
        
        return {
            "success": success,
            "time": elapsed,
            "query": query,
        }
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        
        for i in range(num_requests):
            query = random.choice(SAMPLE_QUERIES)
            futures.append(executor.submit(run_request, query))
        
        for future in as_completed(futures):
            result = future.result()
            
            with lock:
                results["total"] += 1
                results["times"].append(result["time"])
                
                if result["success"]:
                    results["success"] += 1
                else:
                    results["failure"] += 1
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    times = results["times"]
    avg_time = sum(times) / len(times) if times else 0
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0
    
    sorted_times = sorted(times)
    p50 = sorted_times[len(sorted_times) // 2] if sorted_times else 0
    p95 = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
    p99 = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
    
    print("\n" + "=" * 50)
    print("LOAD TEST RESULTS")
    print("=" * 50)
    print(f"Total Requests:  {results['total']}")
    print(f"Successful:      {results['success']}")
    print(f"Failed:          {results['failure']}")
    print(f"Success Rate:    {results['success'] / results['total'] * 100:.1f}%")
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Requests/sec:    {results['total'] / total_time:.2f}")
    print()
    print("Response Times:")
    print(f"  Average:       {avg_time * 1000:.2f}ms")
    print(f"  Min:           {min_time * 1000:.2f}ms")
    print(f"  Max:           {max_time * 1000:.2f}ms")
    print(f"  P50:           {p50 * 1000:.2f}ms")
    print(f"  P95:           {p95 * 1000:.2f}ms")
    print(f"  P99:           {p99 * 1000:.2f}ms")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SQL Agent Load Tests")
    parser.add_argument("--simple", action="store_true", help="Run simple load test")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent users")
    
    args = parser.parse_args()
    
    if args.simple:
        run_simple_load_test(args.requests, args.concurrency)
    else:
        if LOCUST_AVAILABLE:
            print("Run Locust with: locust -f tests/load/locustfile.py")
            print("Then open http://localhost:8089 to configure and start the test")
        else:
            print("Locust not available. Running simple load test instead.")
            run_simple_load_test(args.requests, args.concurrency)
