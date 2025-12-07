#!/usr/bin/env python3
"""
Basic Usage Example for SQL Agent System

This example demonstrates:
1. Creating a pipeline with SQLite
2. Generating SQL from natural language
3. Handling responses
"""
import sys
import os

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import (
    create_pipeline,
    SQLGenerationRequest,
    setup_logging,
    RequestStatus,
)


def main():
    # Setup logging
    setup_logging(level="INFO")
    
    print("=" * 60)
    print("SQL Agent System - Basic Usage Example")
    print("=" * 60)
    
    # Create pipeline with SQLite in-memory database
    print("\n1. Creating pipeline with SQLite...")
    pipeline = create_pipeline(
        db_type="sqlite",
        database=":memory:",
        sqlite_path=":memory:",
    )
    
    # Setup test database schema
    print("2. Setting up test database schema...")
    adapter = pipeline.adapter
    adapter.connect()
    
    adapter.execute_query("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            city TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    adapter.execute_query("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id),
            product_name TEXT NOT NULL,
            quantity INTEGER DEFAULT 1,
            total_amount DECIMAL(10,2),
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert sample data
    adapter.execute_query("INSERT INTO users (name, email, age, city) VALUES ('Alice', 'alice@example.com', 30, 'New York')")
    adapter.execute_query("INSERT INTO users (name, email, age, city) VALUES ('Bob', 'bob@example.com', 25, 'Los Angeles')")
    adapter.execute_query("INSERT INTO users (name, email, age, city) VALUES ('Charlie', 'charlie@example.com', 35, 'Chicago')")
    
    adapter.execute_query("INSERT INTO orders (user_id, product_name, quantity, total_amount, status) VALUES (1, 'Laptop', 1, 999.99, 'completed')")
    adapter.execute_query("INSERT INTO orders (user_id, product_name, quantity, total_amount, status) VALUES (1, 'Mouse', 2, 49.98, 'completed')")
    adapter.execute_query("INSERT INTO orders (user_id, product_name, quantity, total_amount, status) VALUES (2, 'Keyboard', 1, 79.99, 'pending')")
    
    print("   Created tables: users, orders")
    print("   Inserted sample data")
    
    # Show schema
    print("\n3. Database Schema:")
    schema = adapter.get_schema()
    print(schema.to_schema_string())
    
    # Example queries
    example_queries = [
        "Show me all users",
        "Get all orders with total amount greater than 50 dollars",
        "Find users who have placed orders",
        "Show the total spending per user",
        "List orders that are still pending",
    ]
    
    print("\n4. Generating SQL for example queries...")
    print("-" * 60)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        # Create request
        request = SQLGenerationRequest(
            user_query=query,
            database_type="sqlite",
        )
        
        # Process request
        # Note: This will fail without actual LLM credentials
        # In a real scenario, the pipeline would call AWS Bedrock
        try:
            # For demo, we'll just show the request
            print(f"   Request ID: {request.request_id}")
            print(f"   Would generate SQL for: '{query}'")
            
            # Uncomment below to actually run (requires AWS credentials):
            # response = pipeline.process(request)
            # if response.status == RequestStatus.COMPLETED:
            #     print(f"   Generated SQL: {response.final_sql}")
            # else:
            #     print(f"   Error: {response.error_message}")
            
        except Exception as e:
            print(f"   Error: {str(e)}")
    
    # Cleanup
    print("\n5. Cleaning up...")
    pipeline.close()
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("""
To run with actual LLM calls:
1. Set AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
2. Set AWS_REGION (e.g., us-east-1)
3. Ensure you have access to Bedrock Claude models
4. Uncomment the pipeline.process() call above
""")


if __name__ == "__main__":
    main()
