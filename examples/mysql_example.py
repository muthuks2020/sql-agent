#!/usr/bin/env python3
"""
MySQL Example for SQL Agent System

This example demonstrates:
1. Connecting to MySQL database
2. Generating MySQL-specific SQL
3. Using table and column hints
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import (
    create_pipeline,
    SQLGenerationRequest,
    DatabaseConfig,
    LLMConfig,
    SystemConfig,
    SQLAgentPipeline,
    PipelineBuilder,
    DatabaseType,
    setup_logging,
)
from pydantic import SecretStr


def main():
    setup_logging(level="INFO")
    
    print("=" * 60)
    print("SQL Agent System - MySQL Example")
    print("=" * 60)
    
    # Configuration from environment variables
    mysql_host = os.getenv("MYSQL_HOST", "localhost")
    mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user = os.getenv("MYSQL_USER", "root")
    mysql_password = os.getenv("MYSQL_PASSWORD", "")
    mysql_database = os.getenv("MYSQL_DATABASE", "testdb")
    
    print(f"\nConnecting to MySQL: {mysql_host}:{mysql_port}/{mysql_database}")
    
    # Method 1: Using create_pipeline convenience function
    print("\n1. Creating pipeline using convenience function...")
    try:
        pipeline = create_pipeline(
            db_type="mysql",
            database=mysql_database,
            host=mysql_host,
            port=mysql_port,
            username=mysql_user,
            password=mysql_password,
        )
        print("   Pipeline created successfully!")
    except Exception as e:
        print(f"   Note: MySQL connection failed (expected if MySQL not running): {e}")
        print("   Showing configuration example instead...")
        pipeline = None
    
    # Method 2: Using PipelineBuilder for more control
    print("\n2. Creating pipeline using builder pattern...")
    db_config = DatabaseConfig(
        db_type=DatabaseType.MYSQL,
        host=mysql_host,
        port=mysql_port,
        database=mysql_database,
        username=mysql_user,
        password=SecretStr(mysql_password) if mysql_password else None,
        connection_pool_size=10,
        query_timeout=60,
        ssl_enabled=False,
    )
    
    llm_config = LLMConfig(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        temperature=0.1,
        max_tokens=4096,
    )
    
    print("   Configuration created:")
    print(f"   - Database: {db_config.db_type.value}")
    print(f"   - LLM Model: {llm_config.model_id}")
    print(f"   - AWS Region: {llm_config.aws_region}")
    
    # Example queries for MySQL
    print("\n3. Example MySQL-specific queries:")
    example_queries = [
        {
            "query": "Show all users registered in the last 30 days",
            "hints": {
                "table_hints": ["users"],
                "column_hints": ["created_at", "registration_date"],
            }
        },
        {
            "query": "Get top 10 customers by total order value",
            "hints": {
                "table_hints": ["customers", "orders"],
                "additional_context": "Use MySQL-specific functions like IFNULL",
            }
        },
        {
            "query": "Find products with sales above average",
            "hints": {
                "table_hints": ["products", "order_items"],
            }
        },
    ]
    
    for i, example in enumerate(example_queries, 1):
        print(f"\n   Query {i}: {example['query']}")
        print(f"   Table hints: {example['hints'].get('table_hints', [])}")
        
        if pipeline:
            request = SQLGenerationRequest(
                user_query=example['query'],
                database_type="mysql",
                table_hints=example['hints'].get('table_hints'),
                column_hints=example['hints'].get('column_hints'),
                additional_context=example['hints'].get('additional_context'),
            )
            print(f"   Request ID: {request.request_id}")
    
    # MySQL dialect hints
    print("\n4. MySQL Dialect Information:")
    from src.adapters import MySQLAdapter
    
    dummy_config = DatabaseConfig(
        db_type=DatabaseType.MYSQL,
        database="dummy",
    )
    adapter = MySQLAdapter(dummy_config)
    hints = adapter.get_sql_dialect_hints()
    
    print(f"   Dialect: {hints['dialect']}")
    print(f"   String concatenation: {hints['string_concatenation']}")
    print(f"   LIMIT syntax: {hints['limit_syntax']}")
    print(f"   NULL handling: {hints['null_handling']}")
    print(f"   Identifier quoting: {hints['identifier_quoting']}")
    
    print("\n" + "=" * 60)
    print("MySQL example completed!")
    print("=" * 60)
    print("""
To run with actual MySQL:
1. Start MySQL server
2. Create database: CREATE DATABASE testdb;
3. Set environment variables:
   export MYSQL_HOST=localhost
   export MYSQL_PORT=3306
   export MYSQL_USER=root
   export MYSQL_PASSWORD=yourpassword
   export MYSQL_DATABASE=testdb
4. Set AWS credentials for Bedrock
5. Run this script
""")


if __name__ == "__main__":
    main()
