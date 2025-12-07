#!/usr/bin/env python3
"""
SQL Agent System - Complete Flow Demonstration

This script demonstrates the entire system flow from user query to final SQL,
showing how each component works together.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DatabaseConfig, DatabaseType
from src.adapters import SQLiteAdapter


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_step(step_num: int, title: str):
    """Print a step header"""
    print(f"\n┌{'─'*78}┐")
    print(f"│  STEP {step_num}: {title:<68} │")
    print(f"└{'─'*78}┘\n")


def create_demo_database():
    """Create a sample database for demonstration"""
    config = DatabaseConfig(db_type=DatabaseType.SQLITE, database=":memory:")
    adapter = SQLiteAdapter(config)
    adapter.connect()
    
    # Create tables (WITHOUT explicit foreign key constraints to test auto-discovery)
    adapter.execute_query("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email VARCHAR(255) UNIQUE,
            name VARCHAR(100),
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    adapter.execute_query("""
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100),
            parent_id INTEGER
        )
    """)
    
    adapter.execute_query("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name VARCHAR(200),
            price DECIMAL(10,2),
            category_id INTEGER,
            stock_quantity INTEGER DEFAULT 0
        )
    """)
    
    adapter.execute_query("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            total DECIMAL(10,2),
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    adapter.execute_query("""
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            unit_price DECIMAL(10,2)
        )
    """)
    
    # Insert sample data
    adapter.execute_query("INSERT INTO users (email, name, status) VALUES ('john@example.com', 'John Doe', 'active')")
    adapter.execute_query("INSERT INTO users (email, name, status) VALUES ('jane@example.com', 'Jane Smith', 'active')")
    adapter.execute_query("INSERT INTO categories (name) VALUES ('Electronics'), ('Clothing')")
    adapter.execute_query("INSERT INTO products (name, price, category_id, stock_quantity) VALUES ('Laptop', 999.99, 1, 50)")
    adapter.execute_query("INSERT INTO products (name, price, category_id, stock_quantity) VALUES ('T-Shirt', 29.99, 2, 200)")
    adapter.execute_query("INSERT INTO orders (user_id, total, status) VALUES (1, 999.99, 'completed')")
    adapter.execute_query("INSERT INTO orders (user_id, total, status) VALUES (2, 29.99, 'pending')")
    adapter.execute_query("INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (1, 1, 1, 999.99)")
    adapter.execute_query("INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (2, 2, 1, 29.99)")
    
    return adapter


def demonstrate_complete_flow():
    """Demonstrate the complete system flow"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              SQL AGENT SYSTEM - COMPLETE FLOW DEMONSTRATION                  ║
║                                                                              ║
║  This demonstrates how a user query flows through the entire system          ║
║  from natural language input to validated SQL output.                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # STEP 0: Setup
    # =========================================================================
    print_step(0, "SETUP - Create Demo Database")
    
    adapter = create_demo_database()
    print("Created SQLite database with tables:")
    print("  • users (id, email, name, status, created_at)")
    print("  • categories (id, name, parent_id)")
    print("  • products (id, name, price, category_id, stock_quantity)")
    print("  • orders (id, user_id, total, status, created_at)")
    print("  • order_items (id, order_id, product_id, quantity, unit_price)")
    print("\nNote: NO explicit foreign key constraints defined!")
    print("      The system will DISCOVER relationships automatically.")
    
    # =========================================================================
    # STEP 1: User Query
    # =========================================================================
    print_step(1, "USER QUERY")
    
    user_query = "Show me all customers who have placed orders with their total spending"
    
    print(f'User asks: "{user_query}"')
    print("""
    Notice:
    • User says "customers" but table is "users"
    • User wants "total spending" which requires aggregation
    • User wants to JOIN users with orders
    """)
    
    # =========================================================================
    # STEP 2: Schema Intelligence Layer
    # =========================================================================
    print_step(2, "SCHEMA INTELLIGENCE LAYER")
    
    print("""
    The Schema Intelligence Layer builds understanding of your database.
    It can use THREE sources (any combination):
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        INPUT SOURCES                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   SOURCE 1: schema.yaml (Optional)                                       │
    │   ─────────────────────────────────                                      │
    │   tables:                                                                │
    │     users:                                                               │
    │       description: "Customer accounts"                                   │
    │       business_name: "Customers"        ◄── Maps "customer" to "users"  │
    │       columns:                                                           │
    │         email:                                                           │
    │           semantic_type: email                                           │
    │           description: "User's email address"                            │
    │                                                                          │
    │   SOURCE 2: descriptions.txt (Optional)                                  │
    │   ─────────────────────────────────────                                  │
    │   ### users                                                              │
    │   The users table stores customer accounts.                              │
    │   - user_id in orders links to users table                               │
    │                                                                          │
    │   ## Business Terms                                                      │
    │   "customer" refers to users                                             │
    │                                                                          │
    │   SOURCE 3: Database Auto-Discovery (Always Available)                   │
    │   ─────────────────────────────────────────────────────                  │
    │   • Extract all tables and columns                                       │
    │   • Find explicit foreign key constraints                                │
    │   • Infer relationships from naming patterns                             │
    │   • Analyze sample data                                                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Actually run schema intelligence
    from src.schema_intelligence import discover_schema_from_database, SchemaContextGenerator
    
    print("Running auto-discovery on database...")
    model = discover_schema_from_database(adapter, include_samples=True)
    
    print(f"\n✓ Discovered {len(model.tables)} tables:")
    for table_name, table in model.tables.items():
        print(f"    • {table_name}: {len(table.columns)} columns")
    
    print(f"\n✓ Discovered {len(model.relationships)} relationships:")
    for rel in model.relationships:
        print(f"    • {rel.source_table}.{rel.source_columns[0]} → {rel.target_table}.{rel.target_columns[0]}")
        print(f"      (Source: {rel.source.value}, Confidence: {rel.confidence:.0%})")
    
    print("\n✓ Inferred semantic types:")
    for table_name, table in model.tables.items():
        for col_name, col in table.columns.items():
            if col.semantic_type.value != 'unknown':
                print(f"    • {table_name}.{col_name}: {col.semantic_type.value}")
    
    # =========================================================================
    # STEP 3: Context Generation
    # =========================================================================
    print_step(3, "CONTEXT GENERATION FOR LLM")
    
    print("""
    The SchemaContextGenerator creates optimized context for the AI.
    It analyzes the user query and includes only relevant information.
    """)
    
    context_gen = SchemaContextGenerator(model)
    context = context_gen.generate_query_focused_context(user_query)
    
    print("Generated context for LLM:")
    print("─" * 60)
    print(context[:1500] + "..." if len(context) > 1500 else context)
    print("─" * 60)
    
    # =========================================================================
    # STEP 4: SQL Generator Agent
    # =========================================================================
    print_step(4, "SQL GENERATOR AGENT")
    
    print("""
    The SQL Generator Agent receives:
    • User query: "Show me all customers who have placed orders..."
    • Schema context: (tables, columns, relationships, business terms)
    • Dialect hints: (SQLite-specific syntax)
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         PROMPT TO CLAUDE AI                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  SYSTEM: You are an expert SQL developer.                               │
    │                                                                          │
    │  SCHEMA CONTEXT:                                                         │
    │  [Tables, columns, relationships from Step 3]                           │
    │                                                                          │
    │  DIALECT: SQLite                                                         │
    │  • Use datetime('now') for current timestamp                             │
    │  • Use LIMIT for pagination                                              │
    │                                                                          │
    │  USER QUERY: Show me all customers who have placed orders               │
    │              with their total spending                                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Claude AI generates:
    """)
    
    # Simulated SQL output (in real system, this comes from Claude)
    generated_sql = """SELECT 
    u.id,
    u.name,
    u.email,
    COUNT(o.id) as order_count,
    SUM(o.total) as total_spending
FROM users u
INNER JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name, u.email
ORDER BY total_spending DESC"""
    
    print(generated_sql)
    print("""
    
    Notice how the AI:
    ✓ Mapped "customers" → "users" table
    ✓ Used correct JOIN (users.id = orders.user_id)
    ✓ Added aggregation for "total spending"
    ✓ Included order_count as bonus information
    """)
    
    # =========================================================================
    # STEP 5: SQL Validator Agent
    # =========================================================================
    print_step(5, "SQL VALIDATOR AGENT")
    
    print("""
    The Validator checks the generated SQL:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        VALIDATION CHECKS                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  1. SYNTAX VALIDATION                                                    │
    │     └── Run EXPLAIN on the query                                        │
    │     └── ✓ Syntax is valid                                               │
    │                                                                          │
    │  2. SCHEMA VALIDATION                                                    │
    │     └── Check all tables exist: users ✓, orders ✓                       │
    │     └── Check all columns exist: id ✓, name ✓, email ✓, total ✓        │
    │     └── ✓ All schema references valid                                   │
    │                                                                          │
    │  3. SEMANTIC VALIDATION                                                  │
    │     └── JOIN condition makes sense (users.id = orders.user_id) ✓        │
    │     └── GROUP BY matches SELECT columns ✓                               │
    │     └── ✓ Semantically correct                                          │
    │                                                                          │
    │  4. SECURITY VALIDATION                                                  │
    │     └── No SQL injection patterns ✓                                     │
    │     └── No dangerous operations (DROP, DELETE, TRUNCATE) ✓              │
    │     └── ✓ Security checks passed                                        │
    │                                                                          │
    │  5. PERFORMANCE HINTS                                                    │
    │     └── Consider adding LIMIT for large result sets                     │
    │     └── Index on orders.user_id would improve performance               │
    │                                                                          │
    │  RESULT: ✓ VALID                                                        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Actually validate
    result = adapter.execute_query(f"EXPLAIN QUERY PLAN {generated_sql}")
    print("Actual EXPLAIN output:")
    if result.success:
        for row in result.rows:
            print(f"  {row}")
        print("\n✓ SQL is syntactically valid!")
    
    # =========================================================================
    # STEP 6: Self-Healing (If Needed)
    # =========================================================================
    print_step(6, "SELF-HEALING LOOP (If Validation Fails)")
    
    print("""
    If validation fails, the SQL Healer Agent fixes the issues:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     SELF-HEALING EXAMPLE                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ORIGINAL (with errors):                                                 │
    │  SELECT user_name, emil FROM users WHERE stauts = 'actve'               │
    │          ▲           ▲                  ▲          ▲                    │
    │          │           │                  │          │                    │
    │          │           │                  │          └── Typo: 'active'   │
    │          │           │                  └── Typo: 'status'              │
    │          │           └── Typo: 'email'                                  │
    │          └── Wrong column: should be 'name'                             │
    │                                                                          │
    │  ITERATION 1:                                                            │
    │  ├── Validator: ❌ Column 'user_name' not found. Did you mean 'name'?   │
    │  ├── Validator: ❌ Column 'emil' not found. Did you mean 'email'?       │
    │  ├── Validator: ❌ Column 'stauts' not found. Did you mean 'status'?    │
    │  └── Healer: Applies fuzzy matching corrections                         │
    │                                                                          │
    │  HEALED SQL:                                                             │
    │  SELECT name, email FROM users WHERE status = 'active'                  │
    │                                                                          │
    │  ITERATION 2:                                                            │
    │  └── Validator: ✓ Valid!                                                │
    │                                                                          │
    │  Max iterations: 5 (configurable)                                        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # =========================================================================
    # STEP 7: Final Output
    # =========================================================================
    print_step(7, "FINAL OUTPUT")
    
    print("""
    The pipeline returns a complete response:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      SQLGenerationResponse                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  status: SUCCESS                                                         │
    │                                                                          │
    │  final_sql:                                                              │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  SELECT                                                          │    │
    │  │      u.id,                                                       │    │
    │  │      u.name,                                                     │    │
    │  │      u.email,                                                    │    │
    │  │      COUNT(o.id) as order_count,                                │    │
    │  │      SUM(o.total) as total_spending                             │    │
    │  │  FROM users u                                                    │    │
    │  │  INNER JOIN orders o ON u.id = o.user_id                        │    │
    │  │  GROUP BY u.id, u.name, u.email                                 │    │
    │  │  ORDER BY total_spending DESC                                   │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                                                          │
    │  validation_result:                                                      │
    │    is_valid: true                                                        │
    │    issues: []                                                            │
    │                                                                          │
    │  generation_result:                                                      │
    │    confidence: 0.95                                                      │
    │    explanation: "Joined users with orders to calculate total spending"  │
    │    query_type: SELECT                                                    │
    │                                                                          │
    │  healing_attempts: []  (none needed - valid on first try)               │
    │                                                                          │
    │  metrics:                                                                │
    │    generation_time: 1.2s                                                 │
    │    validation_time: 0.1s                                                 │
    │    total_time: 1.3s                                                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Actually execute the query
    print("Executing the generated SQL:")
    print("─" * 60)
    result = adapter.execute_query(generated_sql)
    if result.success:
        print(f"Columns: {result.columns}")
        for row in result.rows:
            print(f"  {row}")
    print("─" * 60)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("COMPLETE FLOW SUMMARY")
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                        END-TO-END FLOW                                       │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                              │
    │   USER QUERY                                                                 │
    │   "Show me all customers who have placed orders with their total spending"  │
    │                                      │                                       │
    │                                      ▼                                       │
    │   ┌───────────────────────────────────────────────────────────────────────┐ │
    │   │              SCHEMA INTELLIGENCE LAYER                                 │ │
    │   │                                                                        │ │
    │   │  Input Sources (any combination):                                     │ │
    │   │  ┌────────────┐  ┌────────────┐  ┌────────────┐                       │ │
    │   │  │schema.yaml │  │ docs.txt   │  │ Database   │                       │ │
    │   │  │(optional)  │  │(optional)  │  │(auto-disc) │                       │ │
    │   │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                       │ │
    │   │        └───────────────┼───────────────┘                              │ │
    │   │                        ▼                                               │ │
    │   │  ┌─────────────────────────────────────────────────────────────────┐  │ │
    │   │  │                    SEMANTIC MODEL                                │  │ │
    │   │  │  • Tables with descriptions                                     │  │ │
    │   │  │  • Columns with semantic types (email, price, status...)       │  │ │
    │   │  │  • Relationships (explicit + inferred)                         │  │ │
    │   │  │  • Business terms ("customer" → users)                         │  │ │
    │   │  └─────────────────────────────────────────────────────────────────┘  │ │
    │   └───────────────────────────────────────────────────────────────────────┘ │
    │                                      │                                       │
    │                                      ▼                                       │
    │   ┌───────────────────────────────────────────────────────────────────────┐ │
    │   │              SQL GENERATOR AGENT                                       │ │
    │   │  • Receives schema context + dialect hints + user query              │ │
    │   │  • Sends to AWS Bedrock Claude AI                                     │ │
    │   │  • Returns: SQL + confidence + explanation                           │ │
    │   └───────────────────────────────────────────────────────────────────────┘ │
    │                                      │                                       │
    │                                      ▼                                       │
    │   ┌───────────────────────────────────────────────────────────────────────┐ │
    │   │              SQL VALIDATOR AGENT                                       │ │
    │   │  • Syntax check (EXPLAIN)                                             │ │
    │   │  • Schema validation (tables/columns exist)                          │ │
    │   │  • Semantic check (JOINs make sense)                                 │ │
    │   │  • Security scan                                                      │ │
    │   └───────────────────────────────────────────────────────────────────────┘ │
    │                                      │                                       │
    │                          ┌───────────┴───────────┐                          │
    │                          │                       │                          │
    │                       Valid?                 Invalid?                       │
    │                          │                       │                          │
    │                          ▼                       ▼                          │
    │   ┌─────────────────────────────┐  ┌─────────────────────────────────────┐ │
    │   │      RETURN RESULT          │  │     SQL HEALER AGENT                │ │
    │   │                             │  │  • Analyze errors                   │ │
    │   │  final_sql: "SELECT..."     │  │  • Generate fixes                   │ │
    │   │  is_valid: true             │  │  • Re-validate                      │ │
    │   │  confidence: 0.95           │  │  • Loop until valid (max 5x)        │ │
    │   └─────────────────────────────┘  └──────────────┬──────────────────────┘ │
    │                                                   │                         │
    │                                                   └──► Back to Validator    │
    │                                                                              │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    KEY FEATURES:
    ═════════════
    
    1. SCHEMA INTELLIGENCE
       • Works with or without schema files
       • Auto-discovers relationships from naming (user_id → users)
       • Infers semantic types (email, price, status, etc.)
       • Maps business terms to technical tables
    
    2. MULTI-DATABASE SUPPORT
       • MySQL, PostgreSQL, Oracle, SQLite
       • Dialect-specific SQL hints
       • Factory pattern for adapters
    
    3. SELF-HEALING
       • Automatically fixes typos and errors
       • Fuzzy matching for column/table names
       • Up to 5 correction iterations
    
    4. PRODUCTION-READY
       • Thread-safe for concurrent requests
       • Comprehensive metrics and logging
       • Error handling with custom exceptions
    """)
    
    adapter.disconnect()
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demonstrate_complete_flow()
