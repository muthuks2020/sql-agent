#!/usr/bin/env python3
"""
Schema Intelligence Examples

This script demonstrates all the ways to use the Schema Intelligence system:

SCENARIO A: I have nothing - just a database connection
SCENARIO B: I have schema files with metadata
SCENARIO C: I have text documentation files  
SCENARIO D: I have both files AND database (hybrid - best option)
SCENARIO E: Analyze unknown database and export schema

Run this script to see each scenario in action with a sample SQLite database.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import SQLiteAdapter
from src.schema_intelligence import (
    # Main entry points
    create_schema_model,
    discover_schema_from_database,
    load_schema,
    analyze_database,
    
    # Manager for advanced usage
    SchemaIntelligenceManager,
    SchemaIntelligenceConfig,
    
    # Context generation
    SchemaContextGenerator,
    
    # Models
    SemanticModel,
)


def create_sample_database() -> SQLiteAdapter:
    """Create a sample SQLite database with realistic e-commerce schema"""
    adapter = SQLiteAdapter(":memory:")
    adapter.connect()
    
    # Create tables
    adapter.execute_query("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email VARCHAR(255) NOT NULL UNIQUE,
            name VARCHAR(100),
            password_hash VARCHAR(255),
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login_at TIMESTAMP
        )
    """)
    
    adapter.execute_query("""
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            slug VARCHAR(100) UNIQUE,
            parent_id INTEGER REFERENCES categories(id)
        )
    """)
    
    adapter.execute_query("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            sku VARCHAR(50) NOT NULL UNIQUE,
            name VARCHAR(200) NOT NULL,
            description TEXT,
            price DECIMAL(10,2) NOT NULL,
            cost DECIMAL(10,2),
            category_id INTEGER REFERENCES categories(id),
            stock_quantity INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    adapter.execute_query("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            order_number VARCHAR(50) NOT NULL UNIQUE,
            user_id INTEGER NOT NULL REFERENCES users(id),
            status VARCHAR(20) DEFAULT 'pending',
            subtotal DECIMAL(10,2),
            tax DECIMAL(10,2),
            shipping DECIMAL(10,2),
            total DECIMAL(10,2),
            shipping_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
    """)
    
    adapter.execute_query("""
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL REFERENCES orders(id),
            product_id INTEGER NOT NULL REFERENCES products(id),
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL,
            line_total DECIMAL(10,2)
        )
    """)
    
    # Insert sample data
    adapter.execute_query("""
        INSERT INTO users (email, name, status) VALUES
        ('john@example.com', 'John Doe', 'active'),
        ('jane@example.com', 'Jane Smith', 'active'),
        ('bob@test.com', 'Bob Wilson', 'inactive')
    """)
    
    adapter.execute_query("""
        INSERT INTO categories (name, slug) VALUES
        ('Electronics', 'electronics'),
        ('Clothing', 'clothing'),
        ('Books', 'books')
    """)
    
    adapter.execute_query("""
        INSERT INTO products (sku, name, description, price, cost, category_id, stock_quantity) VALUES
        ('ELEC-001', 'Wireless Mouse', 'Ergonomic wireless mouse', 29.99, 15.00, 1, 100),
        ('ELEC-002', 'USB Keyboard', 'Mechanical keyboard', 79.99, 40.00, 1, 50),
        ('CLTH-001', 'Cotton T-Shirt', 'Comfortable cotton t-shirt', 19.99, 8.00, 2, 200)
    """)
    
    adapter.execute_query("""
        INSERT INTO orders (order_number, user_id, status, subtotal, tax, total) VALUES
        ('ORD-2024-001', 1, 'delivered', 109.98, 8.80, 118.78),
        ('ORD-2024-002', 2, 'shipped', 29.99, 2.40, 32.39)
    """)
    
    adapter.execute_query("""
        INSERT INTO order_items (order_id, product_id, quantity, unit_price, line_total) VALUES
        (1, 1, 1, 29.99, 29.99),
        (1, 2, 1, 79.99, 79.99),
        (2, 1, 1, 29.99, 29.99)
    """)
    
    return adapter


def create_sample_schema_file(directory: str) -> str:
    """Create a sample schema.yaml file"""
    schema_content = """
# E-commerce Database Schema
database:
  name: ecommerce
  type: sqlite
  domain: E-commerce
  description: Online retail platform database

tables:
  users:
    description: Customer accounts and authentication
    purpose: Stores user credentials and profile data
    business_name: Customers
    columns:
      id:
        type: integer
        primary_key: true
        description: Unique user identifier
      email:
        type: varchar(255)
        semantic_type: email
        description: User's email address for login and notifications
      name:
        type: varchar(100)
        semantic_type: name
        description: User's display name
      status:
        type: varchar(20)
        semantic_type: status
        allowed_values: ["active", "inactive", "suspended"]
      created_at:
        type: timestamp
        semantic_type: created_at

  products:
    description: Product catalog with pricing
    business_name: Products
    columns:
      id:
        type: integer
        primary_key: true
      price:
        type: decimal
        semantic_type: price
        description: Current selling price in USD
      category_id:
        type: integer
        foreign_key: true
        references_table: categories
        references_column: id

  orders:
    description: Customer orders
    business_name: Purchases
    columns:
      user_id:
        type: integer
        foreign_key: true
        references_table: users
        references_column: id
        description: Customer who placed the order

relationships:
  - name: user_orders
    from: orders.user_id
    to: users.id
    type: many_to_one
    description: Each order belongs to one customer

business_context:
  terminology:
    customer: users
    purchase: orders
    item: products
  key_entities:
    - users
    - orders
    - products
"""
    path = os.path.join(directory, "schema.yaml")
    with open(path, 'w') as f:
        f.write(schema_content)
    return path


def create_sample_description_file(directory: str) -> str:
    """Create a sample descriptions.txt file"""
    content = """
# Database Overview

This is an e-commerce database for an online retail platform.
It handles customer accounts, product catalog, and order management.

## Tables

### users
The users table stores customer account information.
- id: Unique identifier for each customer
- email: Customer's email address (must be unique, used for login)
- name: Full name of the customer
- status: Account status (active, inactive, suspended)
- created_at: When the account was created

### products
Products available in the store catalog.
- id: Product identifier
- sku: Stock keeping unit (unique product code)
- price: Selling price in USD
- category_id: Links to the categories table

### orders
Customer orders and their status.
- id: Order identifier
- user_id: Links to the users table (the customer who placed it)
- status: Order status (pending, paid, shipped, delivered, cancelled)
- total: Final order amount

### order_items
Individual items within each order.
- order_id: Links to the orders table
- product_id: Links to the products table
- quantity: Number of units

## Relationships
- Each user can have multiple orders (users -> orders via user_id)
- Each order has multiple items (orders -> order_items via order_id)
- Each item references one product (order_items -> products via product_id)

## Business Terms
- "customer" refers to a user
- "purchase" or "transaction" refers to an order
"""
    path = os.path.join(directory, "descriptions.txt")
    with open(path, 'w') as f:
        f.write(content)
    return path


def print_model_summary(model: SemanticModel, title: str):
    """Print a summary of the semantic model"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"Database: {model.database_name} ({model.database_type})")
    print(f"Tables: {len(model.tables)}")
    print(f"Relationships: {len(model.relationships)}")
    print(f"Sources: {', '.join(model.sources)}")
    
    if model.business_context:
        print(f"Domain: {model.business_context.domain}")
    
    print("\nTables:")
    for name, table in model.tables.items():
        desc = table.description[:50] + "..." if table.description and len(table.description) > 50 else table.description
        print(f"  - {name}: {len(table.columns)} columns - {desc or 'No description'}")
    
    print("\nRelationships:")
    for rel in model.relationships:
        print(f"  - {rel.source_table}.{rel.source_columns[0]} -> {rel.target_table}.{rel.target_columns[0]} ({rel.source.value})")


def scenario_a_database_only():
    """
    SCENARIO A: Have nothing - just database connection
    
    The system will:
    1. Connect to database
    2. Extract all table structures
    3. Find explicit foreign keys
    4. Infer relationships from naming (user_id -> users)
    5. Analyze sample data to infer semantic types
    """
    print("\n" + "="*70)
    print(" SCENARIO A: Database Only (Auto-Discovery)")
    print("="*70)
    print("""
    You have: Just a database connection
    You need: Understanding of the schema with relationships
    
    The system will automatically:
    - Extract all tables and columns
    - Find explicit foreign key constraints
    - Infer relationships from naming patterns (user_id -> users)
    - Detect semantic types (email, price, status, etc.)
    - Get sample data for better understanding
    """)
    
    # Create sample database
    adapter = create_sample_database()
    
    # Method 1: Using discover_schema_from_database
    print("\n--- Using discover_schema_from_database() ---")
    model = discover_schema_from_database(
        adapter=adapter,
        include_samples=True,
        discover_relationships=True,
    )
    print_model_summary(model, "Discovered Model")
    
    # Method 2: Using create_schema_model
    print("\n--- Using create_schema_model() ---")
    model = create_schema_model(adapter=adapter)
    print(f"Tables found: {list(model.tables.keys())}")
    
    # Show discovered relationships
    print("\nAuto-discovered relationships:")
    for rel in model.relationships:
        print(f"  {rel.source_table}.{rel.source_columns[0]} -> {rel.target_table}")
        print(f"    Source: {rel.source.value}, Confidence: {rel.confidence:.0%}")
    
    adapter.disconnect()


def scenario_b_schema_file_only():
    """
    SCENARIO B: Have schema files with metadata
    
    Load schema from YAML/JSON file with all metadata.
    No database connection required!
    """
    print("\n" + "="*70)
    print(" SCENARIO B: Schema File Only")
    print("="*70)
    print("""
    You have: A schema.yaml file with table definitions
    You need: A semantic model for SQL generation
    
    The system will:
    - Load all metadata from the YAML file
    - Use provided descriptions and semantic types
    - Use defined relationships
    """)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        schema_file = create_sample_schema_file(tmpdir)
        
        # Method 1: Using create_schema_model
        print(f"\n--- Loading from: {schema_file} ---")
        model = create_schema_model(
            schema_file=schema_file,
            auto_discover=False,  # No database
        )
        print_model_summary(model, "Model from Schema File")
        
        # Show business context
        if model.business_context:
            print("\nBusiness Context:")
            print(f"  Terminology: {model.business_context.terminology}")


def scenario_c_text_descriptions_only():
    """
    SCENARIO C: Have text documentation files
    
    Parse natural language documentation to extract schema info.
    """
    print("\n" + "="*70)
    print(" SCENARIO C: Text Descriptions Only")
    print("="*70)
    print("""
    You have: A text file with natural language documentation
    You need: A semantic model for SQL generation
    
    The system will:
    - Parse the text file for table/column descriptions
    - Extract relationship hints from text
    - Identify business terminology
    """)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        desc_file = create_sample_description_file(tmpdir)
        
        print(f"\n--- Loading from: {desc_file} ---")
        model = create_schema_model(
            description_file=desc_file,
            auto_discover=False,
        )
        print_model_summary(model, "Model from Text Descriptions")


def scenario_d_hybrid_best_option():
    """
    SCENARIO D: Have files AND database (BEST OPTION!)
    
    This gives the richest model:
    - Schema files provide descriptions and business context
    - Database provides actual types and additional tables
    - Auto-discovery fills gaps and finds more relationships
    """
    print("\n" + "="*70)
    print(" SCENARIO D: Hybrid - Files + Database (RECOMMENDED)")
    print("="*70)
    print("""
    You have: Schema files + Database connection
    You need: The most complete semantic model
    
    This is the BEST approach because:
    - Schema files provide human-written descriptions
    - Database provides accurate types and constraints
    - Auto-discovery finds additional relationships
    - All sources are merged intelligently
    """)
    
    adapter = create_sample_database()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        schema_file = create_sample_schema_file(tmpdir)
        desc_file = create_sample_description_file(tmpdir)
        
        # Using SchemaIntelligenceManager for full control
        print("\n--- Using SchemaIntelligenceManager ---")
        manager = SchemaIntelligenceManager(
            adapter=adapter,
            schema_file=schema_file,
            description_file=desc_file,
            config=SchemaIntelligenceConfig(
                auto_discover_from_db=True,
                discover_relationships=True,
                use_naming_conventions=True,
                use_data_patterns=True,
                include_sample_data=True,
            ),
        )
        
        model = manager.build_model()
        print_model_summary(model, "Hybrid Model (Best Quality)")
        
        # Show merged data
        print("\nSources used:", manager.get_sources_used())
        
        # Validate the model
        diagnostics = manager.validate_model()
        print(f"Model valid: {diagnostics['valid']}")
        if diagnostics['warnings']:
            print(f"Warnings: {diagnostics['warnings']}")
        
        # Generate context for a query
        context = manager.get_schema_context("Show me all customers with their orders")
        print("\n--- Context for 'Show me all customers with their orders' ---")
        print(context[:500] + "..." if len(context) > 500 else context)
    
    adapter.disconnect()


def scenario_e_analyze_and_export():
    """
    SCENARIO E: Analyze unknown database and export schema
    
    Perfect for when you're working with a new database:
    1. Analyze and discover everything
    2. Export to YAML for review/editing
    3. Use edited file for future runs
    """
    print("\n" + "="*70)
    print(" SCENARIO E: Analyze Database & Export Schema")
    print("="*70)
    print("""
    You have: An unknown database you need to understand
    You need: Documentation + schema file for future use
    
    The system will:
    - Analyze the entire database
    - Discover all relationships
    - Export everything to schema.yaml for review
    - You can then edit and use this file!
    """)
    
    adapter = create_sample_database()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = os.path.join(tmpdir, "discovered_schema.yaml")
        
        # Analyze and export
        print("\n--- Analyzing database... ---")
        model, report = analyze_database(
            adapter=adapter,
            output_path=export_path,
        )
        
        print_model_summary(model, "Analyzed Model")
        
        # Show discovery report
        print("\nDiscovery Report:")
        for key, value in report.items():
            print(f"  {key}: {value}")
        
        # Show exported file
        print(f"\n--- Exported to: {export_path} ---")
        with open(export_path, 'r') as f:
            content = f.read()
            # Show first 50 lines
            lines = content.split('\n')[:50]
            print('\n'.join(lines))
            if len(content.split('\n')) > 50:
                print("... (truncated)")
        
        print("\nYou can now:")
        print("1. Review the exported schema.yaml")
        print("2. Add descriptions and business context")
        print("3. Use it for future runs with create_schema_model(schema_file='...')")
    
    adapter.disconnect()


def scenario_f_smart_loader():
    """
    SCENARIO F: Using SmartSchemaLoader
    
    The SmartSchemaLoader automatically figures out what's available
    and does the right thing.
    """
    print("\n" + "="*70)
    print(" SCENARIO F: Smart Schema Loader (Auto-Detection)")
    print("="*70)
    print("""
    You have: Maybe files, maybe database, you're not sure
    You need: Just give me a model!
    
    SmartSchemaLoader will:
    - Scan for schema files in a directory
    - Check if database is available
    - Use whatever it finds
    - Merge everything together
    """)
    
    adapter = create_sample_database()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files in the directory
        create_sample_schema_file(tmpdir)
        create_sample_description_file(tmpdir)
        
        # SmartSchemaLoader will find them!
        print(f"\n--- Scanning directory: {tmpdir} ---")
        model = load_schema(
            adapter=adapter,
            schema_dir=tmpdir,  # Will auto-find files
        )
        
        print_model_summary(model, "Smart-Loaded Model")
    
    adapter.disconnect()


def main():
    """Run all scenarios"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              SCHEMA INTELLIGENCE DEMONSTRATION                        ║
║                                                                      ║
║  This demonstrates all ways to load and understand database schemas  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run each scenario
    scenario_a_database_only()
    scenario_b_schema_file_only()
    scenario_c_text_descriptions_only()
    scenario_d_hybrid_best_option()
    scenario_e_analyze_and_export()
    scenario_f_smart_loader()
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print("""
    Choose your approach:
    
    1. HAVE NOTHING? Use database auto-discovery:
       model = discover_schema_from_database(adapter)
    
    2. HAVE SCHEMA FILES? Load them directly:
       model = create_schema_model(schema_file="schema.yaml")
    
    3. HAVE BOTH? Use hybrid for best results:
       model = create_schema_model(
           adapter=adapter,
           schema_file="schema.yaml",
           description_file="docs.txt"
       )
    
    4. WANT TO ANALYZE & EXPORT? Use analyze_database:
       model, report = analyze_database(adapter, output_path="schema.yaml")
    
    5. NOT SURE WHAT YOU HAVE? Use load_schema:
       model = load_schema(adapter=adapter, schema_dir="./")
    """)


if __name__ == "__main__":
    main()
