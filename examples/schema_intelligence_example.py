"""
Schema Intelligence Examples

Demonstrates all three scenarios for loading database schema:
1. Schema file + description file + database
2. Database only (auto-discovery)
3. From directory (auto-detect files)

Run this example:
    python examples/schema_intelligence_example.py
"""
import os
import sys
import sqlite3
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import SQLiteAdapter
from src.schema_intelligence import (
    SchemaIntelligenceManager,
    SchemaIntelligenceConfig,
    create_schema_model,
    discover_schema_from_database,
    SchemaContextGenerator,
)
from src.config import DatabaseConfig, DatabaseType


def create_sample_database() -> str:
    """Create a sample SQLite database for testing"""
    db_path = "/tmp/schema_intel_demo.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript("""
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS categories;
        DROP TABLE IF EXISTS users;
        
        -- Users table
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(100),
            password_hash VARCHAR(255),
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login_at TIMESTAMP
        );
        
        -- Categories table (self-referential)
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            slug VARCHAR(100) UNIQUE,
            parent_id INTEGER REFERENCES categories(id)
        );
        
        -- Products table
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku VARCHAR(50) UNIQUE NOT NULL,
            name VARCHAR(200) NOT NULL,
            description TEXT,
            price DECIMAL(10,2) NOT NULL,
            cost DECIMAL(10,2),
            category_id INTEGER REFERENCES categories(id),
            stock_quantity INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Orders table
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_number VARCHAR(50) UNIQUE,
            user_id INTEGER NOT NULL REFERENCES users(id),
            status VARCHAR(20) DEFAULT 'pending',
            subtotal DECIMAL(10,2),
            tax DECIMAL(10,2),
            shipping DECIMAL(10,2),
            total DECIMAL(10,2),
            shipping_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        );
        
        -- Order items table
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL REFERENCES orders(id),
            product_id INTEGER NOT NULL REFERENCES products(id),
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL,
            line_total DECIMAL(10,2)
        );
        
        -- Insert sample data
        INSERT INTO users (email, name, status) VALUES
            ('alice@example.com', 'Alice Johnson', 'active'),
            ('bob@example.com', 'Bob Smith', 'active'),
            ('carol@example.com', 'Carol White', 'inactive');
        
        INSERT INTO categories (name, slug, parent_id) VALUES
            ('Electronics', 'electronics', NULL),
            ('Clothing', 'clothing', NULL),
            ('Phones', 'phones', 1),
            ('Laptops', 'laptops', 1);
        
        INSERT INTO products (sku, name, description, price, cost, category_id, stock_quantity) VALUES
            ('PHONE-001', 'Smartphone X', 'Latest smartphone', 999.99, 600.00, 3, 50),
            ('LAPTOP-001', 'Pro Laptop', 'Professional laptop', 1499.99, 900.00, 4, 25),
            ('SHIRT-001', 'Cotton T-Shirt', 'Comfortable cotton shirt', 29.99, 10.00, 2, 100);
        
        INSERT INTO orders (order_number, user_id, status, subtotal, tax, shipping, total) VALUES
            ('ORD-2024-001', 1, 'delivered', 999.99, 80.00, 10.00, 1089.99),
            ('ORD-2024-002', 1, 'shipped', 29.99, 2.40, 5.00, 37.39),
            ('ORD-2024-003', 2, 'pending', 1499.99, 120.00, 0.00, 1619.99);
        
        INSERT INTO order_items (order_id, product_id, quantity, unit_price, line_total) VALUES
            (1, 1, 1, 999.99, 999.99),
            (2, 3, 1, 29.99, 29.99),
            (3, 2, 1, 1499.99, 1499.99);
    """)
    
    conn.commit()
    conn.close()
    
    return db_path


def scenario_a_with_files():
    """
    SCENARIO A: Schema file + description file + database
    
    This is the richest scenario where you have:
    - schema.yaml: Detailed table/column definitions with descriptions
    - descriptions.txt: Natural language documentation
    - Database connection: For verification and samples
    """
    print("\n" + "="*70)
    print("SCENARIO A: Schema File + Description File + Database")
    print("="*70)
    
    # Create sample database
    db_path = create_sample_database()
    
    # Create adapter
    adapter = SQLiteAdapter(DatabaseConfig(
        db_type=DatabaseType.SQLITE,
        database=db_path,
    ))
    adapter.connect()
    
    # Get paths to example files
    examples_dir = Path(__file__).parent
    schema_file = examples_dir / "schema.yaml"
    description_file = examples_dir / "descriptions.txt"
    
    # Build model with all sources
    manager = SchemaIntelligenceManager(
        adapter=adapter,
        schema_file=str(schema_file) if schema_file.exists() else None,
        description_file=str(description_file) if description_file.exists() else None,
        config=SchemaIntelligenceConfig(
            auto_discover_from_db=True,
            discover_relationships=True,
            include_sample_data=True,
        ),
    )
    
    model = manager.build_model()
    
    print(f"\n✓ Built model from sources: {manager.get_sources_used()}")
    print(f"✓ Tables: {len(model.tables)}")
    print(f"✓ Relationships: {len(model.relationships)}")
    
    # Show some details
    print("\nTables discovered:")
    for table_name, table in model.tables.items():
        desc = table.description or "(no description)"
        print(f"  - {table_name}: {desc[:60]}...")
    
    print("\nRelationships discovered:")
    for rel in model.relationships:
        print(f"  - {rel.source_table}.{rel.source_columns[0]} -> "
              f"{rel.target_table}.{rel.target_columns[0]} ({rel.source.value})")
    
    # Generate context for a query
    context_gen = SchemaContextGenerator(model)
    context = context_gen.generate_query_focused_context(
        "Show me all orders for customers who registered last month"
    )
    print(f"\nGenerated context for query (first 500 chars):")
    print(context[:500] + "...")
    
    adapter.disconnect()


def scenario_b_database_only():
    """
    SCENARIO B: Database only (auto-discovery)
    
    When you have no schema files, the system will:
    - Extract all tables and columns from the database
    - Find explicit foreign key constraints
    - Infer relationships from naming conventions (user_id -> users)
    - Analyze sample data for semantic type inference
    """
    print("\n" + "="*70)
    print("SCENARIO B: Database Only (Auto-Discovery)")
    print("="*70)
    
    # Create sample database
    db_path = create_sample_database()
    
    # Create adapter
    adapter = SQLiteAdapter(DatabaseConfig(
        db_type=DatabaseType.SQLITE,
        database=db_path,
    ))
    adapter.connect()
    
    # Use the quick helper for database-only discovery
    model = discover_schema_from_database(
        adapter=adapter,
        include_samples=True,
        discover_relationships=True,
    )
    
    print(f"\n✓ Auto-discovered {len(model.tables)} tables")
    print(f"✓ Found {len(model.relationships)} relationships")
    
    # Show discovered tables with inferred info
    print("\nAuto-discovered tables:")
    for table_name, table in model.tables.items():
        print(f"\n  {table_name}:")
        print(f"    Columns: {len(table.columns)}")
        if table.row_count:
            print(f"    Rows: {table.row_count}")
        
        # Show FK columns discovered
        fk_cols = [c for c in table.columns.values() if c.is_foreign_key]
        if fk_cols:
            print(f"    Foreign Keys:")
            for col in fk_cols:
                print(f"      - {col.name} -> {col.references_table}.{col.references_column}")
    
    # Show inferred relationships
    print("\nInferred relationships:")
    for rel in model.relationships:
        conf = f"{rel.confidence:.0%}" if rel.confidence < 1.0 else "explicit"
        print(f"  - {rel.source_table}.{rel.source_columns[0]} -> "
              f"{rel.target_table} ({rel.source.value}, {conf})")
    
    adapter.disconnect()


def scenario_c_from_directory():
    """
    SCENARIO C: From directory (auto-detect files)
    
    Point to a directory and the system will automatically find:
    - schema.yaml, schema.json, database.yaml, etc.
    - descriptions.txt, docs.txt, README.md, etc.
    
    Then combine with database discovery.
    """
    print("\n" + "="*70)
    print("SCENARIO C: From Directory (Auto-Detect Files)")
    print("="*70)
    
    # Create sample database
    db_path = create_sample_database()
    
    # Create adapter
    adapter = SQLiteAdapter(DatabaseConfig(
        db_type=DatabaseType.SQLITE,
        database=db_path,
    ))
    adapter.connect()
    
    # Point to examples directory
    examples_dir = Path(__file__).parent
    
    # Auto-detect files in directory
    manager = SchemaIntelligenceManager.from_directory(
        directory=str(examples_dir),
        adapter=adapter,
    )
    
    model = manager.build_model()
    
    print(f"\n✓ Built model from directory: {examples_dir}")
    print(f"✓ Sources used: {manager.get_sources_used()}")
    print(f"✓ Tables: {len(model.tables)}")
    print(f"✓ Relationships: {len(model.relationships)}")
    
    # Validate the model
    diagnostics = manager.validate_model()
    print(f"\nModel validation:")
    print(f"  Valid: {diagnostics['valid']}")
    if diagnostics['warnings']:
        print(f"  Warnings: {diagnostics['warnings']}")
    if diagnostics['errors']:
        print(f"  Errors: {diagnostics['errors']}")
    
    adapter.disconnect()


def export_discovered_schema():
    """
    Bonus: Export discovered schema
    
    Useful for generating initial schema files that you can then
    edit to add descriptions and business context.
    """
    print("\n" + "="*70)
    print("BONUS: Export Discovered Schema")
    print("="*70)
    
    # Create sample database
    db_path = create_sample_database()
    
    # Create adapter
    adapter = SQLiteAdapter(DatabaseConfig(
        db_type=DatabaseType.SQLITE,
        database=db_path,
    ))
    adapter.connect()
    
    # Export path
    export_path = "/tmp/discovered_schema.yaml"
    
    # Discover and export
    model = discover_schema_from_database(
        adapter=adapter,
        include_samples=True,
        discover_relationships=True,
        export_path=export_path,
    )
    
    print(f"\n✓ Exported schema to: {export_path}")
    print(f"\nYou can now edit this file to add:")
    print("  - Table descriptions and purposes")
    print("  - Column descriptions and semantic types")
    print("  - Business context and terminology")
    print("  - Common query patterns")
    
    # Show first 50 lines of exported file
    print(f"\nFirst 30 lines of {export_path}:")
    print("-" * 50)
    with open(export_path) as f:
        for i, line in enumerate(f):
            if i >= 30:
                print("...")
                break
            print(line.rstrip())
    
    adapter.disconnect()


def demonstrate_context_generation():
    """
    Demonstrate how schema context helps SQL generation
    """
    print("\n" + "="*70)
    print("DEMO: Context Generation for SQL Agent")
    print("="*70)
    
    # Create sample database
    db_path = create_sample_database()
    
    # Create adapter
    adapter = SQLiteAdapter(DatabaseConfig(
        db_type=DatabaseType.SQLITE,
        database=db_path,
    ))
    adapter.connect()
    
    # Build model
    model = discover_schema_from_database(adapter, discover_relationships=True)
    context_gen = SchemaContextGenerator(model)
    
    # Example queries and their generated contexts
    queries = [
        "Show all users",
        "Get orders for user alice@example.com",
        "What products are low in stock?",
        "Calculate revenue by category",
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        context = context_gen.generate_query_focused_context(query)
        # Show just the relevant tables section
        lines = context.split('\n')
        for line in lines[:25]:  # First 25 lines
            print(line)
        if len(lines) > 25:
            print("...")
    
    adapter.disconnect()


if __name__ == "__main__":
    print("Schema Intelligence Examples")
    print("="*70)
    
    # Run all scenarios
    scenario_a_with_files()
    scenario_b_database_only()
    scenario_c_from_directory()
    export_discovered_schema()
    demonstrate_context_generation()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
