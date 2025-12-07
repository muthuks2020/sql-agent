#!/usr/bin/env python3
"""
Query Understanding Demo

Demonstrates how the Query Understanding module analyzes natural language
queries before SQL generation, improving the quality of generated SQL.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query_understanding import (
    QueryUnderstanding,
    QueryAnalysis,
    SQLOperationType,
    AggregationType,
    FilterType,
    TimeFrame,
    analyze_query,
)


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_analysis(query: str, analysis: QueryAnalysis):
    """Pretty print query analysis results"""
    print(f"\n📝 Query: \"{query}\"")
    print("-" * 60)
    
    # Operation type with emoji
    op_emoji = {
        SQLOperationType.SELECT: "🔍",
        SQLOperationType.INSERT: "➕",
        SQLOperationType.UPDATE: "✏️",
        SQLOperationType.DELETE: "🗑️",
    }
    print(f"  {op_emoji.get(analysis.operation, '❓')} Operation: {analysis.operation.value}")
    
    # Tables
    if analysis.potential_tables:
        print(f"  📋 Tables: {', '.join(analysis.potential_tables)}")
    
    # Columns
    if analysis.potential_columns:
        print(f"  📊 Columns: {', '.join(analysis.potential_columns)}")
    
    # Features
    features = []
    if analysis.has_aggregation:
        agg_str = ', '.join(a.value for a in analysis.aggregation_types)
        features.append(f"Aggregation ({agg_str})")
    if analysis.has_filtering:
        features.append("Filtering")
    if analysis.has_time_filter:
        time_str = ', '.join(t.frame.value for t in analysis.time_references)
        features.append(f"Time ({time_str})")
    if analysis.has_ordering:
        features.append("Ordering")
    if analysis.has_grouping:
        features.append("Grouping")
    if analysis.has_limit:
        features.append(f"Limit ({analysis.limit_value})")
    if analysis.needs_distinct:
        features.append("Distinct")
    if analysis.needs_join:
        features.append("Joins")
    
    if features:
        print(f"  🔧 Features: {', '.join(features)}")
    
    # Confidence
    conf_bar = "█" * int(analysis.confidence * 10) + "░" * (10 - int(analysis.confidence * 10))
    print(f"  📈 Confidence: [{conf_bar}] {analysis.confidence:.0%}")
    
    # Warnings
    if analysis.warnings:
        print(f"  ⚠️  Warnings:")
        for w in analysis.warnings:
            print(f"      - {w}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║              QUERY UNDERSTANDING MODULE - DEMONSTRATION                  ║
║                                                                          ║
║  This module analyzes natural language queries BEFORE SQL generation     ║
║  to help the AI understand user intent and generate better SQL.          ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Define schema context
    schema_tables = ['users', 'orders', 'products', 'categories', 'order_items']
    schema_columns = {
        'users': ['id', 'name', 'email', 'status', 'created_at'],
        'orders': ['id', 'user_id', 'total', 'status', 'created_at'],
        'products': ['id', 'name', 'price', 'category_id', 'stock_quantity'],
        'categories': ['id', 'name', 'parent_id'],
        'order_items': ['id', 'order_id', 'product_id', 'quantity', 'unit_price'],
    }
    business_terms = {
        'customer': 'users',
        'client': 'users',
        'buyer': 'users',
        'purchase': 'orders',
        'sale': 'orders',
        'item': 'products',
    }
    
    # Create analyzer with context
    analyzer = QueryUnderstanding(
        schema_tables=schema_tables,
        schema_columns=schema_columns,
        business_terms=business_terms,
    )
    
    # =========================================================================
    # Test Cases
    # =========================================================================
    
    print_header("1. SELECT QUERIES")
    
    queries = [
        "Show me all customers",
        "List all active users",
        "What are the product names and prices?",
        "Get me the email addresses of all clients",
    ]
    
    for q in queries:
        analysis = analyzer.analyze(q)
        print_analysis(q, analysis)
    
    print_header("2. AGGREGATION QUERIES")
    
    queries = [
        "How many orders were placed last month?",
        "What is the total revenue?",
        "Show me the average order value",
        "What's the maximum price of products?",
        "Count of customers by status",
    ]
    
    for q in queries:
        analysis = analyzer.analyze(q)
        print_analysis(q, analysis)
    
    print_header("3. TIME-BASED QUERIES")
    
    queries = [
        "Show orders from today",
        "List users who signed up yesterday",
        "Sales from this week",
        "Products added last month",
        "Orders from the last 30 days",
    ]
    
    for q in queries:
        analysis = analyzer.analyze(q)
        print_analysis(q, analysis)
    
    print_header("4. FILTERING QUERIES")
    
    queries = [
        "Show customers where status is active",
        "Products with price greater than 100",
        "Orders between yesterday and today",
        "Users containing 'john' in their name",
        "Products with missing category",
    ]
    
    for q in queries:
        analysis = analyzer.analyze(q)
        print_analysis(q, analysis)
    
    print_header("5. SORTING & LIMITING QUERIES")
    
    queries = [
        "Top 10 customers by total orders",
        "Show the 5 most expensive products",
        "Latest orders sorted by date",
        "First 100 users alphabetically",
        "Highest spending customers",
    ]
    
    for q in queries:
        analysis = analyzer.analyze(q)
        print_analysis(q, analysis)
    
    print_header("6. JOIN QUERIES")
    
    queries = [
        "Show customers with their orders",
        "Products and their categories",
        "Users along with order totals",
        "Customer's purchase history",
    ]
    
    for q in queries:
        analysis = analyzer.analyze(q)
        print_analysis(q, analysis)
    
    print_header("7. COMPLEX QUERIES")
    
    queries = [
        "Show me the top 10 customers who spent more than $1000 last month, sorted by total spending",
        "Count unique products ordered by category this year",
        "List active users who haven't placed any orders",
        "Average order value per customer for the last 6 months",
    ]
    
    for q in queries:
        analysis = analyzer.analyze(q)
        print_analysis(q, analysis)
    
    print_header("8. UPDATE/DELETE QUERIES")
    
    queries = [
        "Update the status of inactive users",
        "Change the price of product 123 to 99.99",
        "Delete all orders older than a year",
        "Remove cancelled orders from last month",
    ]
    
    for q in queries:
        analysis = analyzer.analyze(q)
        print_analysis(q, analysis)
    
    # =========================================================================
    # Show Prompt Context Example
    # =========================================================================
    
    print_header("EXAMPLE: PROMPT CONTEXT GENERATION")
    
    query = "Show me the top 10 customers who spent more than $1000 last month with their order count"
    analysis = analyzer.analyze(query)
    
    print(f"\nQuery: \"{query}\"")
    print("\nGenerated context for LLM prompt:")
    print("-" * 60)
    print(analysis.to_prompt_context())
    print("-" * 60)
    
    print("""
This context is added to the LLM prompt, helping Claude understand:
• What SQL operation to use (SELECT)
• Which tables are relevant (users, orders)
• What features are needed (Aggregation, Filtering, Sorting, Limit)
• Time constraints (last_month)

This results in MORE ACCURATE SQL generation!
    """)
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print_header("SUMMARY: WHAT QUERY UNDERSTANDING DETECTS")
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DETECTION CAPABILITIES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  OPERATION TYPE                                                              │
│  ─────────────                                                               │
│  • SELECT: show, list, get, find, display, what, who, how many              │
│  • INSERT: add, create, new, insert, register                               │
│  • UPDATE: update, change, modify, edit, set, fix                           │
│  • DELETE: delete, remove, drop, erase, clear                               │
│                                                                              │
│  AGGREGATIONS                                                                │
│  ────────────                                                                │
│  • COUNT: count, how many, number of                                        │
│  • SUM: sum, total, combined                                                │
│  • AVG: average, mean, typical                                              │
│  • MIN: minimum, lowest, smallest, cheapest                                 │
│  • MAX: maximum, highest, largest, most expensive                           │
│                                                                              │
│  TIME FILTERS                                                                │
│  ────────────                                                                │
│  • today, yesterday, this week, last week                                   │
│  • this month, last month, this year, last year                             │
│  • last N days/weeks/months                                                 │
│                                                                              │
│  FILTERS                                                                     │
│  ───────                                                                     │
│  • Equality: is, equals, exactly                                            │
│  • Comparison: greater than, more than, less than, at least                 │
│  • Range: between, from...to                                                │
│  • Pattern: contains, starts with, ends with, like                          │
│  • Null: missing, empty, null, blank                                        │
│                                                                              │
│  SORTING                                                                     │
│  ───────                                                                     │
│  • DESC: highest, largest, most, top, newest, latest                        │
│  • ASC: lowest, smallest, least, bottom, oldest, first                      │
│                                                                              │
│  JOINS                                                                       │
│  ─────                                                                       │
│  • "with their", "along with", "including"                                  │
│  • Possessives: "customer's orders", "user's profile"                       │
│  • Multiple tables mentioned                                                 │
│                                                                              │
│  LIMIT                                                                       │
│  ─────                                                                       │
│  • "top 10", "first 5", "limit 100"                                         │
│  • "N results", "N rows", "N records"                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
