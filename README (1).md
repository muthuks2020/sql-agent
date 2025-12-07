# SQL Agent

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Convert natural language to SQL using AI with automatic validation and self-healing.**

Transform plain English questions into accurate, production-ready SQL queries. Supports MySQL, PostgreSQL, Oracle, and SQLite with zero configuration.

```
"Show me customers who spent more than $1000 last month"
                          |
                          v
SELECT u.name, u.email, SUM(o.total) as total_spent
FROM users u
JOIN orders o ON u.id = o.user_id  
WHERE o.created_at >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
GROUP BY u.id HAVING total_spent > 1000
```

## Features

- **Natural Language to SQL** - Ask questions in plain English
- **Self-Healing** - Automatically fixes SQL errors and typos
- **Multi-Database** - MySQL, PostgreSQL, Oracle, SQLite
- **Schema Intelligence** - Auto-discovers tables, relationships, and business terms
- **SQL Injection Protection** - Built-in security validation
- **Production Ready** - Thread-safe, metrics, logging, load-tested
- **AWS Bedrock Claude** - Powered by Anthropic's Claude AI

## Quick Start

### Installation

```bash
pip install sql-agent
```

Or install from source:

```bash
git clone https://github.com/YOUR_USERNAME/sql-agent.git
cd sql-agent
pip install -e .
```

### Basic Usage

```python
from sql_agent import create_pipeline, SQLGenerationRequest

# Connect to your database
pipeline = create_pipeline(
    db_type="postgresql",
    database="myapp",
    host="localhost",
    username="user",
    password="pass"
)

# Generate SQL from natural language
response = pipeline.process(
    SQLGenerationRequest(user_query="Show all active users with their orders")
)

print(response.final_sql)
# SELECT u.*, o.* FROM users u
# JOIN orders o ON u.id = o.user_id
# WHERE u.status = 'active'
```

### With AWS Credentials

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

## Architecture

```
+-------------------------------------------------------------+
|                    SQL AGENT PIPELINE                        |
+-------------------------------------------------------------+
|                                                              |
|  "Show me top customers"                                     |
|           |                                                  |
|           v                                                  |
|  +-----------------+                                         |
|  | Query           |  Understands English intent             |
|  | Understanding   |  -> tables, filters, aggregations       |
|  +--------+--------+                                         |
|           |                                                  |
|           v                                                  |
|  +-----------------+                                         |
|  | Schema          |  Auto-discovers database structure      |
|  | Intelligence    |  -> relationships, business terms       |
|  +--------+--------+                                         |
|           |                                                  |
|           v                                                  |
|  +-----------------+                                         |
|  | SQL Generator   |  Claude AI generates SQL                |
|  | (Claude)        |  -> dialect-specific syntax             |
|  +--------+--------+                                         |
|           |                                                  |
|           v                                                  |
|  +-----------------+                                         |
|  | SQL Validator   |  Validates against real database        |
|  |                 |  -> syntax, tables, columns, security   |
|  +--------+--------+                                         |
|           |                                                  |
|      +----+----+                                             |
|      v         v                                             |
|   VALID    INVALID --> SQL Healer --> Retry                  |
|      |                  (auto-fix)                           |
|      v                                                       |
|  [Final SQL]                                                 |
|                                                              |
+-------------------------------------------------------------+
```

## Supported Databases

| Database | Status | Adapter |
|----------|--------|---------|
| MySQL | Full Support | `MySQLAdapter` |
| PostgreSQL | Full Support | `PostgreSQLAdapter` |
| Oracle | Full Support | `OracleAdapter` |
| SQLite | Full Support | `SQLiteAdapter` |

## Schema Intelligence

SQL Agent automatically understands your database:

### Auto-Discovery (Zero Config)

```python
pipeline = create_pipeline(db_type="mysql", database="myapp", ...)
# Automatically discovers:
# - All tables and columns
# - Relationships (orders.user_id -> users.id)
# - Data types and constraints
```

### With Schema Files (Enhanced)

```python
pipeline = create_pipeline(
    db_type="mysql",
    database="myapp",
    schema_file="schema.yaml",      # Column descriptions
    description_file="docs.txt"     # Business term mappings
)
```

### Example schema.yaml

```yaml
tables:
  users:
    description: "Customer accounts"
    business_terms: ["customer", "client", "buyer"]
    columns:
      email:
        description: "Primary contact email"
        semantic_type: email
      
  orders:
    description: "Purchase transactions"
    columns:
      total:
        description: "Order total in USD"
        semantic_type: price
```

## Advanced Usage

### Custom Configuration

```python
from sql_agent import PipelineBuilder, DatabaseConfig, LLMConfig

pipeline = (
    PipelineBuilder()
    .with_database(DatabaseConfig(
        db_type="postgresql",
        host="localhost",
        database="myapp",
        username="user",
        password="pass"
    ))
    .with_llm(LLMConfig(
        provider="bedrock",
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature=0.1
    ))
    .with_schema_file("schema.yaml")
    .with_self_healing(max_iterations=5)
    .build()
)
```

### Analyze Unknown Database

```python
from sql_agent import analyze_database, create_adapter

adapter = create_adapter(db_type="mysql", database="legacy_db", ...)
model, report = analyze_database(adapter, output_path="discovered_schema.yaml")

print(f"Found {len(model.tables)} tables")
print(f"Found {len(model.relationships)} relationships")
```

### Query Understanding

```python
from sql_agent import analyze_query

analysis = analyze_query("Show me top 10 customers by revenue last month")

print(analysis.operation)        # SELECT
print(analysis.potential_tables) # ['customers', 'users']
print(analysis.has_aggregation)  # True
print(analysis.has_time_filter)  # True
print(analysis.has_limit)        # True (10)
```

### SQL Parsing and Safety

```python
from sql_agent import parse_sql, is_safe_query

# Parse SQL structure
structure = parse_sql("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
print(structure.tables)  # ['users', 'orders']
print(structure.joins)   # [('INNER', 'orders')]

# Safety check
is_safe, issues = is_safe_query("DELETE FROM users; DROP TABLE users;")
# is_safe = False
# issues = ['DROP TABLE detected', 'Multiple statements']
```

## Metrics and Monitoring

```python
from sql_agent import get_metrics_collector

metrics = get_metrics_collector()

# Get statistics
stats = metrics.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")
```

## Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Load tests
locust -f tests/load/locustfile.py
```

## Project Structure

```
sql-agent/
├── src/
│   ├── query_understanding/    # NLP: English to Intent
│   ├── schema_intelligence/    # DB to Knowledge Graph
│   ├── sql_understanding/      # SQL to Parsed Structure
│   ├── agents/                 # Generator, Validator, Healer
│   ├── adapters/               # MySQL, PostgreSQL, Oracle, SQLite
│   ├── orchestration/          # Pipeline coordination
│   └── utils/                  # Logging, metrics, errors
├── examples/
├── tests/
└── docs/
```

## Security

- SQL injection detection and prevention
- Dangerous operation warnings (DROP, TRUNCATE, DELETE without WHERE)
- Query complexity analysis
- Multiple statement detection
- Parameterized query support

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Anthropic Claude](https://www.anthropic.com/) - AI model powering SQL generation
- [AWS Bedrock](https://aws.amazon.com/bedrock/) - Managed AI service
- [sqlparse](https://github.com/andialbrecht/sqlparse) - SQL parsing library

## Contact

- GitHub Issues: [Report a bug](https://github.com/YOUR_USERNAME/sql-agent/issues)
- Discussions: [Ask questions](https://github.com/YOUR_USERNAME/sql-agent/discussions)

---

**If this project helps you, please give it a star!**

## Keywords

`natural-language-to-sql` `text-to-sql` `nl2sql` `text2sql` `sql-generator` `ai-sql` `llm-sql` `claude-sql` `aws-bedrock` `sql-agent` `query-generator` `database-ai` `sql-automation` `self-healing-sql` `multi-database` `mysql` `postgresql` `oracle` `sqlite`
