# Contributing to SQL Agent System

Thank you for your interest in contributing to SQL Agent System! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful** - Treat everyone with respect and kindness
- **Be inclusive** - Welcome newcomers and help them get started
- **Be constructive** - Provide helpful feedback and accept criticism gracefully
- **Be collaborative** - Work together to achieve the best outcomes

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- AWS Account with Bedrock access
- A database for testing (SQLite works for basic tests)

### Development Setup

1. **Fork the repository**
   
   Click the "Fork" button on GitHub to create your own copy.

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sql-agent-system.git
   cd sql-agent-system
   ```

3. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

5. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

6. **Set up AWS credentials**
   ```bash
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   export AWS_DEFAULT_REGION="us-east-1"
   ```

7. **Run tests to verify setup**
   ```bash
   pytest tests/unit/
   ```

## How to Contribute

### Reporting Bugs

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, database type)
   - Relevant logs or error messages

### Suggesting Features

1. **Check existing issues/discussions** for similar ideas
2. **Create a new discussion** or issue with:
   - Clear description of the feature
   - Use case and motivation
   - Potential implementation approach
   - Any alternatives considered

### Contributing Code

1. **Find an issue** to work on (look for "good first issue" labels)
2. **Comment on the issue** to claim it
3. **Create a branch** for your work
4. **Write code** following our standards
5. **Write tests** for your changes
6. **Submit a pull request**

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**
   ```bash
   pytest
   ```

2. **Run linting**
   ```bash
   flake8 src/ tests/
   black --check src/ tests/
   ```

3. **Update documentation** if needed

4. **Add to CHANGELOG.md** if applicable

### PR Guidelines

1. **Branch naming**
   - `feature/description` - New features
   - `fix/description` - Bug fixes
   - `docs/description` - Documentation
   - `refactor/description` - Code refactoring

2. **Commit messages**
   ```
   type: short description
   
   Longer description if needed.
   
   Fixes #123
   ```
   
   Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

3. **PR description**
   - Reference related issues
   - Describe what changed and why
   - Include testing instructions
   - Add screenshots if UI changes

### Review Process

1. All PRs require at least one review
2. Address review feedback promptly
3. Keep PRs focused and reasonably sized
4. Squash commits before merging if requested

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **Black** for formatting (line length 100)
- Use **type hints** for all functions
- Maximum line length: 100 characters

### Code Organization

```python
# Standard library imports
import os
from typing import Optional, List

# Third-party imports
import boto3
from pydantic import BaseModel

# Local imports
from .config import Settings
from .utils import logger
```

### Docstrings

Use Google style docstrings:

```python
def process_query(
    query: str,
    validate: bool = True,
    timeout: int = 30
) -> SQLGenerationResponse:
    """Process a natural language query and generate SQL.
    
    Args:
        query: Natural language query to convert to SQL.
        validate: Whether to validate the generated SQL.
        timeout: Maximum time in seconds for processing.
    
    Returns:
        SQLGenerationResponse containing the generated SQL
        and validation results.
    
    Raises:
        ValidationError: If the generated SQL is invalid.
        TimeoutError: If processing exceeds timeout.
    
    Example:
        >>> response = process_query("Show all users")
        >>> print(response.final_sql)
        'SELECT * FROM users'
    """
```

### Type Hints

```python
from typing import Optional, List, Dict, Any, Union

def get_schema(
    adapter: BaseDatabaseAdapter,
    include_samples: bool = False
) -> Dict[str, TableInfo]:
    ...
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                  # Unit tests (no external deps)
│   ├── test_adapters.py
│   ├── test_agents.py
│   └── test_schemas.py
├── integration/           # Integration tests
│   └── test_pipeline.py
└── load/                  # Load/performance tests
    └── locustfile.py
```

### Writing Tests

```python
import pytest
from sql_agent import create_pipeline

class TestSQLGenerator:
    """Tests for SQL Generator Agent."""
    
    def test_simple_select(self):
        """Test generating a simple SELECT query."""
        # Arrange
        pipeline = create_pipeline(db_type="sqlite", database=":memory:")
        
        # Act
        result = pipeline.generate("Show all users")
        
        # Assert
        assert "SELECT" in result.sql
        assert result.is_valid
    
    def test_invalid_table_raises_error(self):
        """Test that invalid table references are caught."""
        with pytest.raises(ValidationError):
            pipeline.generate("SELECT * FROM nonexistent")
    
    @pytest.mark.parametrize("query,expected_tables", [
        ("Show users", ["users"]),
        ("Show orders with users", ["orders", "users"]),
    ])
    def test_table_detection(self, query, expected_tables):
        """Test that correct tables are identified."""
        result = pipeline.generate(query)
        assert all(t in result.tables_used for t in expected_tables)
```

### Test Coverage

- Aim for **80%+ coverage**
- Test edge cases and error conditions
- Include both positive and negative tests

```bash
# Run with coverage
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Documentation

### When to Update Docs

- New features
- API changes
- Configuration changes
- New examples needed

### Documentation Files

- `README.md` - Project overview
- `docs/ARCHITECTURE.md` - Technical architecture
- `docs/API.md` - API reference
- `docs/EXAMPLES.md` - Usage examples
- `CHANGELOG.md` - Version history

### Docstring Requirements

All public functions and classes must have docstrings:

- Purpose/description
- Args with types and descriptions
- Returns description
- Raises (if applicable)
- Example (for complex functions)

## Questions?

- **GitHub Discussions** - General questions
- **GitHub Issues** - Bug reports, feature requests
- **Pull Request comments** - Code-specific questions

Thank you for contributing! 🎉
