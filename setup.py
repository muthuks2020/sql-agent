"""
SQL Agent System - Setup Configuration
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() 
        for line in fh.readlines() 
        if line.strip() and not line.startswith("#")
    ]

# Core requirements (without optional dependencies)
core_requirements = [
    "pydantic>=2.0.0",
    "pyyaml>=6.0.0",
    "boto3>=1.28.0",
    "botocore>=1.31.0",
    "python-dotenv>=1.0.0",
]

# Optional database drivers
mysql_requirements = ["mysql-connector-python>=8.0.0"]
postgresql_requirements = ["psycopg2-binary>=2.9.0"]
oracle_requirements = ["oracledb>=1.3.0"]

# Strands SDK (recommended)
strands_requirements = ["strands-agents>=0.1.0"]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

# Documentation requirements
docs_requirements = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
    "mkdocstrings>=0.22.0",
]

setup(
    name="sql-agent-system",
    version="1.0.0",
    author="SQL Agent Contributors",
    author_email="",
    description="Multi-agent AI system for natural language to SQL conversion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sql-agent-system",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/sql-agent-system/issues",
        "Documentation": "https://github.com/yourusername/sql-agent-system#readme",
        "Source Code": "https://github.com/yourusername/sql-agent-system",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=core_requirements,
    extras_require={
        "mysql": mysql_requirements,
        "postgresql": postgresql_requirements,
        "oracle": oracle_requirements,
        "strands": strands_requirements,
        "all-databases": mysql_requirements + postgresql_requirements + oracle_requirements,
        "dev": dev_requirements,
        "docs": docs_requirements,
        "all": (
            mysql_requirements + 
            postgresql_requirements + 
            oracle_requirements + 
            strands_requirements + 
            dev_requirements + 
            docs_requirements
        ),
    },
    entry_points={
        "console_scripts": [
            "sql-agent=sql_agent.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "sql_agent": ["py.typed"],
    },
    keywords=[
        "sql",
        "ai",
        "agent",
        "natural-language",
        "database",
        "bedrock",
        "claude",
        "strands",
    ],
)
