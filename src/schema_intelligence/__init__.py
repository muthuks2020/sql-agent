"""
Schema Intelligence Module

Provides intelligent schema understanding for the SQL Agent System:
- Load schemas from multiple sources (files, text, database)
- Auto-discover relationships between tables
- Infer semantic types for columns
- Generate enriched context for SQL generation

USAGE SCENARIOS:
================

Scenario A: Schema file + description file + database
    manager = SchemaIntelligenceManager(
        adapter=db_adapter,
        schema_file="schema.yaml",
        description_file="docs.txt"
    )
    model = manager.build_model()

Scenario B: Database only (auto-discovery)
    manager = SchemaIntelligenceManager.from_database_only(adapter)
    model = manager.build_model()

Scenario C: From directory (auto-detect files)
    manager = SchemaIntelligenceManager.from_directory(
        "/path/to/schema/",
        adapter=db_adapter
    )
    model = manager.build_model()

Quick helpers:
    # Simplest usage
    model = create_schema_model(adapter=db_adapter)
    
    # Auto-discover and export
    model = discover_schema_from_database(
        adapter,
        export_path="discovered_schema.yaml"
    )
"""

# Core models
from .models import (
    SemanticType,
    RelationshipType,
    RelationshipSource,
    ColumnSemantics,
    TableSemantics,
    Relationship,
    BusinessContext,
    SemanticModel,
)

# Schema providers
from .providers import (
    BaseSchemaProvider,
    FileSchemaProvider,
    TextDescriptionProvider,
    DatabaseSchemaProvider,
    HybridSchemaProvider,
    SchemaFileConfig,
)

# Relationship discovery
from .relationship_discovery import (
    RelationshipCandidate,
    NamingConventionAnalyzer,
    DataPatternMatcher,
    LLMRelationshipInferrer,
    RelationshipDiscoveryEngine,
)

# Model builder
from .model_builder import (
    ModelBuilderConfig,
    SemanticModelBuilder,
    QuickModelBuilder,
    SchemaContextGenerator,
)

# Main manager
from .manager import (
    SchemaIntelligenceConfig,
    SchemaIntelligenceManager,
    create_schema_model,
    discover_schema_from_database,
)

# Smart loader (alternative entry point)
from .smart_loader import (
    SmartSchemaLoader,
    LoaderConfig,
    SchemaSource,
    load_schema,
    analyze_database,
)

__all__ = [
    # Models
    "SemanticType",
    "RelationshipType",
    "RelationshipSource",
    "ColumnSemantics",
    "TableSemantics",
    "Relationship",
    "BusinessContext",
    "SemanticModel",
    
    # Providers
    "BaseSchemaProvider",
    "FileSchemaProvider",
    "TextDescriptionProvider",
    "DatabaseSchemaProvider",
    "HybridSchemaProvider",
    "SchemaFileConfig",
    
    # Relationship Discovery
    "RelationshipCandidate",
    "NamingConventionAnalyzer",
    "DataPatternMatcher",
    "LLMRelationshipInferrer",
    "RelationshipDiscoveryEngine",
    
    # Model Builder
    "ModelBuilderConfig",
    "SemanticModelBuilder",
    "QuickModelBuilder",
    "SchemaContextGenerator",
    
    # Manager (main entry point)
    "SchemaIntelligenceConfig",
    "SchemaIntelligenceManager",
    "create_schema_model",
    "discover_schema_from_database",
    
    # Smart Loader (alternative entry point)
    "SmartSchemaLoader",
    "LoaderConfig",
    "SchemaSource",
    "load_schema",
    "analyze_database",
]
