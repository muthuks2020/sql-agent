"""
Schema Intelligence Manager

The main entry point for schema intelligence. Handles all scenarios:
1. Schema files available (YAML/JSON with full metadata)
2. Text description files available (natural language docs)
3. No files available - pure database auto-discovery

Usage:
    # Scenario A: With schema files
    manager = SchemaIntelligenceManager(
        adapter=db_adapter,
        schema_file="schema.yaml",
        description_file="docs.txt"
    )
    model = manager.build_model()
    
    # Scenario B: Database only (auto-discovery)
    manager = SchemaIntelligenceManager(adapter=db_adapter)
    model = manager.build_model()
    
    # Scenario C: From directory (auto-detect files)
    manager = SchemaIntelligenceManager.from_directory(
        "/path/to/schema/",
        adapter=db_adapter
    )
    model = manager.build_model()
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import json

from .models import (
    BusinessContext,
    ColumnSemantics,
    Relationship,
    RelationshipSource,
    RelationshipType,
    SemanticModel,
    SemanticType,
    TableSemantics,
)
from .providers import (
    DatabaseSchemaProvider,
    FileSchemaProvider,
    HybridSchemaProvider,
    TextDescriptionProvider,
)
from .relationship_discovery import (
    DataPatternMatcher,
    LLMRelationshipInferrer,
    NamingConventionAnalyzer,
    RelationshipCandidate,
    RelationshipDiscoveryEngine,
)
from ..adapters.base import BaseDatabaseAdapter
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaIntelligenceConfig:
    """Configuration for schema intelligence"""
    # File paths
    schema_file: Optional[str] = None
    description_file: Optional[str] = None
    schema_directory: Optional[str] = None
    
    # Auto-discovery options
    auto_discover_from_db: bool = True
    discover_relationships: bool = True
    use_naming_conventions: bool = True
    use_data_patterns: bool = True
    use_llm_inference: bool = False
    
    # Confidence thresholds
    min_relationship_confidence: float = 0.5
    
    # Sample data
    include_sample_data: bool = True
    sample_size: int = 5
    
    # Statistics
    include_row_counts: bool = True
    include_column_stats: bool = False
    
    # Output
    cache_model: bool = True
    cache_path: Optional[str] = None
    export_discovered_schema: bool = False
    export_path: Optional[str] = None


class SchemaIntelligenceManager:
    """
    Main manager for schema intelligence
    
    Handles the complete flow of:
    1. Detecting available schema sources
    2. Loading from files if available
    3. Auto-discovering from database if needed
    4. Merging all sources into a unified model
    5. Discovering relationships
    6. Generating enriched context for SQL generation
    """
    
    def __init__(
        self,
        adapter: Optional[BaseDatabaseAdapter] = None,
        schema_file: Optional[str] = None,
        description_file: Optional[str] = None,
        llm_client: Optional[Any] = None,
        config: Optional[SchemaIntelligenceConfig] = None,
    ):
        self.adapter = adapter
        self.llm_client = llm_client
        self.config = config or SchemaIntelligenceConfig(
            schema_file=schema_file,
            description_file=description_file,
        )
        
        # Override config with explicit parameters
        if schema_file:
            self.config.schema_file = schema_file
        if description_file:
            self.config.description_file = description_file
        
        self._model: Optional[SemanticModel] = None
        self._sources_used: List[str] = []
    
    @classmethod
    def from_directory(
        cls,
        directory: str,
        adapter: Optional[BaseDatabaseAdapter] = None,
        llm_client: Optional[Any] = None,
    ) -> "SchemaIntelligenceManager":
        """
        Create manager by scanning a directory for schema files
        
        Looks for:
        - schema.yaml, schema.json, database.yaml, etc.
        - descriptions.txt, docs.txt, README.md, data_dictionary.txt, etc.
        """
        dir_path = Path(directory)
        
        schema_file = None
        description_file = None
        
        # Schema file patterns
        schema_patterns = [
            'schema.yaml', 'schema.yml', 'schema.json',
            'database.yaml', 'database.yml', 'database.json',
            'tables.yaml', 'tables.yml', 'tables.json',
            'metadata.yaml', 'metadata.json',
        ]
        
        # Description file patterns
        desc_patterns = [
            'descriptions.txt', 'descriptions.md',
            'docs.txt', 'docs.md',
            'README.md', 'readme.md',
            'data_dictionary.txt', 'data_dictionary.md',
            'database_docs.txt', 'database_docs.md',
            'schema_description.txt',
        ]
        
        for name in schema_patterns:
            path = dir_path / name
            if path.exists():
                schema_file = str(path)
                logger.info(f"Found schema file: {schema_file}")
                break
        
        for name in desc_patterns:
            path = dir_path / name
            if path.exists():
                description_file = str(path)
                logger.info(f"Found description file: {description_file}")
                break
        
        return cls(
            adapter=adapter,
            schema_file=schema_file,
            description_file=description_file,
            llm_client=llm_client,
            config=SchemaIntelligenceConfig(
                schema_directory=directory,
            ),
        )
    
    @classmethod
    def from_database_only(
        cls,
        adapter: BaseDatabaseAdapter,
        llm_client: Optional[Any] = None,
        use_llm_inference: bool = False,
    ) -> "SchemaIntelligenceManager":
        """
        Create manager for pure database auto-discovery
        
        No files needed - extracts everything from database:
        - Table structures
        - Column types and constraints
        - Foreign key relationships
        - Inferred relationships from naming patterns
        - Sample data for semantic inference
        """
        return cls(
            adapter=adapter,
            llm_client=llm_client,
            config=SchemaIntelligenceConfig(
                auto_discover_from_db=True,
                discover_relationships=True,
                use_naming_conventions=True,
                use_data_patterns=True,
                use_llm_inference=use_llm_inference,
                include_sample_data=True,
                include_row_counts=True,
            ),
        )
    
    def build_model(self) -> SemanticModel:
        """
        Build the complete semantic model
        
        Flow:
        1. Check what sources are available
        2. Load from files if available
        3. Auto-discover from database if needed
        4. Merge all sources
        5. Discover relationships
        6. Enrich with samples and statistics
        7. Cache/export if configured
        """
        logger.info("Building semantic model...")
        self._sources_used = []
        
        # Step 1: Determine available sources
        has_schema_file = self._file_exists(self.config.schema_file)
        has_description_file = self._file_exists(self.config.description_file)
        has_database = self.adapter is not None
        
        logger.info(f"Available sources - Schema file: {has_schema_file}, "
                   f"Description file: {has_description_file}, Database: {has_database}")
        
        # Step 2: Load from available sources
        if has_schema_file or has_description_file:
            # Use hybrid provider to combine files + database
            self._model = self._load_from_files_and_db()
        elif has_database:
            # Pure database auto-discovery
            self._model = self._auto_discover_from_database()
        else:
            raise ValueError(
                "No schema sources available. Provide either:\n"
                "- schema_file (YAML/JSON with table definitions)\n"
                "- description_file (text with table descriptions)\n"
                "- adapter (database connection for auto-discovery)"
            )
        
        if not self._model:
            raise ValueError("Failed to build semantic model from available sources")
        
        # Step 3: Discover relationships
        if self.config.discover_relationships:
            self._discover_and_apply_relationships()
        
        # Step 4: Enrich with additional data
        if self.adapter:
            self._enrich_model()
        
        # Step 5: Cache/export if configured
        if self.config.export_discovered_schema and self.config.export_path:
            self._export_model()
        
        self._model.sources = self._sources_used
        self._model.last_updated = datetime.utcnow()
        
        logger.info(
            f"Model built successfully: {len(self._model.tables)} tables, "
            f"{len(self._model.relationships)} relationships"
        )
        
        return self._model
    
    def _file_exists(self, path: Optional[str]) -> bool:
        """Check if a file exists"""
        return path is not None and os.path.exists(path)
    
    def _load_from_files_and_db(self) -> SemanticModel:
        """Load schema from files, optionally enriched with database info"""
        logger.info("Loading schema from files...")
        
        provider = HybridSchemaProvider(
            adapter=self.adapter if self.config.auto_discover_from_db else None,
            schema_file=self.config.schema_file,
            description_file=self.config.description_file,
        )
        
        model = provider.load()
        
        if self.config.schema_file and self._file_exists(self.config.schema_file):
            self._sources_used.append(f"schema_file:{self.config.schema_file}")
        if self.config.description_file and self._file_exists(self.config.description_file):
            self._sources_used.append(f"description_file:{self.config.description_file}")
        if self.adapter and self.config.auto_discover_from_db:
            self._sources_used.append("database_auto_discovery")
        
        return model
    
    def _auto_discover_from_database(self) -> SemanticModel:
        """
        Pure database auto-discovery
        
        This is the fallback when no files are available.
        Extracts everything from the database structure.
        """
        logger.info("Auto-discovering schema from database...")
        
        if not self.adapter:
            raise ValueError("Database adapter required for auto-discovery")
        
        # Ensure connection
        if not self.adapter.is_connected():
            self.adapter.connect()
        
        # Create database provider
        provider = DatabaseSchemaProvider(self.adapter)
        
        if self.config.include_sample_data:
            model = provider.load_with_samples(self.config.sample_size)
        else:
            model = provider.load()
        
        if not model:
            raise ValueError("Failed to discover schema from database")
        
        self._sources_used.append("database_auto_discovery")
        
        # Add row counts if configured
        if self.config.include_row_counts:
            self._add_row_counts(model)
        
        return model
    
    def _add_row_counts(self, model: SemanticModel) -> None:
        """Add row counts to tables"""
        if not self.adapter:
            return
        
        for table_name, table in model.tables.items():
            try:
                result = self.adapter.execute_query(f"SELECT COUNT(*) FROM {table_name}")
                if result.success and result.rows:
                    table.row_count = result.rows[0][0]
            except Exception as e:
                logger.debug(f"Could not get row count for {table_name}: {e}")
    
    def _discover_and_apply_relationships(self) -> None:
        """Discover and apply relationships to the model"""
        if not self._model:
            return
        
        logger.info("Discovering relationships...")
        
        engine = RelationshipDiscoveryEngine(
            model=self._model,
            adapter=self.adapter,
            llm_client=self.llm_client if self.config.use_llm_inference else None,
        )
        
        # Discover relationships
        candidates = engine.discover_all(
            use_naming_conventions=self.config.use_naming_conventions,
            use_data_patterns=self.config.use_data_patterns and self.adapter is not None,
            use_llm=self.config.use_llm_inference and self.llm_client is not None,
            min_confidence=self.config.min_relationship_confidence,
        )
        
        # Apply to model
        if candidates:
            self._model = engine.apply_to_model(candidates)
            logger.info(f"Discovered {len(candidates)} relationships")
        
        # Identify junction tables
        junction_tables = engine.discover_junction_tables()
        if junction_tables:
            logger.info(f"Identified junction tables: {junction_tables}")
    
    def _enrich_model(self) -> None:
        """Enrich model with additional data from database"""
        if not self._model or not self.adapter:
            return
        
        logger.info("Enriching model with database statistics...")
        
        for table_name, table in self._model.tables.items():
            # Add sample data if not already present
            if self.config.include_sample_data:
                self._add_sample_data(table_name, table)
            
            # Infer semantic types from samples
            self._infer_semantic_types(table)
    
    def _add_sample_data(self, table_name: str, table: TableSemantics) -> None:
        """Add sample data to table columns"""
        try:
            result = self.adapter.get_sample_data(table_name, self.config.sample_size)
            
            if result.success and result.rows:
                for i, col_name in enumerate(result.columns):
                    if col_name in table.columns:
                        col = table.columns[col_name]
                        if not col.sample_values:
                            col.sample_values = [
                                row[i] for row in result.rows
                                if row[i] is not None
                            ][:self.config.sample_size]
        except Exception as e:
            logger.debug(f"Could not get samples for {table_name}: {e}")
    
    def _infer_semantic_types(self, table: TableSemantics) -> None:
        """Infer semantic types from sample data"""
        for col in table.columns.values():
            if col.semantic_type == SemanticType.UNKNOWN and col.sample_values:
                inferred = self._infer_type_from_samples(col.sample_values)
                if inferred != SemanticType.UNKNOWN:
                    col.semantic_type = inferred
    
    def _infer_type_from_samples(self, samples: List[Any]) -> SemanticType:
        """Infer semantic type from sample values"""
        if not samples:
            return SemanticType.UNKNOWN
        
        # Filter to string samples for pattern matching
        str_samples = [str(s) for s in samples if s is not None]
        if not str_samples:
            return SemanticType.UNKNOWN
        
        # Email pattern
        email_pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
        if all(email_pattern.match(s) for s in str_samples):
            return SemanticType.EMAIL
        
        # URL pattern
        url_pattern = re.compile(r'^https?://|^www\.')
        if all(url_pattern.match(s) for s in str_samples):
            return SemanticType.URL
        
        # Phone pattern
        phone_pattern = re.compile(r'^[\d\s\-\+\(\)]{10,}$')
        if all(phone_pattern.match(s) for s in str_samples):
            return SemanticType.PHONE
        
        # UUID pattern
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
        if all(uuid_pattern.match(s) for s in str_samples):
            return SemanticType.UUID
        
        # Status-like (few distinct short strings)
        unique = set(str_samples)
        if len(unique) <= 10 and all(len(s) < 30 for s in unique):
            # Could be status/enum
            return SemanticType.STATUS
        
        return SemanticType.UNKNOWN
    
    def _export_model(self) -> None:
        """Export discovered model to file"""
        if not self._model or not self.config.export_path:
            return
        
        self._model.save(self.config.export_path)
        logger.info(f"Exported model to {self.config.export_path}")
    
    def get_model(self) -> Optional[SemanticModel]:
        """Get the built model (build if not already built)"""
        if self._model is None:
            self.build_model()
        return self._model
    
    def refresh_model(self) -> SemanticModel:
        """Force rebuild the model"""
        self._model = None
        return self.build_model()
    
    def get_schema_context(self, user_query: Optional[str] = None) -> str:
        """
        Get schema context for SQL generation
        
        If user_query is provided, returns focused context for that query.
        Otherwise returns full schema context.
        """
        if self._model is None:
            self.build_model()
        
        if not self._model:
            raise ValueError("No model available")
        
        from .model_builder import SchemaContextGenerator
        
        generator = SchemaContextGenerator(self._model)
        
        if user_query:
            return generator.generate_query_focused_context(user_query)
        else:
            return generator.generate_full_context()
    
    def get_sources_used(self) -> List[str]:
        """Get list of sources used to build the model"""
        return self._sources_used.copy()
    
    def validate_model(self) -> Dict[str, Any]:
        """Validate the built model and return diagnostics"""
        if self._model is None:
            return {"valid": False, "error": "No model built"}
        
        diagnostics = {
            "valid": True,
            "tables": len(self._model.tables),
            "relationships": len(self._model.relationships),
            "sources": self._sources_used,
            "warnings": [],
            "errors": [],
        }
        
        # Check for tables without columns
        for table_name, table in self._model.tables.items():
            if not table.columns:
                diagnostics["warnings"].append(f"Table '{table_name}' has no columns")
        
        # Check for orphan relationships
        table_names = set(self._model.tables.keys())
        for rel in self._model.relationships:
            if rel.source_table not in table_names:
                diagnostics["errors"].append(
                    f"Relationship '{rel.name}' references unknown table '{rel.source_table}'"
                )
                diagnostics["valid"] = False
            if rel.target_table not in table_names:
                diagnostics["errors"].append(
                    f"Relationship '{rel.name}' references unknown table '{rel.target_table}'"
                )
                diagnostics["valid"] = False
        
        # Check for FK columns without relationships
        for table_name, table in self._model.tables.items():
            for col_name, col in table.columns.items():
                if col.is_foreign_key and not col.references_table:
                    diagnostics["warnings"].append(
                        f"Column '{table_name}.{col_name}' marked as FK but has no reference"
                    )
        
        return diagnostics


# Convenience functions for common use cases
def create_schema_model(
    adapter: Optional[BaseDatabaseAdapter] = None,
    schema_file: Optional[str] = None,
    description_file: Optional[str] = None,
    auto_discover: bool = True,
) -> SemanticModel:
    """
    Create a semantic model from available sources
    
    This is the simplest way to create a model:
    
        # From database only
        model = create_schema_model(adapter=my_adapter)
        
        # From files + database
        model = create_schema_model(
            adapter=my_adapter,
            schema_file="schema.yaml",
            description_file="docs.txt"
        )
        
        # From files only (no database)
        model = create_schema_model(
            schema_file="schema.yaml",
            auto_discover=False
        )
    """
    manager = SchemaIntelligenceManager(
        adapter=adapter,
        schema_file=schema_file,
        description_file=description_file,
        config=SchemaIntelligenceConfig(
            auto_discover_from_db=auto_discover and adapter is not None,
        ),
    )
    return manager.build_model()


def discover_schema_from_database(
    adapter: BaseDatabaseAdapter,
    include_samples: bool = True,
    discover_relationships: bool = True,
    export_path: Optional[str] = None,
) -> SemanticModel:
    """
    Auto-discover schema from database connection
    
    Use when you have no schema files and want to extract
    everything from the database:
    
        model = discover_schema_from_database(adapter)
        
        # Export for future use
        model = discover_schema_from_database(
            adapter,
            export_path="discovered_schema.yaml"
        )
    """
    manager = SchemaIntelligenceManager.from_database_only(
        adapter=adapter,
    )
    manager.config.include_sample_data = include_samples
    manager.config.discover_relationships = discover_relationships
    manager.config.export_discovered_schema = export_path is not None
    manager.config.export_path = export_path
    
    return manager.build_model()
