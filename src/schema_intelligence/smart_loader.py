"""
Smart Schema Loader

Automatically detects available schema sources and builds the best possible
semantic model. Handles all scenarios:

1. Schema files available → Load and use them
2. Text descriptions available → Parse and enrich
3. Nothing available → Auto-discover from database

Usage:
    # Automatic - figures out what's available
    loader = SmartSchemaLoader(adapter=db_adapter)
    model = loader.load()  # Auto-discovers everything
    
    # With hints about file locations
    loader = SmartSchemaLoader(
        adapter=db_adapter,
        schema_dir="/path/to/schemas",  # Will look for schema.yaml, docs.txt, etc.
    )
    model = loader.load()
    
    # Export discovered model for review/editing
    loader.export_model("discovered_schema.yaml")
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

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
    TextDescriptionProvider,
    HybridSchemaProvider,
)
from .relationship_discovery import (
    RelationshipDiscoveryEngine,
    NamingConventionAnalyzer,
    DataPatternMatcher,
)
from ..adapters.base import BaseDatabaseAdapter
from ..llm_client import BaseLLMClient
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaSource:
    """Represents a discovered schema source"""
    source_type: str  # "schema_file", "description_file", "database"
    path: Optional[str] = None
    available: bool = False
    priority: int = 0  # Lower = higher priority


@dataclass 
class LoaderConfig:
    """Configuration for the smart schema loader"""
    # File discovery
    schema_dir: Optional[str] = None
    schema_file: Optional[str] = None
    description_file: Optional[str] = None
    
    # Auto-discovery options
    auto_discover_relationships: bool = True
    use_naming_conventions: bool = True
    use_data_patterns: bool = True
    use_llm_inference: bool = False
    
    # Confidence thresholds
    min_relationship_confidence: float = 0.5
    
    # Sample data
    include_samples: bool = True
    sample_size: int = 5
    
    # Export
    auto_export: bool = False
    export_path: Optional[str] = None


class SmartSchemaLoader:
    """
    Intelligent schema loader that automatically discovers and loads
    the best available schema information.
    
    Priority order:
    1. Explicit schema files (YAML/JSON) - most authoritative
    2. Text description files - adds context
    3. Database auto-discovery - fills gaps
    
    All sources are merged to create the richest possible model.
    """
    
    # File patterns to look for
    SCHEMA_FILE_PATTERNS = [
        "schema.yaml", "schema.yml", "schema.json",
        "database.yaml", "database.yml", "database.json",
        "tables.yaml", "tables.yml", "tables.json",
        "metadata.yaml", "metadata.yml", "metadata.json",
    ]
    
    DESCRIPTION_FILE_PATTERNS = [
        "descriptions.txt", "descriptions.md",
        "docs.txt", "docs.md",
        "README.md", "readme.md",
        "data_dictionary.txt", "data_dictionary.md",
        "schema_docs.txt", "schema_docs.md",
    ]
    
    def __init__(
        self,
        adapter: Optional[BaseDatabaseAdapter] = None,
        llm_client: Optional[BaseLLMClient] = None,
        config: Optional[LoaderConfig] = None,
    ):
        self.adapter = adapter
        self.llm_client = llm_client
        self.config = config or LoaderConfig()
        
        self._sources: List[SchemaSource] = []
        self._model: Optional[SemanticModel] = None
        self._discovery_report: Dict[str, Any] = {}
    
    def discover_sources(self) -> List[SchemaSource]:
        """
        Discover all available schema sources
        
        Returns list of available sources sorted by priority
        """
        self._sources = []
        
        # Check explicit file paths first
        if self.config.schema_file:
            self._sources.append(SchemaSource(
                source_type="schema_file",
                path=self.config.schema_file,
                available=os.path.exists(self.config.schema_file),
                priority=1,
            ))
        
        if self.config.description_file:
            self._sources.append(SchemaSource(
                source_type="description_file",
                path=self.config.description_file,
                available=os.path.exists(self.config.description_file),
                priority=2,
            ))
        
        # Scan directory for schema files
        if self.config.schema_dir:
            self._scan_directory(self.config.schema_dir)
        
        # Check database availability
        if self.adapter:
            try:
                if not self.adapter.is_connected():
                    self.adapter.connect()
                db_available = self.adapter.test_connection()[0]
            except:
                db_available = False
            
            self._sources.append(SchemaSource(
                source_type="database",
                path=None,
                available=db_available,
                priority=10,  # Lowest priority but always useful
            ))
        
        # Sort by priority
        self._sources.sort(key=lambda s: s.priority)
        
        # Log what we found
        available = [s for s in self._sources if s.available]
        logger.info(f"Discovered {len(available)} schema sources:")
        for source in available:
            logger.info(f"  - {source.source_type}: {source.path or 'live connection'}")
        
        return self._sources
    
    def _scan_directory(self, directory: str) -> None:
        """Scan directory for schema and description files"""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Schema directory not found: {directory}")
            return
        
        # Look for schema files
        for pattern in self.SCHEMA_FILE_PATTERNS:
            path = dir_path / pattern
            if path.exists() and not any(s.path == str(path) for s in self._sources):
                self._sources.append(SchemaSource(
                    source_type="schema_file",
                    path=str(path),
                    available=True,
                    priority=1,
                ))
                break  # Only use first match
        
        # Look for description files
        for pattern in self.DESCRIPTION_FILE_PATTERNS:
            path = dir_path / pattern
            if path.exists() and not any(s.path == str(path) for s in self._sources):
                self._sources.append(SchemaSource(
                    source_type="description_file",
                    path=str(path),
                    available=True,
                    priority=2,
                ))
                break
    
    def load(self) -> SemanticModel:
        """
        Load schema from all available sources and build semantic model
        
        This is the main entry point. It will:
        1. Discover available sources
        2. Load from each source
        3. Merge into unified model
        4. Discover relationships if enabled
        5. Enrich with samples and semantics
        
        Returns:
            Complete SemanticModel
        """
        logger.info("Starting smart schema load...")
        
        # Step 1: Discover what's available
        self.discover_sources()
        
        available_sources = [s for s in self._sources if s.available]
        
        if not available_sources:
            raise ValueError(
                "No schema sources available. Provide schema files, "
                "description files, or a database connection."
            )
        
        # Step 2: Load from sources
        self._model = self._load_from_sources(available_sources)
        
        # Step 3: Discover relationships
        if self.config.auto_discover_relationships:
            self._discover_relationships()
        
        # Step 4: Enrich with samples
        if self.config.include_samples and self.adapter:
            self._add_sample_data()
        
        # Step 5: Infer semantic types
        self._infer_semantics()
        
        # Step 6: Generate discovery report
        self._generate_report()
        
        # Step 7: Auto-export if configured
        if self.config.auto_export and self.config.export_path:
            self.export_model(self.config.export_path)
        
        logger.info(
            f"Schema load complete: {len(self._model.tables)} tables, "
            f"{len(self._model.relationships)} relationships"
        )
        
        return self._model
    
    def _load_from_sources(self, sources: List[SchemaSource]) -> SemanticModel:
        """Load and merge from all available sources"""
        models = []
        
        for source in sources:
            if not source.available:
                continue
            
            model = None
            
            if source.source_type == "schema_file":
                provider = FileSchemaProvider(source.path)
                model = provider.load()
                if model:
                    logger.info(f"Loaded schema from file: {source.path}")
            
            elif source.source_type == "description_file":
                provider = TextDescriptionProvider(source.path)
                model = provider.load()
                if model:
                    logger.info(f"Loaded descriptions from: {source.path}")
            
            elif source.source_type == "database" and self.adapter:
                provider = DatabaseSchemaProvider(self.adapter)
                model = provider.load()
                if model:
                    logger.info(f"Loaded schema from database")
            
            if model:
                models.append(model)
        
        if not models:
            raise ValueError("Failed to load schema from any source")
        
        # Merge models
        if len(models) == 1:
            return models[0]
        
        # Use first model as base, merge others
        merged = models[0]
        for model in models[1:]:
            merged = self._merge_models(merged, model)
        
        merged.sources = list(set(
            source for model in models for source in model.sources
        ))
        
        return merged
    
    def _merge_models(self, base: SemanticModel, overlay: SemanticModel) -> SemanticModel:
        """Merge two models, with overlay providing descriptions"""
        # Update database info if missing
        if base.database_name == "unknown":
            base.database_name = overlay.database_name
        if base.database_type == "unknown":
            base.database_type = overlay.database_type
        
        # Merge tables
        for table_name, overlay_table in overlay.tables.items():
            if table_name in base.tables:
                base_table = base.tables[table_name]
                
                # Descriptions from overlay take precedence
                if overlay_table.description:
                    base_table.description = overlay_table.description
                if overlay_table.purpose:
                    base_table.purpose = overlay_table.purpose
                if overlay_table.business_name:
                    base_table.business_name = overlay_table.business_name
                if overlay_table.domain:
                    base_table.domain = overlay_table.domain
                
                # Merge columns
                for col_name, overlay_col in overlay_table.columns.items():
                    if col_name in base_table.columns:
                        base_col = base_table.columns[col_name]
                        
                        if overlay_col.description:
                            base_col.description = overlay_col.description
                        if overlay_col.business_name:
                            base_col.business_name = overlay_col.business_name
                        if overlay_col.semantic_type != SemanticType.UNKNOWN:
                            base_col.semantic_type = overlay_col.semantic_type
                        if overlay_col.references_table:
                            base_col.references_table = overlay_col.references_table
                            base_col.references_column = overlay_col.references_column
                            base_col.is_foreign_key = True
                    else:
                        base_table.columns[col_name] = overlay_col
            else:
                base.tables[table_name] = overlay_table
        
        # Merge relationships
        existing_rels = {
            (r.source_table, tuple(r.source_columns), r.target_table)
            for r in base.relationships
        }
        for rel in overlay.relationships:
            key = (rel.source_table, tuple(rel.source_columns), rel.target_table)
            if key not in existing_rels:
                base.relationships.append(rel)
        
        # Merge business context
        if overlay.business_context:
            if not base.business_context:
                base.business_context = overlay.business_context
            else:
                if overlay.business_context.domain:
                    base.business_context.domain = overlay.business_context.domain
                if overlay.business_context.description:
                    base.business_context.description = overlay.business_context.description
                base.business_context.terminology.update(
                    overlay.business_context.terminology
                )
                base.business_context.key_entities = list(set(
                    base.business_context.key_entities + 
                    overlay.business_context.key_entities
                ))
                base.business_context.common_queries.update(
                    overlay.business_context.common_queries
                )
                base.business_context.business_rules.extend(
                    overlay.business_context.business_rules
                )
        
        return base
    
    def _discover_relationships(self) -> None:
        """Discover relationships between tables"""
        if not self._model:
            return
        
        engine = RelationshipDiscoveryEngine(
            model=self._model,
            adapter=self.adapter,
            llm_client=self.llm_client if self.config.use_llm_inference else None,
        )
        
        candidates = engine.discover_all(
            use_naming_conventions=self.config.use_naming_conventions,
            use_data_patterns=self.config.use_data_patterns and self.adapter is not None,
            use_llm=self.config.use_llm_inference and self.llm_client is not None,
            min_confidence=self.config.min_relationship_confidence,
        )
        
        if candidates:
            self._model = engine.apply_to_model(candidates)
            
            # Track what was discovered
            self._discovery_report["relationships_discovered"] = len(candidates)
            self._discovery_report["relationship_sources"] = {}
            for c in candidates:
                source = c.source.value
                self._discovery_report["relationship_sources"][source] = \
                    self._discovery_report["relationship_sources"].get(source, 0) + 1
        
        # Identify junction tables
        junction_tables = engine.discover_junction_tables()
        self._discovery_report["junction_tables"] = junction_tables
    
    def _add_sample_data(self) -> None:
        """Add sample data to columns for better inference"""
        if not self._model or not self.adapter:
            return
        
        for table_name, table in self._model.tables.items():
            try:
                result = self.adapter.get_sample_data(
                    table_name, 
                    self.config.sample_size
                )
                
                if result.success and result.rows:
                    for i, col_name in enumerate(result.columns):
                        if col_name in table.columns:
                            samples = [
                                row[i] for row in result.rows
                                if row[i] is not None
                            ][:self.config.sample_size]
                            table.columns[col_name].sample_values = samples
            except Exception as e:
                logger.debug(f"Could not get samples for {table_name}: {e}")
    
    def _infer_semantics(self) -> None:
        """Infer semantic types for columns"""
        if not self._model:
            return
        
        inferred_count = 0
        
        for table in self._model.tables.values():
            for col in table.columns.values():
                if col.semantic_type == SemanticType.UNKNOWN:
                    inferred = self._infer_column_type(col)
                    if inferred != SemanticType.UNKNOWN:
                        col.semantic_type = inferred
                        inferred_count += 1
        
        self._discovery_report["semantic_types_inferred"] = inferred_count
    
    def _infer_column_type(self, column: ColumnSemantics) -> SemanticType:
        """Infer semantic type from column name, type, and samples"""
        import re
        
        name_lower = column.name.lower()
        type_lower = column.data_type.lower()
        
        # Check samples first (most accurate)
        if column.sample_values:
            inferred = self._infer_from_samples(column.sample_values)
            if inferred != SemanticType.UNKNOWN:
                return inferred
        
        # Name-based inference
        # IDs
        if name_lower == 'id' or name_lower == 'pk':
            return SemanticType.PRIMARY_KEY
        if name_lower.endswith('_id') and name_lower != 'id':
            return SemanticType.FOREIGN_KEY
        if 'uuid' in name_lower or 'guid' in name_lower:
            return SemanticType.UUID
        
        # Personal info
        if 'email' in name_lower:
            return SemanticType.EMAIL
        if 'phone' in name_lower or 'tel' in name_lower or 'mobile' in name_lower:
            return SemanticType.PHONE
        if name_lower in ('name', 'full_name', 'fullname', 'display_name'):
            return SemanticType.NAME
        if name_lower in ('first_name', 'firstname', 'fname', 'given_name'):
            return SemanticType.FIRST_NAME
        if name_lower in ('last_name', 'lastname', 'lname', 'surname', 'family_name'):
            return SemanticType.LAST_NAME
        
        # Address
        if 'address' in name_lower and 'email' not in name_lower:
            return SemanticType.ADDRESS
        if name_lower in ('city', 'town'):
            return SemanticType.CITY
        if name_lower in ('state', 'province', 'region'):
            return SemanticType.STATE
        if name_lower in ('country', 'nation'):
            return SemanticType.COUNTRY
        if 'zip' in name_lower or 'postal' in name_lower:
            return SemanticType.ZIP_CODE
        
        # Temporal
        if 'created' in name_lower and ('at' in name_lower or 'date' in name_lower or 'time' in name_lower):
            return SemanticType.CREATED_AT
        if 'updated' in name_lower or 'modified' in name_lower:
            return SemanticType.UPDATED_AT
        if 'deleted' in name_lower:
            return SemanticType.DELETED_AT
        
        # Financial
        if any(term in name_lower for term in ['price', 'cost', 'fee', 'amount', 'total', 'subtotal']):
            return SemanticType.PRICE
        if 'currency' in name_lower:
            return SemanticType.CURRENCY
        if 'percent' in name_lower or name_lower.endswith('_rate'):
            return SemanticType.PERCENTAGE
        
        # Status
        if 'status' in name_lower or 'state' in name_lower:
            return SemanticType.STATUS
        if 'bool' in type_lower or name_lower.startswith('is_') or name_lower.startswith('has_'):
            return SemanticType.BOOLEAN_FLAG
        
        # Content
        if name_lower in ('title', 'subject', 'headline', 'heading'):
            return SemanticType.TITLE
        if name_lower in ('description', 'desc', 'summary', 'bio', 'about'):
            return SemanticType.DESCRIPTION
        if name_lower in ('content', 'body', 'text', 'message'):
            return SemanticType.CONTENT
        if 'url' in name_lower or 'link' in name_lower:
            return SemanticType.URL
        if 'image' in name_lower or 'photo' in name_lower or 'avatar' in name_lower:
            return SemanticType.IMAGE_URL
        
        # Metrics
        if name_lower.endswith('_count') or name_lower.startswith('num_'):
            return SemanticType.COUNT
        if 'quantity' in name_lower or 'qty' in name_lower:
            return SemanticType.QUANTITY
        if 'score' in name_lower:
            return SemanticType.SCORE
        if 'rating' in name_lower:
            return SemanticType.RATING
        
        # Security
        if 'password' in name_lower or 'hash' in name_lower:
            return SemanticType.PASSWORD_HASH
        if 'token' in name_lower or 'secret' in name_lower or 'api_key' in name_lower:
            return SemanticType.TOKEN
        
        return SemanticType.UNKNOWN
    
    def _infer_from_samples(self, samples: List[Any]) -> SemanticType:
        """Infer type from sample values"""
        import re
        
        if not samples:
            return SemanticType.UNKNOWN
        
        str_samples = [s for s in samples if isinstance(s, str)]
        
        if not str_samples:
            return SemanticType.UNKNOWN
        
        # Email pattern
        email_pattern = re.compile(r'^[\w\.\-\+]+@[\w\.\-]+\.\w{2,}$')
        if all(email_pattern.match(s) for s in str_samples):
            return SemanticType.EMAIL
        
        # URL pattern
        url_pattern = re.compile(r'^https?://|^www\.')
        if all(url_pattern.match(s) for s in str_samples):
            return SemanticType.URL
        
        # Phone pattern (various formats)
        phone_pattern = re.compile(r'^[\d\s\-\+\(\)\.]{7,}$')
        if all(phone_pattern.match(s) for s in str_samples):
            return SemanticType.PHONE
        
        # UUID pattern
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        if all(uuid_pattern.match(s) for s in str_samples):
            return SemanticType.UUID
        
        # Status-like (few distinct short values)
        unique = set(str_samples)
        if len(unique) <= 10 and all(len(s) < 30 for s in unique):
            return SemanticType.STATUS
        
        return SemanticType.UNKNOWN
    
    def _generate_report(self) -> None:
        """Generate discovery report"""
        if not self._model:
            return
        
        self._discovery_report.update({
            "timestamp": datetime.utcnow().isoformat(),
            "database_name": self._model.database_name,
            "database_type": self._model.database_type,
            "tables_count": len(self._model.tables),
            "total_columns": sum(
                len(t.columns) for t in self._model.tables.values()
            ),
            "relationships_count": len(self._model.relationships),
            "sources_used": self._model.sources,
        })
    
    def get_discovery_report(self) -> Dict[str, Any]:
        """Get the discovery report"""
        return self._discovery_report.copy()
    
    def export_model(self, path: str, format: str = "auto") -> None:
        """
        Export the discovered model to a file
        
        This allows you to:
        1. Review the auto-discovered schema
        2. Edit and add descriptions
        3. Use as input for future runs
        
        Args:
            path: Output file path
            format: "yaml", "json", or "auto" (from extension)
        """
        if not self._model:
            raise ValueError("No model to export. Call load() first.")
        
        if format == "auto":
            if path.endswith('.json'):
                format = "json"
            else:
                format = "yaml"
        
        # Convert to exportable dict
        export_data = self._model_to_export_format()
        
        with open(path, 'w') as f:
            if format == "json":
                json.dump(export_data, f, indent=2, default=str)
            else:
                yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Exported model to {path}")
    
    def _model_to_export_format(self) -> Dict[str, Any]:
        """Convert model to human-friendly export format"""
        if not self._model:
            return {}
        
        export = {
            "database": {
                "name": self._model.database_name,
                "type": self._model.database_type,
            },
            "tables": {},
            "relationships": [],
        }
        
        # Add business context
        if self._model.business_context:
            export["database"]["domain"] = self._model.business_context.domain
            export["database"]["description"] = self._model.business_context.description
            export["business_context"] = {
                "terminology": self._model.business_context.terminology,
                "key_entities": self._model.business_context.key_entities,
                "common_queries": self._model.business_context.common_queries,
                "business_rules": self._model.business_context.business_rules,
            }
        
        # Export tables
        for table_name, table in self._model.tables.items():
            table_export = {
                "description": table.description or f"TODO: Add description for {table_name}",
                "columns": {},
            }
            
            if table.purpose:
                table_export["purpose"] = table.purpose
            if table.business_name:
                table_export["business_name"] = table.business_name
            if table.is_lookup_table:
                table_export["is_lookup_table"] = True
            if table.is_junction_table:
                table_export["is_junction_table"] = True
            if table.row_count:
                table_export["row_count"] = table.row_count
            
            for col_name, col in table.columns.items():
                col_export = {
                    "type": col.data_type,
                }
                
                if col.description:
                    col_export["description"] = col.description
                else:
                    col_export["description"] = f"TODO: Add description"
                
                if col.semantic_type != SemanticType.UNKNOWN:
                    col_export["semantic_type"] = col.semantic_type.value
                
                if col.is_primary_key:
                    col_export["primary_key"] = True
                if col.is_foreign_key:
                    col_export["foreign_key"] = True
                    if col.references_table:
                        col_export["references_table"] = col.references_table
                        col_export["references_column"] = col.references_column
                if col.is_unique:
                    col_export["unique"] = True
                if not col.nullable:
                    col_export["nullable"] = False
                if col.allowed_values:
                    col_export["allowed_values"] = col.allowed_values
                
                table_export["columns"][col_name] = col_export
            
            export["tables"][table_name] = table_export
        
        # Export relationships
        for rel in self._model.relationships:
            rel_export = {
                "name": rel.name,
                "from": f"{rel.source_table}.{rel.source_columns[0]}",
                "to": f"{rel.target_table}.{rel.target_columns[0]}",
                "type": rel.relationship_type.value,
            }
            if rel.description:
                rel_export["description"] = rel.description
            if rel.confidence < 1.0:
                rel_export["confidence"] = rel.confidence
                rel_export["discovered_by"] = rel.source.value
            
            export["relationships"].append(rel_export)
        
        return export


# Convenience functions
def load_schema(
    adapter: Optional[BaseDatabaseAdapter] = None,
    schema_file: Optional[str] = None,
    description_file: Optional[str] = None,
    schema_dir: Optional[str] = None,
    **kwargs
) -> SemanticModel:
    """
    Smart schema loading with automatic source detection
    
    Args:
        adapter: Database adapter (for live DB access)
        schema_file: Path to YAML/JSON schema file
        description_file: Path to text description file
        schema_dir: Directory to scan for schema files
        **kwargs: Additional LoaderConfig options
    
    Returns:
        SemanticModel with all available information
        
    Examples:
        # From database only (auto-discovers everything)
        model = load_schema(adapter=db_adapter)
        
        # From files only
        model = load_schema(schema_file="schema.yaml")
        
        # Hybrid (best option)
        model = load_schema(
            adapter=db_adapter,
            schema_file="schema.yaml",
            description_file="docs.txt"
        )
    """
    config = LoaderConfig(
        schema_file=schema_file,
        description_file=description_file,
        schema_dir=schema_dir,
        **kwargs
    )
    
    loader = SmartSchemaLoader(adapter=adapter, config=config)
    return loader.load()


def analyze_database(
    adapter: BaseDatabaseAdapter,
    output_path: Optional[str] = None,
    llm_client: Optional[BaseLLMClient] = None,
) -> Tuple[SemanticModel, Dict[str, Any]]:
    """
    Analyze a database and discover its structure
    
    Useful for:
    - Understanding an unknown database
    - Generating initial schema documentation
    - Finding relationships between tables
    
    Args:
        adapter: Connected database adapter
        output_path: Optional path to export discovered schema
        llm_client: Optional LLM for enhanced inference
    
    Returns:
        Tuple of (SemanticModel, discovery_report)
    """
    config = LoaderConfig(
        auto_discover_relationships=True,
        use_naming_conventions=True,
        use_data_patterns=True,
        use_llm_inference=llm_client is not None,
        include_samples=True,
        auto_export=output_path is not None,
        export_path=output_path,
    )
    
    loader = SmartSchemaLoader(
        adapter=adapter,
        llm_client=llm_client,
        config=config,
    )
    
    model = loader.load()
    report = loader.get_discovery_report()
    
    return model, report
