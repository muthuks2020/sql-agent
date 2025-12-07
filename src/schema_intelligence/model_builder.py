"""
Semantic Model Builder

Orchestrates the complete process of building a semantic model:
1. Load schema from available sources (files, text, database)
2. Discover relationships automatically
3. Infer semantic types
4. Generate enriched context for SQL generation
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    BusinessContext,
    ColumnSemantics,
    Relationship,
    SemanticModel,
    SemanticType,
    TableSemantics,
)
from .providers import (
    BaseSchemaProvider,
    DatabaseSchemaProvider,
    FileSchemaProvider,
    HybridSchemaProvider,
    TextDescriptionProvider,
)
from .relationship_discovery import RelationshipDiscoveryEngine
from ..adapters.base import BaseDatabaseAdapter
from ..llm_client import BaseLLMClient
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelBuilderConfig:
    """Configuration for model building"""
    # Schema sources
    schema_file: Optional[str] = None  # Path to YAML/JSON schema file
    description_file: Optional[str] = None  # Path to text description file
    
    # Discovery options
    auto_discover_relationships: bool = True
    use_naming_conventions: bool = True
    use_data_patterns: bool = True
    use_llm_inference: bool = False  # Expensive, use carefully
    min_relationship_confidence: float = 0.5
    
    # Sample data options
    include_sample_data: bool = True
    sample_size: int = 5
    
    # Output options
    export_model: bool = False
    export_path: Optional[str] = None
    export_format: str = "yaml"  # yaml or json


class SemanticModelBuilder:
    """
    Builds a comprehensive semantic model from multiple sources
    
    Usage:
        builder = SemanticModelBuilder(adapter)
        builder.add_schema_file("schema.yaml")
        builder.add_description_file("docs.txt")
        model = builder.build()
    """
    
    def __init__(
        self,
        adapter: Optional[BaseDatabaseAdapter] = None,
        llm_client: Optional[BaseLLMClient] = None,
        config: Optional[ModelBuilderConfig] = None,
    ):
        self.adapter = adapter
        self.llm_client = llm_client
        self.config = config or ModelBuilderConfig()
        
        self.schema_file: Optional[str] = self.config.schema_file
        self.description_file: Optional[str] = self.config.description_file
        self._model: Optional[SemanticModel] = None
    
    def add_schema_file(self, path: str) -> "SemanticModelBuilder":
        """Add schema file path"""
        self.schema_file = path
        return self
    
    def add_description_file(self, path: str) -> "SemanticModelBuilder":
        """Add description file path"""
        self.description_file = path
        return self
    
    def build(self) -> SemanticModel:
        """Build the complete semantic model"""
        logger.info("Starting semantic model build...")
        
        # Step 1: Load schema from sources
        self._model = self._load_schema()
        
        if not self._model:
            raise ValueError("No schema sources available. Provide schema files or database adapter.")
        
        # Step 2: Discover relationships
        if self.config.auto_discover_relationships:
            self._discover_relationships()
        
        # Step 3: Identify junction tables
        if self._model.relationships:
            self._identify_junction_tables()
        
        # Step 4: Enrich with sample data
        if self.config.include_sample_data and self.adapter:
            self._add_sample_data()
        
        # Step 5: Infer missing semantic types
        self._infer_semantic_types()
        
        # Step 6: Generate business context if missing
        if not self._model.business_context:
            self._generate_business_context()
        
        # Step 7: Export if configured
        if self.config.export_model and self.config.export_path:
            self._export_model()
        
        self._model.last_updated = datetime.utcnow()
        logger.info(
            f"Model build complete: {len(self._model.tables)} tables, "
            f"{len(self._model.relationships)} relationships"
        )
        
        return self._model
    
    def _load_schema(self) -> Optional[SemanticModel]:
        """Load schema from available sources"""
        # Use hybrid provider to combine all sources
        provider = HybridSchemaProvider(
            adapter=self.adapter,
            schema_file=self.schema_file,
            description_file=self.description_file,
        )
        
        if not provider.is_available():
            logger.error("No schema sources available")
            return None
        
        return provider.load()
    
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
    
    def _identify_junction_tables(self) -> None:
        """Identify many-to-many junction tables"""
        if not self._model:
            return
        
        engine = RelationshipDiscoveryEngine(self._model)
        junction_tables = engine.discover_junction_tables()
        
        for table_name in junction_tables:
            logger.info(f"Marked {table_name} as junction table")
    
    def _add_sample_data(self) -> None:
        """Add sample data to columns"""
        if not self._model or not self.adapter:
            return
        
        for table_name, table in self._model.tables.items():
            try:
                result = self.adapter.get_sample_data(table_name, self.config.sample_size)
                
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
    
    def _infer_semantic_types(self) -> None:
        """Infer semantic types for columns without them"""
        if not self._model:
            return
        
        for table in self._model.tables.values():
            for col in table.columns.values():
                if col.semantic_type == SemanticType.UNKNOWN:
                    # Try to infer from sample values
                    inferred = self._infer_from_samples(col)
                    if inferred != SemanticType.UNKNOWN:
                        col.semantic_type = inferred
    
    def _infer_from_samples(self, column: ColumnSemantics) -> SemanticType:
        """Infer semantic type from sample values"""
        if not column.sample_values:
            return SemanticType.UNKNOWN
        
        samples = column.sample_values
        
        # Check for email pattern
        import re
        email_pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
        if all(isinstance(s, str) and email_pattern.match(s) for s in samples):
            return SemanticType.EMAIL
        
        # Check for URL pattern
        url_pattern = re.compile(r'^https?://|^www\.')
        if all(isinstance(s, str) and url_pattern.match(s) for s in samples):
            return SemanticType.URL
        
        # Check for phone pattern
        phone_pattern = re.compile(r'^[\d\s\-\+\(\)]{10,}$')
        if all(isinstance(s, str) and phone_pattern.match(s) for s in samples):
            return SemanticType.PHONE
        
        # Check for status-like values (few distinct strings)
        if all(isinstance(s, str) for s in samples):
            unique = set(samples)
            if len(unique) <= 5 and all(len(s) < 20 for s in unique):
                return SemanticType.STATUS
        
        return SemanticType.UNKNOWN
    
    def _generate_business_context(self) -> None:
        """Generate basic business context from schema"""
        if not self._model:
            return
        
        # Extract key entities (tables with most relationships)
        relationship_counts: Dict[str, int] = {}
        for rel in self._model.relationships:
            relationship_counts[rel.source_table] = relationship_counts.get(rel.source_table, 0) + 1
            relationship_counts[rel.target_table] = relationship_counts.get(rel.target_table, 0) + 1
        
        key_entities = sorted(
            relationship_counts.keys(),
            key=lambda t: relationship_counts[t],
            reverse=True
        )[:5]
        
        self._model.business_context = BusinessContext(
            key_entities=key_entities,
        )
    
    def _export_model(self) -> None:
        """Export model to file"""
        if not self._model or not self.config.export_path:
            return
        
        path = self.config.export_path
        if self.config.export_format == "yaml":
            if not path.endswith(('.yaml', '.yml')):
                path += '.yaml'
        else:
            if not path.endswith('.json'):
                path += '.json'
        
        self._model.save(path)
        logger.info(f"Exported model to {path}")


class QuickModelBuilder:
    """
    Convenience class for quick model building
    
    Usage:
        # From database only
        model = QuickModelBuilder.from_database(adapter)
        
        # From files only
        model = QuickModelBuilder.from_files("schema.yaml", "docs.txt")
        
        # From database with file overlays
        model = QuickModelBuilder.from_hybrid(
            adapter=adapter,
            schema_file="schema.yaml"
        )
    """
    
    @staticmethod
    def from_database(
        adapter: BaseDatabaseAdapter,
        discover_relationships: bool = True,
        include_samples: bool = True,
    ) -> SemanticModel:
        """Build model from database only"""
        builder = SemanticModelBuilder(
            adapter=adapter,
            config=ModelBuilderConfig(
                auto_discover_relationships=discover_relationships,
                include_sample_data=include_samples,
            ),
        )
        return builder.build()
    
    @staticmethod
    def from_files(
        schema_file: Optional[str] = None,
        description_file: Optional[str] = None,
    ) -> SemanticModel:
        """Build model from files only"""
        builder = SemanticModelBuilder(
            config=ModelBuilderConfig(
                schema_file=schema_file,
                description_file=description_file,
                auto_discover_relationships=False,  # No DB to verify
                include_sample_data=False,
            ),
        )
        return builder.build()
    
    @staticmethod
    def from_hybrid(
        adapter: BaseDatabaseAdapter,
        schema_file: Optional[str] = None,
        description_file: Optional[str] = None,
        llm_client: Optional[BaseLLMClient] = None,
        use_llm_inference: bool = False,
    ) -> SemanticModel:
        """Build model from database with file overlays"""
        builder = SemanticModelBuilder(
            adapter=adapter,
            llm_client=llm_client,
            config=ModelBuilderConfig(
                schema_file=schema_file,
                description_file=description_file,
                use_llm_inference=use_llm_inference,
            ),
        )
        return builder.build()
    
    @staticmethod
    def from_directory(
        directory: str,
        adapter: Optional[BaseDatabaseAdapter] = None,
    ) -> SemanticModel:
        """
        Build model by scanning a directory for schema files
        
        Looks for:
        - schema.yaml / schema.json
        - descriptions.txt / descriptions.md / README.md
        """
        dir_path = Path(directory)
        
        schema_file = None
        description_file = None
        
        # Look for schema files
        for name in ['schema.yaml', 'schema.yml', 'schema.json', 'database.yaml', 'database.json']:
            path = dir_path / name
            if path.exists():
                schema_file = str(path)
                break
        
        # Look for description files
        for name in ['descriptions.txt', 'descriptions.md', 'README.md', 'docs.txt', 'data_dictionary.txt']:
            path = dir_path / name
            if path.exists():
                description_file = str(path)
                break
        
        builder = SemanticModelBuilder(
            adapter=adapter,
            config=ModelBuilderConfig(
                schema_file=schema_file,
                description_file=description_file,
            ),
        )
        return builder.build()


# Context generator for SQL agent
class SchemaContextGenerator:
    """
    Generates optimized schema context for LLM prompts
    
    Creates context strings that help the LLM understand:
    - Table structures and purposes
    - Relationships between tables
    - How to join tables correctly
    - Business terminology
    """
    
    def __init__(self, model: SemanticModel):
        self.model = model
    
    def generate_full_context(self) -> str:
        """Generate complete schema context for complex queries"""
        return self.model.to_schema_prompt(include_business_context=True)
    
    def generate_minimal_context(self, tables: List[str]) -> str:
        """Generate context for specific tables only"""
        lines = [f"DATABASE: {self.model.database_name} ({self.model.database_type})", ""]
        
        # Add specified tables
        for table_name in tables:
            table = self.model.get_table(table_name)
            if table:
                lines.append(table.to_ddl_string())
                lines.append("")
        
        # Add relationships between these tables
        relevant_rels = [
            r for r in self.model.relationships
            if r.source_table in tables and r.target_table in tables
        ]
        
        if relevant_rels:
            lines.append("RELATIONSHIPS:")
            for rel in relevant_rels:
                lines.append(f"  {rel.source_table} -> {rel.target_table}: {rel.get_join_sql()}")
        
        return "\n".join(lines)
    
    def generate_query_focused_context(self, user_query: str) -> str:
        """
        Generate context optimized for a specific query
        
        Includes:
        - Tables likely relevant to the query
        - Relationships between those tables
        - Business terminology mappings
        """
        # Find relevant tables by keyword matching
        relevant_tables = self._find_relevant_tables(user_query)
        
        # Add related tables (via relationships)
        expanded_tables = set(relevant_tables)
        for table_name in relevant_tables:
            related = self.model.get_related_tables(table_name)
            expanded_tables.update(related)
        
        lines = [f"DATABASE: {self.model.database_name}", ""]
        
        # Business context
        if self.model.business_context:
            lines.append("CONTEXT:")
            if self.model.business_context.terminology:
                for biz_term, tech_term in self.model.business_context.terminology.items():
                    if biz_term.lower() in user_query.lower():
                        lines.append(f"  Note: '{biz_term}' refers to '{tech_term}' table")
            lines.append("")
        
        # Tables
        lines.append("RELEVANT TABLES:")
        for table_name in sorted(expanded_tables):
            table = self.model.get_table(table_name)
            if table:
                lines.append(table.to_ddl_string(include_descriptions=True))
                lines.append("")
        
        # Relationships
        relevant_rels = [
            r for r in self.model.relationships
            if r.source_table in expanded_tables or r.target_table in expanded_tables
        ]
        
        if relevant_rels:
            lines.append("HOW TO JOIN TABLES:")
            for rel in relevant_rels:
                lines.append(f"  JOIN {rel.target_table} ON {rel.get_join_sql()}")
        
        return "\n".join(lines)
    
    def _find_relevant_tables(self, query: str) -> List[str]:
        """Find tables relevant to a query by keyword matching"""
        query_lower = query.lower()
        relevant = []
        
        # Match table names
        for table_name in self.model.tables:
            if table_name.lower() in query_lower:
                relevant.append(table_name)
        
        # Match business names
        for table_name, table in self.model.tables.items():
            if table.business_name and table.business_name.lower() in query_lower:
                if table_name not in relevant:
                    relevant.append(table_name)
        
        # Match business terminology
        if self.model.business_context:
            for biz_term, tech_term in self.model.business_context.terminology.items():
                if biz_term.lower() in query_lower:
                    if tech_term in self.model.tables and tech_term not in relevant:
                        relevant.append(tech_term)
        
        # If no matches, return most connected tables
        if not relevant:
            relationship_counts = {}
            for rel in self.model.relationships:
                relationship_counts[rel.source_table] = relationship_counts.get(rel.source_table, 0) + 1
                relationship_counts[rel.target_table] = relationship_counts.get(rel.target_table, 0) + 1
            
            relevant = sorted(
                relationship_counts.keys(),
                key=lambda t: relationship_counts[t],
                reverse=True
            )[:3]
        
        return relevant
    
    def get_join_path(self, from_table: str, to_table: str) -> Optional[str]:
        """Get SQL JOIN path between two tables"""
        path = self.model.find_join_path(from_table, to_table)
        
        if not path:
            return None
        
        joins = []
        current_table = from_table
        
        for rel in path:
            if rel.source_table == current_table:
                join_table = rel.target_table
            else:
                join_table = rel.source_table
            
            joins.append(f"JOIN {join_table} ON {rel.get_join_sql()}")
            current_table = join_table
        
        return " ".join(joins)
