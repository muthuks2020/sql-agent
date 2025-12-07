"""
Schema Providers for Schema Intelligence

Provides multiple ways to load schema information:
1. FileSchemaProvider - From YAML/JSON schema files
2. TextDescriptionProvider - From natural language text files
3. DatabaseSchemaProvider - Auto-discovery from database connection
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
from ..adapters.base import BaseDatabaseAdapter, DatabaseSchema
from ..utils import get_logger

logger = get_logger(__name__)


class BaseSchemaProvider(ABC):
    """Abstract base class for schema providers"""
    
    @abstractmethod
    def load(self) -> Optional[SemanticModel]:
        """Load schema information and return a SemanticModel"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider can load data"""
        pass


@dataclass
class SchemaFileConfig:
    """Configuration for schema file provider"""
    schema_file: Optional[str] = None  # Path to schema YAML/JSON
    descriptions_file: Optional[str] = None  # Path to text descriptions
    business_context_file: Optional[str] = None  # Path to business context


class FileSchemaProvider(BaseSchemaProvider):
    """
    Loads schema from structured YAML/JSON files
    
    Expected file format:
    ```yaml
    database:
      name: myapp
      type: postgresql
      domain: E-commerce
      description: Main application database
    
    tables:
      users:
        description: User accounts
        purpose: Stores user authentication and profile data
        columns:
          id:
            type: integer
            primary_key: true
            description: Unique user identifier
          email:
            type: varchar(255)
            unique: true
            semantic_type: email
            description: User's email address
          created_at:
            type: timestamp
            semantic_type: created_at
    
    relationships:
      - name: user_orders
        from: orders.user_id
        to: users.id
        type: many_to_one
        description: Each order belongs to one user
    
    business_context:
      terminology:
        customer: users
        purchase: orders
      key_entities:
        - users
        - orders
        - products
    ```
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def is_available(self) -> bool:
        return os.path.exists(self.file_path)
    
    def load(self) -> Optional[SemanticModel]:
        if not self.is_available():
            logger.warning(f"Schema file not found: {self.file_path}")
            return None
        
        try:
            with open(self.file_path, 'r') as f:
                if self.file_path.endswith('.yaml') or self.file_path.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            return self._parse_schema_file(data)
        
        except Exception as e:
            logger.error(f"Error loading schema file: {e}")
            return None
    
    def _parse_schema_file(self, data: Dict[str, Any]) -> SemanticModel:
        """Parse schema file into SemanticModel"""
        db_info = data.get("database", {})
        
        model = SemanticModel(
            database_name=db_info.get("name", "unknown"),
            database_type=db_info.get("type", "unknown"),
            sources=["schema_file"],
        )
        
        # Parse tables
        for table_name, table_data in data.get("tables", {}).items():
            table = self._parse_table(table_name, table_data)
            model.tables[table_name] = table
        
        # Parse relationships
        for rel_data in data.get("relationships", []):
            rel = self._parse_relationship(rel_data)
            if rel:
                model.relationships.append(rel)
        
        # Parse business context
        if "business_context" in data:
            model.business_context = self._parse_business_context(data["business_context"])
        elif db_info.get("domain") or db_info.get("description"):
            model.business_context = BusinessContext(
                domain=db_info.get("domain"),
                description=db_info.get("description"),
            )
        
        return model
    
    def _parse_table(self, name: str, data: Dict[str, Any]) -> TableSemantics:
        """Parse table definition"""
        table = TableSemantics(
            name=name,
            description=data.get("description"),
            purpose=data.get("purpose"),
            business_name=data.get("business_name"),
            domain=data.get("domain"),
            is_lookup_table=data.get("is_lookup_table", False),
            is_junction_table=data.get("is_junction_table", False),
            row_count=data.get("row_count"),
            primary_key=data.get("primary_key", []),
        )
        
        for col_name, col_data in data.get("columns", {}).items():
            col = self._parse_column(col_name, col_data)
            table.columns[col_name] = col
            
            # Track primary key
            if col.is_primary_key and col_name not in table.primary_key:
                table.primary_key.append(col_name)
        
        return table
    
    def _parse_column(self, name: str, data: Dict[str, Any]) -> ColumnSemantics:
        """Parse column definition"""
        semantic_type = SemanticType.UNKNOWN
        if "semantic_type" in data:
            try:
                semantic_type = SemanticType(data["semantic_type"])
            except ValueError:
                pass
        
        return ColumnSemantics(
            name=name,
            data_type=data.get("type", data.get("data_type", "unknown")),
            semantic_type=semantic_type,
            description=data.get("description"),
            nullable=data.get("nullable", True),
            is_primary_key=data.get("primary_key", False),
            is_foreign_key=data.get("foreign_key", False),
            is_unique=data.get("unique", False),
            is_indexed=data.get("indexed", False),
            references_table=data.get("references_table"),
            references_column=data.get("references_column"),
            default_value=data.get("default"),
            business_name=data.get("business_name"),
            allowed_values=data.get("allowed_values"),
        )
    
    def _parse_relationship(self, data: Dict[str, Any]) -> Optional[Relationship]:
        """Parse relationship definition"""
        try:
            # Parse "from: table.column" format
            from_parts = data.get("from", "").split(".")
            to_parts = data.get("to", "").split(".")
            
            if len(from_parts) != 2 or len(to_parts) != 2:
                # Try alternative format
                source_table = data.get("source_table")
                source_cols = data.get("source_columns", [])
                target_table = data.get("target_table")
                target_cols = data.get("target_columns", [])
            else:
                source_table = from_parts[0]
                source_cols = [from_parts[1]]
                target_table = to_parts[0]
                target_cols = [to_parts[1]]
            
            if not source_table or not target_table:
                return None
            
            rel_type = RelationshipType.ONE_TO_MANY
            if "type" in data:
                try:
                    rel_type = RelationshipType(data["type"])
                except ValueError:
                    pass
            
            return Relationship(
                name=data.get("name", f"{source_table}_{target_table}"),
                source_table=source_table,
                source_columns=source_cols,
                target_table=target_table,
                target_columns=target_cols,
                relationship_type=rel_type,
                source=RelationshipSource.MANUAL,
                description=data.get("description"),
            )
        except Exception as e:
            logger.warning(f"Error parsing relationship: {e}")
            return None
    
    def _parse_business_context(self, data: Dict[str, Any]) -> BusinessContext:
        """Parse business context"""
        return BusinessContext(
            domain=data.get("domain"),
            description=data.get("description"),
            terminology=data.get("terminology", {}),
            key_entities=data.get("key_entities", []),
            common_queries=data.get("common_queries", {}),
            business_rules=data.get("business_rules", []),
        )


class TextDescriptionProvider(BaseSchemaProvider):
    """
    Loads schema descriptions from natural language text files
    
    Supports formats like:
    
    ```
    # Database Overview
    This is an e-commerce database for an online retail platform.
    
    ## Tables
    
    ### users
    The users table stores customer account information.
    - id: Unique identifier for each user
    - email: Customer's email address (must be unique)
    - name: Full name of the customer
    - created_at: When the account was created
    
    ### orders
    Orders placed by customers.
    - id: Order identifier
    - user_id: Links to the users table (the customer who placed the order)
    - total: Total order amount in USD
    - status: Order status (pending, paid, shipped, delivered, cancelled)
    
    ## Relationships
    - Each user can have multiple orders (users -> orders via user_id)
    - Each order belongs to exactly one user
    
    ## Business Terms
    - "customer" refers to a user
    - "purchase" refers to an order
    ```
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def is_available(self) -> bool:
        return os.path.exists(self.file_path)
    
    def load(self) -> Optional[SemanticModel]:
        if not self.is_available():
            logger.warning(f"Description file not found: {self.file_path}")
            return None
        
        try:
            with open(self.file_path, 'r') as f:
                content = f.read()
            
            return self._parse_text_description(content)
        
        except Exception as e:
            logger.error(f"Error loading description file: {e}")
            return None
    
    def _parse_text_description(self, content: str) -> SemanticModel:
        """Parse natural language description into SemanticModel"""
        model = SemanticModel(
            database_name="unknown",
            database_type="unknown",
            sources=["text_description"],
        )
        
        # Extract database overview
        overview_match = re.search(
            r'#\s*Database\s+Overview\s*\n(.*?)(?=\n#|\Z)',
            content, re.IGNORECASE | re.DOTALL
        )
        if overview_match:
            overview = overview_match.group(1).strip()
            model.business_context = BusinessContext(description=overview)
        
        # Extract table descriptions
        table_sections = re.findall(
            r'###\s+(\w+)\s*\n(.*?)(?=\n###|\n##|\Z)',
            content, re.DOTALL
        )
        
        for table_name, table_content in table_sections:
            table = self._parse_table_description(table_name, table_content)
            model.tables[table_name] = table
        
        # Extract relationships section
        rel_match = re.search(
            r'##\s*Relationships?\s*\n(.*?)(?=\n##|\Z)',
            content, re.IGNORECASE | re.DOTALL
        )
        if rel_match:
            relationships = self._parse_relationships_text(rel_match.group(1))
            model.relationships.extend(relationships)
        
        # Extract business terms
        terms_match = re.search(
            r'##\s*Business\s+Terms?\s*\n(.*?)(?=\n##|\Z)',
            content, re.IGNORECASE | re.DOTALL
        )
        if terms_match and model.business_context:
            model.business_context.terminology = self._parse_terminology(terms_match.group(1))
        
        return model
    
    def _parse_table_description(self, name: str, content: str) -> TableSemantics:
        """Parse a table description section"""
        lines = content.strip().split('\n')
        
        # First non-empty line is usually the description
        description = None
        for line in lines:
            line = line.strip()
            if line and not line.startswith('-'):
                description = line
                break
        
        table = TableSemantics(
            name=name,
            description=description,
        )
        
        # Parse column descriptions (lines starting with -)
        for line in lines:
            line = line.strip()
            if line.startswith('-'):
                col_match = re.match(r'-\s*(\w+):\s*(.+)', line)
                if col_match:
                    col_name = col_match.group(1)
                    col_desc = col_match.group(2)
                    
                    # Infer semantic type from description
                    semantic_type = self._infer_semantic_type(col_name, col_desc)
                    
                    # Check for FK hints
                    references_table = None
                    references_column = None
                    fk_match = re.search(r'links?\s+to\s+(?:the\s+)?(\w+)', col_desc, re.IGNORECASE)
                    if fk_match:
                        references_table = fk_match.group(1)
                        references_column = "id"  # Default assumption
                    
                    table.columns[col_name] = ColumnSemantics(
                        name=col_name,
                        data_type="unknown",  # Will be filled by DB adapter
                        semantic_type=semantic_type,
                        description=col_desc,
                        is_foreign_key=references_table is not None,
                        references_table=references_table,
                        references_column=references_column,
                    )
        
        return table
    
    def _infer_semantic_type(self, name: str, description: str) -> SemanticType:
        """Infer semantic type from column name and description"""
        name_lower = name.lower()
        desc_lower = description.lower()
        
        # Check for common patterns
        if name_lower in ('id', 'pk') or 'identifier' in desc_lower or 'primary' in desc_lower:
            return SemanticType.PRIMARY_KEY
        if name_lower.endswith('_id') or 'foreign' in desc_lower or 'links to' in desc_lower:
            return SemanticType.FOREIGN_KEY
        if 'email' in name_lower or 'email' in desc_lower:
            return SemanticType.EMAIL
        if 'phone' in name_lower or 'phone' in desc_lower:
            return SemanticType.PHONE
        if name_lower in ('name', 'full_name') or 'full name' in desc_lower:
            return SemanticType.NAME
        if 'created' in name_lower or 'created' in desc_lower:
            return SemanticType.CREATED_AT
        if 'updated' in name_lower or 'updated' in desc_lower:
            return SemanticType.UPDATED_AT
        if 'status' in name_lower or 'status' in desc_lower:
            return SemanticType.STATUS
        if any(term in name_lower for term in ['price', 'cost', 'amount', 'total']):
            return SemanticType.PRICE
        if 'description' in name_lower:
            return SemanticType.DESCRIPTION
        if 'url' in name_lower or 'link' in name_lower:
            return SemanticType.URL
        
        return SemanticType.UNKNOWN
    
    def _parse_relationships_text(self, content: str) -> List[Relationship]:
        """Parse relationship descriptions from text"""
        relationships = []
        
        # Pattern: "table1 -> table2 via column"
        pattern = r'(\w+)\s*->\s*(\w+)\s+via\s+(\w+)'
        for match in re.finditer(pattern, content, re.IGNORECASE):
            source = match.group(1)
            target = match.group(2)
            column = match.group(3)
            
            relationships.append(Relationship(
                name=f"{source}_{target}",
                source_table=target,  # The table with the FK
                source_columns=[column],
                target_table=source,  # The table being referenced
                target_columns=["id"],
                relationship_type=RelationshipType.MANY_TO_ONE,
                source=RelationshipSource.MANUAL,
            ))
        
        return relationships
    
    def _parse_terminology(self, content: str) -> Dict[str, str]:
        """Parse business terminology mappings"""
        terminology = {}
        
        # Pattern: "term" refers to "table"
        pattern = r'"(\w+)"\s+refers?\s+to\s+(?:a\s+|an\s+)?(\w+)'
        for match in re.finditer(pattern, content, re.IGNORECASE):
            business_term = match.group(1)
            technical_term = match.group(2)
            terminology[business_term] = technical_term
        
        return terminology


class DatabaseSchemaProvider(BaseSchemaProvider):
    """
    Auto-discovers schema from database connection
    
    This is the fallback when no schema files are available.
    It connects to the database and extracts:
    - Table structures
    - Column types and constraints
    - Foreign key relationships
    - Sample data for semantic inference
    """
    
    def __init__(self, adapter: BaseDatabaseAdapter):
        self.adapter = adapter
    
    def is_available(self) -> bool:
        try:
            return self.adapter.is_connected() or self.adapter.test_connection()[0]
        except:
            return False
    
    def load(self) -> Optional[SemanticModel]:
        try:
            if not self.adapter.is_connected():
                self.adapter.connect()
            
            # Get raw schema from adapter
            raw_schema = self.adapter.get_schema(force_refresh=True)
            
            # Convert to semantic model
            return self._convert_to_semantic_model(raw_schema)
        
        except Exception as e:
            logger.error(f"Error loading schema from database: {e}")
            return None
    
    def _convert_to_semantic_model(self, schema: DatabaseSchema) -> SemanticModel:
        """Convert DatabaseSchema to SemanticModel with inferred semantics"""
        model = SemanticModel(
            database_name=schema.database_name,
            database_type=schema.database_type.value,
            sources=["database_auto_discovery"],
        )
        
        # Convert tables
        for table_name, raw_table in schema.tables.items():
            table = TableSemantics(
                name=table_name,
                row_count=raw_table.row_count,
                primary_key=raw_table.primary_key,
            )
            
            # Convert columns with semantic inference
            for raw_col in raw_table.columns:
                semantic_type = self._infer_column_semantic_type(raw_col.name, raw_col.data_type)
                
                col = ColumnSemantics(
                    name=raw_col.name,
                    data_type=raw_col.data_type,
                    semantic_type=semantic_type,
                    nullable=raw_col.nullable,
                    is_primary_key=raw_col.is_primary_key,
                    is_foreign_key=raw_col.is_foreign_key,
                    is_unique=raw_col.is_unique,
                    is_indexed=raw_col.is_indexed,
                    default_value=raw_col.default_value,
                )
                table.columns[raw_col.name] = col
            
            # Process foreign keys
            for fk in raw_table.foreign_keys:
                for i, col_name in enumerate(fk.columns):
                    if col_name in table.columns:
                        table.columns[col_name].is_foreign_key = True
                        table.columns[col_name].references_table = fk.referenced_table
                        if i < len(fk.referenced_columns):
                            table.columns[col_name].references_column = fk.referenced_columns[i]
            
            model.tables[table_name] = table
        
        # Convert foreign key relationships
        for table_name, raw_table in schema.tables.items():
            for fk in raw_table.foreign_keys:
                rel = Relationship(
                    name=fk.name,
                    source_table=table_name,
                    source_columns=fk.columns,
                    target_table=fk.referenced_table,
                    target_columns=fk.referenced_columns,
                    relationship_type=RelationshipType.MANY_TO_ONE,
                    source=RelationshipSource.EXPLICIT_FK,
                )
                model.relationships.append(rel)
        
        return model
    
    def _infer_column_semantic_type(self, name: str, data_type: str) -> SemanticType:
        """Infer semantic type from column name and data type"""
        name_lower = name.lower()
        type_lower = data_type.lower()
        
        # ID columns
        if name_lower == 'id' or name_lower == 'pk':
            return SemanticType.PRIMARY_KEY
        if name_lower.endswith('_id') or name_lower.endswith('id'):
            return SemanticType.FOREIGN_KEY
        if 'uuid' in type_lower or 'guid' in name_lower:
            return SemanticType.UUID
        
        # Personal info
        if 'email' in name_lower:
            return SemanticType.EMAIL
        if 'phone' in name_lower or 'tel' in name_lower:
            return SemanticType.PHONE
        if name_lower in ('name', 'full_name', 'fullname'):
            return SemanticType.NAME
        if name_lower in ('first_name', 'firstname', 'fname'):
            return SemanticType.FIRST_NAME
        if name_lower in ('last_name', 'lastname', 'lname', 'surname'):
            return SemanticType.LAST_NAME
        
        # Address
        if 'address' in name_lower:
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
        if 'created' in name_lower:
            return SemanticType.CREATED_AT
        if 'updated' in name_lower or 'modified' in name_lower:
            return SemanticType.UPDATED_AT
        if 'deleted' in name_lower:
            return SemanticType.DELETED_AT
        if 'date' in type_lower and 'date' in name_lower:
            return SemanticType.DATE
        if 'timestamp' in type_lower or 'datetime' in type_lower:
            return SemanticType.DATETIME
        
        # Financial
        if any(term in name_lower for term in ['price', 'cost', 'fee', 'amount', 'total', 'subtotal']):
            return SemanticType.PRICE
        if 'currency' in name_lower:
            return SemanticType.CURRENCY
        if 'percent' in name_lower or 'rate' in name_lower:
            return SemanticType.PERCENTAGE
        
        # Status
        if 'status' in name_lower or 'state' in name_lower:
            return SemanticType.STATUS
        if 'bool' in type_lower or name_lower.startswith('is_') or name_lower.startswith('has_'):
            return SemanticType.BOOLEAN_FLAG
        if 'enum' in type_lower:
            return SemanticType.ENUM
        
        # Content
        if name_lower in ('title', 'subject', 'headline'):
            return SemanticType.TITLE
        if name_lower in ('description', 'desc', 'summary', 'bio', 'about'):
            return SemanticType.DESCRIPTION
        if name_lower in ('content', 'body', 'text', 'message'):
            return SemanticType.CONTENT
        if 'url' in name_lower or 'link' in name_lower:
            return SemanticType.URL
        if 'image' in name_lower or 'photo' in name_lower or 'avatar' in name_lower:
            return SemanticType.IMAGE_URL
        if 'path' in name_lower or 'file' in name_lower:
            return SemanticType.FILE_PATH
        if 'json' in type_lower or 'jsonb' in type_lower:
            return SemanticType.JSON_DATA
        
        # Metrics
        if 'count' in name_lower or 'num' in name_lower:
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
        if 'token' in name_lower or 'secret' in name_lower or 'key' in name_lower:
            return SemanticType.TOKEN
        if 'ip' in name_lower:
            return SemanticType.IP_ADDRESS
        if 'user_agent' in name_lower or 'useragent' in name_lower:
            return SemanticType.USER_AGENT
        
        return SemanticType.UNKNOWN
    
    def load_with_samples(self, sample_limit: int = 5) -> Optional[SemanticModel]:
        """Load schema with sample data for better inference"""
        model = self.load()
        if not model:
            return None
        
        # Fetch sample data for each table
        for table_name, table in model.tables.items():
            try:
                result = self.adapter.get_sample_data(table_name, sample_limit)
                if result.success and result.rows:
                    for i, col_name in enumerate(result.columns):
                        if col_name in table.columns:
                            samples = [row[i] for row in result.rows if row[i] is not None]
                            table.columns[col_name].sample_values = samples[:5]
            except Exception as e:
                logger.debug(f"Could not get samples for {table_name}: {e}")
        
        return model


class HybridSchemaProvider(BaseSchemaProvider):
    """
    Combines multiple schema sources with priority
    
    Priority order:
    1. Schema files (YAML/JSON) - most authoritative
    2. Text descriptions - adds business context
    3. Database auto-discovery - fills in gaps
    
    Merges information from all available sources.
    """
    
    def __init__(
        self,
        adapter: Optional[BaseDatabaseAdapter] = None,
        schema_file: Optional[str] = None,
        description_file: Optional[str] = None,
    ):
        self.providers: List[BaseSchemaProvider] = []
        
        # Add providers in priority order
        if schema_file:
            self.providers.append(FileSchemaProvider(schema_file))
        if description_file:
            self.providers.append(TextDescriptionProvider(description_file))
        if adapter:
            self.providers.append(DatabaseSchemaProvider(adapter))
    
    def is_available(self) -> bool:
        return any(p.is_available() for p in self.providers)
    
    def load(self) -> Optional[SemanticModel]:
        models = []
        
        # Load from all available providers
        for provider in self.providers:
            if provider.is_available():
                model = provider.load()
                if model:
                    models.append(model)
                    logger.info(f"Loaded schema from {provider.__class__.__name__}")
        
        if not models:
            logger.warning("No schema sources available")
            return None
        
        # Merge models (first model is base, others add/override)
        merged = models[0]
        for model in models[1:]:
            merged = self._merge_models(merged, model)
        
        merged.sources = list(set(
            source for model in models for source in model.sources
        ))
        
        return merged
    
    def _merge_models(self, base: SemanticModel, overlay: SemanticModel) -> SemanticModel:
        """Merge two models, with overlay taking precedence for descriptions"""
        # Update database info if missing
        if base.database_name == "unknown":
            base.database_name = overlay.database_name
        if base.database_type == "unknown":
            base.database_type = overlay.database_type
        
        # Merge tables
        for table_name, overlay_table in overlay.tables.items():
            if table_name in base.tables:
                base_table = base.tables[table_name]
                
                # Overlay descriptions take precedence
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
                        
                        # Overlay descriptions
                        if overlay_col.description:
                            base_col.description = overlay_col.description
                        if overlay_col.business_name:
                            base_col.business_name = overlay_col.business_name
                        if overlay_col.semantic_type != SemanticType.UNKNOWN:
                            base_col.semantic_type = overlay_col.semantic_type
                        if overlay_col.references_table:
                            base_col.references_table = overlay_col.references_table
                            base_col.references_column = overlay_col.references_column
                    else:
                        # Add new column from overlay
                        base_table.columns[col_name] = overlay_col
            else:
                # Add new table from overlay
                base.tables[table_name] = overlay_table
        
        # Merge relationships (avoid duplicates)
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
                base.business_context.terminology.update(overlay.business_context.terminology)
                base.business_context.key_entities = list(set(
                    base.business_context.key_entities + overlay.business_context.key_entities
                ))
                base.business_context.common_queries.update(overlay.business_context.common_queries)
                base.business_context.business_rules.extend(overlay.business_context.business_rules)
        
        return base
