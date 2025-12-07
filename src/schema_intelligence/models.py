"""
Semantic Model Definitions for Schema Intelligence

These models represent the enriched understanding of database structure,
going beyond raw schema to include business context and relationships.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import json
import yaml


class SemanticType(str, Enum):
    """Semantic types for columns - what the data represents"""
    # Identifiers
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    UNIQUE_ID = "unique_id"
    UUID = "uuid"
    
    # Personal Information
    EMAIL = "email"
    PHONE = "phone"
    NAME = "name"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    ADDRESS = "address"
    CITY = "city"
    STATE = "state"
    COUNTRY = "country"
    ZIP_CODE = "zip_code"
    
    # Financial
    CURRENCY = "currency"
    PRICE = "price"
    AMOUNT = "amount"
    PERCENTAGE = "percentage"
    
    # Temporal
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    DELETED_AT = "deleted_at"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    YEAR = "year"
    MONTH = "month"
    
    # Status/State
    STATUS = "status"
    BOOLEAN_FLAG = "boolean_flag"
    ENUM = "enum"
    
    # Content
    TITLE = "title"
    DESCRIPTION = "description"
    CONTENT = "content"
    URL = "url"
    IMAGE_URL = "image_url"
    FILE_PATH = "file_path"
    JSON_DATA = "json_data"
    
    # Metrics
    COUNT = "count"
    QUANTITY = "quantity"
    SCORE = "score"
    RATING = "rating"
    
    # Other
    PASSWORD_HASH = "password_hash"
    TOKEN = "token"
    IP_ADDRESS = "ip_address"
    USER_AGENT = "user_agent"
    
    UNKNOWN = "unknown"


class RelationshipType(str, Enum):
    """Types of relationships between tables"""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class RelationshipSource(str, Enum):
    """How the relationship was discovered"""
    EXPLICIT_FK = "explicit_fk"           # From foreign key constraint
    NAMING_CONVENTION = "naming_convention"  # From column naming (user_id -> users)
    DATA_PATTERN = "data_pattern"          # From analyzing actual data
    LLM_INFERRED = "llm_inferred"          # LLM analyzed and suggested
    MANUAL = "manual"                       # From schema file


@dataclass
class ColumnSemantics:
    """Semantic information about a database column"""
    name: str
    data_type: str
    semantic_type: SemanticType = SemanticType.UNKNOWN
    description: Optional[str] = None
    
    # Constraints
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_unique: bool = False
    is_indexed: bool = False
    
    # Foreign key details
    references_table: Optional[str] = None
    references_column: Optional[str] = None
    
    # Value information
    default_value: Optional[str] = None
    sample_values: List[Any] = field(default_factory=list)
    distinct_count: Optional[int] = None
    null_percentage: Optional[float] = None
    
    # Business context
    business_name: Optional[str] = None  # Human-readable name
    business_rules: List[str] = field(default_factory=list)
    allowed_values: Optional[List[str]] = None  # For enums
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "semantic_type": self.semantic_type.value,
            "description": self.description,
            "nullable": self.nullable,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "is_unique": self.is_unique,
            "references_table": self.references_table,
            "references_column": self.references_column,
            "business_name": self.business_name,
            "sample_values": self.sample_values[:5],  # Limit samples
        }
    
    def to_prompt_string(self) -> str:
        """Generate string for LLM prompt"""
        parts = [f"{self.name} ({self.data_type})"]
        
        if self.is_primary_key:
            parts.append("PRIMARY KEY")
        if self.is_foreign_key and self.references_table:
            parts.append(f"FK -> {self.references_table}.{self.references_column}")
        if not self.nullable:
            parts.append("NOT NULL")
        if self.description:
            parts.append(f"-- {self.description}")
        elif self.business_name:
            parts.append(f"-- {self.business_name}")
            
        return " ".join(parts)


@dataclass
class TableSemantics:
    """Semantic information about a database table"""
    name: str
    columns: Dict[str, ColumnSemantics] = field(default_factory=dict)
    
    # Descriptions
    description: Optional[str] = None
    purpose: Optional[str] = None
    
    # Business context
    business_name: Optional[str] = None
    domain: Optional[str] = None  # e.g., "Sales", "Users", "Inventory"
    is_lookup_table: bool = False
    is_junction_table: bool = False  # Many-to-many bridge
    
    # Statistics
    row_count: Optional[int] = None
    
    # Keys
    primary_key: List[str] = field(default_factory=list)
    
    # Sample queries
    common_queries: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "purpose": self.purpose,
            "business_name": self.business_name,
            "domain": self.domain,
            "is_lookup_table": self.is_lookup_table,
            "is_junction_table": self.is_junction_table,
            "row_count": self.row_count,
            "primary_key": self.primary_key,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }
    
    def to_ddl_string(self, include_descriptions: bool = True) -> str:
        """Generate DDL-like string for LLM context"""
        lines = []
        
        # Table header with description
        if self.description and include_descriptions:
            lines.append(f"-- {self.description}")
        if self.purpose and include_descriptions:
            lines.append(f"-- Purpose: {self.purpose}")
        
        lines.append(f"TABLE {self.name} (")
        
        for col in self.columns.values():
            lines.append(f"    {col.to_prompt_string()},")
        
        if self.primary_key:
            lines.append(f"    PRIMARY KEY ({', '.join(self.primary_key)})")
        
        # Remove trailing comma if needed
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]
        
        lines.append(");")
        
        if self.row_count is not None and include_descriptions:
            lines.append(f"-- Approximate rows: {self.row_count:,}")
        
        return "\n".join(lines)
    
    def get_column(self, name: str) -> Optional[ColumnSemantics]:
        """Get column by name (case-insensitive)"""
        if name in self.columns:
            return self.columns[name]
        name_lower = name.lower()
        for col_name, col in self.columns.items():
            if col_name.lower() == name_lower:
                return col
        return None


@dataclass
class Relationship:
    """Represents a relationship between two tables"""
    name: str
    source_table: str
    source_columns: List[str]
    target_table: str
    target_columns: List[str]
    relationship_type: RelationshipType
    source: RelationshipSource
    
    # Metadata
    description: Optional[str] = None
    confidence: float = 1.0  # 0-1, how confident we are in this relationship
    
    # Join hints
    join_condition: Optional[str] = None  # Pre-built JOIN condition
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_table": self.source_table,
            "source_columns": self.source_columns,
            "target_table": self.target_table,
            "target_columns": self.target_columns,
            "relationship_type": self.relationship_type.value,
            "source": self.source.value,
            "description": self.description,
            "confidence": self.confidence,
            "join_condition": self.join_condition,
        }
    
    def get_join_sql(self) -> str:
        """Generate SQL JOIN condition"""
        if self.join_condition:
            return self.join_condition
        
        conditions = []
        for src_col, tgt_col in zip(self.source_columns, self.target_columns):
            conditions.append(f"{self.source_table}.{src_col} = {self.target_table}.{tgt_col}")
        
        return " AND ".join(conditions)


@dataclass
class BusinessContext:
    """Business/domain context for the database"""
    domain: Optional[str] = None  # e.g., "E-commerce", "Healthcare", "Finance"
    description: Optional[str] = None
    
    # Terminology mappings (business term -> technical term)
    terminology: Dict[str, str] = field(default_factory=dict)
    # e.g., {"customer": "users", "purchase": "orders", "item": "products"}
    
    # Key entities
    key_entities: List[str] = field(default_factory=list)
    
    # Common query patterns
    common_queries: Dict[str, str] = field(default_factory=dict)
    # e.g., {"active users": "SELECT * FROM users WHERE status = 'active'"}
    
    # Business rules
    business_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "description": self.description,
            "terminology": self.terminology,
            "key_entities": self.key_entities,
            "common_queries": self.common_queries,
            "business_rules": self.business_rules,
        }
    
    def to_prompt_string(self) -> str:
        """Generate context string for LLM"""
        lines = []
        
        if self.domain:
            lines.append(f"Domain: {self.domain}")
        if self.description:
            lines.append(f"Description: {self.description}")
        
        if self.terminology:
            lines.append("\nTerminology Mappings:")
            for business_term, technical_term in self.terminology.items():
                lines.append(f"  - '{business_term}' refers to '{technical_term}'")
        
        if self.key_entities:
            lines.append(f"\nKey Entities: {', '.join(self.key_entities)}")
        
        if self.business_rules:
            lines.append("\nBusiness Rules:")
            for rule in self.business_rules:
                lines.append(f"  - {rule}")
        
        return "\n".join(lines)


@dataclass
class SemanticModel:
    """
    Complete semantic model of a database
    
    This is the enriched understanding of the database that goes beyond
    raw schema to include business context, relationships, and semantics.
    """
    database_name: str
    database_type: str
    
    # Tables with full semantic information
    tables: Dict[str, TableSemantics] = field(default_factory=dict)
    
    # Relationships between tables
    relationships: List[Relationship] = field(default_factory=list)
    
    # Business context
    business_context: Optional[BusinessContext] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"
    
    # Source tracking
    sources: List[str] = field(default_factory=list)  # Where data came from
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "database_name": self.database_name,
            "database_type": self.database_type,
            "tables": {k: v.to_dict() for k, v in self.tables.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "business_context": self.business_context.to_dict() if self.business_context else None,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "version": self.version,
            "sources": self.sources,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export as JSON"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Export as YAML"""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def save(self, path: str) -> None:
        """Save model to file (JSON or YAML based on extension)"""
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                f.write(self.to_yaml())
        else:
            with open(path, 'w') as f:
                f.write(self.to_json())
    
    @classmethod
    def load(cls, path: str) -> "SemanticModel":
        """Load model from file"""
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticModel":
        """Create model from dictionary"""
        model = cls(
            database_name=data.get("database_name", "unknown"),
            database_type=data.get("database_type", "unknown"),
            version=data.get("version", "1.0"),
            sources=data.get("sources", []),
        )
        
        # Load tables
        for table_name, table_data in data.get("tables", {}).items():
            table = TableSemantics(
                name=table_name,
                description=table_data.get("description"),
                purpose=table_data.get("purpose"),
                business_name=table_data.get("business_name"),
                domain=table_data.get("domain"),
                is_lookup_table=table_data.get("is_lookup_table", False),
                is_junction_table=table_data.get("is_junction_table", False),
                row_count=table_data.get("row_count"),
                primary_key=table_data.get("primary_key", []),
            )
            
            for col_name, col_data in table_data.get("columns", {}).items():
                col = ColumnSemantics(
                    name=col_name,
                    data_type=col_data.get("data_type", "unknown"),
                    semantic_type=SemanticType(col_data.get("semantic_type", "unknown")),
                    description=col_data.get("description"),
                    nullable=col_data.get("nullable", True),
                    is_primary_key=col_data.get("is_primary_key", False),
                    is_foreign_key=col_data.get("is_foreign_key", False),
                    is_unique=col_data.get("is_unique", False),
                    references_table=col_data.get("references_table"),
                    references_column=col_data.get("references_column"),
                    business_name=col_data.get("business_name"),
                )
                table.columns[col_name] = col
            
            model.tables[table_name] = table
        
        # Load relationships
        for rel_data in data.get("relationships", []):
            rel = Relationship(
                name=rel_data.get("name", ""),
                source_table=rel_data.get("source_table", ""),
                source_columns=rel_data.get("source_columns", []),
                target_table=rel_data.get("target_table", ""),
                target_columns=rel_data.get("target_columns", []),
                relationship_type=RelationshipType(rel_data.get("relationship_type", "one_to_many")),
                source=RelationshipSource(rel_data.get("source", "manual")),
                description=rel_data.get("description"),
                confidence=rel_data.get("confidence", 1.0),
            )
            model.relationships.append(rel)
        
        # Load business context
        if data.get("business_context"):
            bc_data = data["business_context"]
            model.business_context = BusinessContext(
                domain=bc_data.get("domain"),
                description=bc_data.get("description"),
                terminology=bc_data.get("terminology", {}),
                key_entities=bc_data.get("key_entities", []),
                common_queries=bc_data.get("common_queries", {}),
                business_rules=bc_data.get("business_rules", []),
            )
        
        return model
    
    def get_table(self, name: str) -> Optional[TableSemantics]:
        """Get table by name (case-insensitive)"""
        if name in self.tables:
            return self.tables[name]
        name_lower = name.lower()
        for table_name, table in self.tables.items():
            if table_name.lower() == name_lower:
                return table
        return None
    
    def get_relationships_for_table(self, table_name: str) -> List[Relationship]:
        """Get all relationships involving a table"""
        return [
            r for r in self.relationships
            if r.source_table == table_name or r.target_table == table_name
        ]
    
    def get_related_tables(self, table_name: str) -> Set[str]:
        """Get all tables related to the given table"""
        related = set()
        for r in self.relationships:
            if r.source_table == table_name:
                related.add(r.target_table)
            elif r.target_table == table_name:
                related.add(r.source_table)
        return related
    
    def find_join_path(self, from_table: str, to_table: str) -> Optional[List[Relationship]]:
        """Find the join path between two tables (BFS)"""
        if from_table == to_table:
            return []
        
        from collections import deque
        
        visited = {from_table}
        queue = deque([(from_table, [])])
        
        while queue:
            current, path = queue.popleft()
            
            for rel in self.get_relationships_for_table(current):
                if rel.source_table == current:
                    next_table = rel.target_table
                else:
                    next_table = rel.source_table
                
                if next_table == to_table:
                    return path + [rel]
                
                if next_table not in visited:
                    visited.add(next_table)
                    queue.append((next_table, path + [rel]))
        
        return None
    
    def to_schema_prompt(self, include_business_context: bool = True) -> str:
        """Generate comprehensive schema prompt for LLM"""
        lines = [
            f"DATABASE: {self.database_name} ({self.database_type})",
            "=" * 60,
            ""
        ]
        
        # Business context first
        if include_business_context and self.business_context:
            lines.append("BUSINESS CONTEXT:")
            lines.append(self.business_context.to_prompt_string())
            lines.append("")
        
        # Tables
        lines.append("TABLES:")
        lines.append("-" * 40)
        for table in self.tables.values():
            lines.append(table.to_ddl_string())
            lines.append("")
        
        # Relationships
        if self.relationships:
            lines.append("RELATIONSHIPS:")
            lines.append("-" * 40)
            for rel in self.relationships:
                rel_desc = (
                    f"{rel.source_table}.{','.join(rel.source_columns)} "
                    f"-> {rel.target_table}.{','.join(rel.target_columns)} "
                    f"({rel.relationship_type.value})"
                )
                if rel.description:
                    rel_desc += f" -- {rel.description}"
                lines.append(rel_desc)
            lines.append("")
        
        return "\n".join(lines)
