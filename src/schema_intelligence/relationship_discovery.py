"""
Relationship Discovery Engine

Automatically discovers relationships between tables using multiple strategies:
1. Explicit foreign keys (from database constraints)
2. Naming conventions (user_id -> users.id)
3. Data pattern matching (same values in columns)
4. LLM inference (analyze schema and suggest relationships)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    Relationship,
    RelationshipSource,
    RelationshipType,
    SemanticModel,
    SemanticType,
    TableSemantics,
)
from ..adapters.base import BaseDatabaseAdapter
from ..llm_client import BaseLLMClient
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class RelationshipCandidate:
    """A potential relationship discovered by analysis"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    confidence: float  # 0.0 to 1.0
    reason: str
    source: RelationshipSource


class NamingConventionAnalyzer:
    """
    Discovers relationships based on column naming patterns
    
    Common patterns:
    - user_id -> users.id
    - customer_id -> customers.id
    - fk_user -> users.id
    - user_fk -> users.id
    - order_user_id -> users.id (compound patterns)
    """
    
    # Patterns for FK column names
    FK_PATTERNS = [
        # Standard: table_id (e.g., user_id -> users)
        (r'^(\w+)_id$', lambda m: [m.group(1), f"{m.group(1)}s"]),
        
        # Prefixed: fk_table (e.g., fk_user -> users)
        (r'^fk_(\w+)$', lambda m: [m.group(1), f"{m.group(1)}s"]),
        
        # Suffixed: table_fk (e.g., user_fk -> users)
        (r'^(\w+)_fk$', lambda m: [m.group(1), f"{m.group(1)}s"]),
        
        # ID prefix: id_table (e.g., id_user -> users)
        (r'^id_(\w+)$', lambda m: [m.group(1), f"{m.group(1)}s"]),
        
        # Compound: prefix_table_id (e.g., primary_user_id -> users)
        (r'^(?:\w+_)?(\w+)_id$', lambda m: [m.group(1), f"{m.group(1)}s"]),
        
        # Ref suffix: table_ref (e.g., user_ref -> users)
        (r'^(\w+)_ref$', lambda m: [m.group(1), f"{m.group(1)}s"]),
    ]
    
    # Common singular -> plural mappings for irregular nouns
    IRREGULAR_PLURALS = {
        "person": "people",
        "child": "children",
        "category": "categories",
        "company": "companies",
        "country": "countries",
        "city": "cities",
        "status": "statuses",
        "address": "addresses",
        "process": "processes",
        "class": "classes",
        "analysis": "analyses",
    }
    
    def __init__(self, model: SemanticModel):
        self.model = model
        self.table_names = set(model.tables.keys())
        self.table_names_lower = {t.lower(): t for t in self.table_names}
    
    def discover(self) -> List[RelationshipCandidate]:
        """Discover relationships based on naming conventions"""
        candidates = []
        
        for table_name, table in self.model.tables.items():
            for col_name, column in table.columns.items():
                # Skip if already a known FK
                if column.is_foreign_key and column.references_table:
                    continue
                
                # Skip primary keys (they're targets, not sources)
                if column.is_primary_key:
                    continue
                
                # Try to find a matching table
                matches = self._find_matching_tables(col_name)
                
                for target_table, confidence in matches:
                    if target_table != table_name:  # Don't self-reference
                        candidates.append(RelationshipCandidate(
                            source_table=table_name,
                            source_column=col_name,
                            target_table=target_table,
                            target_column="id",  # Assume PK is 'id'
                            confidence=confidence,
                            reason=f"Column '{col_name}' matches table '{target_table}' naming pattern",
                            source=RelationshipSource.NAMING_CONVENTION,
                        ))
        
        return candidates
    
    def _find_matching_tables(self, column_name: str) -> List[Tuple[str, float]]:
        """Find tables that match the column name pattern"""
        matches = []
        col_lower = column_name.lower()
        
        for pattern, get_candidates in self.FK_PATTERNS:
            match = re.match(pattern, col_lower)
            if match:
                potential_tables = get_candidates(match)
                
                for potential in potential_tables:
                    # Check direct match
                    if potential in self.table_names_lower:
                        matches.append((self.table_names_lower[potential], 0.9))
                    
                    # Check irregular plural
                    if potential in self.IRREGULAR_PLURALS:
                        plural = self.IRREGULAR_PLURALS[potential]
                        if plural in self.table_names_lower:
                            matches.append((self.table_names_lower[plural], 0.85))
                    
                    # Check if singular form exists
                    singular = self._singularize(potential)
                    if singular in self.table_names_lower:
                        matches.append((self.table_names_lower[singular], 0.8))
        
        # Deduplicate keeping highest confidence
        seen = {}
        for table, conf in matches:
            if table not in seen or seen[table] < conf:
                seen[table] = conf
        
        return [(t, c) for t, c in seen.items()]
    
    def _singularize(self, word: str) -> str:
        """Simple singularization (reverse of common plural rules)"""
        if word.endswith('ies'):
            return word[:-3] + 'y'
        if word.endswith('es') and len(word) > 3:
            return word[:-2]
        if word.endswith('s') and not word.endswith('ss'):
            return word[:-1]
        return word


class DataPatternMatcher:
    """
    Discovers relationships by analyzing actual data patterns
    
    Looks for columns with matching values that could indicate FK relationships.
    """
    
    def __init__(self, adapter: BaseDatabaseAdapter, model: SemanticModel):
        self.adapter = adapter
        self.model = model
    
    def discover(self, sample_size: int = 100) -> List[RelationshipCandidate]:
        """Discover relationships by comparing column values"""
        candidates = []
        
        # Get potential ID columns (PKs and unique integer columns)
        pk_columns = self._get_pk_columns()
        
        # Get potential FK columns (integer columns ending in _id)
        fk_columns = self._get_potential_fk_columns()
        
        # Compare values between potential FKs and PKs
        for (fk_table, fk_col), fk_values in fk_columns.items():
            for (pk_table, pk_col), pk_values in pk_columns.items():
                if fk_table == pk_table:
                    continue  # Skip same table
                
                # Calculate overlap
                overlap = len(fk_values & pk_values)
                if overlap == 0:
                    continue
                
                # Calculate confidence based on overlap ratio
                fk_coverage = overlap / len(fk_values) if fk_values else 0
                pk_coverage = overlap / len(pk_values) if pk_values else 0
                
                # FK should be subset of PK (high fk_coverage, any pk_coverage)
                if fk_coverage > 0.7:
                    confidence = min(0.95, fk_coverage)
                    candidates.append(RelationshipCandidate(
                        source_table=fk_table,
                        source_column=fk_col,
                        target_table=pk_table,
                        target_column=pk_col,
                        confidence=confidence,
                        reason=f"{fk_coverage:.0%} of {fk_table}.{fk_col} values exist in {pk_table}.{pk_col}",
                        source=RelationshipSource.DATA_PATTERN,
                    ))
        
        return candidates
    
    def _get_pk_columns(self) -> Dict[Tuple[str, str], Set[Any]]:
        """Get primary key column values from all tables"""
        pk_columns = {}
        
        for table_name, table in self.model.tables.items():
            for col_name, column in table.columns.items():
                if column.is_primary_key or column.is_unique:
                    values = self._get_column_values(table_name, col_name)
                    if values:
                        pk_columns[(table_name, col_name)] = values
        
        return pk_columns
    
    def _get_potential_fk_columns(self) -> Dict[Tuple[str, str], Set[Any]]:
        """Get potential foreign key column values"""
        fk_columns = {}
        
        for table_name, table in self.model.tables.items():
            for col_name, column in table.columns.items():
                # Look for integer columns that might be FKs
                if column.is_primary_key:
                    continue
                
                col_lower = col_name.lower()
                type_lower = column.data_type.lower()
                
                # Check if it looks like an FK
                is_potential_fk = (
                    col_lower.endswith('_id') or
                    col_lower.endswith('_fk') or
                    col_lower.startswith('fk_') or
                    column.is_foreign_key
                )
                
                is_integer_like = any(t in type_lower for t in ['int', 'integer', 'bigint', 'smallint'])
                
                if is_potential_fk or (is_integer_like and col_lower != 'id'):
                    values = self._get_column_values(table_name, col_name)
                    if values:
                        fk_columns[(table_name, col_name)] = values
        
        return fk_columns
    
    def _get_column_values(self, table_name: str, column_name: str, limit: int = 100) -> Set[Any]:
        """Get distinct values from a column"""
        try:
            sql = f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT {limit}"
            result = self.adapter.execute_query(sql)
            
            if result.success and result.rows:
                return {row[0] for row in result.rows}
        except Exception as e:
            logger.debug(f"Could not get values for {table_name}.{column_name}: {e}")
        
        return set()


class LLMRelationshipInferrer:
    """
    Uses LLM to analyze schema and suggest relationships
    
    Particularly useful for:
    - Unusual naming conventions
    - Junction tables (many-to-many)
    - Non-obvious relationships
    """
    
    ANALYSIS_PROMPT = """Analyze this database schema and identify relationships between tables.

SCHEMA:
{schema}

For each relationship you identify, provide:
1. Source table and column
2. Target table and column
3. Relationship type (one_to_one, one_to_many, many_to_many)
4. Confidence level (high, medium, low)
5. Brief explanation

Format your response as a list with this structure:
RELATIONSHIP: source_table.source_column -> target_table.target_column
TYPE: relationship_type
CONFIDENCE: high/medium/low
REASON: explanation

Only include relationships you're confident about. Focus on:
- Columns that look like foreign keys but aren't explicitly defined
- Junction tables that enable many-to-many relationships
- Any patterns suggesting relationships between tables

If a table appears to be a junction table (many-to-many bridge), describe both relationships it creates.
"""
    
    def __init__(self, llm_client: BaseLLMClient, model: SemanticModel):
        self.llm_client = llm_client
        self.model = model
    
    def discover(self) -> List[RelationshipCandidate]:
        """Use LLM to discover relationships"""
        try:
            schema_str = self.model.to_schema_prompt(include_business_context=False)
            
            prompt = self.ANALYSIS_PROMPT.format(schema=schema_str)
            
            response = self.llm_client.invoke(
                prompt=prompt,
                system_prompt="You are a database schema analyst. Identify relationships between tables based on naming conventions, data types, and common patterns.",
                max_tokens=2000,
            )
            
            return self._parse_llm_response(response.content)
        
        except Exception as e:
            logger.warning(f"LLM relationship inference failed: {e}")
            return []
    
    def _parse_llm_response(self, content: str) -> List[RelationshipCandidate]:
        """Parse LLM response into RelationshipCandidates"""
        candidates = []
        
        # Parse each relationship block
        blocks = re.split(r'\n(?=RELATIONSHIP:)', content)
        
        for block in blocks:
            if not block.strip() or 'RELATIONSHIP:' not in block:
                continue
            
            try:
                # Parse relationship line
                rel_match = re.search(
                    r'RELATIONSHIP:\s*(\w+)\.(\w+)\s*->\s*(\w+)\.(\w+)',
                    block
                )
                if not rel_match:
                    continue
                
                source_table = rel_match.group(1)
                source_col = rel_match.group(2)
                target_table = rel_match.group(3)
                target_col = rel_match.group(4)
                
                # Verify tables exist
                if source_table not in self.model.tables or target_table not in self.model.tables:
                    continue
                
                # Parse type
                type_match = re.search(r'TYPE:\s*(\w+)', block)
                rel_type = RelationshipType.ONE_TO_MANY
                if type_match:
                    type_str = type_match.group(1).lower()
                    if 'one_to_one' in type_str or 'onetoone' in type_str:
                        rel_type = RelationshipType.ONE_TO_ONE
                    elif 'many_to_many' in type_str or 'manytomany' in type_str:
                        rel_type = RelationshipType.MANY_TO_MANY
                
                # Parse confidence
                conf_match = re.search(r'CONFIDENCE:\s*(\w+)', block)
                confidence = 0.7  # Default
                if conf_match:
                    conf_str = conf_match.group(1).lower()
                    if 'high' in conf_str:
                        confidence = 0.9
                    elif 'medium' in conf_str:
                        confidence = 0.7
                    elif 'low' in conf_str:
                        confidence = 0.5
                
                # Parse reason
                reason_match = re.search(r'REASON:\s*(.+)', block, re.DOTALL)
                reason = reason_match.group(1).strip() if reason_match else "LLM inferred"
                reason = reason.split('\n')[0]  # First line only
                
                candidates.append(RelationshipCandidate(
                    source_table=source_table,
                    source_column=source_col,
                    target_table=target_table,
                    target_column=target_col,
                    confidence=confidence,
                    reason=f"LLM: {reason}",
                    source=RelationshipSource.LLM_INFERRED,
                ))
            
            except Exception as e:
                logger.debug(f"Failed to parse relationship block: {e}")
        
        return candidates


class RelationshipDiscoveryEngine:
    """
    Main engine that orchestrates all relationship discovery methods
    
    Usage:
        engine = RelationshipDiscoveryEngine(model, adapter, llm_client)
        relationships = engine.discover_all()
        model = engine.apply_to_model(relationships)
    """
    
    def __init__(
        self,
        model: SemanticModel,
        adapter: Optional[BaseDatabaseAdapter] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.model = model
        self.adapter = adapter
        self.llm_client = llm_client
    
    def discover_all(
        self,
        use_naming_conventions: bool = True,
        use_data_patterns: bool = True,
        use_llm: bool = True,
        min_confidence: float = 0.5,
    ) -> List[RelationshipCandidate]:
        """
        Discover relationships using all available methods
        
        Returns candidates sorted by confidence
        """
        all_candidates = []
        
        # 1. Naming convention analysis
        if use_naming_conventions:
            try:
                analyzer = NamingConventionAnalyzer(self.model)
                candidates = analyzer.discover()
                all_candidates.extend(candidates)
                logger.info(f"Naming convention analysis found {len(candidates)} candidates")
            except Exception as e:
                logger.warning(f"Naming convention analysis failed: {e}")
        
        # 2. Data pattern matching
        if use_data_patterns and self.adapter:
            try:
                matcher = DataPatternMatcher(self.adapter, self.model)
                candidates = matcher.discover()
                all_candidates.extend(candidates)
                logger.info(f"Data pattern matching found {len(candidates)} candidates")
            except Exception as e:
                logger.warning(f"Data pattern matching failed: {e}")
        
        # 3. LLM inference
        if use_llm and self.llm_client:
            try:
                inferrer = LLMRelationshipInferrer(self.llm_client, self.model)
                candidates = inferrer.discover()
                all_candidates.extend(candidates)
                logger.info(f"LLM inference found {len(candidates)} candidates")
            except Exception as e:
                logger.warning(f"LLM inference failed: {e}")
        
        # Filter by confidence and deduplicate
        filtered = self._filter_and_deduplicate(all_candidates, min_confidence)
        
        # Sort by confidence (descending)
        filtered.sort(key=lambda c: c.confidence, reverse=True)
        
        return filtered
    
    def _filter_and_deduplicate(
        self,
        candidates: List[RelationshipCandidate],
        min_confidence: float,
    ) -> List[RelationshipCandidate]:
        """Filter by confidence and keep highest confidence for duplicates"""
        # Filter by min confidence
        filtered = [c for c in candidates if c.confidence >= min_confidence]
        
        # Deduplicate - keep highest confidence for each unique relationship
        seen: Dict[Tuple, RelationshipCandidate] = {}
        for candidate in filtered:
            key = (
                candidate.source_table,
                candidate.source_column,
                candidate.target_table,
                candidate.target_column,
            )
            
            if key not in seen or seen[key].confidence < candidate.confidence:
                seen[key] = candidate
        
        return list(seen.values())
    
    def apply_to_model(
        self,
        candidates: List[RelationshipCandidate],
    ) -> SemanticModel:
        """Apply discovered relationships to the model"""
        # Check which relationships already exist
        existing = {
            (r.source_table, tuple(r.source_columns), r.target_table)
            for r in self.model.relationships
        }
        
        for candidate in candidates:
            key = (
                candidate.source_table,
                (candidate.source_column,),
                candidate.target_table,
            )
            
            if key not in existing:
                # Create relationship
                rel = Relationship(
                    name=f"{candidate.source_table}_{candidate.source_column}_fk",
                    source_table=candidate.source_table,
                    source_columns=[candidate.source_column],
                    target_table=candidate.target_table,
                    target_columns=[candidate.target_column],
                    relationship_type=RelationshipType.MANY_TO_ONE,
                    source=candidate.source,
                    confidence=candidate.confidence,
                    description=candidate.reason,
                )
                self.model.relationships.append(rel)
                
                # Update column metadata
                table = self.model.get_table(candidate.source_table)
                if table:
                    col = table.get_column(candidate.source_column)
                    if col:
                        col.is_foreign_key = True
                        col.references_table = candidate.target_table
                        col.references_column = candidate.target_column
                
                logger.info(
                    f"Added relationship: {candidate.source_table}.{candidate.source_column} "
                    f"-> {candidate.target_table}.{candidate.target_column} "
                    f"(confidence: {candidate.confidence:.0%})"
                )
        
        return self.model
    
    def discover_junction_tables(self) -> List[str]:
        """Identify tables that are likely junction/bridge tables for M:N relationships"""
        junction_tables = []
        
        for table_name, table in self.model.tables.items():
            # Characteristics of junction tables:
            # 1. Usually have 2 FK columns
            # 2. Often have compound primary key
            # 3. Few or no other columns (maybe just timestamps)
            # 4. Name often combines two other table names
            
            fk_columns = [
                col for col in table.columns.values()
                if col.is_foreign_key or col.name.lower().endswith('_id')
            ]
            
            non_fk_columns = [
                col for col in table.columns.values()
                if not col.is_foreign_key
                and not col.name.lower().endswith('_id')
                and not col.is_primary_key
                and col.semantic_type not in (
                    SemanticType.CREATED_AT,
                    SemanticType.UPDATED_AT,
                )
            ]
            
            # Strong signal: exactly 2 FK columns and few other columns
            if len(fk_columns) == 2 and len(non_fk_columns) <= 2:
                junction_tables.append(table_name)
                table.is_junction_table = True
                logger.info(f"Identified junction table: {table_name}")
        
        return junction_tables
