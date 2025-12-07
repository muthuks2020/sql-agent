"""
Query Understanding Module

Analyzes natural language queries to understand user intent before SQL generation.
This helps the LLM generate better SQL by providing structured analysis of:
- Operation type (SELECT, INSERT, UPDATE, DELETE)
- Required tables
- Aggregation needs
- Filtering requirements
- Ordering/sorting
- Time-based queries
- Join requirements
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta


class SQLOperationType(Enum):
    """Types of SQL operations"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    UNKNOWN = "UNKNOWN"


class AggregationType(Enum):
    """Types of aggregations detected"""
    COUNT = "COUNT"
    SUM = "SUM"
    AVERAGE = "AVG"
    MINIMUM = "MIN"
    MAXIMUM = "MAX"
    GROUP = "GROUP"
    DISTINCT = "DISTINCT"


class FilterType(Enum):
    """Types of filters detected"""
    EQUALITY = "equality"           # is, equals, =
    COMPARISON = "comparison"       # greater, less, more, fewer
    RANGE = "range"                 # between, from...to
    LIKE = "like"                   # contains, starts with, ends with
    IN_LIST = "in_list"             # one of, in, any of
    NULL_CHECK = "null_check"       # is null, is not null, missing, empty
    TIME_BASED = "time_based"       # last week, yesterday, this month


class TimeFrame(Enum):
    """Detected time frames"""
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"
    LAST_N_DAYS = "last_n_days"
    CUSTOM_RANGE = "custom_range"


class SortDirection(Enum):
    """Sort directions"""
    ASCENDING = "ASC"
    DESCENDING = "DESC"


@dataclass
class TimeReference:
    """Represents a detected time reference"""
    frame: TimeFrame
    field_hint: Optional[str] = None  # e.g., "created_at", "order_date"
    n_value: Optional[int] = None     # for "last N days"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class SortRequirement:
    """Represents a detected sorting requirement"""
    direction: SortDirection
    field_hint: Optional[str] = None  # e.g., "price", "date", "name"
    limit: Optional[int] = None       # for "top 10", "first 5"


@dataclass
class FilterRequirement:
    """Represents a detected filter requirement"""
    filter_type: FilterType
    field_hint: Optional[str] = None
    value_hint: Optional[str] = None
    operator: Optional[str] = None


@dataclass
class JoinRequirement:
    """Represents a detected join requirement"""
    tables: List[str]
    relationship_hint: Optional[str] = None  # e.g., "with their orders"


@dataclass
class QueryAnalysis:
    """Complete analysis of a natural language query"""
    
    # Original query
    original_query: str
    
    # Operation type
    operation: SQLOperationType = SQLOperationType.SELECT
    
    # Detected entities
    potential_tables: List[str] = field(default_factory=list)
    potential_columns: List[str] = field(default_factory=list)
    
    # Aggregations
    has_aggregation: bool = False
    aggregation_types: List[AggregationType] = field(default_factory=list)
    
    # Filtering
    has_filtering: bool = False
    filter_requirements: List[FilterRequirement] = field(default_factory=list)
    
    # Time-based
    has_time_filter: bool = False
    time_references: List[TimeReference] = field(default_factory=list)
    
    # Ordering
    has_ordering: bool = False
    sort_requirements: List[SortRequirement] = field(default_factory=list)
    
    # Joins
    needs_join: bool = False
    join_hints: List[JoinRequirement] = field(default_factory=list)
    
    # Grouping
    has_grouping: bool = False
    group_by_hints: List[str] = field(default_factory=list)
    
    # Limit
    has_limit: bool = False
    limit_value: Optional[int] = None
    
    # Distinct
    needs_distinct: bool = False
    
    # Confidence
    confidence: float = 0.0
    
    # Keywords found
    keywords_found: Dict[str, List[str]] = field(default_factory=dict)
    
    # Warnings/notes
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_query": self.original_query,
            "operation": self.operation.value,
            "potential_tables": self.potential_tables,
            "potential_columns": self.potential_columns,
            "has_aggregation": self.has_aggregation,
            "aggregation_types": [a.value for a in self.aggregation_types],
            "has_filtering": self.has_filtering,
            "has_time_filter": self.has_time_filter,
            "has_ordering": self.has_ordering,
            "has_grouping": self.has_grouping,
            "has_limit": self.has_limit,
            "limit_value": self.limit_value,
            "needs_distinct": self.needs_distinct,
            "needs_join": self.needs_join,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }
    
    def to_prompt_context(self) -> str:
        """Generate context string for LLM prompt"""
        lines = ["## Query Analysis"]
        
        lines.append(f"Operation Type: {self.operation.value}")
        
        if self.potential_tables:
            lines.append(f"Likely Tables: {', '.join(self.potential_tables)}")
        
        if self.potential_columns:
            lines.append(f"Likely Columns: {', '.join(self.potential_columns)}")
        
        features = []
        if self.has_aggregation:
            agg_str = ', '.join(a.value for a in self.aggregation_types)
            features.append(f"Aggregation ({agg_str})")
        if self.has_filtering:
            features.append("Filtering")
        if self.has_time_filter:
            time_str = ', '.join(t.frame.value for t in self.time_references)
            features.append(f"Time Filter ({time_str})")
        if self.has_ordering:
            sort_str = ', '.join(s.direction.value for s in self.sort_requirements)
            features.append(f"Ordering ({sort_str})")
        if self.has_grouping:
            features.append("Grouping")
        if self.has_limit:
            features.append(f"Limit ({self.limit_value})")
        if self.needs_distinct:
            features.append("Distinct")
        if self.needs_join:
            features.append("Joins Required")
        
        if features:
            lines.append(f"Features Needed: {', '.join(features)}")
        
        if self.warnings:
            lines.append(f"Notes: {'; '.join(self.warnings)}")
        
        return '\n'.join(lines)


class QueryUnderstanding:
    """
    Analyzes natural language queries to understand user intent.
    
    This helps the SQL Generator by providing:
    - Detected operation type
    - Potential tables/columns mentioned
    - Required SQL features (aggregation, filtering, joins, etc.)
    - Time-based query detection
    - Sorting/limit requirements
    """
    
    # Keywords for operation detection
    OPERATION_KEYWORDS = {
        SQLOperationType.SELECT: [
            "show", "display", "get", "find", "list", "fetch", "retrieve",
            "select", "view", "see", "give me", "what", "which", "who",
            "how many", "how much", "report", "query", "search", "look up"
        ],
        SQLOperationType.INSERT: [
            "insert", "add", "create", "new", "put", "register", "save",
            "store", "enter", "record"
        ],
        SQLOperationType.UPDATE: [
            "update", "change", "modify", "edit", "set", "alter", "fix",
            "correct", "adjust", "replace"
        ],
        SQLOperationType.DELETE: [
            "delete", "remove", "drop", "erase", "clear", "purge",
            "eliminate", "destroy"
        ],
    }
    
    # Keywords for aggregation detection
    AGGREGATION_KEYWORDS = {
        AggregationType.COUNT: [
            "count", "how many", "number of", "total number", "quantity of"
        ],
        AggregationType.SUM: [
            "sum", "total", "add up", "combined", "altogether"
        ],
        AggregationType.AVERAGE: [
            "average", "avg", "mean", "typical"
        ],
        AggregationType.MINIMUM: [
            "minimum", "min", "lowest", "smallest", "least", "cheapest"
        ],
        AggregationType.MAXIMUM: [
            "maximum", "max", "highest", "largest", "biggest", "most expensive", "top"
        ],
        AggregationType.DISTINCT: [
            "distinct", "unique", "different", "various"
        ],
    }
    
    # Keywords for time detection
    TIME_KEYWORDS = {
        TimeFrame.TODAY: ["today", "current day", "this day"],
        TimeFrame.YESTERDAY: ["yesterday", "previous day", "day before"],
        TimeFrame.THIS_WEEK: ["this week", "current week"],
        TimeFrame.LAST_WEEK: ["last week", "previous week", "past week"],
        TimeFrame.THIS_MONTH: ["this month", "current month"],
        TimeFrame.LAST_MONTH: ["last month", "previous month", "past month"],
        TimeFrame.THIS_YEAR: ["this year", "current year"],
        TimeFrame.LAST_YEAR: ["last year", "previous year", "past year"],
    }
    
    # Keywords for sorting detection
    SORT_KEYWORDS = {
        SortDirection.DESCENDING: [
            "highest", "largest", "biggest", "most", "top", "best",
            "newest", "latest", "recent", "descending", "desc",
            "maximum", "greatest"
        ],
        SortDirection.ASCENDING: [
            "lowest", "smallest", "least", "bottom", "worst",
            "oldest", "earliest", "first", "ascending", "asc",
            "minimum", "fewest"
        ],
    }
    
    # Keywords for filtering
    FILTER_KEYWORDS = {
        FilterType.EQUALITY: [
            "is", "equals", "equal to", "exactly", "specifically", "named"
        ],
        FilterType.COMPARISON: [
            "greater than", "more than", "less than", "fewer than",
            "at least", "at most", "over", "under", "above", "below"
        ],
        FilterType.RANGE: [
            "between", "from", "ranging", "in the range"
        ],
        FilterType.LIKE: [
            "contains", "containing", "includes", "like", "similar",
            "starts with", "ends with", "matching"
        ],
        FilterType.IN_LIST: [
            "one of", "any of", "either", "in"
        ],
        FilterType.NULL_CHECK: [
            "missing", "empty", "null", "blank", "no", "without",
            "not set", "undefined"
        ],
    }
    
    # Keywords for grouping
    GROUP_KEYWORDS = [
        "by", "per", "each", "every", "grouped by", "group by",
        "for each", "breakdown", "categorized"
    ]
    
    # Keywords for join detection
    JOIN_KEYWORDS = [
        "with", "and their", "along with", "including", "together with",
        "associated", "related", "linked", "connected"
    ]
    
    # Common table name patterns
    COMMON_ENTITIES = [
        "user", "users", "customer", "customers", "client", "clients",
        "order", "orders", "purchase", "purchases", "transaction", "transactions",
        "product", "products", "item", "items", "inventory",
        "employee", "employees", "staff", "worker", "workers",
        "category", "categories", "type", "types",
        "payment", "payments", "invoice", "invoices",
        "account", "accounts", "profile", "profiles",
        "message", "messages", "comment", "comments", "review", "reviews",
        "post", "posts", "article", "articles", "blog",
        "log", "logs", "event", "events", "activity", "activities",
        "session", "sessions", "visit", "visits",
        "sale", "sales", "revenue",
        "report", "reports", "metric", "metrics",
        "address", "addresses", "location", "locations",
        "department", "departments", "team", "teams",
        "project", "projects", "task", "tasks",
        "file", "files", "document", "documents",
        "notification", "notifications", "alert", "alerts",
        "subscription", "subscriptions", "plan", "plans",
        "setting", "settings", "config", "configuration",
        "role", "roles", "permission", "permissions",
    ]
    
    # Common column name patterns
    COMMON_COLUMNS = [
        "id", "name", "email", "phone", "address", "city", "state", "country",
        "price", "cost", "amount", "total", "quantity", "count",
        "date", "time", "created", "updated", "modified", "deleted",
        "status", "state", "type", "category", "level", "priority",
        "description", "title", "content", "body", "text", "note",
        "active", "enabled", "visible", "public", "verified",
        "first_name", "last_name", "full_name", "username", "password",
        "age", "gender", "birthday", "birth_date",
        "rating", "score", "rank", "points",
        "url", "link", "image", "photo", "avatar",
        "parent", "child", "owner", "author", "creator",
    ]
    
    def __init__(
        self,
        schema_tables: Optional[List[str]] = None,
        schema_columns: Optional[Dict[str, List[str]]] = None,
        business_terms: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize QueryUnderstanding.
        
        Args:
            schema_tables: List of actual table names in the database
            schema_columns: Dict of table_name -> list of column names
            business_terms: Dict of business term -> technical term mappings
        """
        self.schema_tables = schema_tables or []
        self.schema_columns = schema_columns or {}
        self.business_terms = business_terms or {}
        
        # Build lowercase versions for matching
        self._tables_lower = {t.lower(): t for t in self.schema_tables}
        self._all_columns = set()
        for cols in self.schema_columns.values():
            self._all_columns.update(c.lower() for c in cols)
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a natural language query.
        
        Args:
            query: The natural language query to analyze
            
        Returns:
            QueryAnalysis with detected features
        """
        analysis = QueryAnalysis(original_query=query)
        query_lower = query.lower()
        words = self._tokenize(query_lower)
        
        # Detect operation type
        analysis.operation = self._detect_operation(query_lower)
        
        # Detect potential tables
        analysis.potential_tables = self._detect_tables(query_lower, words)
        
        # Detect potential columns
        analysis.potential_columns = self._detect_columns(query_lower, words)
        
        # Detect aggregations
        agg_result = self._detect_aggregations(query_lower)
        analysis.has_aggregation = agg_result[0]
        analysis.aggregation_types = agg_result[1]
        
        # Detect time filters
        time_result = self._detect_time_references(query_lower)
        analysis.has_time_filter = time_result[0]
        analysis.time_references = time_result[1]
        
        # Detect filtering
        filter_result = self._detect_filters(query_lower)
        analysis.has_filtering = filter_result[0] or analysis.has_time_filter
        analysis.filter_requirements = filter_result[1]
        
        # Detect sorting
        sort_result = self._detect_sorting(query_lower)
        analysis.has_ordering = sort_result[0]
        analysis.sort_requirements = sort_result[1]
        
        # Detect limit
        limit_result = self._detect_limit(query_lower)
        analysis.has_limit = limit_result[0]
        analysis.limit_value = limit_result[1]
        
        # Detect grouping
        group_result = self._detect_grouping(query_lower, analysis)
        analysis.has_grouping = group_result[0]
        analysis.group_by_hints = group_result[1]
        
        # Detect distinct
        analysis.needs_distinct = self._detect_distinct(query_lower)
        
        # Detect join requirements
        join_result = self._detect_joins(query_lower, analysis)
        analysis.needs_join = join_result[0]
        analysis.join_hints = join_result[1]
        
        # Calculate confidence
        analysis.confidence = self._calculate_confidence(analysis)
        
        # Add warnings
        analysis.warnings = self._generate_warnings(analysis)
        
        return analysis
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove punctuation except apostrophes
        cleaned = re.sub(r"[^\w\s']", " ", text)
        return cleaned.split()
    
    def _detect_operation(self, query: str) -> SQLOperationType:
        """Detect the SQL operation type"""
        # Check each operation type
        scores = {}
        
        for op_type, keywords in self.OPERATION_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    # Weight by keyword position (earlier = more likely intent)
                    pos = query.find(keyword)
                    weight = 1.0 - (pos / len(query)) * 0.5  # Earlier keywords weighted higher
                    score += weight
            scores[op_type] = score
        
        # Default to SELECT if no clear operation
        if max(scores.values()) == 0:
            return SQLOperationType.SELECT
        
        return max(scores, key=scores.get)
    
    def _detect_tables(self, query: str, words: List[str]) -> List[str]:
        """Detect potential table names"""
        tables = []
        
        # Check against actual schema tables
        for table_lower, table_actual in self._tables_lower.items():
            if table_lower in query or any(w == table_lower for w in words):
                tables.append(table_actual)
        
        # Check business terms
        for term, technical in self.business_terms.items():
            if term.lower() in query:
                if technical in self._tables_lower.values():
                    tables.append(technical)
        
        # Check common entity names
        for entity in self.COMMON_ENTITIES:
            if entity in words:
                # Check if matches a schema table
                if entity in self._tables_lower:
                    tables.append(self._tables_lower[entity])
                elif entity + 's' in self._tables_lower:
                    tables.append(self._tables_lower[entity + 's'])
                elif entity.rstrip('s') in self._tables_lower:
                    tables.append(self._tables_lower[entity.rstrip('s')])
                else:
                    # Add as hint even if not in schema
                    tables.append(entity)
        
        return list(set(tables))
    
    def _detect_columns(self, query: str, words: List[str]) -> List[str]:
        """Detect potential column names"""
        columns = []
        
        # Check against actual schema columns
        for col in self._all_columns:
            if col in query or col in words:
                columns.append(col)
        
        # Check common column patterns
        for col in self.COMMON_COLUMNS:
            if col in words:
                columns.append(col)
        
        # Detect column patterns like "user's email" -> email
        possessive_pattern = r"(\w+)'s\s+(\w+)"
        matches = re.findall(possessive_pattern, query)
        for _, col in matches:
            if col in self._all_columns or col in self.COMMON_COLUMNS:
                columns.append(col)
        
        return list(set(columns))
    
    def _detect_aggregations(self, query: str) -> Tuple[bool, List[AggregationType]]:
        """Detect aggregation requirements"""
        found = []
        
        for agg_type, keywords in self.AGGREGATION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    found.append(agg_type)
                    break
        
        return (len(found) > 0, list(set(found)))
    
    def _detect_time_references(self, query: str) -> Tuple[bool, List[TimeReference]]:
        """Detect time-based filters"""
        references = []
        
        for time_frame, keywords in self.TIME_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    ref = TimeReference(frame=time_frame)
                    
                    # Try to detect the field
                    for field in ["created", "updated", "date", "time", "ordered", "placed"]:
                        if field in query:
                            ref.field_hint = field
                            break
                    
                    references.append(ref)
                    break
        
        # Detect "last N days/weeks/months"
        last_n_pattern = r"last\s+(\d+)\s+(day|week|month|year)s?"
        matches = re.findall(last_n_pattern, query)
        for n, unit in matches:
            ref = TimeReference(
                frame=TimeFrame.LAST_N_DAYS,
                n_value=int(n) * {'day': 1, 'week': 7, 'month': 30, 'year': 365}.get(unit, 1)
            )
            references.append(ref)
        
        return (len(references) > 0, references)
    
    def _detect_filters(self, query: str) -> Tuple[bool, List[FilterRequirement]]:
        """Detect filtering requirements"""
        filters = []
        
        for filter_type, keywords in self.FILTER_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    req = FilterRequirement(filter_type=filter_type)
                    
                    # Try to extract value hints
                    # Pattern: "status is active" -> field=status, value=active
                    after_keyword = query.split(keyword)[-1].strip()
                    if after_keyword:
                        words = after_keyword.split()
                        if words:
                            req.value_hint = words[0]
                    
                    filters.append(req)
                    break
        
        # Detect comparison patterns: "price > 100", "age >= 18"
        comparison_pattern = r"(\w+)\s*([<>=!]+)\s*(\d+)"
        matches = re.findall(comparison_pattern, query)
        for field, op, value in matches:
            filters.append(FilterRequirement(
                filter_type=FilterType.COMPARISON,
                field_hint=field,
                operator=op,
                value_hint=value
            ))
        
        return (len(filters) > 0, filters)
    
    def _detect_sorting(self, query: str) -> Tuple[bool, List[SortRequirement]]:
        """Detect sorting requirements"""
        sorts = []
        
        for direction, keywords in self.SORT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    req = SortRequirement(direction=direction)
                    
                    # Try to detect what to sort by
                    # "highest price" -> sort by price DESC
                    # "most recent" -> sort by date DESC
                    idx = query.find(keyword)
                    context = query[max(0, idx-20):min(len(query), idx+30)]
                    
                    for col in self.COMMON_COLUMNS + ["price", "date", "amount", "count", "sales"]:
                        if col in context:
                            req.field_hint = col
                            break
                    
                    sorts.append(req)
                    break
        
        # Detect "order by" / "sort by" explicit patterns
        sort_pattern = r"(?:order|sort|sorted)\s+by\s+(\w+)"
        matches = re.findall(sort_pattern, query)
        for field in matches:
            sorts.append(SortRequirement(
                direction=SortDirection.DESCENDING,  # Default to DESC
                field_hint=field
            ))
        
        return (len(sorts) > 0, sorts)
    
    def _detect_limit(self, query: str) -> Tuple[bool, Optional[int]]:
        """Detect limit requirements"""
        # Patterns: "top 10", "first 5", "limit 100"
        patterns = [
            r"top\s+(\d+)",
            r"first\s+(\d+)",
            r"limit\s+(\d+)",
            r"(\d+)\s+(?:results?|rows?|records?|items?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return (True, int(match.group(1)))
        
        # Keywords that imply limit
        if any(kw in query for kw in ["top", "first", "one", "single", "a "]):
            return (True, None)  # Limit needed but value unknown
        
        return (False, None)
    
    def _detect_grouping(self, query: str, analysis: QueryAnalysis) -> Tuple[bool, List[str]]:
        """Detect grouping requirements"""
        hints = []
        
        # Check for GROUP BY indicators
        has_grouping = any(kw in query for kw in self.GROUP_KEYWORDS)
        
        # If aggregation + entity mention, likely needs grouping
        if analysis.has_aggregation and analysis.potential_tables:
            has_grouping = True
        
        # Extract grouping fields
        # "sales by region" -> group by region
        # "count per category" -> group by category
        group_pattern = r"(?:by|per|each|every)\s+(\w+)"
        matches = re.findall(group_pattern, query)
        hints.extend(matches)
        
        return (has_grouping, list(set(hints)))
    
    def _detect_distinct(self, query: str) -> bool:
        """Detect if DISTINCT is needed"""
        distinct_keywords = ["distinct", "unique", "different", "no duplicates"]
        return any(kw in query for kw in distinct_keywords)
    
    def _detect_joins(self, query: str, analysis: QueryAnalysis) -> Tuple[bool, List[JoinRequirement]]:
        """Detect if joins are needed"""
        hints = []
        
        # Multiple tables mentioned
        needs_join = len(analysis.potential_tables) > 1
        
        # Join keywords present
        for keyword in self.JOIN_KEYWORDS:
            if keyword in query:
                needs_join = True
                
                # Try to extract related tables
                # "users with their orders" -> join users and orders
                idx = query.find(keyword)
                context = query[idx:min(len(query), idx+30)]
                
                for table in self.COMMON_ENTITIES + list(self._tables_lower.keys()):
                    if table in context:
                        hints.append(JoinRequirement(
                            tables=analysis.potential_tables + [table],
                            relationship_hint=keyword
                        ))
                        break
        
        # Possessive patterns: "customer's orders" -> join customers and orders
        possessive_pattern = r"(\w+)'s\s+(\w+)"
        matches = re.findall(possessive_pattern, query)
        for entity, related in matches:
            if entity in self.COMMON_ENTITIES or related in self.COMMON_ENTITIES:
                needs_join = True
                hints.append(JoinRequirement(
                    tables=[entity, related],
                    relationship_hint="possessive"
                ))
        
        return (needs_join, hints)
    
    def _calculate_confidence(self, analysis: QueryAnalysis) -> float:
        """Calculate confidence score for the analysis"""
        score = 0.5  # Base score
        
        # Boost for detected tables
        if analysis.potential_tables:
            score += 0.1 * min(len(analysis.potential_tables), 3)
        
        # Boost for clear operation
        if analysis.operation != SQLOperationType.UNKNOWN:
            score += 0.1
        
        # Boost for detected features
        if analysis.has_aggregation:
            score += 0.05
        if analysis.has_filtering:
            score += 0.05
        if analysis.has_ordering:
            score += 0.05
        if analysis.has_time_filter:
            score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _generate_warnings(self, analysis: QueryAnalysis) -> List[str]:
        """Generate warnings for ambiguous queries"""
        warnings = []
        
        if not analysis.potential_tables:
            warnings.append("No specific tables detected - may need clarification")
        
        if analysis.operation == SQLOperationType.DELETE and not analysis.has_filtering:
            warnings.append("DELETE without filter may affect all rows")
        
        if analysis.operation == SQLOperationType.UPDATE and not analysis.has_filtering:
            warnings.append("UPDATE without filter may affect all rows")
        
        if analysis.has_aggregation and not analysis.has_grouping:
            warnings.append("Aggregation detected but no GROUP BY hint found")
        
        if len(analysis.potential_tables) > 3:
            warnings.append("Multiple tables detected - verify join requirements")
        
        return warnings


# Convenience function
def analyze_query(
    query: str,
    schema_tables: Optional[List[str]] = None,
    schema_columns: Optional[Dict[str, List[str]]] = None,
    business_terms: Optional[Dict[str, str]] = None,
) -> QueryAnalysis:
    """
    Analyze a natural language query.
    
    Args:
        query: Natural language query
        schema_tables: List of table names
        schema_columns: Dict of table -> columns
        business_terms: Business term mappings
        
    Returns:
        QueryAnalysis object
    """
    analyzer = QueryUnderstanding(
        schema_tables=schema_tables,
        schema_columns=schema_columns,
        business_terms=business_terms,
    )
    return analyzer.analyze(query)
