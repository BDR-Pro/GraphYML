"""
Query engine for GraphYML.
Provides a simple query language for searching and filtering graph nodes.
"""
import re
import operator
from typing import Dict, List, Any, Callable, Tuple, Set, Optional, Union

# Comparison operators
OPERATORS = {
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "contains": lambda a, b: b in a if isinstance(a, (list, str)) else False,
    "in": lambda a, b: a in b if isinstance(b, (list, str)) else False,
    "startswith": lambda a, b: a.startswith(b) if isinstance(a, str) else False,
    "endswith": lambda a, b: a.endswith(b) if isinstance(a, str) else False,
    "matches": lambda a, b: bool(re.search(b, a)) if isinstance(a, str) else False
}

# Logical operators
AND = "AND"
OR = "OR"
NOT = "NOT"


class QueryCondition:
    """Represents a single condition in a query."""
    
    def __init__(
        self, 
        field: str, 
        operator_str: str, 
        value: Any, 
        negate: bool = False
    ):
        """
        Initialize a query condition.
        
        Args:
            field: Field name to compare
            operator_str: String representation of the operator
            value: Value to compare against
            negate: Whether to negate the condition
        """
        self.field = field
        self.operator_str = operator_str
        self.value = value
        self.negate = negate
        
        if operator_str not in OPERATORS:
            raise ValueError(f"Unknown operator: {operator_str}")
        
        self.operator_func = OPERATORS[operator_str]
    
    def evaluate(self, node: Dict[str, Any]) -> bool:
        """
        Evaluate the condition against a node.
        
        Args:
            node: Node to evaluate against
            
        Returns:
            bool: Whether the condition is satisfied
        """
        # Handle nested fields with dot notation
        field_parts = self.field.split('.')
        value = node
        
        for part in field_parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                # Field doesn't exist
                return False
        
        try:
            result = self.operator_func(value, self.value)
            return not result if self.negate else result
        except (TypeError, ValueError):
            # Type mismatch or other error
            return False


class Query:
    """Represents a complete query with conditions and logical operators."""
    
    def __init__(self):
        """Initialize an empty query."""
        self.conditions = []
        self.operators = []  # AND/OR operators between conditions
    
    def add_condition(
        self, 
        field: str, 
        operator_str: str, 
        value: Any, 
        logical_op: str = AND, 
        negate: bool = False
    ):
        """
        Add a condition to the query.
        
        Args:
            field: Field name to compare
            operator_str: String representation of the operator
            value: Value to compare against
            logical_op: Logical operator (AND/OR) to combine with previous conditions
            negate: Whether to negate the condition
        """
        condition = QueryCondition(field, operator_str, value, negate)
        
        if self.conditions:
            if logical_op not in (AND, OR):
                raise ValueError(f"Unknown logical operator: {logical_op}")
            self.operators.append(logical_op)
        
        self.conditions.append(condition)
    
    def evaluate(self, node: Dict[str, Any]) -> bool:
        """
        Evaluate the query against a node.
        
        Args:
            node: Node to evaluate against
            
        Returns:
            bool: Whether the node satisfies the query
        """
        if not self.conditions:
            return True
        
        results = [condition.evaluate(node) for condition in self.conditions]
        
        if not self.operators:
            return results[0]
        
        # Evaluate with proper precedence (AND before OR)
        result = results[0]
        for i, op in enumerate(self.operators):
            if op == AND:
                result = result and results[i + 1]
            else:  # OR
                result = result or results[i + 1]
        
        return result


class QueryParser:
    """Parser for the GraphYML query language."""
    
    @staticmethod
    def parse(query_str: str) -> Query:
        """
        Parse a query string into a Query object.
        
        Args:
            query_str: Query string to parse
            
        Returns:
            Query: Parsed query
            
        Example query format:
            title contains "inception" AND year > 2010
            genres contains "Sci-Fi" OR genres contains "Action"
            director = "Christopher Nolan" AND NOT rating < 8.5
        """
        query = Query()
        
        if not query_str.strip():
            return query
        
        # Tokenize the query
        tokens = QueryParser._tokenize(query_str)
        
        # Parse tokens into conditions
        i = 0
        logical_op = AND  # Default logical operator
        negate = False
        
        while i < len(tokens):
            token = tokens[i]
            
            if token.upper() == NOT:
                negate = True
                i += 1
                continue
            
            if token.upper() in (AND, OR):
                logical_op = token.upper()
                i += 1
                continue
            
            # Parse field, operator, and value
            if i + 2 < len(tokens):
                field = token
                op = tokens[i + 1]
                value = tokens[i + 2]
                
                # Convert value to appropriate type
                value = QueryParser._convert_value(value)
                
                query.add_condition(field, op, value, logical_op, negate)
                
                negate = False
                i += 3
            else:
                raise ValueError(f"Incomplete condition at position {i}")
        
        return query
    
    @staticmethod
    def _tokenize(query_str: str) -> List[str]:
        """
        Tokenize a query string.
        
        Args:
            query_str: Query string to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        # Handle quoted strings
        in_quotes = False
        quote_char = None
        tokens = []
        current_token = ""
        
        for char in query_str:
            if char in ('"', "'") and (not in_quotes or quote_char == char):
                in_quotes = not in_quotes
                quote_char = char if in_quotes else None
                current_token += char
            elif char.isspace() and not in_quotes:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    @staticmethod
    def _convert_value(value: str) -> Any:
        """
        Convert a string value to the appropriate type.
        
        Args:
            value: String value to convert
            
        Returns:
            Any: Converted value
        """
        # Remove quotes from string literals
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        # Try to convert to numeric types
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            # Handle boolean values
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            
            # Return as string if no conversion applies
            return value


class QueryEngine:
    """Engine for executing queries against the graph."""
    
    def __init__(self, graph: Dict[str, Dict[str, Any]]):
        """
        Initialize the query engine.
        
        Args:
            graph: Graph to query
        """
        self.graph = graph
    
    def execute_query(self, query_str: str) -> List[str]:
        """
        Execute a query and return matching node keys.
        
        Args:
            query_str: Query string to execute
            
        Returns:
            List[str]: List of matching node keys
        """
        query = QueryParser.parse(query_str)
        
        results = []
        for key, node in self.graph.items():
            if query.evaluate(node):
                results.append(key)
        
        return results
    
    def find_by_field(
        self, 
        field: str, 
        value: Any, 
        operator_str: str = "="
    ) -> List[str]:
        """
        Find nodes by field value.
        
        Args:
            field: Field name to compare
            value: Value to compare against
            operator_str: String representation of the operator
            
        Returns:
            List[str]: List of matching node keys
        """
        query = Query()
        query.add_condition(field, operator_str, value)
        
        results = []
        for key, node in self.graph.items():
            if query.evaluate(node):
                results.append(key)
        
        return results
    
    def find_by_embedding_similarity(
        self, 
        embedding: List[float], 
        threshold: float = 0.7, 
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find nodes by embedding similarity.
        
        Args:
            embedding: Embedding vector to compare against
            threshold: Minimum similarity threshold
            limit: Maximum number of results
            
        Returns:
            List[Tuple[str, float]]: List of (node_key, similarity) tuples
        """
        from src.models.embeddings import embedding_similarity
        
        results = []
        for key, node in self.graph.items():
            if "embedding" in node:
                similarity = embedding_similarity(embedding, node["embedding"])
                if similarity >= threshold:
                    results.append((key, similarity))
        
        # Sort by similarity (descending) and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def traverse_graph(
        self, 
        start_key: str, 
        max_depth: int = 2, 
        filter_query: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Traverse the graph from a starting node.
        
        Args:
            start_key: Key of the starting node
            max_depth: Maximum traversal depth
            filter_query: Optional query to filter traversed nodes
            
        Returns:
            Dict[str, Dict[str, Any]]: Subgraph of traversed nodes
        """
        if start_key not in self.graph:
            return {}
        
        filter_query_obj = None
        if filter_query:
            filter_query_obj = QueryParser.parse(filter_query)
        
        visited = {start_key: self.graph[start_key]}
        frontier = [(start_key, 0)]  # (node_key, depth)
        
        while frontier:
            key, depth = frontier.pop(0)
            
            if depth >= max_depth:
                continue
            
            node = self.graph[key]
            for link in node.get("links", []):
                if link in self.graph and link not in visited:
                    linked_node = self.graph[link]
                    
                    # Apply filter if provided
                    if filter_query_obj and not filter_query_obj.evaluate(linked_node):
                        continue
                    
                    visited[link] = linked_node
                    frontier.append((link, depth + 1))
        
        return visited

