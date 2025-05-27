"""
ORM-like interface for GraphYML database.
Provides a more user-friendly API for common database operations.
"""
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import re

from src.models.query_engine import QueryEngine


class GraphORM:
    """
    ORM-like interface for GraphYML database.
    Provides a more user-friendly API for common database operations.
    """
    
    def __init__(self, database):
        """
        Initialize the ORM interface.
        
        Args:
            database: Database instance
        """
        self.db = database
        self.graph = database.graph
        self.query_engine = database.query_engine
    
    def find_by_id(self, node_id: str, user) -> Optional[Dict[str, Any]]:
        """
        Find a node by ID.
        
        Args:
            node_id: Node ID
            user: User making the request
            
        Returns:
            Optional[Dict[str, Any]]: Node data or None
        """
        node, error = self.db.get_node(node_id, user)
        return node
    
    def find_by_title(self, title: str, user) -> List[Dict[str, Any]]:
        """
        Find nodes by title.
        
        Args:
            title: Title to search for
            user: User making the request
            
        Returns:
            List[Dict[str, Any]]: List of matching nodes
        """
        results = self.query_engine.execute_query(f'title = "{title}"')
        return [self.graph[key] for key in results if key in self.graph]
    
    def find_by_field(
        self, 
        field: str, 
        value: Any, 
        operator: str = "=", 
        user=None
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by field value.
        
        Args:
            field: Field name
            value: Field value
            operator: Comparison operator
            user: User making the request
            
        Returns:
            List[Dict[str, Any]]: List of matching nodes
        """
        # Handle string values
        if isinstance(value, str):
            value = f'"{value}"'
        
        results = self.query_engine.execute_query(f'{field} {operator} {value}')
        return [self.graph[key] for key in results if key in self.graph]
    
    def find_by_tag(self, tag: str, user=None) -> List[Dict[str, Any]]:
        """
        Find nodes by tag.
        
        Args:
            tag: Tag to search for
            user: User making the request
            
        Returns:
            List[Dict[str, Any]]: List of matching nodes
        """
        results = self.query_engine.execute_query(f'tags contains "{tag}"')
        return [self.graph[key] for key in results if key in self.graph]
    
    def find_by_genre(self, genre: str, user=None) -> List[Dict[str, Any]]:
        """
        Find nodes by genre.
        
        Args:
            genre: Genre to search for
            user: User making the request
            
        Returns:
            List[Dict[str, Any]]: List of matching nodes
        """
        results = self.query_engine.execute_query(f'genres contains "{genre}"')
        return [self.graph[key] for key in results if key in self.graph]
    
    def find_by_year_range(
        self, 
        min_year: int, 
        max_year: int, 
        user=None
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by year range.
        
        Args:
            min_year: Minimum year
            max_year: Maximum year
            user: User making the request
            
        Returns:
            List[Dict[str, Any]]: List of matching nodes
        """
        results = self.query_engine.execute_query(
            f'year >= {min_year} AND year <= {max_year}'
        )
        return [self.graph[key] for key in results if key in self.graph]
    
    def find_by_rating(
        self, 
        min_rating: float, 
        user=None
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by minimum rating.
        
        Args:
            min_rating: Minimum rating
            user: User making the request
            
        Returns:
            List[Dict[str, Any]]: List of matching nodes
        """
        results = self.query_engine.execute_query(f'rating >= {min_rating}')
        return [self.graph[key] for key in results if key in self.graph]
    
    def find_by_text_search(
        self, 
        text: str, 
        fields: List[str] = ["title", "overview", "tagline"], 
        user=None
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by text search across multiple fields.
        
        Args:
            text: Text to search for
            fields: Fields to search in
            user: User making the request
            
        Returns:
            List[Dict[str, Any]]: List of matching nodes
        """
        query_parts = []
        for field in fields:
            query_parts.append(f'{field} contains "{text}"')
        
        query = " OR ".join(query_parts)
        results = self.query_engine.execute_query(query)
        return [self.graph[key] for key in results if key in self.graph]
    
    def find_by_similarity(
        self,
        text: str = None,
        node_id: str = None,
        embedding: List[float] = None,
        threshold: float = 0.7,
        limit: int = 10,
        user=None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find nodes by embedding similarity.
        
        Args:
            text: Text to generate embedding from
            node_id: ID of node to use as reference
            embedding: Embedding vector to compare against
            threshold: Minimum similarity threshold
            limit: Maximum number of results
            user: User making the request
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: List of (node, similarity) tuples
        """
        if embedding is None:
            if text is not None:
                # Generate embedding from text
                embedding, error = self.db.embedding_generator.generate_embedding(text)
                if error:
                    return []
            elif node_id is not None:
                # Get embedding from node
                node, error = self.db.get_node(node_id, user)
                if error or "embedding" not in node:
                    return []
                embedding = node["embedding"]
            else:
                return []
        
        # Find similar nodes
        results, error = self.db.search_by_embedding(
            embedding if isinstance(embedding, str) else embedding,
            threshold,
            limit,
            user
        )
        
        if error:
            return []
        
        # Return nodes with similarity scores
        return [(self.graph[key], score) for key, score in results if key in self.graph]
    
    def find_by_combined_criteria(
        self,
        genres: List[str] = None,
        tags: List[str] = None,
        year_range: Tuple[int, int] = None,
        min_rating: float = None,
        director: str = None,
        text_search: str = None,
        similar_to: str = None,
        similarity_threshold: float = 0.7,
        user=None
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by multiple criteria.
        
        Args:
            genres: List of genres to match
            tags: List of tags to match
            year_range: (min_year, max_year) tuple
            min_rating: Minimum rating
            director: Director name
            text_search: Text to search for
            similar_to: ID of node to find similar nodes to
            similarity_threshold: Minimum similarity threshold
            user: User making the request
            
        Returns:
            List[Dict[str, Any]]: List of matching nodes
        """
        query_parts = []
        
        # Add genre conditions
        if genres:
            genre_parts = []
            for genre in genres:
                genre_parts.append(f'genres contains "{genre}"')
            query_parts.append("(" + " OR ".join(genre_parts) + ")")
        
        # Add tag conditions
        if tags:
            tag_parts = []
            for tag in tags:
                tag_parts.append(f'tags contains "{tag}"')
            query_parts.append("(" + " OR ".join(tag_parts) + ")")
        
        # Add year range condition
        if year_range:
            min_year, max_year = year_range
            query_parts.append(f'year >= {min_year} AND year <= {max_year}')
        
        # Add rating condition
        if min_rating is not None:
            query_parts.append(f'rating >= {min_rating}')
        
        # Add director condition
        if director:
            query_parts.append(f'director = "{director}"')
        
        # Add text search condition
        if text_search:
            text_parts = []
            for field in ["title", "overview", "tagline"]:
                text_parts.append(f'{field} contains "{text_search}"')
            query_parts.append("(" + " OR ".join(text_parts) + ")")
        
        # Build and execute query
        if query_parts:
            query = " AND ".join(query_parts)
            results = self.query_engine.execute_query(query)
            nodes = [self.graph[key] for key in results if key in self.graph]
        else:
            # If no criteria specified, return all nodes
            nodes = list(self.graph.values())
        
        # Filter by similarity if requested
        if similar_to:
            similar_results = self.find_by_similarity(
                node_id=similar_to,
                threshold=similarity_threshold,
                limit=len(self.graph),
                user=user
            )
            
            if similar_results:
                # Get IDs of similar nodes
                similar_ids = {node["id"] for node, _ in similar_results}
                
                # Filter nodes by similarity
                nodes = [node for node in nodes if node["id"] in similar_ids]
        
        return nodes
    
    def find_connected_nodes(
        self,
        node_id: str,
        max_depth: int = 2,
        filter_query: str = None,
        user=None
    ) -> List[Dict[str, Any]]:
        """
        Find nodes connected to a given node.
        
        Args:
            node_id: ID of the starting node
            max_depth: Maximum traversal depth
            filter_query: Optional query to filter traversed nodes
            user: User making the request
            
        Returns:
            List[Dict[str, Any]]: List of connected nodes
        """
        subgraph = self.query_engine.traverse_graph(node_id, max_depth, filter_query)
        return list(subgraph.values())
    
    def find_path_between(
        self,
        start_id: str,
        end_id: str,
        user=None
    ) -> List[Dict[str, Any]]:
        """
        Find a path between two nodes.
        
        Args:
            start_id: ID of the starting node
            end_id: ID of the ending node
            user: User making the request
            
        Returns:
            List[Dict[str, Any]]: List of nodes in the path
        """
        from src.models.graph_ops import a_star
        
        path = a_star(self.graph, start_id, end_id)
        
        if path:
            return [self.graph[node_id] for node_id in path if node_id in self.graph]
        
        return []
    
    def count_by_query(self, query: str, user=None) -> int:
        """
        Count nodes matching a query.
        
        Args:
            query: Query string
            user: User making the request
            
        Returns:
            int: Number of matching nodes
        """
        results = self.query_engine.execute_query(query)
        return len(results)
    
    def group_by_field(
        self,
        field: str,
        user=None
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """
        Group nodes by field value.
        
        Args:
            field: Field to group by
            user: User making the request
            
        Returns:
            Dict[Any, List[Dict[str, Any]]]: Dictionary of field value -> nodes
        """
        result = {}
        
        for node in self.graph.values():
            # Handle nested fields with dot notation
            value = node
            for part in field.split('.'):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            
            if value is not None:
                # Convert non-hashable values to strings
                if isinstance(value, (list, dict)):
                    value = str(value)
                
                if value not in result:
                    result[value] = []
                
                result[value].append(node)
        
        return result
    
    def aggregate_by_field(
        self,
        field: str,
        aggregation: str = "count",
        user=None
    ) -> Dict[Any, Union[int, float]]:
        """
        Aggregate nodes by field value.
        
        Args:
            field: Field to group by
            aggregation: Aggregation function (count, avg, sum, min, max)
            user: User making the request
            
        Returns:
            Dict[Any, Union[int, float]]: Dictionary of field value -> aggregated value
        """
        grouped = self.group_by_field(field, user)
        result = {}
        
        for value, nodes in grouped.items():
            if aggregation == "count":
                result[value] = len(nodes)
            elif aggregation in ("avg", "sum", "min", "max"):
                # These aggregations require a numeric field
                # For simplicity, we'll use the "rating" field
                ratings = [node.get("rating", 0) for node in nodes if "rating" in node]
                
                if ratings:
                    if aggregation == "avg":
                        result[value] = sum(ratings) / len(ratings)
                    elif aggregation == "sum":
                        result[value] = sum(ratings)
                    elif aggregation == "min":
                        result[value] = min(ratings)
                    elif aggregation == "max":
                        result[value] = max(ratings)
                else:
                    result[value] = 0
        
        return result

