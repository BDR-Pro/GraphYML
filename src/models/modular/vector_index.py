"""
Vector index implementation for modular indexing system.
"""
from typing import Dict, List, Any, Optional, Set, Tuple

from .base_index import BaseIndex
from ..embeddings import embedding_similarity


class VectorIndex(BaseIndex):
    """Vector index for semantic search."""
    
    def __init__(self, name: str, field: str):
        """
        Initialize the vector index.
        
        Args:
            name: Name of the index
            field: Field to index
        """
        super().__init__(name, field)
        self.index = {}  # Map of node_id -> embedding
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from a graph.
        
        Args:
            graph: Graph to build index from
        """
        # Clear the index
        self.index = {}
        
        # Build the index
        for node_id, node_data in graph.items():
            self.update(node_id, node_data)
        
        self.is_built = True
    
    def search(self, query_vector: List[float], threshold: float = 0.7, limit: int = 10, **kwargs) -> List[Tuple[str, float]]:
        """
        Search the index using vector similarity.
        
        Args:
            query_vector: Vector to search for
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of (node_id, similarity) tuples
        """
        if not self.is_built:
            return []
        
        results = []
        
        for node_id, embedding in self.index.items():
            similarity = embedding_similarity(query_vector, embedding)
            if similarity >= threshold:
                results.append((node_id, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def update(self, node_id: str, node_data: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update the index with a new or modified node.
        
        Args:
            node_id: ID of the node to update
            node_data: Node data
            is_delete: Whether this is a delete operation
        """
        # Handle delete
        if is_delete:
            if node_id in self.index:
                del self.index[node_id]
            return
        
        # Get field value
        field_value = self._get_field_value(node_data)
        
        # Skip if field not found or not a list
        if field_value is None or not isinstance(field_value, list):
            return
        
        # Add to index
        self.index[node_id] = field_value

