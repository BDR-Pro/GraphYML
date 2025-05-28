"""
Hash index implementation for modular indexing system.
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from .base_index import BaseIndex


class HashIndex(BaseIndex):
    """Hash index for exact match searches."""
    
    def __init__(self, name: str, field: str):
        """
        Initialize the hash index.
        
        Args:
            name: Name of the index
            field: Field to index
        """
        super().__init__(name, field)
        self.index = defaultdict(list)
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from a graph.
        
        Args:
            graph: Graph to build index from
        """
        # Clear the index
        self.index = defaultdict(list)
        
        # Build the index
        for node_id, node_data in graph.items():
            self.update(node_id, node_data)
        
        self.is_built = True
    
    def search(self, query: Any, **kwargs) -> List[str]:
        """
        Search the index.
        
        Args:
            query: Query to search for
            **kwargs: Additional search parameters
            
        Returns:
            List of node IDs matching the query
        """
        if not self.is_built:
            return []
        
        # Get nodes with the query value
        return self.index.get(query, [])
    
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
            # Remove node from all values
            for values in self.index.values():
                if node_id in values:
                    values.remove(node_id)
            return
        
        # Get field value
        field_value = self._get_field_value(node_data)
        
        # Skip if field not found
        if field_value is None:
            return
        
        # Handle list values
        if isinstance(field_value, list):
            # Add node to each value
            for value in field_value:
                if node_id not in self.index[value]:
                    self.index[value].append(node_id)
        else:
            # Add node to value
            if node_id not in self.index[field_value]:
                self.index[field_value].append(node_id)

