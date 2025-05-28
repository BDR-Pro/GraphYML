"""
BTree index implementation for modular indexing system.
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import bisect

from .base_index import BaseIndex
from src.utils.index_utils import make_hashable, normalize_query


class BTreeIndex(BaseIndex):
    """BTree index for prefix and range searches."""
    
    def __init__(self, name: str, field: str):
        """
        Initialize the BTree index.
        
        Args:
            name: Name of the index
            field: Field to index
        """
        super().__init__(name, field)
        self.index = {}  # Map of node_id -> field_value
        self.sorted_keys = []  # Sorted list of field values
        self.value_to_nodes = {}  # Map of field_value -> list of node_ids
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from a graph.
        
        Args:
            graph: Graph to build index from
        """
        # Clear the index
        self.index = {}
        self.sorted_keys = []
        self.value_to_nodes = {}
        
        # Build the index
        for node_id, node_data in graph.items():
            self.update(node_id, node_data)
        
        # Sort the keys
        self.sorted_keys = sorted(self.value_to_nodes.keys())
        
        self.is_built = True
    
    def search(self, query: str, prefix: bool = False, **kwargs) -> List[str]:
        """
        Search the index.
        
        Args:
            query: Query to search for
            prefix: Whether to do a prefix search
            **kwargs: Additional search parameters
            
        Returns:
            List of node IDs matching the query
        """
        if not self.is_built:
            return []
        
        # Normalize query for consistent comparison
        normalized_query = normalize_query(query)
        
        # Make query hashable for dictionary lookup
        hashable_query = make_hashable(normalized_query)
        
        if prefix:
            # Find all keys that start with the query
            results = []
            for key in self.sorted_keys:
                if isinstance(key, str) and isinstance(hashable_query, str) and key.startswith(hashable_query):
                    results.extend(self.value_to_nodes.get(key, []))
            return results
        else:
            # Exact match - try both original and normalized query
            results = self.value_to_nodes.get(hashable_query, [])
            if not results and isinstance(query, str):
                # Try case-insensitive match for strings
                for key, nodes in self.value_to_nodes.items():
                    if isinstance(key, str) and key.lower() == query.lower():
                        results.extend(nodes)
            return results
    
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
                # Remove from value_to_nodes
                value = self.index[node_id]
                if value in self.value_to_nodes and node_id in self.value_to_nodes[value]:
                    self.value_to_nodes[value].remove(node_id)
                    if not self.value_to_nodes[value]:
                        del self.value_to_nodes[value]
                        # Remove from sorted_keys
                        if value in self.sorted_keys:
                            self.sorted_keys.remove(value)
                
                # Remove from index
                del self.index[node_id]
            return
        
        # Get field value
        field_value = self._get_field_value(node_data)
        
        # Skip if field not found
        if field_value is None:
            return
        
        # Make field value hashable for dictionary keys
        hashable_value = make_hashable(field_value)
        
        # Add to index
        self.index[node_id] = hashable_value
        
        # Add to value_to_nodes
        if hashable_value not in self.value_to_nodes:
            self.value_to_nodes[hashable_value] = []
            # Add to sorted_keys - will be properly sorted in build()
            if hashable_value not in self.sorted_keys:
                self.sorted_keys.append(hashable_value)
        
        if node_id not in self.value_to_nodes[hashable_value]:
            self.value_to_nodes[hashable_value].append(node_id)
    
    def _get_serializable_index(self) -> Dict[str, Any]:
        """
        Get a serializable version of the index.
        
        Returns:
            Dict[str, Any]: Serializable index
        """
        return {
            "index": self.index,
            "sorted_keys": self.sorted_keys,
            "value_to_nodes": self.value_to_nodes
        }
    
    def _set_index_from_serialized(self, serialized_index: Dict[str, Any]) -> None:
        """
        Set the index from a serialized version.
        
        Args:
            serialized_index: Serialized index
        """
        self.index = serialized_index.get("index", {})
        self.sorted_keys = serialized_index.get("sorted_keys", [])
        self.value_to_nodes = serialized_index.get("value_to_nodes", {})
