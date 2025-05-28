"""
Base index implementation for modular indexing system.
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import os
import json
import pickle
from abc import ABC, abstractmethod


class BaseIndex(ABC):
    """Base class for all index types."""
    
    def __init__(self, name: str, field: str):
        """
        Initialize the index.
        
        Args:
            name: Name of the index
            field: Field to index
        """
        self.name = name
        self.field = field
        self.is_built = False
    
    @abstractmethod
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from a graph.
        
        Args:
            graph: Graph to build index from
        """
        pass
    
    @abstractmethod
    def search(self, query: Any, **kwargs) -> List[str]:
        """
        Search the index.
        
        Args:
            query: Query to search for
            **kwargs: Additional search parameters
            
        Returns:
            List of node IDs matching the query
        """
        pass
    
    @abstractmethod
    def update(self, node_id: str, node_data: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update the index with a new or modified node.
        
        Args:
            node_id: ID of the node to update
            node_data: Node data
            is_delete: Whether this is a delete operation
        """
        pass
    
    def save(self, path: str) -> bool:
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save metadata
            metadata_path = f"{path}.meta"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "name": self.name,
                    "field": self.field,
                    "type": self.__class__.__name__.lower().replace("index", "")
                }, f)
            
            # Save index
            with open(path, 'wb') as f:
                pickle.dump({
                    "name": self.name,
                    "field": self.field,
                    "is_built": self.is_built,
                    "index": self._get_serializable_index()
                }, f)
            
            return True
        except Exception as e:
            print(f"Error saving index: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                return False
            
            # Load index
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
                # Update attributes
                self.name = data['name']
                self.field = data['field']
                self.is_built = data['is_built']
                self._set_index_from_serialized(data['index'])
            
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def _get_serializable_index(self) -> Any:
        """
        Get a serializable version of the index.
        
        Returns:
            Any: Serializable index
        """
        # By default, return the index attribute
        # Subclasses can override this if needed
        return getattr(self, 'index', None)
    
    def _set_index_from_serialized(self, serialized_index: Any) -> None:
        """
        Set the index from a serialized version.
        
        Args:
            serialized_index: Serialized index
        """
        # By default, set the index attribute
        # Subclasses can override this if needed
        setattr(self, 'index', serialized_index)
    
    def _get_field_value(self, node: Dict[str, Any]) -> Optional[Any]:
        """
        Get the field value from a node.
        
        Args:
            node: Node to get field value from
            
        Returns:
            Optional[Any]: Field value or None if not found
        """
        return node.get(self.field, None)

