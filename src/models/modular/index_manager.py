"""
Index manager implementation for modular indexing system.
"""
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import os
import json
from enum import Enum
import re

from .base_index import BaseIndex
from .hash_index import HashIndex
from .btree_index import BTreeIndex
from .fulltext_index import FullTextIndex
from .vector_index import VectorIndex


class IndexType(Enum):
    """Index types."""
    HASH = "hash"
    BTREE = "btree"
    FULLTEXT = "fulltext"
    VECTOR = "vector"


class IndexManager:
    """Manager for multiple indexes."""
    
    def __init__(self, graph=None, config=None, index_dir: Optional[str] = None, **kwargs):
        """
        Initialize the index manager.
        
        Args:
            graph: Optional graph data
            config: Optional configuration
            index_dir: Directory to store indexes
            **kwargs: Additional parameters
        """
        self.indexes = {}  # Map of name -> index
        self.index_dir = index_dir
        self.graph = graph
        self.config = config or {}
    
    def create_index(self, name: str, field: str, index_type: Union[IndexType, str]) -> BaseIndex:
        """
        Create a new index.
        
        Args:
            name: Name of the index
            field: Field to index
            index_type: Type of index to create
            
        Returns:
            BaseIndex: Created index
            
        Raises:
            ValueError: If index type is invalid or name is invalid
        """
        # Validate name
        if not name or not isinstance(name, str):
            raise ValueError(f"Invalid index name: {name}. Name must be a non-empty string.")
        
        # Check if name contains only valid characters
        if not re.match(r'^[a-zA-Z0-9_]+$', name):
            raise ValueError(f"Invalid index name: {name}. Name must contain only letters, numbers, and underscores.")
        
        # Check if index already exists
        if name in self.indexes:
            raise ValueError(f"Index already exists: {name}")
        
        # Convert string to enum
        if isinstance(index_type, str):
            try:
                index_type = IndexType(index_type)
            except ValueError:
                valid_types = [t.value for t in IndexType]
                raise ValueError(f"Invalid index type: {index_type}. Valid types are: {valid_types}")
        
        # Validate index_type is an IndexType
        if not isinstance(index_type, IndexType):
            valid_types = [t.value for t in IndexType]
            raise ValueError(f"Invalid index type: {index_type}. Valid types are: {valid_types}")
        
        # Create the index
        if index_type == IndexType.HASH:
            index = HashIndex(name, field)
        elif index_type == IndexType.BTREE:
            index = BTreeIndex(name, field)
        elif index_type == IndexType.FULLTEXT:
            index = FullTextIndex(name, field)
        elif index_type == IndexType.VECTOR:
            index = VectorIndex(name, field)
        else:
            # This should never happen due to the validation above
            raise ValueError(f"Invalid index type: {index_type}")
        
        # Add to manager
        self.indexes[name] = index
        
        return index
    
    def get_index(self, name: str) -> BaseIndex:
        """
        Get an index by name.
        
        Args:
            name: Name of the index
            
        Returns:
            BaseIndex: Index
            
        Raises:
            ValueError: If index not found
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"Invalid index name: {name}. Name must be a non-empty string.")
        
        if name not in self.indexes:
            raise ValueError(f"Index not found: {name}")
        
        return self.indexes[name]
    
    def drop_index(self, name: str) -> None:
        """
        Drop an index.
        
        Args:
            name: Name of the index
            
        Raises:
            ValueError: If index not found
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"Invalid index name: {name}. Name must be a non-empty string.")
        
        if name not in self.indexes:
            raise ValueError(f"Index not found: {name}")
        
        del self.indexes[name]
    
    def rebuild_indexes(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Rebuild all indexes.
        
        Args:
            graph: Graph to build indexes from
        """
        for index in self.indexes.values():
            index.build(graph)
    
    def update_indexes(self, node_id: str, node_data: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update all indexes with a new or modified node.
        
        Args:
            node_id: ID of the node to update
            node_data: Node data
            is_delete: Whether this is a delete operation
        """
        for index in self.indexes.values():
            index.update(node_id, node_data, is_delete)
    
    def search(self, name: str, query: Any, **kwargs) -> List[Any]:
        """
        Search an index.
        
        Args:
            name: Name of the index
            query: Query to search for
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        if not name or not isinstance(name, str):
            return []
        
        if name not in self.indexes:
            return []
        
        return self.indexes[name].search(query, **kwargs)
    
    def save_indexes(self) -> bool:
        """
        Save all indexes to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.index_dir:
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Save each index
        success = True
        
        for name, index in self.indexes.items():
            index_path = os.path.join(self.index_dir, f"{name}.idx")
            if not index.save(index_path):
                success = False
        
        return success
    
    def load_indexes(self) -> bool:
        """
        Load all indexes from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.index_dir or not os.path.exists(self.index_dir):
            return False
        
        # Load each index
        success = True
        
        for filename in os.listdir(self.index_dir):
            if filename.endswith(".idx"):
                index_path = os.path.join(self.index_dir, filename)
                metadata_path = f"{index_path}.meta"
                name = filename[:-4]  # Remove .idx extension
                
                # Load metadata
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            
                            # Create the appropriate index type
                            index_type = metadata.get("type", "hash")
                            field = metadata.get("field", "")
                            
                            try:
                                index = self.create_index(name, field, index_type)
                                
                                # Load index
                                if not index.load(index_path):
                                    success = False
                            except ValueError as e:
                                print(f"Error loading index {name}: {e}")
                                success = False
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Error loading index metadata {metadata_path}: {e}")
                        success = False
                else:
                    print(f"Metadata file not found: {metadata_path}")
                    success = False
        
        return success

