"""
Indexing support for GraphYML.
Provides indexes for efficient querying of graph nodes.
"""
import os
import json
import time
import threading
from enum import Enum
from typing import Dict, List, Any, Set, Optional, Tuple, Callable, Union
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree


class IndexType(Enum):
    """Types of indexes supported."""
    HASH = "hash"  # Exact match lookups
    BTREE = "btree"  # Range queries
    FULLTEXT = "fulltext"  # Text search
    VECTOR = "vector"  # Embedding similarity


class Index:
    """Base class for all index types."""
    
    def __init__(
        self, 
        name: str, 
        field_path: str, 
        index_type: IndexType
    ):
        """
        Initialize an index.
        
        Args:
            name: Index name
            field_path: Path to the field to index (dot notation)
            index_type: Type of index
        """
        self.name = name
        self.field_path = field_path
        self.index_type = index_type
        self.last_updated = 0
        self.is_dirty = True
    
    def get_field_value(self, node: Dict[str, Any]) -> Any:
        """
        Get the value of the indexed field from a node.
        
        Args:
            node: Node to extract value from
            
        Returns:
            Any: Field value or None if not found
        """
        parts = self.field_path.split('.')
        value = node
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def build(self, graph: Dict[str, Dict[str, Any]]):
        """
        Build the index from the graph.
        
        Args:
            graph: Graph to index
        """
        raise NotImplementedError("Subclasses must implement build()")
    
    def search(self, query: Any) -> List[str]:
        """
        Search the index.
        
        Args:
            query: Search query
            
        Returns:
            List[str]: List of matching node keys
        """
        raise NotImplementedError("Subclasses must implement search()")
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False):
        """
        Update the index for a single node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def save(self, path: Path):
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
        """
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self, path: Path) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: Success flag
        """
        raise NotImplementedError("Subclasses must implement load()")


class HashIndex(Index):
    """Hash index for exact match lookups."""
    
    def __init__(self, name: str, field_path: str):
        """
        Initialize a hash index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
        """
        super().__init__(name, field_path, IndexType.HASH)
        self.index = {}  # value -> set of node keys
    
    def build(self, graph: Dict[str, Dict[str, Any]]):
        """
        Build the index from the graph.
        
        Args:
            graph: Graph to index
        """
        self.index = {}
        
        for key, node in graph.items():
            value = self.get_field_value(node)
            
            if value is not None:
                # Handle list values
                if isinstance(value, list):
                    for item in value:
                        self._add_to_index(item, key)
                else:
                    self._add_to_index(value, key)
        
        self.last_updated = time.time()
        self.is_dirty = False
    
    def _add_to_index(self, value: Any, key: str):
        """
        Add a value to the index.
        
        Args:
            value: Value to index
            key: Node key
        """
        # Convert value to hashable type
        if isinstance(value, (list, dict)):
            value = str(value)
        
        if value not in self.index:
            self.index[value] = set()
        
        self.index[value].add(key)
    
    def search(self, query: Any) -> List[str]:
        """
        Search the index for exact matches.
        
        Args:
            query: Value to search for
            
        Returns:
            List[str]: List of matching node keys
        """
        # Convert query to hashable type
        if isinstance(query, (list, dict)):
            query = str(query)
        
        if query in self.index:
            return list(self.index[query])
        
        return []
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False):
        """
        Update the index for a single node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        if is_delete:
            # Remove from all value sets
            for value_set in self.index.values():
                if key in value_set:
                    value_set.remove(key)
            
            # Clean up empty sets
            self.index = {k: v for k, v in self.index.items() if v}
        else:
            value = self.get_field_value(node)
            
            if value is not None:
                # Handle list values
                if isinstance(value, list):
                    for item in value:
                        self._add_to_index(item, key)
                else:
                    self._add_to_index(value, key)
        
        self.is_dirty = True
    
    def save(self, path: Path):
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
        """
        # Convert sets to lists for JSON serialization
        serializable = {
            "name": self.name,
            "field_path": self.field_path,
            "type": self.index_type.value,
            "last_updated": self.last_updated,
            "index": {k: list(v) for k, v in self.index.items()}
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        
        self.is_dirty = False
    
    def load(self, path: Path) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: Success flag
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.name = data["name"]
            self.field_path = data["field_path"]
            self.last_updated = data["last_updated"]
            
            # Convert lists back to sets
            self.index = {k: set(v) for k, v in data["index"].items()}
            
            self.is_dirty = False
            return True
        
        except Exception as e:
            print(f"Error loading index: {e}")
            return False


class BTreeIndex(Index):
    """B-tree index for range queries."""
    
    def __init__(self, name: str, field_path: str):
        """
        Initialize a B-tree index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
        """
        super().__init__(name, field_path, IndexType.BTREE)
        self.values = []  # Sorted list of values
        self.key_map = {}  # value -> set of node keys
    
    def build(self, graph: Dict[str, Dict[str, Any]]):
        """
        Build the index from the graph.
        
        Args:
            graph: Graph to index
        """
        self.key_map = {}
        
        for key, node in graph.items():
            value = self.get_field_value(node)
            
            if value is not None and isinstance(value, (int, float)):
                if value not in self.key_map:
                    self.key_map[value] = set()
                
                self.key_map[value].add(key)
        
        # Sort values for binary search
        self.values = sorted(self.key_map.keys())
        
        self.last_updated = time.time()
        self.is_dirty = False
    
    def search(
        self, 
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        include_min: bool = True,
        include_max: bool = True
    ) -> List[str]:
        """
        Search the index for values in a range.
        
        Args:
            min_value: Minimum value (inclusive if include_min is True)
            max_value: Maximum value (inclusive if include_max is True)
            include_min: Whether to include the minimum value
            include_max: Whether to include the maximum value
            
        Returns:
            List[str]: List of matching node keys
        """
        if not self.values:
            return []
        
        # Find range of matching values
        matching_values = []
        
        for value in self.values:
            if min_value is not None:
                if include_min and value < min_value:
                    continue
                elif not include_min and value <= min_value:
                    continue
            
            if max_value is not None:
                if include_max and value > max_value:
                    continue
                elif not include_max and value >= max_value:
                    continue
            
            matching_values.append(value)
        
        # Collect matching keys
        result = set()
        for value in matching_values:
            result.update(self.key_map[value])
        
        return list(result)
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False):
        """
        Update the index for a single node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        if is_delete:
            # Remove from all value sets
            for value_set in self.key_map.values():
                if key in value_set:
                    value_set.remove(key)
            
            # Clean up empty sets
            self.key_map = {k: v for k, v in self.key_map.items() if v}
            
            # Update sorted values
            self.values = sorted(self.key_map.keys())
        else:
            value = self.get_field_value(node)
            
            if value is not None and isinstance(value, (int, float)):
                if value not in self.key_map:
                    self.key_map[value] = set()
                
                self.key_map[value].add(key)
                
                # Update sorted values if needed
                if value not in self.values:
                    self.values.append(value)
                    self.values.sort()
        
        self.is_dirty = True
    
    def save(self, path: Path):
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
        """
        # Convert sets to lists for JSON serialization
        serializable = {
            "name": self.name,
            "field_path": self.field_path,
            "type": self.index_type.value,
            "last_updated": self.last_updated,
            "values": self.values,
            "key_map": {str(k): list(v) for k, v in self.key_map.items()}
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        
        self.is_dirty = False
    
    def load(self, path: Path) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: Success flag
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.name = data["name"]
            self.field_path = data["field_path"]
            self.last_updated = data["last_updated"]
            self.values = data["values"]
            
            # Convert lists back to sets and string keys to numeric
            self.key_map = {}
            for k, v in data["key_map"].items():
                try:
                    # Try to convert back to number
                    if '.' in k:
                        k = float(k)
                    else:
                        k = int(k)
                except ValueError:
                    pass
                
                self.key_map[k] = set(v)
            
            self.is_dirty = False
            return True
        
        except Exception as e:
            print(f"Error loading index: {e}")
            return False


class FullTextIndex(Index):
    """Full-text index for text search."""
    
    def __init__(self, name: str, field_path: str):
        """
        Initialize a full-text index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
        """
        super().__init__(name, field_path, IndexType.FULLTEXT)
        self.token_map = {}  # token -> set of node keys
    
    def build(self, graph: Dict[str, Dict[str, Any]]):
        """
        Build the index from the graph.
        
        Args:
            graph: Graph to index
        """
        self.token_map = {}
        
        for key, node in graph.items():
            value = self.get_field_value(node)
            
            if value is not None and isinstance(value, str):
                self._index_text(value, key)
        
        self.last_updated = time.time()
        self.is_dirty = False
    
    def _index_text(self, text: str, key: str):
        """
        Index a text string.
        
        Args:
            text: Text to index
            key: Node key
        """
        # Tokenize and normalize
        tokens = self._tokenize(text)
        
        # Add to token map
        for token in tokens:
            if token not in self.token_map:
                self.token_map[token] = set()
            
            self.token_map[token].add(key)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        # Simple tokenization - split on whitespace and punctuation
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace punctuation with spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
            'at', 'from', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'to', 'of', 'in', 'on', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'this', 'that', 'these', 'those', 'it', 'its', 'it\'s'
        }
        
        tokens = [t for t in tokens if t not in stop_words]
        
        return tokens
    
    def search(self, query: str) -> List[str]:
        """
        Search the index for text matches.
        
        Args:
            query: Text to search for
            
        Returns:
            List[str]: List of matching node keys
        """
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Find matches for each token
        matches = []
        for token in query_tokens:
            if token in self.token_map:
                matches.append(self.token_map[token])
        
        if not matches:
            return []
        
        # Intersect results (AND semantics)
        result = matches[0]
        for match_set in matches[1:]:
            result = result.intersection(match_set)
        
        return list(result)
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False):
        """
        Update the index for a single node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        if is_delete:
            # Remove from all token sets
            for token_set in self.token_map.values():
                if key in token_set:
                    token_set.remove(key)
            
            # Clean up empty sets
            self.token_map = {k: v for k, v in self.token_map.items() if v}
        else:
            value = self.get_field_value(node)
            
            if value is not None and isinstance(value, str):
                self._index_text(value, key)
        
        self.is_dirty = True
    
    def save(self, path: Path):
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
        """
        # Convert sets to lists for JSON serialization
        serializable = {
            "name": self.name,
            "field_path": self.field_path,
            "type": self.index_type.value,
            "last_updated": self.last_updated,
            "token_map": {k: list(v) for k, v in self.token_map.items()}
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        
        self.is_dirty = False
    
    def load(self, path: Path) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: Success flag
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.name = data["name"]
            self.field_path = data["field_path"]
            self.last_updated = data["last_updated"]
            
            # Convert lists back to sets
            self.token_map = {k: set(v) for k, v in data["token_map"].items()}
            
            self.is_dirty = False
            return True
        
        except Exception as e:
            print(f"Error loading index: {e}")
            return False


class VectorIndex(Index):
    """Vector index for embedding similarity search."""
    
    def __init__(self, name: str, field_path: str):
        """
        Initialize a vector index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
        """
        super().__init__(name, field_path, IndexType.VECTOR)
        self.keys = []  # List of node keys
        self.vectors = []  # List of embedding vectors
        self.tree = None  # BallTree for efficient similarity search
    
    def build(self, graph: Dict[str, Dict[str, Any]]):
        """
        Build the index from the graph.
        
        Args:
            graph: Graph to index
        """
        self.keys = []
        self.vectors = []
        
        for key, node in graph.items():
            value = self.get_field_value(node)
            
            if value is not None and isinstance(value, list):
                try:
                    # Convert to numpy array
                    vector = np.array(value, dtype=np.float32)
                    
                    self.keys.append(key)
                    self.vectors.append(vector)
                except (ValueError, TypeError):
                    # Skip invalid vectors
                    pass
        
        if self.vectors:
            # Build BallTree for efficient similarity search
            self.tree = BallTree(np.array(self.vectors), leaf_size=40)
        else:
            self.tree = None
        
        self.last_updated = time.time()
        self.is_dirty = False
    
    def search(
        self, 
        query_vector: List[float], 
        k: int = 10, 
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Search the index for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            threshold: Optional similarity threshold (0-1)
            
        Returns:
            List[Tuple[str, float]]: List of (node_key, similarity) tuples
        """
        if not self.tree or not self.keys:
            return []
        
        try:
            # Convert to numpy array
            query = np.array([query_vector], dtype=np.float32)
            
            # Find k nearest neighbors
            distances, indices = self.tree.query(query, k=min(k, len(self.keys)))
            
            # Convert distances to similarities (cosine distance is 1 - cosine similarity)
            similarities = 1 - distances[0]
            
            # Apply threshold if provided
            results = []
            for i, similarity in enumerate(similarities):
                if threshold is None or similarity >= threshold:
                    results.append((self.keys[indices[0][i]], float(similarity)))
            
            return results
        
        except Exception as e:
            print(f"Error searching vector index: {e}")
            return []
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False):
        """
        Update the index for a single node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        # Vector indexes require a full rebuild for updates
        # This is a limitation of the BallTree data structure
        self.is_dirty = True
    
    def save(self, path: Path):
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
        """
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "name": self.name,
            "field_path": self.field_path,
            "type": self.index_type.value,
            "last_updated": self.last_updated,
            "keys": self.keys
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Save vectors to a separate file
        if self.vectors:
            np.save(str(path) + ".npy", np.array(self.vectors))
        
        self.is_dirty = False
    
    def load(self, path: Path) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: Success flag
        """
        try:
            # Load metadata
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.name = data["name"]
            self.field_path = data["field_path"]
            self.last_updated = data["last_updated"]
            self.keys = data["keys"]
            
            # Load vectors
            vectors_path = str(path) + ".npy"
            if os.path.exists(vectors_path):
                self.vectors = np.load(vectors_path)
                
                # Rebuild BallTree
                self.tree = BallTree(self.vectors, leaf_size=40)
            else:
                self.vectors = []
                self.tree = None
            
            self.is_dirty = False
            return True
        
        except Exception as e:
            print(f"Error loading index: {e}")
            return False


class IndexManager:
    """Manages database indexes."""
    
    def __init__(self, graph: Dict[str, Dict[str, Any]], index_dir: str):
        """
        Initialize the index manager.
        
        Args:
            graph: Graph to index
            index_dir: Directory to store indexes
        """
        self.graph = graph
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Indexes by name
        self.indexes = {}
        
        # Load existing indexes
        self._load_indexes()
    
    def create_index(
        self, 
        name: str, 
        field_path: str, 
        index_type: IndexType
    ) -> Index:
        """
        Create a new index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
            index_type: Type of index
            
        Returns:
            Index: New index object
        """
        with self.lock:
            if name in self.indexes:
                raise ValueError(f"Index {name} already exists")
            
            # Create index of the appropriate type
            if index_type == IndexType.HASH:
                index = HashIndex(name, field_path)
            elif index_type == IndexType.BTREE:
                index = BTreeIndex(name, field_path)
            elif index_type == IndexType.FULLTEXT:
                index = FullTextIndex(name, field_path)
            elif index_type == IndexType.VECTOR:
                index = VectorIndex(name, field_path)
            else:
                raise ValueError(f"Unknown index type: {index_type}")
            
            # Build the index
            index.build(self.graph)
            
            # Save the index
            index_path = self.index_dir / f"{name}.json"
            index.save(index_path)
            
            # Add to indexes
            self.indexes[name] = index
            
            return index
    
    def get_index(self, name: str) -> Optional[Index]:
        """
        Get an index by name.
        
        Args:
            name: Index name
            
        Returns:
            Optional[Index]: Index object or None
        """
        with self.lock:
            return self.indexes.get(name)
    
    def drop_index(self, name: str) -> bool:
        """
        Drop an index.
        
        Args:
            name: Index name
            
        Returns:
            bool: Success flag
        """
        with self.lock:
            if name not in self.indexes:
                return False
            
            # Remove from indexes
            del self.indexes[name]
            
            # Remove from disk
            index_path = self.index_dir / f"{name}.json"
            if index_path.exists():
                index_path.unlink()
            
            # Remove vector data if it exists
            vector_path = self.index_dir / f"{name}.json.npy"
            if vector_path.exists():
                vector_path.unlink()
            
            return True
    
    def update_indexes(self, key: str, node: Dict[str, Any], is_delete: bool = False):
        """
        Update all indexes for a node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        with self.lock:
            for index in self.indexes.values():
                index.update(key, node, is_delete)
    
    def rebuild_indexes(self):
        """Rebuild all indexes."""
        with self.lock:
            for name, index in self.indexes.items():
                index.build(self.graph)
                
                # Save the index
                index_path = self.index_dir / f"{name}.json"
                index.save(index_path)
    
    def save_indexes(self):
        """Save all dirty indexes to disk."""
        with self.lock:
            for name, index in self.indexes.items():
                if index.is_dirty:
                    index_path = self.index_dir / f"{name}.json"
                    index.save(index_path)
    
    def _load_indexes(self):
        """Load indexes from disk."""
        with self.lock:
            for index_path in self.index_dir.glob("*.json"):
                # Skip vector data files
                if index_path.name.endswith(".json.npy"):
                    continue
                
                try:
                    # Load index metadata
                    with open(index_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    name = metadata["name"]
                    field_path = metadata["field_path"]
                    index_type = IndexType(metadata["type"])
                    
                    # Create index of the appropriate type
                    if index_type == IndexType.HASH:
                        index = HashIndex(name, field_path)
                    elif index_type == IndexType.BTREE:
                        index = BTreeIndex(name, field_path)
                    elif index_type == IndexType.FULLTEXT:
                        index = FullTextIndex(name, field_path)
                    elif index_type == IndexType.VECTOR:
                        index = VectorIndex(name, field_path)
                    else:
                        continue
                    
                    # Load the index
                    if index.load(index_path):
                        self.indexes[name] = index
                
                except Exception as e:
                    print(f"Error loading index {index_path.name}: {e}")
    
    def get_indexes(self) -> List[Dict[str, Any]]:
        """
        Get information about all indexes.
        
        Returns:
            List[Dict[str, Any]]: List of index information
        """
        with self.lock:
            return [
                {
                    "name": index.name,
                    "field_path": index.field_path,
                    "type": index.index_type.value,
                    "last_updated": index.last_updated
                }
                for index in self.indexes.values()
            ]

