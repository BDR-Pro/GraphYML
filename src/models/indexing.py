"""
Indexing system for GraphYML.
Provides different types of indexes for efficient querying.
"""
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import os
import json
import pickle
import numpy as np


class IndexType(Enum):
    """Types of indexes supported by the system."""
    HASH = "hash"  # For exact matches
    BTREE = "btree"  # For range queries
    FULLTEXT = "fulltext"  # For text search
    VECTOR = "vector"  # For embedding similarity


class BaseIndex:
    """
    Base class for all index implementations.
    Defines the common interface that all indexes must implement.
    """
    
    def __init__(self, name: str, field_path: str):
        """
        Initialize the base index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
        """
        self.name = name
        self.field_path = field_path
        self.index_type = None
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from the graph.
        
        Args:
            graph: Graph dictionary
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement build()")
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update the index with a new or modified node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def search(self, query: Any, limit: int = 10, threshold: float = None) -> List[Union[str, Tuple[str, float]]]:
        """
        Search the index.
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Optional threshold for similarity-based searches
            
        Returns:
            List of matching keys or (key, score) tuples
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement search()")
    
    def save(self, path: str) -> bool:
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
            
        Returns:
            bool: Success flag
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self, path: str) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: Success flag
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    def get_field_value(self, node: Dict[str, Any]) -> Any:
        """
        Get the value of the indexed field from a node.
        
        Args:
            node: Node data
            
        Returns:
            Any: Field value
        """
        # Handle nested fields with dot notation
        parts = self.field_path.split('.')
        value = node
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value


class HashIndex(BaseIndex):
    """
    Hash index for exact matches.
    Maps field values to sets of node keys.
    """
    
    def __init__(self, name: str, field_path: str):
        """
        Initialize the hash index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
        """
        super().__init__(name, field_path)
        self.index_type = IndexType.HASH
        self.index = {}  # {field_value: set(node_keys)}
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from the graph.
        
        Args:
            graph: Graph dictionary
        """
        self.index = {}
        
        for key, node in graph.items():
            self.update(key, node)
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update the index with a new or modified node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        if is_delete:
            # Remove from all index entries
            for value_set in self.index.values():
                if key in value_set:
                    value_set.remove(key)
            return
        
        # Get field value
        value = self.get_field_value(node)
        
        if value is None:
            return
        
        # Handle list values
        if isinstance(value, list):
            for item in value:
                # Convert to string for hashability
                item_str = str(item)
                
                if item_str not in self.index:
                    self.index[item_str] = set()
                
                self.index[item_str].add(key)
        else:
            # Convert to string for hashability
            value_str = str(value)
            
            if value_str not in self.index:
                self.index[value_str] = set()
            
            self.index[value_str].add(key)
    
    def search(self, query: Any, limit: int = 10, threshold: float = None) -> List[str]:
        """
        Search the index for exact matches.
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Not used for hash index
            
        Returns:
            List of matching keys
        """
        # Convert query to string for hashability
        query_str = str(query)
        
        if query_str in self.index:
            # Convert set to list and limit results
            return list(self.index[query_str])[:limit]
        
        return []
    
    def save(self, path: str) -> bool:
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
            
        Returns:
            bool: Success flag
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'name': self.name,
                    'field_path': self.field_path,
                    'index_type': self.index_type.value,
                    'index': self.index
                }, f)
            return True
        except Exception as e:
            print(f"Error saving index {self.name}: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: Success flag
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.name = data['name']
                self.field_path = data['field_path']
                self.index_type = IndexType(data['index_type'])
                self.index = data['index']
            return True
        except Exception as e:
            print(f"Error loading index {self.name}: {e}")
            return False


class BTreeIndex(BaseIndex):
    """
    B-tree index for range queries.
    Maps field values to sets of node keys, maintaining sorted order.
    """
    
    def __init__(self, name: str, field_path: str):
        """
        Initialize the B-tree index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
        """
        super().__init__(name, field_path)
        self.index_type = IndexType.BTREE
        self.index = {}  # {field_value: set(node_keys)}
        self.sorted_keys = []  # Sorted list of field values
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from the graph.
        
        Args:
            graph: Graph dictionary
        """
        self.index = {}
        
        for key, node in graph.items():
            self.update(key, node)
        
        # Sort keys
        self._sort_keys()
    
    def _sort_keys(self) -> None:
        """Sort the keys in the index."""
        try:
            # Try to sort as numbers
            self.sorted_keys = sorted(self.index.keys(), key=float)
        except (ValueError, TypeError):
            # Fall back to string sorting
            self.sorted_keys = sorted(self.index.keys())
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update the index with a new or modified node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        if is_delete:
            # Remove from all index entries
            for value_set in self.index.values():
                if key in value_set:
                    value_set.remove(key)
            
            # Resort keys
            self._sort_keys()
            return
        
        # Get field value
        value = self.get_field_value(node)
        
        if value is None:
            return
        
        # Handle list values
        if isinstance(value, list):
            for item in value:
                # Convert to string for hashability
                item_str = str(item)
                
                if item_str not in self.index:
                    self.index[item_str] = set()
                
                self.index[item_str].add(key)
        else:
            # Convert to string for hashability
            value_str = str(value)
            
            if value_str not in self.index:
                self.index[value_str] = set()
            
            self.index[value_str].add(key)
        
        # Resort keys
        self._sort_keys()
    
    def search(self, query: Dict[str, Any], limit: int = 10, threshold: float = None) -> List[str]:
        """
        Search the index for range queries.
        
        Args:
            query: Search query with operators (e.g., {'min': 10, 'max': 20})
            limit: Maximum number of results
            threshold: Not used for B-tree index
            
        Returns:
            List of matching keys
        """
        results = set()
        
        # Handle range query
        if 'min' in query and 'max' in query:
            min_val = str(query['min'])
            max_val = str(query['max'])
            
            # Find keys in range
            for key in self.sorted_keys:
                if min_val <= key <= max_val:
                    results.update(self.index[key])
        
        # Handle greater than query
        elif 'min' in query:
            min_val = str(query['min'])
            
            # Find keys greater than min
            for key in self.sorted_keys:
                if key >= min_val:
                    results.update(self.index[key])
        
        # Handle less than query
        elif 'max' in query:
            max_val = str(query['max'])
            
            # Find keys less than max
            for key in self.sorted_keys:
                if key <= max_val:
                    results.update(self.index[key])
        
        # Convert set to list and limit results
        return list(results)[:limit]
    
    def save(self, path: str) -> bool:
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
            
        Returns:
            bool: Success flag
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'name': self.name,
                    'field_path': self.field_path,
                    'index_type': self.index_type.value,
                    'index': self.index,
                    'sorted_keys': self.sorted_keys
                }, f)
            return True
        except Exception as e:
            print(f"Error saving index {self.name}: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: Success flag
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.name = data['name']
                self.field_path = data['field_path']
                self.index_type = IndexType(data['index_type'])
                self.index = data['index']
                self.sorted_keys = data['sorted_keys']
            return True
        except Exception as e:
            print(f"Error loading index {self.name}: {e}")
            return False


class FullTextIndex(BaseIndex):
    """
    Full-text index for text search.
    Maps terms to sets of node keys.
    """
    
    def __init__(self, name: str, field_path: str):
        """
        Initialize the full-text index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
        """
        super().__init__(name, field_path)
        self.index_type = IndexType.FULLTEXT
        self.index = {}  # {term: {node_key: frequency}}
        self.document_lengths = {}  # {node_key: document_length}
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from the graph.
        
        Args:
            graph: Graph dictionary
        """
        self.index = {}
        self.document_lengths = {}
        
        for key, node in graph.items():
            self.update(key, node)
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update the index with a new or modified node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        if is_delete:
            # Remove from all index entries
            for term_dict in self.index.values():
                if key in term_dict:
                    del term_dict[key]
            
            # Remove from document lengths
            if key in self.document_lengths:
                del self.document_lengths[key]
            
            return
        
        # Get field value
        value = self.get_field_value(node)
        
        if value is None:
            return
        
        # Convert to string
        if not isinstance(value, str):
            value = str(value)
        
        # Tokenize and count terms
        terms = self._tokenize(value)
        term_counts = {}
        
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        
        # Update index
        for term, count in term_counts.items():
            if term not in self.index:
                self.index[term] = {}
            
            self.index[term][key] = count
        
        # Update document length
        self.document_lengths[key] = len(terms)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of terms
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        import re
        return re.findall(r'\w+', text.lower())
    
    def search(self, query: str, limit: int = 10, threshold: float = None) -> List[Tuple[str, float]]:
        """
        Search the index for text matches.
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Minimum score threshold
            
        Returns:
            List of (key, score) tuples
        """
        # Tokenize query
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return []
        
        # Calculate TF-IDF scores
        scores = {}
        
        for term in query_terms:
            if term not in self.index:
                continue
            
            # Calculate IDF (inverse document frequency)
            idf = np.log(len(self.document_lengths) / len(self.index[term]))
            
            for key, tf in self.index[term].items():
                # Calculate TF (term frequency)
                tf = tf / self.document_lengths[key]
                
                # Calculate TF-IDF
                tfidf = tf * idf
                
                # Update score
                scores[key] = scores.get(key, 0) + tfidf
        
        # Sort by score
        results = [(key, score) for key, score in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply threshold
        if threshold is not None:
            results = [(key, score) for key, score in results if score >= threshold]
        
        # Limit results
        return results[:limit]
    
    def save(self, path: str) -> bool:
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
            
        Returns:
            bool: Success flag
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'name': self.name,
                    'field_path': self.field_path,
                    'index_type': self.index_type.value,
                    'index': self.index,
                    'document_lengths': self.document_lengths
                }, f)
            return True
        except Exception as e:
            print(f"Error saving index {self.name}: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: Success flag
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.name = data['name']
                self.field_path = data['field_path']
                self.index_type = IndexType(data['index_type'])
                self.index = data['index']
                self.document_lengths = data['document_lengths']
            return True
        except Exception as e:
            print(f"Error loading index {self.name}: {e}")
            return False


class VectorIndex(BaseIndex):
    """
    Vector index for embedding similarity.
    Maps node keys to embedding vectors.
    """
    
    def __init__(self, name: str, field_path: str):
        """
        Initialize the vector index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
        """
        super().__init__(name, field_path)
        self.index_type = IndexType.VECTOR
        self.vectors = {}  # {node_key: embedding_vector}
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from the graph.
        
        Args:
            graph: Graph dictionary
        """
        self.vectors = {}
        
        for key, node in graph.items():
            self.update(key, node)
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update the index with a new or modified node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        if is_delete:
            # Remove from vectors
            if key in self.vectors:
                del self.vectors[key]
            return
        
        # Get field value
        value = self.get_field_value(node)
        
        if value is None:
            return
        
        # Store vector
        self.vectors[key] = np.array(value)
    
    def search(self, query: List[float], limit: int = 10, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Search the index for similar vectors.
        
        Args:
            query: Query vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of (key, similarity) tuples
        """
        if not self.vectors:
            return []
        
        # Convert query to numpy array
        query_vector = np.array(query)
        
        # Calculate similarities
        similarities = []
        
        for key, vector in self.vectors.items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_vector, vector)
            
            # Apply threshold
            if similarity >= threshold:
                similarities.append((key, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Limit results
        return similarities[:limit]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            float: Cosine similarity
        """
        # Calculate dot product
        dot_product = np.dot(a, b)
        
        # Calculate magnitudes
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        
        # Avoid division by zero
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        # Calculate cosine similarity
        return dot_product / (magnitude_a * magnitude_b)
    
    def save(self, path: str) -> bool:
        """
        Save the index to disk.
        
        Args:
            path: Path to save to
            
        Returns:
            bool: Success flag
        """
        try:
            # Convert vectors to lists for serialization
            vectors_list = {key: vector.tolist() for key, vector in self.vectors.items()}
            
            with open(path, 'wb') as f:
                pickle.dump({
                    'name': self.name,
                    'field_path': self.field_path,
                    'index_type': self.index_type.value,
                    'vectors': vectors_list
                }, f)
            return True
        except Exception as e:
            print(f"Error saving index {self.name}: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the index from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            bool: Success flag
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.name = data['name']
                self.field_path = data['field_path']
                self.index_type = IndexType(data['index_type'])
                
                # Convert lists back to numpy arrays
                self.vectors = {key: np.array(vector) for key, vector in data['vectors'].items()}
            return True
        except Exception as e:
            print(f"Error loading index {self.name}: {e}")
            return False


class IndexManager:
    """
    Manager for creating, updating, and using indexes.
    """
    
    def __init__(self, graph: Dict[str, Dict[str, Any]], index_dir: str):
        """
        Initialize the index manager.
        
        Args:
            graph: Graph dictionary
            index_dir: Directory to store indexes
        """
        self.graph = graph
        self.index_dir = index_dir
        self.indexes = {}  # {index_name: index_instance}
        
        # Create index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        # Load existing indexes
        self._load_indexes()
    
    def _load_indexes(self) -> None:
        """Load existing indexes from disk."""
        # Check if index metadata file exists
        metadata_path = os.path.join(self.index_dir, 'index_metadata.json')
        
        if not os.path.exists(metadata_path):
            return
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load each index
            for index_info in metadata:
                name = index_info['name']
                field_path = index_info['field_path']
                index_type = IndexType(index_info['type'])
                
                # Create index instance
                index = self._create_index_instance(name, field_path, index_type)
                
                # Load index data
                index_path = os.path.join(self.index_dir, f"{name}.idx")
                
                if os.path.exists(index_path):
                    index.load(index_path)
                    self.indexes[name] = index
        
        except Exception as e:
            print(f"Error loading indexes: {e}")
    
    def _save_metadata(self) -> None:
        """Save index metadata to disk."""
        metadata = []
        
        for name, index in self.indexes.items():
            metadata.append({
                'name': name,
                'field_path': index.field_path,
                'type': index.index_type.value
            })
        
        # Save metadata
        metadata_path = os.path.join(self.index_dir, 'index_metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def _create_index_instance(self, name: str, field_path: str, index_type: IndexType) -> BaseIndex:
        """
        Create an index instance of the specified type.
        
        Args:
            name: Index name
            field_path: Path to the field to index
            index_type: Type of index
            
        Returns:
            BaseIndex: Index instance
            
        Raises:
            ValueError: If the index type is not supported
        """
        if index_type == IndexType.HASH:
            return HashIndex(name, field_path)
        elif index_type == IndexType.BTREE:
            return BTreeIndex(name, field_path)
        elif index_type == IndexType.FULLTEXT:
            return FullTextIndex(name, field_path)
        elif index_type == IndexType.VECTOR:
            return VectorIndex(name, field_path)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def create_index(self, name: str, field_path: str, index_type: IndexType) -> None:
        """
        Create a new index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
            index_type: Type of index
            
        Raises:
            ValueError: If an index with the same name already exists
        """
        if name in self.indexes:
            raise ValueError(f"Index {name} already exists")
        
        # Create index instance
        index = self._create_index_instance(name, field_path, index_type)
        
        # Build index
        index.build(self.graph)
        
        # Save index
        index_path = os.path.join(self.index_dir, f"{name}.idx")
        index.save(index_path)
        
        # Add to indexes
        self.indexes[name] = index
        
        # Save metadata
        self._save_metadata()
    
    def drop_index(self, name: str) -> bool:
        """
        Drop an index.
        
        Args:
            name: Index name
            
        Returns:
            bool: Success flag
        """
        if name not in self.indexes:
            return False
        
        # Remove index file
        index_path = os.path.join(self.index_dir, f"{name}.idx")
        
        if os.path.exists(index_path):
            os.remove(index_path)
        
        # Remove from indexes
        del self.indexes[name]
        
        # Save metadata
        self._save_metadata()
        
        return True
    
    def get_index(self, name: str) -> Optional[BaseIndex]:
        """
        Get an index by name.
        
        Args:
            name: Index name
            
        Returns:
            Optional[BaseIndex]: Index instance or None if not found
        """
        return self.indexes.get(name)
    
    def get_indexes(self) -> List[Dict[str, Any]]:
        """
        Get information about all indexes.
        
        Returns:
            List[Dict[str, Any]]: List of index information
        """
        return [
            {
                'name': name,
                'field_path': index.field_path,
                'type': index.index_type.value
            }
            for name, index in self.indexes.items()
        ]
    
    def update_indexes(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update all indexes with a new or modified node.
        
        Args:
            key: Node key
            node: Node data
            is_delete: Whether this is a deletion
        """
        for index in self.indexes.values():
            index.update(key, node, is_delete)
            
            # Save index
            index_path = os.path.join(self.index_dir, f"{index.name}.idx")
            index.save(index_path)
    
    def rebuild_indexes(self) -> None:
        """Rebuild all indexes from the graph."""
        for index in self.indexes.values():
            # Rebuild index
            index.build(self.graph)
            
            # Save index
            index_path = os.path.join(self.index_dir, f"{index.name}.idx")
            index.save(index_path)
    
    def search(self, index_name: str, query: Any, limit: int = 10, threshold: float = None) -> List[Union[str, Tuple[str, float]]]:
        """
        Search an index.
        
        Args:
            index_name: Index name
            query: Search query
            limit: Maximum number of results
            threshold: Optional threshold for similarity-based searches
            
        Returns:
            List of matching keys or (key, score) tuples
            
        Raises:
            ValueError: If the index is not found
        """
        if index_name not in self.indexes:
            raise ValueError(f"Index {index_name} not found")
        
        return self.indexes[index_name].search(query, limit, threshold)

