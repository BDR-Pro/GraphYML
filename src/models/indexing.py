"""
Indexing module for GraphYML.
Provides classes for indexing and searching graph data.
"""
import os
import json
import pickle
import logging
import traceback
from enum import Enum
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# Set up logging
logger = logging.getLogger(__name__)


class IndexType(Enum):
    """
    Enum for index types.
    """
    HASH = "hash"
    BTREE = "btree"
    FULLTEXT = "fulltext"
    VECTOR = "vector"


class BaseIndex:
    """
    Base class for all indexes.
    """
    
    def __init__(self, name: str, field: str):
        """
        Initialize the base index.
        
        Args:
            name: Index name
            field: Field to index
        """
        self.name = name
        self.field = field
        self.index = {}
        self.is_built = False
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index.
        
        Args:
            graph: Graph to index
        """
        raise NotImplementedError("Subclasses must implement build()")
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update a node in the index.
        
        Args:
            key: Node key
            node: Updated node
            is_delete: Whether to delete the node
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def search(self, query: Any, **kwargs) -> List[Tuple[str, float]]:
        """
        Search the index.
        
        Args:
            query: Query value to search for
            **kwargs: Additional search parameters
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, score) tuples
        """
        raise NotImplementedError("Subclasses must implement search()")
    
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
            
            # Save index
            with open(path, 'wb') as f:
                pickle.dump({
                    'name': self.name,
                    'field': self.field,
                    'index': self.index,
                    'is_built': self.is_built
                }, f)
            
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
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
                self.index = data['index']
                self.is_built = data['is_built']
            
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False


class HashIndex(BaseIndex):
    """
    Hash-based index for exact matches.
    """
    
    def __init__(self, name: str, field: str):
        """
        Initialize the hash index.
        
        Args:
            name: Index name
            field: Field to index
        """
        super().__init__(name, field)
        self.index = defaultdict(list)
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index.
        
        Args:
            graph: Graph to index
        """
        self.index = defaultdict(list)
        
        for key, node in graph.items():
            # Get field value
            parts = self.field.split('.')
            value = node
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            
            if value is None:
                continue
            
            # Handle different value types
            if isinstance(value, list):
                # Index each value in the list
                for item in value:
                    if item is not None:
                        self.index[str(item)].append(key)
            else:
                # Index the value directly
                self.index[str(value)].append(key)
        
        self.is_built = True
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update a node in the index.
        
        Args:
            key: Node key
            node: Updated node
            is_delete: Whether to delete the node
        """
        # Remove old entries
        for value, keys in list(self.index.items()):
            if key in keys:
                keys.remove(key)
                
                # Remove empty entries
                if not keys:
                    del self.index[value]
        
        # Skip adding new entries if deleting
        if is_delete:
            return
        
        # Add new entries
        parts = self.field.split('.')
        value = node
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        
        if value is not None:
            # Handle different value types
            if isinstance(value, list):
                # Index each value in the list
                for item in value:
                    if item is not None:
                        self.index[str(item)].append(key)
            else:
                # Index the value directly
                self.index[str(value)].append(key)
    
    def search(self, query: Any, **kwargs) -> List[Tuple[str, float]]:
        """
        Search the index.
        
        Args:
            query: Query value to search for
            **kwargs: Additional search parameters
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, 1.0) tuples
        """
        if not self.is_built:
            return []
        
        # Convert query to string
        query_str = str(query)
        
        # Get matching nodes
        if query_str in self.index:
            # Return nodes with score 1.0
            return [(node_id, 1.0) for node_id in self.index[query_str]]
        
        return []


class BTreeIndex(BaseIndex):
    """
    B-tree index for range queries.
    """
    
    def __init__(self, name: str, field: str):
        """
        Initialize the B-tree index.
        
        Args:
            name: Index name
            field: Field to index
        """
        super().__init__(name, field)
        self.sorted_keys = []
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index.
        
        Args:
            graph: Graph to index
        """
        self.index = {}
        
        for key, node in graph.items():
            # Get field value
            parts = self.field.split('.')
            value = node
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            
            if value is None:
                continue
            
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue
            
            # Index the value
            if value not in self.index:
                self.index[value] = []
            
            self.index[value].append(key)
        
        # Sort keys for range queries
        self.sorted_keys = sorted(self.index.keys())
        self.is_built = True
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update a node in the index.
        
        Args:
            key: Node key
            node: Updated node
            is_delete: Whether to delete the node
        """
        # Remove old entries
        for value, keys in list(self.index.items()):
            if key in keys:
                keys.remove(key)
                
                # Remove empty entries
                if not keys:
                    del self.index[value]
        
        # Skip adding new entries if deleting
        if is_delete:
            return
        
        # Add new entries
        parts = self.field.split('.')
        value = node
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        
        if value is not None and isinstance(value, (int, float)):
            if value not in self.index:
                self.index[value] = []
            
            self.index[value].append(key)
            
            # Update sorted keys
            self.sorted_keys = sorted(self.index.keys())
    
    def search(self, query: Any, **kwargs) -> List[Tuple[str, float]]:
        """
        Search the index.
        
        Args:
            query: Query value to search for (can be a value or a range dict)
            **kwargs: Additional search parameters
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, 1.0) tuples
        """
        if not self.is_built:
            return []
        
        results = []
        
        # Handle range queries
        if isinstance(query, dict):
            min_val = query.get('min', float('-inf'))
            max_val = query.get('max', float('inf'))
            
            # Find values in range
            for value in self.sorted_keys:
                if min_val <= value <= max_val:
                    results.extend([(node_id, 1.0) for node_id in self.index[value]])
        else:
            # Handle exact match
            if query in self.index:
                results = [(node_id, 1.0) for node_id in self.index[query]]
        
        return results


class FullTextIndex(BaseIndex):
    """
    Full-text index for text search.
    """
    
    def __init__(self, name: str, field: str):
        """
        Initialize the full-text index.
        
        Args:
            name: Index name
            field: Field to index
        """
        super().__init__(name, field)
        self.inverted_index = defaultdict(list)
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index.
        
        Args:
            graph: Graph to index
        """
        self.inverted_index = defaultdict(list)
        
        for key, node in graph.items():
            # Get field value
            parts = self.field.split('.')
            value = node
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            
            if value is None:
                continue
            
            # Skip non-string values
            if not isinstance(value, str):
                continue
            
            # Tokenize text
            tokens = self._tokenize(value)
            
            # Index tokens
            for token in tokens:
                if (key, token.lower()) not in self.inverted_index[token.lower()]:
                    self.inverted_index[token.lower()].append((key, value))
        
        self.is_built = True
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update a node in the index.
        
        Args:
            key: Node key
            node: Updated node
            is_delete: Whether to delete the node
        """
        # Remove old entries
        for token, entries in list(self.inverted_index.items()):
            self.inverted_index[token] = [(k, v) for k, v in entries if k != key]
            
            # Remove empty entries
            if not self.inverted_index[token]:
                del self.inverted_index[token]
        
        # Skip adding new entries if deleting
        if is_delete:
            return
        
        # Add new entries
        parts = self.field.split('.')
        value = node
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        
        if value is not None and isinstance(value, str):
            # Tokenize text
            tokens = self._tokenize(value)
            
            # Index tokens
            for token in tokens:
                if (key, token.lower()) not in self.inverted_index[token.lower()]:
                    self.inverted_index[token.lower()].append((key, value))
    
    def search(self, query: str, **kwargs) -> List[Tuple[str, float]]:
        """
        Search the index.
        
        Args:
            query: Query string to search for
            **kwargs: Additional search parameters
                - match_type: Type of match (any, all, phrase)
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, score) tuples
        """
        if not self.is_built:
            return []
        
        # Get match type
        match_type = kwargs.get("match_type", "any")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Handle different match types
        if match_type == "any":
            # Match any token
            results = {}
            
            for token in query_tokens:
                token = token.lower()
                
                if token in self.inverted_index:
                    for key, text in self.inverted_index[token]:
                        # Calculate score based on token frequency
                        score = text.lower().count(token) / len(text.split())
                        
                        if key in results:
                            results[key] = max(results[key], score)
                        else:
                            results[key] = score
            
            # Sort by score
            return sorted([(key, score) for key, score in results.items()], key=lambda x: x[1], reverse=True)
        
        elif match_type == "all":
            # Match all tokens
            results = {}
            
            # Get documents that contain all tokens
            for token in query_tokens:
                token = token.lower()
                
                if token not in self.inverted_index:
                    return []
                
                for key, text in self.inverted_index[token]:
                    if key in results:
                        results[key] += text.lower().count(token) / len(text.split())
                    else:
                        results[key] = text.lower().count(token) / len(text.split())
            
            # Filter documents that don't contain all tokens
            min_count = len(query_tokens)
            results = {key: score for key, score in results.items() if score >= min_count}
            
            # Sort by score
            return sorted([(key, score) for key, score in results.items()], key=lambda x: x[1], reverse=True)
        
        elif match_type == "phrase":
            # Match exact phrase
            results = {}
            
            # Get documents that contain all tokens
            for token in query_tokens:
                token = token.lower()
                
                if token not in self.inverted_index:
                    return []
                
                for key, text in self.inverted_index[token]:
                    if key in results:
                        results[key].append(text)
                    else:
                        results[key] = [text]
            
            # Filter documents that don't contain all tokens
            results = {key: texts for key, texts in results.items() if len(texts) == len(query_tokens)}
            
            # Check if phrase exists in text
            phrase_results = {}
            
            for key, texts in results.items():
                for text in texts:
                    if query.lower() in text.lower():
                        phrase_results[key] = text.lower().count(query.lower()) / len(text.split())
            
            # Sort by score
            return sorted([(key, score) for key, score in phrase_results.items()], key=lambda x: x[1], reverse=True)
        
        return []
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        for char in '.,;:!?()[]{}"\'':
            text = text.replace(char, ' ')
        
        # Split into tokens
        tokens = text.split()
        
        # Remove duplicates
        return list(set(tokens))


class VectorIndex(BaseIndex):
    """
    Vector index for embedding similarity search.
    """
    
    def __init__(self, name: str, field: str):
        """
        Initialize the vector index.
        
        Args:
            name: Index name
            field: Field to index
        """
        super().__init__(name, field)
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index.
        
        Args:
            graph: Graph to index
        """
        self.index = {}
        
        for key, node in graph.items():
            # Get field value
            parts = self.field.split('.')
            value = node
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            
            if value is None:
                continue
            
            # Skip non-list values
            if not isinstance(value, list):
                continue
            
            # Index the embedding
            self.index[key] = value
        
        self.is_built = True
    
    def update(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update a node in the index.
        
        Args:
            key: Node key
            node: Updated node
            is_delete: Whether to delete the node
        """
        # Remove old entries
        if key in self.index:
            del self.index[key]
        
        # Skip adding new entries if deleting
        if is_delete:
            return
        
        # Add new entries
        parts = self.field.split('.')
        value = node
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        
        if value is not None and isinstance(value, list):
            self.index[key] = value
    
    def search(self, query: List[float], threshold: float = 0.8, max_results: int = 10, **kwargs) -> List[Tuple[str, float]]:
        """
        Search the index.
        
        Args:
            query: Query embedding
            threshold: Similarity threshold
            max_results: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, similarity) tuples
        """
        if not self.is_built:
            return []
        
        # Import embedding_similarity function
        from src.models.embeddings import embedding_similarity
        
        # Calculate similarity for each node
        similarities = []
        
        for key, embedding in self.index.items():
            # Calculate similarity
            similarity = embedding_similarity(query, embedding)
            
            # Add to results if above threshold
            if similarity >= threshold:
                similarities.append((key, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Limit results
        return similarities[:max_results]


class IndexManager:
    """
    Manager for multiple indexes.
    """
    
    def __init__(self, graph: Dict[str, Dict[str, Any]] = None, index_dir: str = None):
        """
        Initialize the index manager.
        
        Args:
            graph: Graph to index
            index_dir: Directory to store indexes
        """
        self.indexes = {}
        self.index_dir = index_dir
        self.graph = graph
    
    def create_index(self, name: str, field: str, index_type: IndexType) -> BaseIndex:
        """
        Create an index.
        
        Args:
            name: Index name
            field: Field to index
            index_type: Index type
            
        Returns:
            BaseIndex: Created index
        """
        # Create index based on type
        if index_type == IndexType.HASH:
            index = HashIndex(name, field)
        elif index_type == IndexType.BTREE:
            index = BTreeIndex(name, field)
        elif index_type == IndexType.FULLTEXT:
            index = FullTextIndex(name, field)
        elif index_type == IndexType.VECTOR:
            index = VectorIndex(name, field)
        else:
            raise ValueError(f"Invalid index type: {index_type}")
        
        # Add index to manager
        self.indexes[name] = index
        
        # Build index if graph is available
        if self.graph:
            index.build(self.graph)
        
        return index
    
    def get_index(self, name: str) -> BaseIndex:
        """
        Get an index by name.
        
        Args:
            name: Index name
            
        Returns:
            BaseIndex: Index
        """
        if name not in self.indexes:
            raise ValueError(f"Index not found: {name}")
        
        return self.indexes[name]
    
    def get_indexes(self) -> Dict[str, BaseIndex]:
        """
        Get all indexes.
        
        Returns:
            Dict[str, BaseIndex]: Dictionary of indexes
        """
        return self.indexes
    
    def drop_index(self, name: str) -> bool:
        """
        Drop an index.
        
        Args:
            name: Index name
            
        Returns:
            bool: True if successful, False otherwise
        """
        if name not in self.indexes:
            return False
        
        # Remove index
        del self.indexes[name]
        
        # Remove index file if it exists
        if self.index_dir:
            index_path = os.path.join(self.index_dir, f"{name}.idx")
            
            if os.path.exists(index_path):
                try:
                    os.remove(index_path)
                except:
                    return False
        
        return True
    
    def update_indexes(self, key: str, node: Dict[str, Any], is_delete: bool = False) -> None:
        """
        Update all indexes with a node.
        
        Args:
            key: Node key
            node: Updated node
            is_delete: Whether to delete the node
        """
        for index in self.indexes.values():
            index.update(key, node, is_delete)
    
    def search(self, query: Any, index_name: str = None, **kwargs) -> List[Tuple[str, float]]:
        """
        Search indexes.
        
        Args:
            query: Query value
            index_name: Index name (if None, search all indexes)
            **kwargs: Additional search parameters
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, score) tuples
        """
        if index_name:
            # Search specific index
            if index_name not in self.indexes:
                raise ValueError(f"Index not found: {index_name}")
            
            return self.indexes[index_name].search(query, **kwargs)
        else:
            # Search all indexes
            results = []
            
            for index in self.indexes.values():
                results.extend(index.search(query, **kwargs))
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Remove duplicates
            seen = set()
            unique_results = []
            
            for node_id, score in results:
                if node_id not in seen:
                    seen.add(node_id)
                    unique_results.append((node_id, score))
            
            return unique_results
    
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
        for name, index in self.indexes.items():
            index_path = os.path.join(self.index_dir, f"{name}.idx")
            
            if not index.save(index_path):
                return False
        
        return True
    
    def load_indexes(self) -> bool:
        """
        Load all indexes from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.index_dir or not os.path.exists(self.index_dir):
            return False
        
        # Load each index
        for filename in os.listdir(self.index_dir):
            if filename.endswith(".idx"):
                index_path = os.path.join(self.index_dir, filename)
                index_name = filename[:-4]  # Remove .idx extension
                
                # Load index
                index_type = BaseIndex.get_index_type(index_path)
                
                if index_type == IndexType.HASH:
                    index = HashIndex.load(index_path)
                elif index_type == IndexType.BTREE:
                    index = BTreeIndex.load(index_path)
                elif index_type == IndexType.FULLTEXT:
                    index = FullTextIndex.load(index_path)
                elif index_type == IndexType.VECTOR:
                    index = VectorIndex.load(index_path)
                else:
                    continue
                
                if index:
                    self.indexes[index_name] = index
        
        return True
