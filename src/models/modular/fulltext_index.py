"""
Full-text index implementation for modular indexing system.
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import re
from collections import defaultdict

from .base_index import BaseIndex


class FullTextIndex(BaseIndex):
    """Full-text index for text search."""
    
    def __init__(self, name: str, field: str):
        """
        Initialize the full-text index.
        
        Args:
            name: Name of the index
            field: Field to index
        """
        super().__init__(name, field)
        self.inverted_index = defaultdict(list)  # Map of term -> list of node_ids
        self.node_terms = {}  # Map of node_id -> list of terms
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from a graph.
        
        Args:
            graph: Graph to build index from
        """
        # Clear the index
        self.inverted_index = defaultdict(list)
        self.node_terms = {}
        
        # Build the index
        for node_id, node_data in graph.items():
            self.update(node_id, node_data)
        
        self.is_built = True
    
    def search(self, query: str, **kwargs) -> List[Tuple[str, float]]:
        """
        Search the index.
        
        Args:
            query: Query to search for
            **kwargs: Additional search parameters
            
        Returns:
            List of (node_id, score) tuples
        """
        if not self.is_built:
            return []
        
        # Check for phrase search
        phrase_match = re.search(r'"([^"]+)"', query)
        if phrase_match:
            # Phrase search
            phrase = phrase_match.group(1)
            return self._phrase_search(phrase)
        
        # Normal search
        terms = self._tokenize(query)
        return self._term_search(terms)
    
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
            if node_id in self.node_terms:
                # Remove from inverted_index
                for term in self.node_terms[node_id]:
                    if node_id in self.inverted_index[term]:
                        self.inverted_index[term].remove(node_id)
                        if not self.inverted_index[term]:
                            del self.inverted_index[term]
                
                # Remove from node_terms
                del self.node_terms[node_id]
            return
        
        # Get field value
        field_value = self._get_field_value(node_data)
        
        # Skip if field not found or not a string
        if field_value is None or not isinstance(field_value, str):
            return
        
        # Tokenize the field value
        terms = self._tokenize(field_value)
        
        # Remove old terms if node already indexed
        if node_id in self.node_terms:
            old_terms = self.node_terms[node_id]
            for term in old_terms:
                if node_id in self.inverted_index[term]:
                    self.inverted_index[term].remove(node_id)
                    if not self.inverted_index[term]:
                        del self.inverted_index[term]
        
        # Add new terms
        self.node_terms[node_id] = terms
        for term in terms:
            if node_id not in self.inverted_index[term]:
                self.inverted_index[term].append(node_id)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of terms
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into terms
        terms = text.split()
        
        return terms
    
    def _term_search(self, terms: List[str]) -> List[Tuple[str, float]]:
        """
        Search for terms.
        
        Args:
            terms: Terms to search for
            
        Returns:
            List of (node_id, score) tuples
        """
        # Get nodes for each term
        term_nodes = {}
        for term in terms:
            term_nodes[term] = set(self.inverted_index.get(term, []))
        
        # Calculate scores
        scores = defaultdict(float)
        for term, nodes in term_nodes.items():
            for node_id in nodes:
                scores[node_id] += 1.0 / len(terms)
        
        # Sort by score
        results = [(node_id, score) for node_id, score in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _phrase_search(self, phrase: str) -> List[Tuple[str, float]]:
        """
        Search for a phrase.
        
        Args:
            phrase: Phrase to search for
            
        Returns:
            List of (node_id, score) tuples
        """
        # Tokenize the phrase
        terms = self._tokenize(phrase)
        
        # Get nodes for each term
        term_nodes = {}
        for term in terms:
            term_nodes[term] = set(self.inverted_index.get(term, []))
        
        # Find nodes that contain all terms
        if not term_nodes:
            return []
        
        # Start with nodes from first term
        common_nodes = term_nodes.get(terms[0], set())
        
        # Intersect with nodes from other terms
        for term in terms[1:]:
            common_nodes &= term_nodes.get(term, set())
        
        # Calculate scores (all nodes get score 1.0 for phrase match)
        results = [(node_id, 1.0) for node_id in common_nodes]
        
        return results
    
    def _get_serializable_index(self) -> Dict[str, Any]:
        """
        Get a serializable version of the index.
        
        Returns:
            Dict[str, Any]: Serializable index
        """
        return {
            "inverted_index": dict(self.inverted_index),
            "node_terms": self.node_terms
        }
    
    def _set_index_from_serialized(self, serialized_index: Dict[str, Any]) -> None:
        """
        Set the index from a serialized version.
        
        Args:
            serialized_index: Serialized index
        """
        self.inverted_index = defaultdict(list)
        for term, nodes in serialized_index.get("inverted_index", {}).items():
            self.inverted_index[term] = nodes
        
        self.node_terms = serialized_index.get("node_terms", {})

