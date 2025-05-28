"""
Full-text index implementation for modular indexing system.
"""
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import re
from collections import defaultdict

from .base_index import BaseIndex
from src.utils.index_utils import tokenize_text, extract_phrases


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
        self.token_to_nodes = defaultdict(list)  # Map of token -> list of (node_id, score)
        self.node_tokens = defaultdict(list)  # Map of node_id -> list of tokens
    
    def build(self, graph: Dict[str, Dict[str, Any]]) -> None:
        """
        Build the index from a graph.
        
        Args:
            graph: Graph to build index from
        """
        # Clear the index
        self.token_to_nodes = defaultdict(list)
        self.node_tokens = defaultdict(list)
        
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
        
        # Extract phrases (quoted text)
        phrases = extract_phrases(query)
        
        # Remove phrases from query
        for phrase in phrases:
            query = query.replace(f'"{phrase}"', '')
        
        # Tokenize query
        tokens = tokenize_text(query)
        
        # Combine tokens and phrases
        all_terms = tokens + phrases
        
        if not all_terms:
            return []
        
        # Find nodes containing all terms
        node_scores = defaultdict(float)
        
        # First, find nodes containing all tokens
        for term in all_terms:
            for node_id, score in self.token_to_nodes.get(term, []):
                node_scores[node_id] += score
        
        # Filter nodes that don't contain all terms
        if kwargs.get('require_all_terms', False):
            for node_id in list(node_scores.keys()):
                for term in all_terms:
                    if node_id not in [n for n, _ in self.token_to_nodes.get(term, [])]:
                        del node_scores[node_id]
                        break
        
        # Check for phrase matches
        if phrases:
            for node_id in list(node_scores.keys()):
                node_text = self._get_node_text(node_id)
                if node_text:
                    for phrase in phrases:
                        if phrase.lower() in node_text.lower():
                            # Boost score for phrase match
                            node_scores[node_id] += 1.0
                        elif kwargs.get('require_phrases', False):
                            # Remove node if it doesn't contain the phrase
                            del node_scores[node_id]
                            break
        
        # Sort by score
        results = [(node_id, score) for node_id, score in node_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
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
            # Remove from token_to_nodes
            for token in self.node_tokens.get(node_id, []):
                self.token_to_nodes[token] = [(n, s) for n, s in self.token_to_nodes[token] if n != node_id]
                if not self.token_to_nodes[token]:
                    del self.token_to_nodes[token]
            
            # Remove from node_tokens
            if node_id in self.node_tokens:
                del self.node_tokens[node_id]
            
            return
        
        # Get field value
        field_value = self._get_field_value(node_data)
        
        # Skip if field not found or not a string
        if not field_value or not isinstance(field_value, str):
            return
        
        # Tokenize field value
        tokens = tokenize_text(field_value)
        
        # Remove old tokens
        for token in self.node_tokens.get(node_id, []):
            self.token_to_nodes[token] = [(n, s) for n, s in self.token_to_nodes[token] if n != node_id]
            if not self.token_to_nodes[token]:
                del self.token_to_nodes[token]
        
        # Add new tokens
        self.node_tokens[node_id] = tokens
        
        # Calculate token frequencies
        token_freq = defaultdict(int)
        for token in tokens:
            token_freq[token] += 1
        
        # Add to token_to_nodes with TF score
        for token, freq in token_freq.items():
            score = freq / len(tokens)  # Term frequency
            self.token_to_nodes[token].append((node_id, score))
    
    def _get_node_text(self, node_id: str) -> Optional[str]:
        """
        Get the text for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Optional[str]: Node text
        """
        # This is a simplified implementation
        # In a real system, you would retrieve the text from the graph
        return ' '.join(self.node_tokens.get(node_id, []))
    
    def _get_serializable_index(self) -> Dict[str, Any]:
        """
        Get a serializable version of the index.
        
        Returns:
            Dict[str, Any]: Serializable index
        """
        return {
            "token_to_nodes": dict(self.token_to_nodes),
            "node_tokens": dict(self.node_tokens)
        }
    
    def _set_index_from_serialized(self, serialized_index: Dict[str, Any]) -> None:
        """
        Set the index from a serialized version.
        
        Args:
            serialized_index: Serialized index
        """
        self.token_to_nodes = defaultdict(list, serialized_index.get("token_to_nodes", {}))
        self.node_tokens = defaultdict(list, serialized_index.get("node_tokens", {}))
