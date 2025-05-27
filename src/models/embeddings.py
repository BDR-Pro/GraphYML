"""
Embeddings module for GraphYML.
Provides functions for working with embeddings.
"""
import math
import logging
import numpy as np
import requests
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

# Set up logging
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Class for generating embeddings.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedding generator.
        
        Args:
            config: Configuration for embedding generation
                - ollama_url: URL for Ollama API
                - ollama_model: Model to use for Ollama API
                - embedding_dimension: Dimension of embeddings
                - allow_fallback: Whether to allow fallback to simple embeddings
        """
        self.config = config or {
            "ollama_url": "http://localhost:11434/api/embeddings",
            "ollama_model": "llama2",
            "embedding_dimension": 384,
            "allow_fallback": True
        }
    
    def generate_embedding(self, text: str) -> Tuple[List[float], Optional[str]]:
        """
        Generate an embedding for text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Tuple[List[float], Optional[str]]: Embedding and error message (if any)
        """
        # Try to generate embedding using Ollama API
        embedding, error = self._generate_ollama_embedding(text)
        
        # Fall back to simple embedding if Ollama fails
        if embedding is None and self.config.get("allow_fallback", True):
            embedding, error = self._generate_fallback_embedding(text)
        
        return embedding, error
    
    def _generate_ollama_embedding(self, text: str) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Generate an embedding using Ollama API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Tuple[Optional[List[float]], Optional[str]]: Embedding and error message (if any)
        """
        try:
            # Make API request
            response = requests.post(
                self.config["ollama_url"],
                json={"model": self.config["ollama_model"], "prompt": text},
                timeout=30
            )
            
            # Check if request was successful
            if response.status_code == 200:
                # Parse response
                data = response.json()
                
                # Return embedding
                return data["embedding"], None
            else:
                # Return error
                error = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error)
                return None, error
        except Exception as e:
            # Return error
            error = f"Error generating Ollama embedding: {str(e)}"
            logger.error(error)
            return None, error
    
    def _generate_fallback_embedding(self, text: str) -> Tuple[List[float], str]:
        """
        Generate a fallback embedding.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Tuple[List[float], str]: Embedding and error message
        """
        # Generate a simple embedding
        embedding = generate_embedding(text)
        
        # Resize to desired dimension
        dimension = self.config.get("embedding_dimension", 384)
        
        if len(embedding) < dimension:
            # Pad with zeros
            embedding = embedding + [0.0] * (dimension - len(embedding))
        else:
            # Truncate
            embedding = embedding[:dimension]
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        # Return embedding with warning
        return embedding, "Using fallback embedding generator"
    
    def batch_generate(self, texts: List[str]) -> List[Tuple[List[float], Optional[str]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: Texts to generate embeddings for
            
        Returns:
            List[Tuple[List[float], Optional[str]]]: List of embeddings and error messages
        """
        return [self.generate_embedding(text) for text in texts]


def embedding_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        float: Similarity score (0-1)
    """
    # Handle empty embeddings
    if not embedding1 or not embedding2:
        return 0.0
    
    # Handle different lengths
    if len(embedding1) != len(embedding2):
        # Pad shorter embedding with zeros
        if len(embedding1) < len(embedding2):
            embedding1 = embedding1 + [0.0] * (len(embedding2) - len(embedding1))
        else:
            embedding2 = embedding2 + [0.0] * (len(embedding1) - len(embedding2))
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in embedding1))
    magnitude2 = math.sqrt(sum(b * b for b in embedding2))
    
    # Handle zero magnitudes
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    return dot_product / (magnitude1 * magnitude2)


def generate_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """
    Generate an embedding for text.
    
    Args:
        text: Text to generate embedding for
        model: Model to use for embedding generation
        
    Returns:
        List[float]: Embedding
    """
    # This is a placeholder function
    # In a real implementation, this would use a model to generate embeddings
    
    # For now, return a simple hash-based embedding
    embedding = []
    
    for i, char in enumerate(text):
        # Use character code as a simple feature
        value = ord(char) / 255.0
        
        # Add position-dependent features
        position_factor = (i + 1) / len(text)
        embedding.append(value * position_factor)
    
    # Pad to fixed length
    embedding = embedding[:128]
    
    if len(embedding) < 128:
        embedding = embedding + [0.0] * (128 - len(embedding))
    
    return embedding


def batch_generate_embeddings(
    graph: Dict[str, Dict[str, Any]],
    embedding_generator: Optional[EmbeddingGenerator] = None
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Generate embeddings for nodes in a graph.
    
    Args:
        graph: Graph to generate embeddings for
        embedding_generator: Embedding generator to use
        
    Returns:
        Tuple[Dict[str, Dict[str, Any]], List[str]]: Updated graph and list of errors
    """
    # Create a copy of the graph
    updated_graph = {}
    errors = []
    
    # Create embedding generator if not provided
    if embedding_generator is None:
        embedding_generator = EmbeddingGenerator()
    
    # Generate embeddings for each node
    for key, node in graph.items():
        # Create a copy of the node
        updated_graph[key] = node.copy()
        
        # Skip if node already has embedding
        if "embedding" in node:
            continue
        
        # Combine text fields
        text = ""
        
        if "title" in node:
            text += node["title"] + " "
        
        if "overview" in node:
            text += node["overview"] + " "
        
        if "tagline" in node:
            text += node["tagline"]
        
        # Generate embedding
        embedding, error = embedding_generator.generate_embedding(text.strip())
        
        if embedding is not None:
            updated_graph[key]["embedding"] = embedding
        
        if error is not None:
            errors.append(f"Error generating embedding for {key}: {error}")
    
    return updated_graph, errors
