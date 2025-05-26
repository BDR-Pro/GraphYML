"""
Embedding generation and similarity computation for GraphYML.
Supports both local Ollama models and remote API services.
"""
import json
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingGenerator:
    """
    Class to handle text embedding generation using various backends.
    Currently supports Ollama and provides a fallback for testing.
    """
    
    def __init__(self, config):
        """
        Initialize the embedding generator with configuration.
        
        Args:
            config (dict): Configuration dictionary with embedding settings
        """
        self.config = config
        self.ollama_url = config.get("ollama_url", "http://localhost:11434/api/embeddings")
        self.ollama_model = config.get("ollama_model", "all-minilm-l6-v2")
        self.embedding_dimension = config.get("embedding_dimension", 384)
        
    def generate_embedding(self, text):
        """
        Generate embedding for the given text.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            tuple: (embedding_vector, error_message)
        """
        if not text:
            return None, "Empty text provided"
            
        # Try Ollama first
        embedding, error = self._generate_ollama_embedding(text)
        
        # If Ollama fails and we're in development/testing, use fallback
        if embedding is None and self.config.get("allow_fallback", False):
            embedding, error = self._generate_fallback_embedding(text)
            
        return embedding, error
    
    def _generate_ollama_embedding(self, text):
        """
        Generate embedding using Ollama API.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            tuple: (embedding_vector, error_message)
        """
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": text},
                timeout=30
            )
            
            if response.status_code != 200:
                return None, f"Ollama API error: {response.status_code} - {response.text}"
                
            result = response.json()
            embedding = result.get("embedding")
            
            if not embedding:
                return None, "No embedding returned from Ollama API"
                
            return embedding, None
            
        except requests.RequestException as e:
            return None, f"Ollama API request failed: {str(e)}"
        except json.JSONDecodeError:
            return None, "Invalid JSON response from Ollama API"
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
    
    def _generate_fallback_embedding(self, text):
        """
        Generate a deterministic pseudo-embedding for testing purposes.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            tuple: (embedding_vector, error_message)
        """
        # Create a deterministic but unique embedding based on text content
        import hashlib
        
        # Get hash of text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to a list of floats
        hash_values = []
        for i in range(0, min(self.embedding_dimension * 2, len(text_hash)), 2):
            if i < len(text_hash) - 1:
                hash_values.append(int(text_hash[i:i+2], 16))
        
        # Pad or truncate to match embedding dimension
        while len(hash_values) < self.embedding_dimension:
            hash_values.append(0)
        hash_values = hash_values[:self.embedding_dimension]
        
        # Normalize to unit length
        embedding = np.array(hash_values, dtype=float)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.tolist(), "Using fallback embedding (not for production)"


def embedding_similarity(a, b):
    """
    Compute cosine similarity between two embedding vectors.
    
    Args:
        a (list): First embedding vector
        b (list): Second embedding vector
        
    Returns:
        float: Cosine similarity (0-1)
    """
    if not a or not b:
        return 0.0
    try:
        a_array = np.array(a).reshape(1, -1)
        b_array = np.array(b).reshape(1, -1)
        return float(cosine_similarity(a_array, b_array)[0][0])
    except (ValueError, TypeError):
        return 0.0


def batch_generate_embeddings(graph, embedding_generator, text_extractor=None):
    """
    Generate embeddings for all nodes in the graph.
    
    Args:
        graph (dict): Graph dictionary
        embedding_generator (EmbeddingGenerator): Embedding generator instance
        text_extractor (callable, optional): Function to extract text from node
            
    Returns:
        tuple: (updated_graph, errors)
    """
    if text_extractor is None:
        # Default extractor uses title, overview, and tagline
        def default_extractor(node):
            text_parts = []
            if node.get("title"):
                text_parts.append(node["title"])
            if node.get("overview"):
                text_parts.append(node["overview"])
            if node.get("tagline"):
                text_parts.append(node["tagline"])
            return " ".join(text_parts)
        
        text_extractor = default_extractor
    
    errors = []
    for key, node in graph.items():
        if "embedding" not in node:
            text = text_extractor(node)
            embedding, error = embedding_generator.generate_embedding(text)
            
            if embedding:
                node["embedding"] = embedding
            else:
                errors.append((key, error))
    
    return graph, errors

