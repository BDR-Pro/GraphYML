"""
Embedding generation and similarity calculation for GraphYML.
"""
import os
import json
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

# Try to import sentence_transformers, but don't fail if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import ollama, but don't fail if not available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def embedding_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        float: Cosine similarity
    """
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


class EmbeddingGenerator:
    """
    Embedding generator for GraphYML.
    Supports multiple embedding providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.embedding_provider = config.get("embedding_provider", "sentence_transformers")
        self.embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_service_url = os.environ.get(
            "EMBEDDING_SERVICE", 
            config.get("embedding_service_url")
        )
        
        # Initialize embedding model
        self.model = None
        
        # Try to initialize the model if using sentence_transformers
        if self.embedding_provider == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.embedding_model)
                print(f"Initialized sentence_transformers model: {self.embedding_model}")
            except Exception as e:
                print(f"Error initializing sentence_transformers model: {e}")
    
    def generate_embedding(
        self, 
        text: str
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Tuple[Optional[List[float]], Optional[str]]: 
                (embedding, error_message)
        """
        # Check if embedding service URL is available
        if self.embedding_service_url:
            return self._generate_embedding_service(text)
        
        # Try different providers
        if self.embedding_provider == "sentence_transformers":
            return self._generate_embedding_sentence_transformers(text)
        elif self.embedding_provider == "ollama":
            return self._generate_embedding_ollama(text)
        elif self.embedding_provider == "openai":
            return self._generate_embedding_openai(text)
        else:
            return None, f"Unsupported embedding provider: {self.embedding_provider}"
    
    def _generate_embedding_service(
        self, 
        text: str
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Generate embedding using the embedding service.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Tuple[Optional[List[float]], Optional[str]]: 
                (embedding, error_message)
        """
        try:
            # Call embedding service
            response = requests.post(
                f"{self.embedding_service_url}",
                json={"text": text}
            )
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                return data["embedding"], None
            else:
                # Try fallback to local embedding generation
                print(f"Embedding service error: {response.status_code}, {response.text}")
                print("Falling back to local embedding generation...")
                
                if self.embedding_provider == "sentence_transformers":
                    return self._generate_embedding_sentence_transformers(text)
                elif self.embedding_provider == "ollama":
                    return self._generate_embedding_ollama(text)
                else:
                    return None, f"Embedding service error: {response.status_code}, {response.text}"
        
        except Exception as e:
            # Try fallback to local embedding generation
            print(f"Embedding service error: {e}")
            print("Falling back to local embedding generation...")
            
            if self.embedding_provider == "sentence_transformers":
                return self._generate_embedding_sentence_transformers(text)
            elif self.embedding_provider == "ollama":
                return self._generate_embedding_ollama(text)
            else:
                return None, f"Embedding service error: {str(e)}"
    
    def _generate_embedding_sentence_transformers(
        self, 
        text: str
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Generate embedding using sentence_transformers.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Tuple[Optional[List[float]], Optional[str]]: 
                (embedding, error_message)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None, "sentence_transformers not available"
        
        try:
            # Initialize model if not already initialized
            if self.model is None:
                self.model = SentenceTransformer(self.embedding_model)
            
            # Generate embedding
            embedding = self.model.encode(text)
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            return embedding_list, None
        
        except Exception as e:
            return None, f"Error generating embedding with sentence_transformers: {str(e)}"
    
    def _generate_embedding_ollama(
        self, 
        text: str
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Generate embedding using Ollama.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Tuple[Optional[List[float]], Optional[str]]: 
                (embedding, error_message)
        """
        if not OLLAMA_AVAILABLE:
            return None, "ollama not available"
        
        try:
            # Generate embedding
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            
            # Extract embedding
            embedding = response["embedding"]
            
            return embedding, None
        
        except Exception as e:
            return None, f"Error generating embedding with ollama: {str(e)}"
    
    def _generate_embedding_openai(
        self, 
        text: str
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Generate embedding using OpenAI API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Tuple[Optional[List[float]], Optional[str]]: 
                (embedding, error_message)
        """
        # Check if OpenAI API key is available
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None, "OpenAI API key not available"
        
        try:
            # Import OpenAI
            import openai
            
            # Set API key
            openai.api_key = api_key
            
            # Generate embedding
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            # Extract embedding
            embedding = response["data"][0]["embedding"]
            
            return embedding, None
        
        except ImportError:
            return None, "openai package not available"
        
        except Exception as e:
            return None, f"Error generating embedding with OpenAI: {str(e)}"
