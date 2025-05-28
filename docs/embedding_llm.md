# Embedding LLMs in GraphYML

This document provides an overview of how Large Language Models (LLMs) are used for embeddings in GraphYML.

## Introduction to Embeddings

Embeddings are numerical representations of text that capture semantic meaning. In GraphYML, we use embeddings to:

1. Enable semantic search across graph nodes
2. Cluster similar nodes together
3. Find relationships between nodes based on content similarity
4. Support pathfinding with semantic awareness

## Embedding Architecture

GraphYML supports multiple embedding providers:

### 1. Local Embedding Service

The local embedding service uses Sentence Transformers to generate embeddings. This is provided through a FastAPI service in `src/embedding_service.py`.

Key features:
- Uses Hugging Face's Sentence Transformers library
- Default model: `all-MiniLM-L6-v2` (384 dimensions)
- Exposed as a REST API
- Containerized for easy deployment

### 2. Ollama Integration

GraphYML can connect to [Ollama](https://ollama.ai/) for embedding generation:

- Default endpoint: `http://localhost:11434/api/embeddings`
- Default model: `all-minilm-l6-v2`
- Configurable through the application settings

### 3. Fallback Simple Embeddings

If external services are unavailable, GraphYML includes a simple fallback embedding generator that:
- Creates basic character-based embeddings
- Normalizes and pads to the expected dimension
- Provides degraded but functional similarity matching

## Implementation Details

### EmbeddingGenerator Class

The main embedding functionality is implemented in `src/models/embeddings.py` through the `EmbeddingGenerator` class:

```python
class EmbeddingGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize with configuration
        
    def generate_embedding(self, text: str) -> Tuple[List[float], Optional[str]]:
        # Generate embedding with error handling
        
    def _generate_ollama_embedding(self, text: str) -> Tuple[Optional[List[float]], Optional[str]]:
        # Use Ollama API
        
    def _generate_fallback_embedding(self, text: str) -> Tuple[List[float], str]:
        # Generate simple fallback embedding
        
    def batch_generate(self, texts: List[str]) -> List[Tuple[List[float], Optional[str]]]:
        # Process multiple texts
```

### Embedding Similarity

Similarity between embeddings is calculated using cosine similarity:

```python
def embedding_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    # Calculate cosine similarity between embeddings
```

## Docker Setup

GraphYML includes a dedicated Dockerfile for the embedding service:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.embedding.txt .
RUN pip install --no-cache-dir -r requirements.embedding.txt

COPY src/embedding_service.py .

ENV MODEL_NAME="all-MiniLM-L6-v2"
ENV PORT=8000

EXPOSE 8000

CMD ["python", "embedding_service.py"]
```

This service can be run independently or as part of the docker-compose setup.

## Configuration

Embedding settings can be configured through the application's configuration file:

```json
{
  "embedding": {
    "ollama_url": "http://localhost:11434",
    "ollama_model": "all-minilm-l6-v2",
    "openai_embedding_model": "text-embedding-3-small",
    "st_model": "all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "allow_fallback": true
  }
}
```

## Adding Custom Embedding Providers

To add a new embedding provider:

1. Create a new method in the `EmbeddingGenerator` class
2. Update the `generate_embedding` method to try your new provider
3. Add appropriate configuration options

Example for adding OpenAI embeddings:

```python
def _generate_openai_embedding(self, text: str) -> Tuple[Optional[List[float]], Optional[str]]:
    """
    Generate an embedding using OpenAI API.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        Tuple[Optional[List[float]], Optional[str]]: Embedding and error message (if any)
    """
    try:
        import openai
        
        # Set API key
        openai.api_key = self.config.get("openai_api_key")
        
        # Make API request
        response = openai.Embedding.create(
            input=text,
            model=self.config.get("openai_embedding_model", "text-embedding-3-small")
        )
        
        # Return embedding
        return response["data"][0]["embedding"], None
    except Exception as e:
        # Return error
        error = f"Error generating OpenAI embedding: {str(e)}"
        logger.error(error)
        return None, error
```

## Best Practices

When working with embeddings in GraphYML:

1. **Text Preparation**: Clean and normalize text before generating embeddings
2. **Caching**: Store embeddings to avoid regenerating them for the same content
3. **Dimensionality**: Be consistent with embedding dimensions across your application
4. **Fallbacks**: Always implement fallback mechanisms for when services are unavailable
5. **Batching**: Use batch processing for multiple embeddings when possible

## Future Improvements

Planned enhancements for embedding functionality:

1. Support for more embedding models and providers
2. Improved caching and persistence of embeddings
3. Fine-tuning capabilities for domain-specific embeddings
4. Hybrid search combining embedding similarity with keyword matching
5. Visualization tools for exploring embedding spaces

