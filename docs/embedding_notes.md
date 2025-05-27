# Embedding LLMs in GraphYML

This document provides information about embedding LLMs (Language Learning Models) in the GraphYML project.

## Overview

Embeddings are vector representations of text that capture semantic meaning. In GraphYML, we use embeddings to:

1. Enable semantic search of nodes
2. Find similar nodes based on content
3. Support clustering and visualization of related nodes

## Supported Embedding Models

GraphYML currently supports the following embedding sources:

### 1. Ollama

[Ollama](https://ollama.ai/) provides a simple way to run LLMs locally. It's the primary embedding source for GraphYML.

**Configuration:**
```python
config = {
    "ollama_url": "http://localhost:11434/api/embeddings",
    "ollama_model": "llama2",
    "embedding_dimension": 384,
    "allow_fallback": True
}
```

**Usage:**
```python
from src.models.embeddings import EmbeddingGenerator

# Create embedding generator
generator = EmbeddingGenerator(config)

# Generate embedding
embedding, error = generator.generate_embedding("Your text here")
```

### 2. Fallback Embedding

When Ollama is unavailable or fails, GraphYML can generate simple fallback embeddings. These are less semantically meaningful but allow the system to continue functioning.

## Integration Points

The embedding system integrates with GraphYML in several key places:

1. **VectorIndex**: Uses embeddings to find semantically similar nodes
2. **graph_ops.find_similar_nodes()**: Finds nodes with similar embeddings
3. **batch_generate_embeddings()**: Processes multiple nodes to add embeddings

## Adding New Embedding Sources

To add a new embedding source:

1. Add a new method to `EmbeddingGenerator` class (e.g., `_generate_openai_embedding`)
2. Update the `generate_embedding` method to try the new source
3. Add appropriate configuration options

Example for adding OpenAI embeddings:

```python
def _generate_openai_embedding(self, text: str) -> Tuple[Optional[List[float]], Optional[str]]:
    """Generate an embedding using OpenAI API."""
    try:
        # Make API request to OpenAI
        response = openai.Embedding.create(
            input=text,
            model=self.config["openai_model"]
        )
        
        # Return embedding
        return response["data"][0]["embedding"], None
    except Exception as e:
        error = f"OpenAI API error: {str(e)}"
        logger.error(error)
        return None, error
```

## Best Practices

1. **Caching**: Consider caching embeddings to avoid regenerating them for the same text
2. **Normalization**: Always normalize embeddings to unit length for consistent similarity calculations
3. **Error Handling**: Provide fallback options when embedding generation fails
4. **Dimensionality**: Be consistent with embedding dimensions across the application

## Performance Considerations

1. Generating embeddings can be computationally expensive
2. Batch processing is more efficient than individual requests
3. Consider using a dedicated service for high-volume embedding generation
4. For large graphs, implement pagination or streaming for embedding operations

## Future Improvements

1. Support for more embedding models (OpenAI, Hugging Face, etc.)
2. Embedding caching and persistence
3. Hybrid search combining embeddings with other index types
4. Customizable embedding generation parameters

