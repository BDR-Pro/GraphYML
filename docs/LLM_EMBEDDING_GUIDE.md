# LLM Embedding Integration Guide

This document explains how GraphYML integrates with language models for text embeddings and provides guidance on setting up and using different embedding providers.

## What are Embeddings?

Embeddings are numerical representations of text that capture semantic meaning. In GraphYML, we use embeddings to:

1. Calculate similarity between nodes
2. Enable semantic search
3. Support clustering and visualization
4. Power the A* pathfinding algorithm

## Embedding Providers

GraphYML supports multiple embedding providers:

### 1. Ollama (Default)

[Ollama](https://ollama.ai/) is a local LLM server that can run various embedding models.

**Setup:**

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull an embedding model:
   ```bash
   ollama pull all-minilm-l6-v2
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```
4. Configure GraphYML to use Ollama:
   - URL: `http://localhost:11434/api/embeddings`
   - Model: `all-minilm-l6-v2`

**Recommended Models:**

- `all-minilm-l6-v2` (384 dimensions, fast)
- `nomic-embed-text` (768 dimensions, more accurate)
- `mxbai-embed-large` (1024 dimensions, highest quality)

### 2. OpenAI

For production use, OpenAI's embedding API provides high-quality embeddings.

**Setup:**

1. Get an API key from [OpenAI](https://platform.openai.com/)
2. Configure GraphYML:
   - URL: `https://api.openai.com/v1/embeddings`
   - Set environment variable: `OPENAI_API_KEY=your_key_here`

**Recommended Models:**

- `text-embedding-3-small` (1536 dimensions)
- `text-embedding-3-large` (3072 dimensions)

### 3. Hugging Face

For self-hosted or open-source options, Hugging Face models can be used.

**Setup:**

1. Install the Hugging Face Transformers library:
   ```bash
   pip install transformers sentence-transformers
   ```
2. Run a local API server (see example in `scripts/hf_embedding_server.py`)
3. Configure GraphYML to use your local server

## Implementing Custom Embedding Providers

To add a new embedding provider:

1. Extend the `EmbeddingGenerator` class in `src/models/embeddings.py`
2. Implement the provider-specific logic in a new method
3. Update the configuration schema in `src/config/settings.py`

Example implementation for a custom provider:

```python
def _generate_custom_embedding(self, text):
    """Generate embedding using a custom API."""
    try:
        response = requests.post(
            self.config["custom_url"],
            json={"text": text, "model": self.config["custom_model"]},
            headers={"Authorization": f"Bearer {self.config['custom_api_key']}"},
            timeout=30
        )
        
        if response.status_code != 200:
            return None, f"API error: {response.status_code} - {response.text}"
            
        result = response.json()
        embedding = result.get("embedding")
        
        if not embedding:
            return None, "No embedding returned from API"
            
        return embedding, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"
```

## Embedding Dimensions

Different models produce embeddings with different dimensions:

| Model | Dimensions | Quality | Speed |
|-------|------------|---------|-------|
| all-minilm-l6-v2 | 384 | Good | Fast |
| nomic-embed-text | 768 | Better | Medium |
| text-embedding-3-small | 1536 | Excellent | Medium |
| text-embedding-3-large | 3072 | Best | Slow |

The dimension is important for:
- Storage requirements
- Processing speed
- Similarity calculation accuracy

## Best Practices

1. **Consistency**: Use the same model for all embeddings in a project
2. **Text Preparation**:
   - Include relevant fields (title, description, tags)
   - Remove unnecessary formatting
   - Keep text length reasonable (under 8,000 tokens)
3. **Batch Processing**: Generate embeddings in batches for efficiency
4. **Caching**: Store embeddings to avoid regeneration
5. **Fallback Strategy**: Implement fallbacks for when the embedding service is unavailable

## Troubleshooting

Common issues and solutions:

1. **Connection Errors**:
   - Ensure the embedding server is running
   - Check network connectivity
   - Verify API keys and authentication

2. **Empty or Invalid Embeddings**:
   - Check input text (empty or too long)
   - Verify model compatibility
   - Look for API rate limiting

3. **Performance Issues**:
   - Use a smaller/faster model for development
   - Implement caching
   - Process embeddings in batches

## Example: Generating Embeddings

```python
from src.models.embeddings import EmbeddingGenerator

# Initialize with configuration
config = {
    "ollama_url": "http://localhost:11434/api/embeddings",
    "ollama_model": "all-minilm-l6-v2"
}
generator = EmbeddingGenerator(config)

# Generate embedding
text = "This is a sample movie about space exploration"
embedding, error = generator.generate_embedding(text)

if embedding:
    print(f"Generated embedding with {len(embedding)} dimensions")
else:
    print(f"Error: {error}")
```

## Future Enhancements

Planned improvements for embedding functionality:

1. Support for more embedding providers (Cohere, Azure, etc.)
2. Automatic dimensionality reduction for large embeddings
3. Hybrid search combining embeddings with keyword matching
4. Progressive enhancement with fallback to simpler methods
5. Embedding visualization tools

