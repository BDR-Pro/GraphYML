# Embedding LLM in GraphYML

This document provides guidance on integrating Large Language Models (LLMs) for embedding generation in the GraphYML project.

## Overview

Embeddings are vector representations of text that capture semantic meaning, allowing for efficient similarity comparisons. In GraphYML, embeddings can be used to:

1. Create semantic search capabilities for nodes
2. Find related content based on meaning rather than keywords
3. Cluster similar nodes together
4. Generate recommendations based on content similarity

## Current Implementation

GraphYML currently has a basic `VectorIndex` implementation in `src/models/modular/vector_index.py` that supports:

- Storing vector embeddings for nodes
- Performing similarity searches using cosine similarity
- Serializing and deserializing embeddings

However, it does not include the actual embedding generation process.

## Recommended LLM Integration Approaches

### 1. Local Embedding Models

For privacy-focused or offline deployments:

```python
# Example using sentence-transformers
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def generate(self, text):
        return self.model.encode(text).tolist()
```

**Pros:**
- No API costs
- Works offline
- Complete privacy

**Cons:**
- Limited by local compute resources
- May not be as powerful as cloud-based models

### 2. OpenAI Embeddings API

For production-quality embeddings with minimal setup:

```python
# Example using OpenAI's embeddings API
import openai

class OpenAIEmbeddingGenerator:
    def __init__(self, api_key, model="text-embedding-3-small"):
        openai.api_key = api_key
        self.model = model
    
    def generate(self, text):
        response = openai.Embedding.create(
            input=text,
            model=self.model
        )
        return response["data"][0]["embedding"]
```

**Pros:**
- High-quality embeddings
- Minimal setup required
- Scales well

**Cons:**
- API costs
- Requires internet connection
- Data privacy considerations

### 3. Hugging Face Inference API

For flexible model selection:

```python
# Example using Hugging Face's inference API
import requests

class HuggingFaceEmbeddingGenerator:
    def __init__(self, api_key, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def generate(self, text):
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text}
        )
        return response.json()
```

**Pros:**
- Wide range of models available
- Community-supported models
- Flexible pricing options

**Cons:**
- Variable model quality
- API reliability depends on model popularity

## Integration with GraphYML

To integrate embedding generation into GraphYML:

1. Create an `EmbeddingService` class in `src/services/embedding_service.py`:

```python
from typing import Dict, List, Any, Optional
import os
from abc import ABC, abstractmethod

class BaseEmbeddingGenerator(ABC):
    @abstractmethod
    def generate(self, text: str) -> List[float]:
        """Generate embedding for text"""
        pass

class EmbeddingService:
    def __init__(self, generator: BaseEmbeddingGenerator):
        self.generator = generator
    
    def generate_node_embeddings(self, graph: Dict[str, Dict[str, Any]], 
                                field: str = "content") -> Dict[str, List[float]]:
        """
        Generate embeddings for all nodes in the graph.
        
        Args:
            graph: Graph to generate embeddings for
            field: Field to generate embeddings from
            
        Returns:
            Dict mapping node IDs to embeddings
        """
        embeddings = {}
        
        for node_id, node_data in graph.items():
            if field in node_data and isinstance(node_data[field], str):
                text = node_data[field]
                embeddings[node_id] = self.generator.generate(text)
        
        return embeddings
    
    def update_graph_with_embeddings(self, graph: Dict[str, Dict[str, Any]], 
                                    embeddings: Dict[str, List[float]],
                                    embedding_field: str = "embedding") -> None:
        """
        Update graph with generated embeddings.
        
        Args:
            graph: Graph to update
            embeddings: Dict mapping node IDs to embeddings
            embedding_field: Field to store embeddings in
        """
        for node_id, embedding in embeddings.items():
            if node_id in graph:
                graph[node_id][embedding_field] = embedding
```

2. Extend the `IndexManager` to support automatic embedding generation:

```python
# In src/models/modular/index_manager.py

def create_vector_index_with_embeddings(self, name: str, content_field: str, 
                                       embedding_field: str = "embedding",
                                       embedding_service: Optional[EmbeddingService] = None) -> VectorIndex:
    """
    Create a vector index with automatically generated embeddings.
    
    Args:
        name: Name of the index
        content_field: Field containing text to generate embeddings from
        embedding_field: Field to store embeddings in
        embedding_service: Service to generate embeddings
        
    Returns:
        VectorIndex: Created index
    """
    # Create the vector index
    index = self.create_index(name, embedding_field, IndexType.VECTOR)
    
    # If embedding service is provided and graph is available, generate embeddings
    if embedding_service and self.graph:
        # Generate embeddings
        embeddings = embedding_service.generate_node_embeddings(self.graph, content_field)
        
        # Update graph with embeddings
        embedding_service.update_graph_with_embeddings(self.graph, embeddings, embedding_field)
        
        # Build the index
        index.build(self.graph)
    
    return index
```

## Configuration

Add embedding configuration to your project's configuration file:

```yaml
# config.yaml
embedding:
  provider: "openai"  # or "local" or "huggingface"
  model: "text-embedding-3-small"  # model name
  api_key: "${OPENAI_API_KEY}"  # environment variable reference
  dimension: 1536  # embedding dimension
  content_field: "content"  # field to generate embeddings from
  embedding_field: "embedding"  # field to store embeddings in
```

## Usage Example

```python
from src.services.embedding_service import EmbeddingService, OpenAIEmbeddingGenerator
from src.models.modular import IndexManager, IndexType

# Create embedding generator based on config
config = load_config("config.yaml")
if config["embedding"]["provider"] == "openai":
    generator = OpenAIEmbeddingGenerator(
        api_key=os.getenv(config["embedding"]["api_key"].strip("${}"), ""),
        model=config["embedding"]["model"]
    )
    embedding_service = EmbeddingService(generator)

# Create index manager with graph
manager = IndexManager(graph=graph, config=config)

# Create vector index with embeddings
vector_index = manager.create_vector_index_with_embeddings(
    name="content_vectors",
    content_field=config["embedding"]["content_field"],
    embedding_field=config["embedding"]["embedding_field"],
    embedding_service=embedding_service
)

# Search for similar nodes
results = vector_index.search([0.1, 0.2, 0.3], top_k=5)
```

## Best Practices

1. **Caching**: Cache embeddings to avoid regenerating them for unchanged content
2. **Batching**: Process embeddings in batches to improve performance
3. **Dimensionality**: Consider dimensionality reduction for large embedding sets
4. **Normalization**: Normalize embeddings before storage for consistent similarity calculations
5. **Error Handling**: Implement robust error handling for API failures
6. **Rate Limiting**: Respect API rate limits when using external services

## Performance Considerations

- Embedding generation can be computationally expensive
- Consider using background workers for large graphs
- Implement incremental updates rather than rebuilding the entire index
- Monitor memory usage when working with large embedding sets

## Security and Privacy

- Store API keys securely using environment variables
- Consider data privacy implications when using external APIs
- Implement access controls for sensitive embeddings
- Document data handling practices for compliance purposes

## Future Enhancements

1. Support for multi-modal embeddings (text + images)
2. Fine-tuning embeddings for domain-specific applications
3. Hybrid search combining keyword and semantic search
4. Clustering and visualization of embedding spaces
5. Incremental embedding updates for changed content only

