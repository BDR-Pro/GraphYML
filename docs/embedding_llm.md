# Embedding LLM in GraphYML

This document provides an overview of how to integrate Large Language Models (LLMs) with the GraphYML system, particularly focusing on embedding capabilities.

## Introduction

GraphYML's modular indexing system provides a powerful foundation for integrating LLM capabilities, especially through the `VectorIndex` component. This integration enables semantic search, content recommendation, and other advanced features that leverage the power of embeddings.

## Embedding Models

### Supported Models

GraphYML can work with various embedding models:

1. **OpenAI Embeddings**
   - Models like `text-embedding-ada-002` or `text-embedding-3-small`
   - High-quality embeddings with 1536 dimensions (Ada) or 1536/3072 dimensions (embedding-3)
   - Requires API key and network access

2. **Sentence Transformers**
   - Open-source models like `all-MiniLM-L6-v2` or `all-mpnet-base-v2`
   - Can run locally without API dependencies
   - Various dimensions depending on the model (384 for MiniLM, 768 for MPNet)

3. **Custom Embedding Models**
   - Any model that produces vector representations of text
   - Must implement the expected interface for the embedding function

## Integration with VectorIndex

The `VectorIndex` class in GraphYML is designed to work with embeddings. Here's how to use it:

```python
from src.models.modular import VectorIndex, IndexManager, IndexType
from src.models.embeddings import get_embedding

# Create an index manager
manager = IndexManager()

# Create a vector index for the "content" field
vector_index = manager.create_index("content_vectors", "content", IndexType.VECTOR)

# Build the index with your graph data
vector_index.build(graph_data)

# Search for semantically similar content
query_embedding = get_embedding("What is knowledge management?")
results = vector_index.search(query_embedding, threshold=0.7)
```

## Embedding Generation

### Embedding at Index Time

For optimal performance, embeddings should be generated during the initial data processing and stored as part of the node data:

```python
from src.models.embeddings import get_embedding

# Process nodes and add embeddings
for node_id, node_data in graph.items():
    if "content" in node_data:
        node_data["embedding"] = get_embedding(node_data["content"])
```

### On-the-fly Embedding

For dynamic queries, embeddings can be generated at search time:

```python
from src.models.embeddings import get_embedding

# Generate embedding for search query
query = "How to implement knowledge graphs?"
query_embedding = get_embedding(query)

# Search using the embedding
results = vector_index.search(query_embedding)
```

## Advanced Features

### Hybrid Search

Combine vector search with traditional search methods for better results:

```python
# Get results from fulltext search
fulltext_results = fulltext_index.search(query)

# Get results from vector search
vector_results = vector_index.search(get_embedding(query))

# Combine and rank results
combined_results = combine_search_results(fulltext_results, vector_results)
```

### Chunking for Long Documents

For long documents, consider chunking the text before embedding:

```python
def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) >= chunk_size / 2:  # Only keep chunks of reasonable size
            chunks.append(chunk)
    return chunks

# Process nodes with chunking
for node_id, node_data in graph.items():
    if "content" in node_data:
        chunks = chunk_text(node_data["content"])
        node_data["chunks"] = chunks
        node_data["chunk_embeddings"] = [get_embedding(chunk) for chunk in chunks]
```

## Performance Considerations

1. **Embedding Dimensionality**: Higher dimensions provide more expressive embeddings but require more storage and computation time.

2. **Batch Processing**: Generate embeddings in batches when possible to improve throughput.

3. **Caching**: Consider caching embeddings for frequently accessed content.

4. **Quantization**: For large-scale deployments, consider quantizing embeddings to reduce storage requirements.

5. **Approximate Nearest Neighbors**: For large vector collections, use approximate nearest neighbor algorithms like HNSW or FAISS.

## Implementation Example

Here's a complete example of implementing embedding-based search:

```python
from src.models.modular import IndexManager, IndexType
from src.models.embeddings import get_embedding

# Initialize index manager
manager = IndexManager(index_dir="./indexes")

# Create indexes
vector_index = manager.create_index("content_vectors", "embedding", IndexType.VECTOR)
fulltext_index = manager.create_index("content_text", "content", IndexType.FULLTEXT)

# Process graph data with embeddings
processed_graph = {}
for node_id, node_data in original_graph.items():
    processed_node = node_data.copy()
    if "content" in node_data:
        processed_node["embedding"] = get_embedding(node_data["content"])
    processed_graph[node_id] = processed_node

# Build indexes
vector_index.build(processed_graph)
fulltext_index.build(processed_graph)

# Save indexes for future use
manager.save_indexes()

# Search example
query = "How to implement knowledge graphs?"
query_embedding = get_embedding(query)

# Vector search
vector_results = vector_index.search(query_embedding, threshold=0.7)

# Text search
text_results = fulltext_index.search(query)

# Combine results (simple approach)
combined_results = list(set([r[0] for r in vector_results] + [r[0] for r in text_results]))
```

## Future Enhancements

1. **Fine-tuning**: Support for fine-tuning embedding models on domain-specific data.

2. **Multi-modal Embeddings**: Extend to support embeddings for images, audio, and other data types.

3. **Embedding Visualization**: Tools for visualizing embedding spaces to understand relationships.

4. **Incremental Updates**: Optimize the update process for large embedding collections.

5. **Distributed Embedding**: Support for distributed embedding generation and storage for large-scale applications.

## Conclusion

Integrating LLM embeddings with GraphYML's modular indexing system provides powerful semantic search capabilities. The `VectorIndex` component makes it straightforward to leverage these embeddings within the existing architecture, enabling advanced knowledge management features.

