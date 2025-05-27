# Embedding LLMs in GraphYML

This guide provides an overview of how to use and integrate embedding models (LLMs) with GraphYML for semantic search and similarity operations.

## Introduction to Embeddings

Embeddings are vector representations of text or other data that capture semantic meaning. In GraphYML, embeddings are used to:

1. Find similar nodes in a graph
2. Perform semantic search across node properties
3. Create vector indexes for efficient similarity lookups

## Current Implementation

GraphYML currently uses a simplified embedding approach for testing and development:

- The `embedding_similarity` function in `src/models/embeddings.py` calculates cosine similarity between vectors
- The `VectorIndex` class in `src/models/indexing.py` provides vector-based indexing and search
- The `find_similar_nodes` function in `src/models/graph_ops.py` finds semantically similar nodes

## Integrating with Real LLM Embeddings

To integrate with real LLM embedding models:

### 1. Choose an Embedding Model

Several options are available:

- **OpenAI Embeddings**: High-quality but requires API access
  ```python
  from openai import OpenAI
  
  client = OpenAI(api_key="your-api-key")
  response = client.embeddings.create(
      model="text-embedding-ada-002",
      input="Your text here"
  )
  embedding = response.data[0].embedding
  ```

- **Sentence Transformers**: Open-source and locally runnable
  ```python
  from sentence_transformers import SentenceTransformer
  
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embedding = model.encode("Your text here")
  ```

- **Hugging Face Transformers**: Wide variety of models
  ```python
  from transformers import AutoTokenizer, AutoModel
  import torch
  
  tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
  model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
  
  inputs = tokenizer("Your text here", return_tensors="pt", padding=True, truncation=True)
  with torch.no_grad():
      outputs = model(**inputs)
  embedding = outputs.last_hidden_state.mean(dim=1).numpy()
  ```

### 2. Create an Embedding Service

Create a dedicated service for generating and managing embeddings:

```python
# src/models/embedding_service.py
from typing import List, Union, Dict, Any
import numpy as np

class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service."""
        self.model_name = model_name
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            )
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not text:
            return []
        
        # Generate embedding
        embedding = self.model.encode(text)
        
        # Convert to list of floats
        return embedding.tolist()
    
    def generate_embeddings_for_graph(
        self, graph: Dict[str, Dict[str, Any]], field: str = "text"
    ) -> Dict[str, Dict[str, Any]]:
        """Generate embeddings for all nodes in a graph."""
        for node_id, node in graph.items():
            if field in node and node[field]:
                node["embedding"] = self.generate_embedding(node[field])
        
        return graph
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
```

### 3. Update the VectorIndex Implementation

Modify the `VectorIndex` class to use the embedding service:

```python
from src.models.embedding_service import EmbeddingService

class VectorIndex(BaseIndex):
    """Vector index for similarity search."""
    
    def __init__(self, name: str, field: str):
        """Initialize the vector index."""
        super().__init__(name, field)
        self.index = {}
        self.embedding_service = EmbeddingService()
    
    # ... rest of implementation ...
    
    def search(self, query: List[float], threshold: float = 0.8, max_results: int = 10, **kwargs) -> List[Tuple[str, float]]:
        """Search the index."""
        # ... existing code ...
        
        # Calculate similarity using the embedding service
        similarities = []
        for key, embedding in self.index.items():
            similarity = self.embedding_service.calculate_similarity(query, embedding)
            if similarity >= threshold:
                similarities.append((key, similarity))
        
        # ... rest of implementation ...
```

## Best Practices for Embedding LLMs

1. **Caching**: Cache embeddings to avoid regenerating them for the same text
2. **Batching**: Process multiple texts in batches for better performance
3. **Dimensionality**: Consider dimensionality reduction for large embeddings
4. **Model Selection**: Choose models based on your specific needs:
   - Smaller models for faster performance
   - Larger models for better quality
5. **Normalization**: Normalize embeddings for consistent similarity calculations
6. **Hybrid Search**: Combine vector search with keyword search for better results

## Performance Considerations

- Embedding generation can be computationally expensive
- Consider using a dedicated service for embedding generation
- For large graphs, use approximate nearest neighbor algorithms
- Consider using a vector database like FAISS, Milvus, or Pinecone for large-scale deployments

## Future Improvements

1. Support for multiple embedding models
2. Integration with vector databases
3. Approximate nearest neighbor search for large graphs
4. Fine-tuning embeddings for specific domains
5. Hybrid search combining vector and keyword search

## Conclusion

Embedding LLMs provide powerful semantic search capabilities for GraphYML. By following this guide, you can integrate state-of-the-art embedding models to enhance the semantic understanding of your graph data.

