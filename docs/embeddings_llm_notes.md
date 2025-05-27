# Embeddings and LLM Integration Notes

This document provides information about embeddings and how to integrate Large Language Models (LLMs) with GraphYML.

## Embeddings

Embeddings are vector representations of data that capture semantic meaning. In GraphYML, embeddings are used for:

1. **Similarity Search**: Finding similar nodes based on semantic meaning
2. **Clustering**: Grouping related nodes together
3. **Visualization**: Projecting high-dimensional data into 2D/3D space

### Current Implementation

The current implementation in `src/models/embeddings.py` provides:

- `embedding_similarity()`: Calculates cosine similarity between two embeddings
- `generate_embedding()`: Creates a simple embedding from text (placeholder)

### Embedding Models

For production use, consider integrating with these embedding models:

| Model | Dimensions | Provider | Notes |
|-------|------------|----------|-------|
| text-embedding-ada-002 | 1536 | OpenAI | Good general-purpose embeddings |
| all-MiniLM-L6-v2 | 384 | Sentence Transformers | Lightweight, open-source |
| all-mpnet-base-v2 | 768 | Sentence Transformers | Higher quality, slower |
| E5-large | 1024 | Microsoft | Strong performance on retrieval tasks |

### Integration Example

```python
from sentence_transformers import SentenceTransformer

def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """Generate embedding using Sentence Transformers."""
    model = SentenceTransformer(model_name)
    embedding = model.encode(text)
    return embedding.tolist()
```

## LLM Integration

Large Language Models can enhance GraphYML in several ways:

### 1. Automatic Node Linking

LLMs can analyze node content to suggest links between semantically related nodes:

```python
def suggest_links(graph: Dict[str, Dict[str, Any]], node_id: str) -> List[Tuple[str, float]]:
    """Suggest links for a node using LLM."""
    node = graph[node_id]
    prompt = f"Analyze this content: {node['content']}. Suggest related topics."
    
    # Call LLM API
    suggestions = llm_client.complete(prompt=prompt)
    
    # Find matching nodes
    matches = []
    for other_id, other_node in graph.items():
        if other_id != node_id and any(s in other_node['content'] for s in suggestions):
            matches.append((other_id, 0.8))  # Confidence score
            
    return matches
```

### 2. Content Generation

LLMs can generate summaries, tags, or additional content for nodes:

```python
def generate_summary(node: Dict[str, Any]) -> str:
    """Generate a summary for a node using LLM."""
    prompt = f"Summarize this content in 2-3 sentences: {node['content']}"
    return llm_client.complete(prompt=prompt)

def generate_tags(node: Dict[str, Any]) -> List[str]:
    """Generate tags for a node using LLM."""
    prompt = f"Generate 5 relevant tags for this content: {node['content']}"
    tags_text = llm_client.complete(prompt=prompt)
    return [tag.strip() for tag in tags_text.split(',')]
```

### 3. Query Enhancement

LLMs can improve query understanding and expansion:

```python
def enhance_query(query_str: str) -> str:
    """Enhance a query using LLM."""
    prompt = f"Expand this search query with relevant terms: {query_str}"
    enhanced_query = llm_client.complete(prompt=prompt)
    return enhanced_query
```

### 4. Knowledge Graph Construction

LLMs can extract structured information to build knowledge graphs:

```python
def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract entities from text using LLM."""
    prompt = f"Extract people, organizations, locations, and concepts from: {text}"
    response = llm_client.complete(prompt=prompt)
    
    # Parse response into entity categories
    entities = {
        "people": [],
        "organizations": [],
        "locations": [],
        "concepts": []
    }
    
    # Parse LLM response to populate entities
    # ...
    
    return entities
```

## Implementation Recommendations

1. **Model Selection**:
   - For embeddings: Use Sentence Transformers for local deployment, OpenAI for cloud
   - For LLMs: Consider Llama 2, Mistral, or GPT-3.5/4 depending on requirements

2. **Deployment Options**:
   - Local: Use libraries like llama.cpp or Hugging Face Transformers
   - Cloud: OpenAI API, Anthropic Claude, or other provider APIs
   - Hybrid: Use local models for sensitive data, cloud for advanced capabilities

3. **Performance Optimization**:
   - Cache embeddings and LLM responses
   - Use batching for multiple requests
   - Consider quantized models for local deployment

4. **Integration Architecture**:
   - Create a dedicated `LLMService` class to abstract provider details
   - Implement fallback mechanisms between providers
   - Add rate limiting and error handling

## Example LLM Service

```python
class LLMService:
    """Service for interacting with LLMs."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo"):
        self.provider = provider
        self.model = model
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        if self.provider == "openai":
            import openai
            return openai
        elif self.provider == "anthropic":
            import anthropic
            return anthropic
        # Add more providers as needed
    
    def complete(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate completion for prompt."""
        try:
            if self.provider == "openai":
                response = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            # Handle other providers
        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            return ""
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            if self.provider == "openai":
                response = self.client.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response.data[0].embedding
            # Handle other providers
        except Exception as e:
            logger.error(f"Embedding API error: {str(e)}")
            return []
```

## Next Steps

1. Implement a proper embedding generation function using a production-ready model
2. Create an LLM service class with provider abstraction
3. Add configuration options for model selection
4. Implement caching for embeddings and LLM responses
5. Add examples of LLM-enhanced graph operations

