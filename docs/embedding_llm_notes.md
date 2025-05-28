# Embedding and LLM Integration Notes

This document provides information about embedding models and LLM (Large Language Model) integration in the GraphYML project.

## Embeddings

### What are Embeddings?

Embeddings are vector representations of text, code, or other data that capture semantic meaning. In the context of GraphYML, embeddings are used to:

1. Represent nodes in a vector space
2. Enable semantic search capabilities
3. Support similarity-based operations

### Embedding Models

Several embedding models can be integrated with GraphYML:

1. **OpenAI Embeddings**
   - Models: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
   - High-quality embeddings with different dimensions (1536 for ada-002, 1536 for 3-small, 3072 for 3-large)
   - Requires API key and network access

2. **Sentence Transformers**
   - Models: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `multi-qa-mpnet-base-dot-v1`
   - Open-source models that can run locally
   - Different dimensions (384 for MiniLM, 768 for mpnet models)
   - Good balance of quality and performance

3. **HuggingFace Models**
   - Various models available through the Hugging Face Hub
   - Can be run locally or via API
   - Wide range of dimensions and specializations

### Implementation in GraphYML

The `VectorIndex` class in the modular indexing system uses embeddings for semantic search:

```python
class VectorIndex(BaseIndex):
    """Vector index for semantic search."""
    
    def search(self, query_vector: List[float], threshold: float = 0.7, limit: int = 10, **kwargs) -> List[Tuple[str, float]]:
        """
        Search the index using vector similarity.
        
        Args:
            query_vector: Vector to search for
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of (node_id, similarity) tuples
        """
        if not self.is_built:
            return []
        
        results = []
        
        for node_id, embedding in self.index.items():
            similarity = embedding_similarity(query_vector, embedding)
            if similarity >= threshold:
                results.append((node_id, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
```

## LLM Integration

### What are LLMs?

Large Language Models (LLMs) are advanced AI models trained on vast amounts of text data that can understand and generate human-like text. In GraphYML, LLMs can be used for:

1. Generating node content
2. Summarizing information
3. Answering questions about the graph
4. Enhancing search capabilities

### LLM Options

Several LLM options can be integrated with GraphYML:

1. **OpenAI Models**
   - Models: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`
   - High-quality text generation and understanding
   - Requires API key and network access

2. **Local Models**
   - Models: `Llama-3`, `Mistral`, `Phi-3`, etc.
   - Can run locally without internet access
   - Various sizes (8B, 13B, 70B parameters)
   - Lower resource requirements for smaller models

3. **Hugging Face Models**
   - Various models available through the Hugging Face Hub
   - Can be run locally or via API
   - Wide range of sizes and specializations

### Integration Strategies

#### 1. Direct API Integration

```python
import openai

def generate_content(prompt, model="gpt-3.5-turbo"):
    """Generate content using OpenAI API."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
```

#### 2. Local Model Integration

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_local_model(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """Load a local LLM model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_with_local_model(prompt, model, tokenizer, max_length=100):
    """Generate text with a local model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### 3. Hybrid Approach

A hybrid approach can be implemented to use local models when possible and fall back to API models when needed:

```python
def smart_generate(prompt, use_local=True, local_model=None, local_tokenizer=None):
    """Generate content using the best available model."""
    if use_local and local_model and local_tokenizer:
        try:
            return generate_with_local_model(prompt, local_model, local_tokenizer)
        except Exception as e:
            print(f"Local model failed: {e}")
    
    # Fall back to API
    return generate_content(prompt)
```

## Recommended Implementation for GraphYML

### Embedding Implementation

1. Add a configurable embedding provider:

```python
class EmbeddingProvider:
    """Provider for text embeddings."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", use_api=False, api_key=None):
        """Initialize the embedding provider."""
        self.model_name = model_name
        self.use_api = use_api
        self.api_key = api_key
        self.model = None
        
        if not use_api:
            # Load local model
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
    
    def get_embedding(self, text):
        """Get embedding for text."""
        if self.use_api:
            # Use API (OpenAI, etc.)
            import openai
            openai.api_key = self.api_key
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        else:
            # Use local model
            return self.model.encode(text).tolist()
```

### LLM Implementation

1. Add an LLM service:

```python
class LLMService:
    """Service for LLM operations."""
    
    def __init__(self, model_name="gpt-3.5-turbo", use_api=True, api_key=None):
        """Initialize the LLM service."""
        self.model_name = model_name
        self.use_api = use_api
        self.api_key = api_key
        self.model = None
        self.tokenizer = None
        
        if not use_api:
            # Load local model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate(self, prompt, max_length=100):
        """Generate text based on prompt."""
        if self.use_api:
            # Use API (OpenAI, etc.)
            import openai
            openai.api_key = self.api_key
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        else:
            # Use local model
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids, 
                max_length=max_length,
                do_sample=True,
                temperature=0.7
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def answer_question(self, question, context):
        """Answer a question based on context."""
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return self.generate(prompt)
    
    def summarize(self, text, max_length=100):
        """Summarize text."""
        prompt = f"Summarize the following text:\n\n{text}"
        return self.generate(prompt, max_length)
```

## Integration with GraphYML

To integrate these components with GraphYML:

1. Add the embedding provider to the graph model
2. Use the LLM service for advanced operations
3. Create a configuration system to manage model settings

Example integration:

```python
from src.models.embeddings import EmbeddingProvider
from src.services.llm_service import LLMService

class GraphYML:
    """Main GraphYML class."""
    
    def __init__(self, config=None):
        """Initialize GraphYML."""
        self.config = config or {}
        self.graph = {}
        self.index_manager = IndexManager()
        
        # Initialize embedding provider
        self.embedding_provider = EmbeddingProvider(
            model_name=self.config.get("embedding_model", "all-MiniLM-L6-v2"),
            use_api=self.config.get("use_api_embeddings", False),
            api_key=self.config.get("api_key")
        )
        
        # Initialize LLM service
        self.llm_service = LLMService(
            model_name=self.config.get("llm_model", "gpt-3.5-turbo"),
            use_api=self.config.get("use_api_llm", True),
            api_key=self.config.get("api_key")
        )
    
    def add_node(self, node_id, data):
        """Add a node to the graph."""
        # Generate embedding if not provided
        if "embedding" not in data and "content" in data:
            data["embedding"] = self.embedding_provider.get_embedding(data["content"])
        
        self.graph[node_id] = data
        
        # Update indexes
        for index_name, index in self.index_manager.indexes.items():
            index.update(node_id, data)
    
    def search(self, query, index_name=None):
        """Search the graph."""
        if index_name:
            return self.index_manager.search(index_name, query)
        
        # If no index specified, try semantic search
        query_embedding = self.embedding_provider.get_embedding(query)
        return self.index_manager.search("vector_index", query_embedding)
    
    def ask(self, question):
        """Ask a question about the graph."""
        # Get relevant nodes
        relevant_nodes = self.search(question)
        
        # Build context from relevant nodes
        context = "\n\n".join([
            f"Node {node_id}: {self.graph[node_id].get('content', '')}"
            for node_id in relevant_nodes
        ])
        
        # Use LLM to answer
        return self.llm_service.answer_question(question, context)
```

## Conclusion

Integrating embeddings and LLMs into GraphYML provides powerful capabilities for semantic search, content generation, and question answering. The implementation can be flexible, supporting both API-based and local models to accommodate different use cases and resource constraints.

By properly abstracting these components, GraphYML can leverage the latest advancements in AI while maintaining a clean and modular architecture.

