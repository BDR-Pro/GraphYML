# GraphYML

A graph-based data management system for YAML files with embedding and querying capabilities.

## Features

- Store and manage graph data in YAML files
- Index data for fast querying
- Generate embeddings for semantic search
- Query data using a simple query language
- Automatically link related nodes
- Find similar nodes using embeddings
- Comprehensive logging system for debugging

## Modules

### 1. Indexing Module

The indexing module provides classes for indexing and searching graph data:

- `BaseIndex`: Base class for all indexes
- `HashIndex`: Hash-based index for exact matches
- `BTreeIndex`: B-tree index for range queries
- `FullTextIndex`: Full-text index for text search
- `VectorIndex`: Vector index for embedding similarity search
- `IndexManager`: Manager for multiple indexes

### 2. Embeddings Module

The embeddings module provides classes and functions for generating and working with embeddings:

- `EmbeddingGenerator`: Class for generating embeddings
- `embedding_similarity`: Function for calculating cosine similarity between embeddings
- `batch_generate_embeddings`: Function for generating embeddings for all nodes in a graph

### 3. Graph Operations Module

The graph operations module provides functions for working with graph data:

- `auto_link_nodes`: Function for automatically linking related nodes
- `tag_similarity`: Function for calculating similarity between tag lists
- `a_star`: Function for finding the shortest path between nodes
- `reconstruct_path`: Function for reconstructing a path from a search
- `find_similar_nodes`: Function for finding nodes similar to a given node

### 4. Query Engine Module

The query engine module provides classes and functions for querying graph data:

- `Condition`: Class for representing a query condition
- `Query`: Class for representing a query
- `QueryParser`: Class for parsing query strings
- `query_graph`: Function for querying a graph using a query string

### 5. Data Handler Module

The data handler module provides functions for loading and saving graph data:

- `validate_node_schema`: Function for validating a node against a schema
- `load_graph_from_folder`: Function for loading graph data from a folder of YAML files
- `save_node_to_yaml`: Function for saving a node to a YAML file
- `create_zip`: Function for creating a ZIP file from a folder
- `flatten_node`: Function for flattening a node by combining text fields
- `query_by_tag`: Function for querying a graph by tag

### 6. Logger Module

The logger module provides functions for setting up and using logging:

- `setup_logger`: Function for setting up a logger with file and console handlers
- `get_logger`: Function for getting a logger with configuration
- `log_function_call`: Function for logging function calls
- `log_function_result`: Function for logging function results
- `log_error`: Function for logging errors

### 7. Decorators Module

The decorators module provides decorators for common functionality:

- `@log_function`: Decorator to log function calls and results
- `@retry`: Decorator to retry a function on failure
- `@cache_result`: Decorator to cache function results
- `@validate_args`: Decorator to validate function arguments

## Embedding LLMs

### Overview

The embedding module supports multiple embedding providers:

1. **Ollama**: Local embedding generation using Ollama API
2. **OpenAI**: Cloud-based embedding generation using OpenAI API
3. **Sentence Transformers**: Local embedding generation using Sentence Transformers library
4. **Fallback**: Random embedding generation as a last resort

### Configuration

You can configure the embedding generator using environment variables or a configuration dictionary:

```python
# Using environment variables
os.environ["OLLAMA_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "all-minilm-l6-v2"
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_EMBEDDING_MODEL"] = "text-embedding-3-small"
os.environ["ST_MODEL"] = "all-MiniLM-L6-v2"

# Using configuration dictionary
config = {
    "ollama_url": "http://localhost:11434",
    "ollama_model": "all-minilm-l6-v2",
    "openai_api_key": "your-api-key",
    "openai_embedding_model": "text-embedding-3-small",
    "st_model": "all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "allow_fallback": True
}

embedding_generator = EmbeddingGenerator(config)
```

### Generating Embeddings

```python
# Generate embedding for a single text
text = "This is a test text for embedding generation."
embedding, error = embedding_generator.generate_embedding(text)

# Generate embeddings for all nodes in a graph
updated_graph, errors = batch_generate_embeddings(
    graph,
    embedding_generator,
    text_fields=["title", "overview", "description"],
    force_update=False
)
```

### Embedding Models

#### Ollama Models

- **all-minilm-l6-v2**: Fast and efficient embedding model
- **nomic-embed-text**: High-quality text embeddings
- **mxbai-embed-large**: Multilingual embedding model

#### OpenAI Models

- **text-embedding-3-small**: Fast and cost-effective embeddings (1536 dimensions)
- **text-embedding-3-large**: High-quality embeddings (3072 dimensions)
- **text-embedding-ada-002**: Legacy model (1536 dimensions)

#### Sentence Transformers Models

- **all-MiniLM-L6-v2**: Fast and efficient embedding model (384 dimensions)
- **all-mpnet-base-v2**: High-quality embeddings (768 dimensions)
- **paraphrase-multilingual-MiniLM-L12-v2**: Multilingual embedding model (384 dimensions)

### Embedding Similarity Search

```python
# Create a vector index
index = VectorIndex("embedding_index", "embedding")

# Build the index
index.build(graph)

# Search for similar embeddings
results = index.search(query_embedding, threshold=0.7, limit=10)

# Find similar nodes
similar_nodes = find_similar_nodes(
    graph,
    node_id,
    similarity_threshold=0.7,
    max_results=10
)
```

## Usage Example

```python
from src.models.indexing import IndexType, IndexManager
from src.models.embeddings import EmbeddingGenerator, batch_generate_embeddings
from src.models.graph_ops import auto_link_nodes, find_similar_nodes
from src.models.query_engine import query_graph
from src.utils.data_handler import load_graph_from_folder, save_node_to_yaml
from src.utils.logger import setup_logger, get_logger, log_function_call, log_function_result, log_error

# Load graph data
graph, errors = load_graph_from_folder("data")

# Generate embeddings
embedding_generator = EmbeddingGenerator()
graph, errors = batch_generate_embeddings(graph, embedding_generator)

# Create indexes
index_manager = IndexManager("indexes")
index_manager.create_index("title_index", "title", IndexType.HASH)
index_manager.create_index("tags_index", "tags", IndexType.HASH)
index_manager.create_index("embedding_index", "embedding", IndexType.VECTOR)
index_manager.build_all(graph)

# Auto-link nodes
linked_graph = auto_link_nodes(graph)

# Query graph
results = query_graph(linked_graph, "title = 'Test Node' AND tags contains 'test'")

# Find similar nodes
similar_nodes = find_similar_nodes(linked_graph, "node1", similarity_threshold=0.7)

# Save node
save_node_to_yaml(linked_graph["node1"], "data")
```

## Viewing Logs

GraphYML includes a comprehensive logging system to help with debugging and monitoring. Logs are stored in the `logs` directory by default.

### Using the Logger Module

```python
from src.utils.logger import get_logger

# Get a logger
logger = get_logger("my_module")

# Log messages
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# Log with context
logger.info("Processing node %s", node_id)
logger.error("Failed to load file: %s", error)
```

### Using Decorators

```python
from src.utils.decorators import log_function, retry, cache_result

# Log function calls and results
@log_function()
def process_node(node_id):
    # Function code here
    return result

# Retry on failure
@retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(ConnectionError,))
def fetch_data(url):
    # Function code here
    return data

# Cache results
@cache_result(ttl=60)  # Cache for 60 seconds
def expensive_calculation(x, y):
    # Function code here
    return result
```

### Viewing Logs

GraphYML includes a script to view and analyze logs:

```bash
# View all logs
python scripts/view_logs.py

# Filter by log level
python scripts/view_logs.py --level ERROR

# Filter by logger name
python scripts/view_logs.py --logger embeddings

# Filter by time range
python scripts/view_logs.py --start-time "2023-01-01 00:00:00" --end-time "2023-01-02 00:00:00"

# Filter by message pattern
python scripts/view_logs.py --message "Error loading file"

# Show only the last 10 lines
python scripts/view_logs.py --tail 10

# Follow log file (like tail -f)
python scripts/view_logs.py --follow

# Custom output format
python scripts/view_logs.py --format "%t - %m"
```

### Log Configuration

You can configure logging in the `config.json` file:

```json
{
  "logging": {
    "log_level": "INFO",
    "log_dir": "logs",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "console_output": true,
    "max_bytes": 10485760,
    "backup_count": 5
  }
}
```
