# GraphYML with Dash

A graph-based data management system for YAML files with embedding and querying capabilities, now with a Dash web interface.

## Features

- Store and manage graph data in YAML files
- Index data for fast querying
- Generate embeddings for semantic search
- Query data using a simple query language
- Automatically link related nodes
- Find similar nodes using embeddings
- Comprehensive logging system for debugging
- Web interface for managing nodes and relationships
- User authentication and permission management
- Backup and restore functionality

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

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (optional, for containerized deployment)

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GraphYML.git
   cd GraphYML
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements_dash.txt
   ```

3. Run the application:
   ```bash
   python run_dash_app.py
   ```

4. Open your browser and navigate to `http://localhost:8050`

### Option 2: Docker Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GraphYML.git
   cd GraphYML
   ```

2. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. Open your browser and navigate to `http://localhost:8050`

## Usage

### Authentication

- Default admin credentials: username `admin`, password `admin`
- Create new users through the User Management interface

### Managing Nodes

1. Navigate to the Node Editor to edit existing nodes
2. Use the Create Node interface to add new nodes
3. Link nodes by adding references in the node content

### Querying

1. Use the Query Interface to search for nodes
2. Perform text search, criteria-based search, or similarity search

### Visualization

1. Navigate to the Visualization tab
2. Choose between clustering or interactive network visualization

### Backup and Restore

1. Navigate to the Management tab
2. Use the Backup & Restore interface to create or restore backups

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

## Development

### Project Structure

- `src/dash_app.py`: Main Dash application
- `src/models/`: Core data models
- `src/visualization/`: Graph visualization utilities
- `src/config/`: Configuration management
- `src/utils/`: Utility functions

### Running Tests

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

