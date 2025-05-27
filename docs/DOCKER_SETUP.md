# Docker Setup Guide

GraphYML can be deployed using Docker Compose, which provides a multi-container setup for better scalability and performance.

## Architecture

The Docker Compose setup includes the following services:

1. **GraphYML Application** (`graphyml`): The main Streamlit application that provides the user interface and core functionality.

2. **Embedding Service** (`embedding-service`): A dedicated service for generating embeddings using Hugging Face models. This service can be scaled independently for better performance.

3. **Ollama Service** (`ollama`): An optional service for local LLM embeddings using Ollama. This is useful for environments without internet access or when you want to use custom models.

## Prerequisites

- Docker and Docker Compose installed on your system
- At least 8GB of RAM for running all services (4GB minimum for just the GraphYML application)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GraphYML.git
   cd GraphYML
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Access the application:
   - GraphYML UI: http://localhost:8501
   - Embedding Service API: http://localhost:8000
   - Ollama API: http://localhost:11434

## Configuration

### Environment Variables

You can configure the services using environment variables in the `docker-compose.yml` file:

#### GraphYML Application

- `EMBEDDING_SERVICE`: URL of the embedding service (default: `http://embedding-service:8000/api/embeddings`)
- `STREAMLIT_SERVER_PORT`: Port for the Streamlit server (default: `8501`)
- `STREAMLIT_SERVER_HEADLESS`: Run Streamlit in headless mode (default: `true`)

#### Embedding Service

- `MODEL_NAME`: Name of the Hugging Face model to use (default: `all-MiniLM-L6-v2`)
- `PORT`: Port for the embedding service (default: `8000`)

#### Ollama Service

- `OLLAMA_HOST`: Host for the Ollama service (default: `0.0.0.0`)

### Volumes

The Docker Compose setup includes the following volumes:

- `./saved_yamls:/app/saved_yamls`: Persists the graph data
- `./graph_config.json:/app/graph_config.json`: Persists the configuration
- `embedding-cache:/root/.cache/huggingface`: Caches the Hugging Face models
- `ollama-models:/root/.ollama`: Stores the Ollama models

## Scaling

You can scale the embedding service for better performance:

```bash
docker-compose up -d --scale embedding-service=3
```

This will start 3 instances of the embedding service, and the GraphYML application will automatically load balance between them.

## Resource Limits

The Docker Compose file includes resource limits for each service:

- GraphYML Application: No explicit limits
- Embedding Service: 4GB memory limit, 2GB memory reservation
- Ollama Service: 8GB memory limit, 4GB memory reservation

You can adjust these limits in the `docker-compose.yml` file based on your system resources.

## Troubleshooting

### Embedding Service Not Available

If the embedding service is not available, the GraphYML application will fall back to local embedding generation using the configured provider. You can check the logs for error messages:

```bash
docker-compose logs embedding-service
```

### Out of Memory Errors

If you encounter out of memory errors, you can adjust the resource limits in the `docker-compose.yml` file or reduce the number of services running:

```bash
# Run only the GraphYML application and embedding service
docker-compose up -d graphyml embedding-service
```

### Slow Embedding Generation

If embedding generation is slow, you can try:

1. Scaling the embedding service:
   ```bash
   docker-compose up -d --scale embedding-service=3
   ```

2. Using a smaller model:
   ```yaml
   embedding-service:
     environment:
       - MODEL_NAME=all-MiniLM-L6-v2  # Smaller, faster model
   ```

3. Using Ollama with a smaller model:
   ```yaml
   ollama:
     environment:
       - OLLAMA_MODEL=all-minilm  # Smaller, faster model
   ```

## Advanced Configuration

### Using OpenAI Embeddings

To use OpenAI embeddings, add the OpenAI API key to the GraphYML application service:

```yaml
graphyml:
  environment:
    - OPENAI_API_KEY=your-api-key
    - EMBEDDING_PROVIDER=openai
```

### Custom Embedding Models

To use a custom embedding model, update the `MODEL_NAME` environment variable:

```yaml
embedding-service:
  environment:
    - MODEL_NAME=your-custom-model
```

### Custom Ollama Models

To use a custom Ollama model, you need to pull it first:

```bash
docker-compose exec ollama ollama pull your-custom-model
```

Then update the configuration to use it:

```yaml
graphyml:
  environment:
    - EMBEDDING_PROVIDER=ollama
    - EMBEDDING_MODEL=your-custom-model
```

