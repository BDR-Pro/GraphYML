# GraphYML with Dash

GraphYML is a graph database system that uses YAML files for storage and provides a web interface built with Dash.

## Features

- Graph database with YAML storage
- Web interface for managing nodes and relationships
- Embedding-based similarity search
- Visualization of graph data
- User authentication and permission management
- Backup and restore functionality

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

## Configuration

Configuration is stored in `graph_config.json` and can be modified through the Settings interface:

- `save_path`: Directory to store YAML files
- `ollama_url`: URL for the Ollama embedding service
- `ollama_model`: Model to use for embeddings
- `edit_inline`: Whether to enable inline editing

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

