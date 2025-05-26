# 🧠 YAML Graph Knowledge DB

A powerful, interactive Streamlit application to explore, edit, visualize, and query a graph-based database of YAML nodes — ideal for movie metadata, research articles, or structured knowledge graphs.

---

## ✨ Features

- 📂 **Upload Support**
  - Drag & drop YAML files (flat or nested folders)
  - Upload ZIP archives for bulk processing

- 🛠 **Graph Construction**
  - Nodes linked by shared tags, genres, or custom logic
  - Auto-link related nodes with overlapping metadata

- 🧠 **LLM-Ready Embeddings**
  - Plug in local models (via Ollama API) for script embeddings
  - Used for similarity, clustering, and pathfinding
  - Support for multiple embedding providers (Ollama, OpenAI, Hugging Face)

- 🔍 **Powerful Querying**
  - Search nodes by tag, genre, director, or custom fields
  - A* pathfinding across similar or connected nodes

- 🧪 **YAML Schema Validation**
  - Checks each node against expected structure
  - Reports missing or invalid fields

- 🧬 **Graph Visualization**
  - View clustered graphs using TSNE + KMeans
  - Interactive network view via PyVis

- 🗺 **Node Editor**
  - View and edit YAML structure directly in-browser
  - Save changes back to file or extended copies

- 📦 **Export**
  - Download selected folder as ZIP archive

---

## 📁 Project Structure

```bash
project-root/
│
├── src/                    # Source code
│   ├── app.py              # Main Streamlit application
│   ├── config/             # Configuration management
│   ├── models/             # Embedding and graph algorithms
│   ├── utils/              # Data handling utilities
│   └── visualization/      # Visualization components
│
├── scripts/                # Utility scripts
│   └── hf_embedding_server.py  # Hugging Face embedding server
│
├── docs/                   # Documentation
│   └── LLM_EMBEDDING_GUIDE.md  # Guide for embedding integration
│
├── cleaned_data/           # Data processing scripts
│   ├── main.py             # CSV to YAML converter
│   └── tmdb-movies.csv     # Sample dataset
│
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── graph_config.json       # Persistent settings
└── README.md               # This file
```

---

## 🚀 How to Run

### 1. Clone and install

```bash
git clone https://github.com/bdr-pro/GraphYML
cd GraphYML
pip install -r requirements.txt
```

### 2. Launch Streamlit

```bash
streamlit run src/app.py
```

### 3. Docker (optional)

```bash
docker build -t graphyml .
docker run -p 8501:8501 graphyml
```

---

## 📦 Upload Format

Each YAML file should have this structure:

```yaml
id: tt1375666
title: Inception
genres: [Action, Sci-Fi]
tags: [dreams, mind-bending, heist]
script: |
  Dom Cobb is a skilled thief, the absolute best in the dangerous art of extraction...
links: []
```

---

## 🔧 Configuration

Settings are saved in `graph_config.json`. You can change:

```json
{
  "save_path": "saved_yamls",
  "ollama_url": "http://localhost:11434/api/embeddings",
  "ollama_model": "all-minilm-l6-v2",
  "edit_inline": true,
  "embedding_dimension": 384,
  "max_cluster_count": 4,
  "perplexity": 30,
  "node_distance": 200
}
```

---

## 🧠 LLM Embedding Integration

GraphYML supports multiple embedding providers:

1. **Ollama** (default) - Run models locally
2. **OpenAI** - High-quality cloud embeddings
3. **Hugging Face** - Self-hosted open-source models

See [LLM Embedding Guide](docs/LLM_EMBEDDING_GUIDE.md) for detailed setup instructions.

---

## 🔄 CSV to YAML Conversion

Convert structured CSV data to YAML nodes:

```bash
python -m src.utils.csv_to_yaml cleaned_data/tmdb-movies.csv output_folder
```

---

## 🔭 Roadmap Ideas

- LLM-generated tag suggestions
- Prompt-based node creation
- Graph database export (Neo4j, RDF)
- Time-based navigation or node evolution
- Multi-modal embeddings (text + image)

---

## 🧠 Built With

- [Streamlit](https://streamlit.io/)
- [PyYAML](https://pyyaml.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Cerberus](https://docs.python-cerberus.org/)
- [PyVis](https://pyvis.readthedocs.io/)
- [Sentence-Transformers](https://www.sbert.net/)

---

## 📝 License

MIT License — use, fork, and build on it freely.

