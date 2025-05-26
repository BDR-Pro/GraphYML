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

- 🔍 **Powerful Querying**
  - Search nodes by tag, genre, director, or custom fields
  - A* pathfinding across similar or connected nodes

- 🧪 **YAML Schema Validation**
  - Checks each node against expected structure
  - Reports missing or invalid fields

- 🧬 **Graph Visualization**
  - View clustered graphs using TSNE + KMeans
  - Interactive network view via PyVis

- 🧾 **Node Editor**
  - View and edit YAML structure directly in-browser
  - Save changes back to file or extended copies

- 📦 **Export**
  - Download selected folder as ZIP archive

---

## 📁 Folder Structure

```bash
project-root/
│
├── db.py                  # Streamlit application
├── requirements.txt       # Python dependencies
├── saved_yamls/           # YAML storage folders (by collection)
├── graph_config.json      # Persistent settings
├── README.md              # This file
````

---

## 🚀 How to Run

### 1. Clone and install

```bash
git clone https://github.com/bdr-pro/GRAPHYML
cd GRAPHYML
pip install -r requirements.txt
```

### 2. Launch Streamlit

```bash
streamlit run db.py
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
  "edit_inline": true
}
```

---

## 🔭 Roadmap Ideas

- LLM-generated tag suggestions
- Prompt-based node creation
- Graph database export (Neo4j, RDF)
- Time-based navigation or node evolution

---

## 🧠 Built With

- [Streamlit](https://streamlit.io/)
- [PyYAML](https://pyyaml.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Cerberus](https://docs.python-cerberus.org/)
- [PyVis](https://pyvis.readthedocs.io/)

---

## 📝 License

MIT License — use, fork, and build on it freely.
