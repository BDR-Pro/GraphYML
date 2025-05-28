# Streamlit App: Refactored YAML Graph DB
"""
A refactored Streamlit app that allows:
- Uploading and browsing YAML graph nodes
- Editing nodes inline
- Exporting selected folders as ZIP
- A* pathfinding
- Tag search
- Clustering using TSNE + KMeans
All logic is now inside functions with docstrings.
"""
# pylint: disable=invalid-name,broad-except

from io import BytesIO
from pathlib import Path
from collections import defaultdict
import heapq
import zipfile
import os
import json
import tempfile

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import yaml
CONFIG_PATH = "graph_config.json"


def load_config():
    """Load or initialize configuration file."""
    # Check if config_path exists and is a file (not a directory)
    if os.path.exists(CONFIG_PATH) and os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config: {e}")
            # Return default config on error
            return {
                "save_path": "saved_yamls",
                "ollama_url": "http://localhost:11434/api/embeddings",
                "ollama_model": "all-minilm-l6-v2",
                "edit_inline": True
            }
    elif os.path.exists(CONFIG_PATH) and not os.path.isfile(CONFIG_PATH):
        # If it exists but is not a file (e.g., it's a directory), use defaults
        print(f"Warning: {CONFIG_PATH} exists but is not a file. Using default configuration.")
        return {
            "save_path": "saved_yamls",
            "ollama_url": "http://localhost:11434/api/embeddings",
            "ollama_model": "all-minilm-l6-v2",
            "edit_inline": True
        }
    return {
        "save_path": "saved_yamls",
        "ollama_url": "http://localhost:11434/api/embeddings",
        "ollama_model": "all-minilm-l6-v2",
        "edit_inline": True
    }


def save_config(config):
    """Save configuration to disk."""
    # Check if CONFIG_PATH is a directory
    if os.path.exists(CONFIG_PATH) and not os.path.isfile(CONFIG_PATH):
        # If it's a directory, remove it and create a file
        try:
            import shutil
            shutil.rmtree(CONFIG_PATH)
        except (IOError, OSError) as e:
            print(f"Error removing directory {CONFIG_PATH}: {e}")
            # Try an alternative path
            alt_config_path = "graph_config_new.json"
            print(f"Using alternative config path: {alt_config_path}")
            with open(alt_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            return
    
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except (IOError, TypeError) as e:
        print(f"Error saving config: {e}")


def create_zip(folder):
    """Compress all YAML files in a folder into a zip archive."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for file in Path(folder).glob("*.yaml"):
            zipf.write(file, arcname=file.name)
    zip_buffer.seek(0)
    return zip_buffer


def flatten_node(d):
    """Flatten nested dict or list values to a flat list of strings."""
    values = []
    if isinstance(d, dict):
        for v in d.values():
            values += flatten_node(v)
    elif isinstance(d, list):
        for item in d:
            values += flatten_node(item)
    elif isinstance(d, str):
        values.append(d)
    return values


def query_by_tag(graph, tag):
    """Return keys where tag is found in any nested field."""
    return [key for key, node in graph.items() if tag in flatten_node(node)]


def embedding_similarity(a, b):
    """Compute cosine similarity between two embedding vectors."""
    if not a or not b:
        return 0.0
    a, b = np.array(a).reshape(1, -1), np.array(b).reshape(1, -1)
    return cosine_similarity(a, b)[0][0]


def a_star(graph, start, goal):
    """Run A* search on graph based on embedding + tag similarity."""
    def heuristic(a, b):
        return 1 - embedding_similarity(graph[a].get('embedding'), graph[b].get('embedding'))

    def tag_sim(a, b):
        return len(set(a) & set(b)) / max(1, len(set(a + b)))

    open_set = [(0, start)]
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    f_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    f_score[start] = heuristic(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph[current].get('links', []):
            if neighbor not in graph:
                continue
            tentative_g = g_score[current] + (1 - tag_sim(
                graph[current].get('tags', []), graph[neighbor].get('tags', [])))
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None


def reconstruct_path(came_from, current):
    """Rebuild full path from goal to start node."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


def cluster_and_plot(graph):
    """Cluster nodes with TSNE + KMeans and draw graph."""
    keys = list(graph.keys())
    embs = [graph[k]['embedding'] for k in keys if 'embedding' in graph[k]]
    if len(embs) <= 1:
        st.warning("Not enough embeddings to cluster.")
        return

    X = np.array(embs)
    perplexity = min(30, len(X) - 1)
    _reduced = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X)
    labels = KMeans(n_clusters=min(4, len(X)), random_state=42).fit_predict(X)

    G = nx.Graph()
    for i, key in enumerate(keys):
        G.add_node(key, group=labels[i])
    for key, data in graph.items():
        for neighbor in data.get('links', []):
            if neighbor in graph:
                G.add_edge(key, neighbor)

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G)
    color_map = [G.nodes[n].get('group', 0.5) for n in G.nodes()]
    nx.draw(G, pos, node_color=color_map, with_labels=True,
            node_size=2000, cmap=plt.cm.viridis, ax=ax)
    st.pyplot(fig)

def handle_zip_upload(uploaded_zip, save_dir):
    """Extract a ZIP file into the configured save directory."""

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "uploaded.zip"
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        for file in Path(tmpdir).rglob("*.yaml"):
            rel_folder = file.relative_to(tmpdir).parent
            dest_folder = Path(save_dir) / rel_folder
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_path = dest_folder / file.name
            with open(dest_path, "wb") as f_out:
                f_out.write(file.read_bytes())


def handle_yaml_uploads(uploaded_files, save_dir):
    """Save uploaded YAML files, using simulated folder paths from their filenames."""
    folders = set()
    for f in uploaded_files:
        folder = Path(f.name).parts[0] if "/" in f.name else "root"
        folders.add(folder)
        folder_path = Path(save_dir) / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        with open(folder_path / Path(f.name).name, "wb") as out:
            out.write(f.getbuffer())
    return folders


def load_graph_from_folder(folder_path):
    """Load all YAML files in a given folder into a graph dictionary."""

    graph = {}
    for yml in Path(folder_path).glob("*.yaml"):
        with open(yml, encoding='utf-8') as f:
            try:
                node = yaml.safe_load(f)
                if node:
                    key = node.get("id") or node.get("title")
                    if key:
                        graph[key] = node
            except Exception as e:
                st.error(f"Failed to load {yml.name}: {e}")
    return graph

import cerberus
import pyvis.network as net

SCHEMA = {
    "id": {"type": "string", "required": True},
    "title": {"type": "string", "required": True},
    "tags": {"type": "list", "schema": {"type": "string"}},
    "genres": {"type": "list", "schema": {"type": "string"}},
    "links": {"type": "list", "schema": {"type": "string"}},
    "embedding": {"type": "list"},
}


def validate_node_schema(node):
    """Validate a YAML node against the expected schema."""
    validator = cerberus.Validator(SCHEMA)
    return validator.validate(node), validator.errors


def auto_link_nodes(graph):
    """Automatically link nodes with overlapping genres or tags."""
    for key, node in graph.items():
        if 'links' not in node:
            node['links'] = []
        for other_key, other_node in graph.items():
            if key == other_key:
                continue
            shared = set(node.get("tags", [])) & set(other_node.get("tags", []))
            shared |= set(node.get("genres", [])) & set(other_node.get("genres", []))
            if shared and other_key not in node['links']:
                node['links'].append(other_key)


def search_graph(graph, query, field):
    """Search graph nodes by a given field value (e.g., director, genre)."""
    results = []
    for key, node in graph.items():
        value = node.get(field)
        if isinstance(value, str) and query.lower() in value.lower():
            results.append((key, value))
        elif isinstance(value, list):
            if any(query.lower() in str(v).lower() for v in value):
                results.append((key, value))
    return results


def visualize_graph(graph):
    """Render graph using PyVis and return HTML content."""
    G = net.Network(height='600px', width='100%', directed=False)
    for key, node in graph.items():
        G.add_node(key, label=key, title=node.get("title", ""))
    for key, node in graph.items():
        for neighbor in node.get("links", []):
            if neighbor in graph:
                G.add_edge(key, neighbor)
    G.repulsion(node_distance=200)
    return G.generate_html()


def show_processing_spinner(label):
    """Show spinner during long processing blocks."""
    return st.spinner(label)


def main():
    """Main entry point for the Streamlit app."""

    st.set_page_config(page_title="YAML Graph DB", layout="wide")
    st.title("🧠 YAML Graph Knowledge DB")

    config = load_config()
    save_path = Path(config["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    uploaded_files = st.file_uploader(
        "📂 Upload YAML or ZIP files (drag folder or select .zip)",
        type=["yaml", "yml", "zip"],
        accept_multiple_files=True
    )

    # Handle uploads
    if uploaded_files:
        for file in uploaded_files:
            if file.name.endswith(".zip"):
                handle_zip_upload(file, save_path)
            else:
                handle_yaml_uploads([file], save_path)

    # Get available folders
    available_folders = sorted(
        [p.name for p in save_path.iterdir() if p.is_dir()]
    )

    if not available_folders:
        st.warning("No folders found. Please upload YAML or ZIP files.")
        return

    folder_context = st.selectbox("🗂 Select Folder", available_folders)
    context_dir = save_path / folder_context
    graph = load_graph_from_folder(context_dir)

    if not graph:
        st.info("No valid YAML nodes found in this folder.")
        return

    st.download_button(
        "📦 Download Folder as ZIP",
        create_zip(context_dir),
        file_name=f"{folder_context}.zip"
    )

    # YAML Editor
    selected = st.selectbox("📝 Edit Node", list(graph))
    if selected:
        current_yaml = yaml.dump(graph[selected], sort_keys=False)
        edited = st.text_area("Edit YAML", current_yaml, height=300)
        if st.button("💾 Save"):
            try:
                updated = yaml.safe_load(edited)
                with open(context_dir / f"{selected}.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(updated, f, sort_keys=False)
                st.success("Node saved.")
            except yaml.YAMLError as e:
                st.error(f"YAML Error: {e}")

    # A* Pathfinding
    st.divider()
    st.subheader("🔁 A* Pathfinding")
    col1, col2 = st.columns(2)
    with col1:
        start = st.selectbox("Start Node", list(graph))
    with col2:
        goal = st.selectbox("Goal Node", list(graph), index=1)
    if st.button("▶️ Find Path"):
        path = a_star(graph, start, goal)
        if path:
            st.success(" → ".join(path))
        else:
            st.warning("No path found.")

    # Tag search
    st.divider()
    tag = st.text_input("🔍 Search by Tag")
    if tag:
        matches = query_by_tag(graph, tag)
        st.info(f"Found {len(matches)} match(es):")
        st.write(matches)

    # Clustering
    st.divider()
    st.subheader("📊 Cluster View")
    if st.button("🧬 Cluster & Visualize"):
        cluster_and_plot(graph)

if __name__ == "__main__":
    main()
