"""
Main Streamlit application for GraphYML.
Integrates all components and provides the user interface.
"""
import streamlit as st
from pathlib import Path

from src.config.settings import load_config, save_config, ensure_directories
from src.utils.data_handler import (
    load_graph_from_folder, create_zip, handle_yaml_uploads,
    handle_zip_upload, save_node_to_yaml, query_by_tag
)
from src.models.embeddings import EmbeddingGenerator, batch_generate_embeddings
from src.models.graph_ops import auto_link_nodes, a_star, find_similar_nodes
from src.visualization.graph_viz import cluster_and_plot, visualize_graph


def init_session_state():
    """Initialize session state variables if they don't exist."""
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
        st.session_state.config = ensure_directories(st.session_state.config)
    
    if 'embedding_generator' not in st.session_state:
        st.session_state.embedding_generator = EmbeddingGenerator(st.session_state.config)
    
    if 'current_graph' not in st.session_state:
        st.session_state.current_graph = {}
    
    if 'current_folder' not in st.session_state:
        st.session_state.current_folder = None
    
    if 'load_errors' not in st.session_state:
        st.session_state.load_errors = []


def show_settings_ui():
    """Display and handle settings UI."""
    st.subheader("⚙️ Settings")
    
    config = st.session_state.config
    
    col1, col2 = st.columns(2)
    
    with col1:
        save_path = st.text_input("Save Path", config["save_path"])
        ollama_url = st.text_input("Ollama API URL", config["ollama_url"])
    
    with col2:
        ollama_model = st.text_input("Ollama Model", config["ollama_model"])
        edit_inline = st.checkbox("Edit Inline", config["edit_inline"])
    
    if st.button("Save Settings"):
        config["save_path"] = save_path
        config["ollama_url"] = ollama_url
        config["ollama_model"] = ollama_model
        config["edit_inline"] = edit_inline
        
        save_config(config)
        st.session_state.config = config
        st.session_state.embedding_generator = EmbeddingGenerator(config)
        
        st.success("Settings saved!")


def handle_uploads():
    """Handle file uploads and return created folders."""
    uploaded_files = st.file_uploader(
        "📂 Upload YAML or ZIP files (drag folder or select .zip)",
        type=["yaml", "yml", "zip"],
        accept_multiple_files=True
    )
    
    folders = set()
    if uploaded_files:
        save_path = Path(st.session_state.config["save_path"])
        
        with st.spinner("Processing uploads..."):
            for file in uploaded_files:
                if file.name.endswith(".zip"):
                    new_folders = handle_zip_upload(file, save_path)
                    folders.update(new_folders)
                else:
                    new_folders = handle_yaml_uploads([file], save_path)
                    folders.update(new_folders)
        
        st.success(f"Uploaded {len(uploaded_files)} file(s) to {len(folders)} folder(s)")
    
    return list(folders)


def show_folder_selector(available_folders):
    """Display folder selector and load selected folder."""
    folder_context = st.selectbox(
        "🗂 Select Folder", 
        available_folders,
        index=0 if available_folders else None
    )
    
    if folder_context:
        context_dir = Path(st.session_state.config["save_path"]) / folder_context
        
        # Only reload if folder changed
        if st.session_state.current_folder != folder_context:
            with st.spinner("Loading graph..."):
                graph, errors = load_graph_from_folder(context_dir)
                
                if graph:
                    st.session_state.current_graph = graph
                    st.session_state.current_folder = folder_context
                    st.session_state.load_errors = errors
                    
                    # Auto-link nodes
                    st.session_state.current_graph = auto_link_nodes(st.session_state.current_graph)
                    
                    st.success(f"Loaded {len(graph)} nodes from {folder_context}")
                    if errors:
                        st.warning(f"Found {len(errors)} errors while loading")
                else:
                    st.error("No valid YAML nodes found in this folder")
        
        # Download button
        st.download_button(
            "📦 Download Folder as ZIP",
            create_zip(context_dir),
            file_name=f"{folder_context}.zip"
        )
        
        return folder_context, context_dir
    
    return None, None


def show_node_editor(context_dir):
    """Display node editor UI."""
    st.subheader("📝 Node Editor")
    
    graph = st.session_state.current_graph
    if not graph:
        st.info("No nodes available to edit")
        return
    
    selected = st.selectbox("Select Node", list(graph))
    if not selected:
        return
    
    current_yaml = st.text_area(
        "Edit YAML", 
        value=st.session_state.get(f"yaml_{selected}", ""),
        height=300
    )
    
    # Store in session state to preserve during reruns
    st.session_state[f"yaml_{selected}"] = current_yaml
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save"):
            try:
                import yaml
                updated = yaml.safe_load(current_yaml)
                
                success, error = save_node_to_yaml(
                    updated, 
                    context_dir, 
                    f"{selected}.yaml"
                )
                
                if success:
                    # Update in-memory graph
                    graph[selected] = updated
                    st.success("Node saved successfully")
                else:
                    st.error(f"Failed to save node: {error}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("🔄 Reset"):
            # Reset to original
            st.session_state[f"yaml_{selected}"] = yaml.dump(
                graph[selected], 
                sort_keys=False
            )
            st.rerun()


def show_pathfinding():
    """Display A* pathfinding UI."""
    st.subheader("🔁 A* Pathfinding")
    
    graph = st.session_state.current_graph
    if not graph:
        st.info("No nodes available for pathfinding")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        start = st.selectbox("Start Node", list(graph), key="path_start")
    
    with col2:
        goal = st.selectbox(
            "Goal Node", 
            list(graph), 
            index=min(1, len(graph)-1),
            key="path_goal"
        )
    
    if st.button("▶️ Find Path"):
        if start == goal:
            st.warning("Start and goal nodes are the same")
            return
            
        with st.spinner("Finding path..."):
            path = a_star(graph, start, goal)
            
        if path:
            st.success(" → ".join(path))
            
            # Show details of path
            st.write("Path details:")
            for i, node_key in enumerate(path):
                node = graph[node_key]
                st.write(f"{i+1}. **{node.get('title', node_key)}**")
                
                # Show tags if available
                if node.get('tags'):
                    st.write(f"   Tags: {', '.join(node['tags'])}")
        else:
            st.warning("No path found between these nodes")


def show_tag_search():
    """Display tag search UI."""
    st.subheader("🔍 Search")
    
    graph = st.session_state.current_graph
    if not graph:
        st.info("No nodes available for search")
        return
    
    search_term = st.text_input("Search Term")
    
    if search_term:
        matches = query_by_tag(graph, search_term)
        
        if matches:
            st.success(f"Found {len(matches)} match(es)")
            
            for key in matches:
                node = graph[key]
                st.write(f"- **{node.get('title', key)}**")
                
                # Show snippet of matching content
                for field in ['overview', 'tagline']:
                    if field in node and search_term.lower() in node[field].lower():
                        st.write(f"  *{field}*: ...{node[field]}...")
        else:
            st.warning("No matches found")


def show_embedding_generator():
    """Display embedding generation UI."""
    st.subheader("🧠 Embedding Generation")
    
    graph = st.session_state.current_graph
    if not graph:
        st.info("No nodes available for embedding generation")
        return
    
    # Count nodes with/without embeddings
    with_emb = sum(1 for node in graph.values() if 'embedding' in node)
    without_emb = len(graph) - with_emb
    
    st.write(f"Nodes with embeddings: {with_emb}")
    st.write(f"Nodes without embeddings: {without_emb}")
    
    if without_emb > 0:
        if st.button("Generate Missing Embeddings"):
            with st.spinner(f"Generating embeddings for {without_emb} nodes..."):
                updated_graph, errors = batch_generate_embeddings(
                    graph,
                    st.session_state.embedding_generator
                )
                
                st.session_state.current_graph = updated_graph
                
                if errors:
                    st.warning(f"Encountered {len(errors)} errors during embedding generation")
                    for node_key, error in errors:
                        st.write(f"- Error for {node_key}: {error}")
                else:
                    st.success("All embeddings generated successfully")


def show_visualization():
    """Display graph visualization UI."""
    st.subheader("📊 Visualization")
    
    graph = st.session_state.current_graph
    if not graph:
        st.info("No nodes available for visualization")
        return
    
    viz_type = st.radio(
        "Visualization Type",
        ["Clustering", "Interactive Network"],
        horizontal=True
    )
    
    if viz_type == "Clustering":
        if st.button("🧮 Cluster & Visualize"):
            with st.spinner("Clustering and plotting..."):
                fig, success = cluster_and_plot(graph, st.session_state.config)
                
                if success and fig:
                    st.pyplot(fig)
                else:
                    st.warning("Not enough embeddings to cluster")
    
    else:  # Interactive Network
        if st.button("🕸️ Generate Interactive Network"):
            with st.spinner("Generating network visualization..."):
                html = visualize_graph(graph)
                
                # Display in an iframe
                st.components.v1.html(html, height=600)


def show_similar_nodes():
    """Display similar nodes UI."""
    st.subheader("🔄 Find Similar Nodes")
    
    graph = st.session_state.current_graph
    if not graph:
        st.info("No nodes available")
        return
    
    # Count nodes with embeddings
    nodes_with_emb = [k for k, v in graph.items() if 'embedding' in v]
    
    if not nodes_with_emb:
        st.warning("No nodes have embeddings. Generate embeddings first.")
        return
    
    selected = st.selectbox("Select Node", nodes_with_emb)
    
    if selected and st.button("Find Similar"):
        with st.spinner("Finding similar nodes..."):
            similar = find_similar_nodes(graph, selected, top_n=5)
            
            if similar:
                st.success(f"Found {len(similar)} similar nodes")
                
                for key, score in similar:
                    node = graph[key]
                    st.write(f"- **{node.get('title', key)}** (Similarity: {score:.2f})")
                    
                    # Show tags if available
                    if node.get('tags'):
                        st.write(f"  Tags: {', '.join(node['tags'])}")
            else:
                st.warning("No similar nodes found")


def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="YAML Graph DB",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 YAML Graph Knowledge DB")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for settings
    with st.sidebar:
        show_settings_ui()
        
        st.divider()
        
        # Show errors if any
        if st.session_state.load_errors:
            with st.expander(f"Loading Errors ({len(st.session_state.load_errors)})"):
                for filename, error in st.session_state.load_errors:
                    st.write(f"- **{filename}**: {error}")
    
    # Main content
    created_folders = handle_uploads()
    
    # Get available folders
    save_path = Path(st.session_state.config["save_path"])
    available_folders = sorted(
        [p.name for p in save_path.iterdir() if p.is_dir()]
    )
    
    if not available_folders:
        st.warning("No folders found. Please upload YAML or ZIP files.")
        return
    
    # Select folder and load graph
    folder_context, context_dir = show_folder_selector(available_folders)
    
    if not folder_context or not st.session_state.current_graph:
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Editor", 
        "🔍 Search & Path", 
        "🧠 Embeddings",
        "📊 Visualization",
        "ℹ️ Info"
    ])
    
    with tab1:
        show_node_editor(context_dir)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            show_tag_search()
        
        with col2:
            show_pathfinding()
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            show_embedding_generator()
        
        with col2:
            show_similar_nodes()
    
    with tab4:
        show_visualization()
    
    with tab5:
        st.subheader("📊 Graph Statistics")
        
        graph = st.session_state.current_graph
        
        st.write(f"**Total Nodes:** {len(graph)}")
        
        # Count links
        total_links = sum(len(node.get('links', [])) for node in graph.values())
        st.write(f"**Total Links:** {total_links}")
        
        # Count nodes with embeddings
        with_emb = sum(1 for node in graph.values() if 'embedding' in node)
        st.write(f"**Nodes with Embeddings:** {with_emb} ({with_emb/len(graph)*100:.1f}%)")
        
        # Most connected nodes
        most_connected = sorted(
            [(k, len(v.get('links', []))) for k, v in graph.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        st.write("**Most Connected Nodes:**")
        for key, count in most_connected:
            node = graph[key]
            st.write(f"- {node.get('title', key)}: {count} links")


if __name__ == "__main__":
    main()

