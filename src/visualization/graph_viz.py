"""
Visualization utilities for GraphYML.
Includes clustering, plotting, and interactive visualization.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pyvis.network as net


def cluster_and_plot(graph, config=None):
    """
    Cluster nodes with TSNE + KMeans and draw graph.
    
    Args:
        graph (dict): Graph dictionary
        config (dict, optional): Configuration dictionary
        
    Returns:
        tuple: (fig, success_flag)
    """
    # Extract keys and embeddings
    keys = list(graph.keys())
    embs = [graph[k].get('embedding', []) for k in keys]
    embs = [e for e in embs if e]  # Filter out empty embeddings
    
    # Check if we have enough data
    if len(embs) <= 1:
        return None, False
    
    # Default configuration
    if config is None:
        config = {
            "perplexity": 30,
            "max_cluster_count": 4
        }
    
    # Prepare data
    X = np.array(embs)
    perplexity = min(config.get("perplexity", 30), len(X) - 1)
    n_clusters = min(config.get("max_cluster_count", 4), len(X))
    
    try:
        # Dimensionality reduction with TSNE
        reduced = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity
        ).fit_transform(X)
        
        # Clustering with KMeans
        labels = KMeans(
            n_clusters=n_clusters, 
            random_state=42
        ).fit_predict(X)
        
        # Create graph
        G = nx.Graph()
        for i, key in enumerate(keys[:len(embs)]):
            G.add_node(key, group=int(labels[i]))
            
        # Add edges
        for key, data in graph.items():
            for neighbor in data.get('links', []):
                if neighbor in graph:
                    G.add_edge(key, neighbor)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42)
        color_map = [G.nodes[n].get('group', 0) for n in G.nodes()]
        
        nx.draw(
            G, pos, 
            node_color=color_map, 
            with_labels=True,
            node_size=2000, 
            cmap=plt.cm.viridis, 
            ax=ax
        )
        
        return fig, True
        
    except Exception as e:
        print(f"Clustering error: {e}")
        return None, False


def visualize_graph(graph, height='600px', width='100%'):
    """
    Render graph using PyVis and return HTML content.
    
    Args:
        graph (dict): Graph dictionary
        height (str): Height of the visualization
        width (str): Width of the visualization
        
    Returns:
        str: HTML content for the visualization
    """
    # Create network
    G = net.Network(height=height, width=width, directed=False)
    
    # Add nodes
    for key, node in graph.items():
        title = f"{node.get('title', key)}"
        if node.get('tagline'):
            title += f"<br>{node.get('tagline')}"
            
        G.add_node(
            key, 
            label=node.get('title', key), 
            title=title
        )
    
    # Add edges
    for key, node in graph.items():
        for neighbor in node.get('links', []):
            if neighbor in graph:
                G.add_edge(key, neighbor)
    
    # Configure physics
    G.repulsion(node_distance=200, central_gravity=0.2)
    G.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "iterations": 100
        }
      }
    }
    """)
    
    # Generate HTML
    return G.generate_html()

