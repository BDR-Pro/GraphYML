"""
Graph operations for GraphYML.
Includes pathfinding, linking, and other graph algorithms.
"""
from collections import defaultdict
import heapq

from src.models.embeddings import embedding_similarity


def auto_link_nodes(graph):
    """
    Automatically link nodes with overlapping genres or tags.
    
    Args:
        graph (dict): Graph dictionary
        
    Returns:
        dict: Updated graph with auto-linked nodes
    """
    for key, node in graph.items():
        if 'links' not in node:
            node['links'] = []
        
        for other_key, other_node in graph.items():
            if key == other_key:
                continue
                
            # Find shared tags or genres
            shared_tags = set(node.get("tags", [])) & set(other_node.get("tags", []))
            shared_genres = set(node.get("genres", [])) & set(other_node.get("genres", []))
            
            # If there are shared elements and not already linked, add link
            if (shared_tags or shared_genres) and other_key not in node['links']:
                node['links'].append(other_key)
    
    return graph


def tag_similarity(a, b):
    """
    Calculate Jaccard similarity between two lists of tags.
    
    Args:
        a (list): First list of tags
        b (list): Second list of tags
        
    Returns:
        float: Similarity score (0-1)
    """
    if not a or not b:
        return 0.0
    
    set_a = set(a)
    set_b = set(b)
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / max(1, union)


def a_star(graph, start, goal):
    """
    Run A* search on graph based on embedding + tag similarity.
    
    Args:
        graph (dict): Graph dictionary
        start (str): Start node key
        goal (str): Goal node key
        
    Returns:
        list or None: Path from start to goal, or None if no path exists
    """
    if start not in graph or goal not in graph:
        return None
        
    def heuristic(a, b):
        """Calculate heuristic distance between nodes."""
        return 1 - embedding_similarity(
            graph[a].get('embedding'), 
            graph[b].get('embedding')
        )

    def edge_cost(a, b):
        """Calculate edge cost based on tag similarity."""
        return 1 - tag_similarity(
            graph[a].get('tags', []), 
            graph[b].get('tags', [])
        )

    # Initialize data structures
    open_set = [(0, start)]  # Priority queue of (f_score, node)
    came_from = {}  # Path reconstruction dictionary
    g_score = defaultdict(lambda: float('inf'))  # Cost from start to node
    f_score = defaultdict(lambda: float('inf'))  # Estimated total cost
    
    g_score[start] = 0
    f_score[start] = heuristic(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)
        
        # Goal reached
        if current == goal:
            return reconstruct_path(came_from, current)

        # Explore neighbors
        for neighbor in graph[current].get('links', []):
            if neighbor not in graph:
                continue
                
            # Calculate tentative g_score
            tentative_g = g_score[current] + edge_cost(current, neighbor)
            
            # If this path is better than previous ones
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                
                # Add to open set if not already there
                if not any(node == neighbor for _, node in open_set):
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return None


def reconstruct_path(came_from, current):
    """
    Rebuild full path from goal to start node.
    
    Args:
        came_from (dict): Dictionary mapping nodes to their predecessors
        current (str): Current (goal) node
        
    Returns:
        list: Path from start to goal
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # Reverse to get start-to-goal order


def find_similar_nodes(graph, node_key, top_n=5):
    """
    Find the most similar nodes to a given node based on embedding similarity.
    
    Args:
        graph (dict): Graph dictionary
        node_key (str): Key of the node to find similar nodes for
        top_n (int): Number of similar nodes to return
        
    Returns:
        list: List of (node_key, similarity_score) tuples
    """
    if node_key not in graph or 'embedding' not in graph[node_key]:
        return []
        
    target_embedding = graph[node_key]['embedding']
    
    similarities = []
    for key, node in graph.items():
        if key == node_key or 'embedding' not in node:
            continue
            
        similarity = embedding_similarity(target_embedding, node['embedding'])
        similarities.append((key, similarity))
    
    # Sort by similarity (descending) and return top_n
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

