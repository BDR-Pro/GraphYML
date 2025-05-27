"""
Graph operations module for GraphYML.
Provides functions for working with graph data.
"""
import heapq
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable


def tag_similarity(tags1: List[str], tags2: List[str]) -> float:
    """
    Calculate similarity between two tag lists.
    
    Args:
        tags1: First tag list
        tags2: Second tag list
        
    Returns:
        float: Similarity score (0-1)
    """
    # Handle empty tags
    if not tags1 or not tags2:
        return 0.0
    
    # Convert to sets
    set1 = set(tags1)
    set2 = set(tags2)
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def auto_link_nodes(graph: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Automatically link nodes based on shared tags or genres.
    
    Args:
        graph: Graph to link
        
    Returns:
        Dict[str, Dict[str, Any]]: Linked graph
    """
    # Create a copy of the graph
    linked_graph = {}
    
    for key, node in graph.items():
        # Create a copy of the node
        linked_graph[key] = node.copy()
        
        # Initialize links if not present
        if "links" not in linked_graph[key]:
            linked_graph[key]["links"] = []
    
    # Link nodes
    for key1, node1 in graph.items():
        for key2, node2 in graph.items():
            # Skip self-links
            if key1 == key2:
                continue
            
            # Check for shared tags
            tags1 = node1.get("tags", [])
            tags2 = node2.get("tags", [])
            
            if tags1 and tags2:
                tag_sim = tag_similarity(tags1, tags2)
                
                if tag_sim > 0.3:  # Threshold for linking
                    if key2 not in linked_graph[key1]["links"]:
                        linked_graph[key1]["links"].append(key2)
            
            # Check for shared genres
            genres1 = node1.get("genres", [])
            genres2 = node2.get("genres", [])
            
            if genres1 and genres2:
                # Check for any shared genre
                for genre in genres1:
                    if genre in genres2:
                        if key2 not in linked_graph[key1]["links"]:
                            linked_graph[key1]["links"].append(key2)
                        break
    
    return linked_graph


def a_star(
    graph: Dict[str, Dict[str, Any]],
    start: str,
    goal: str,
    heuristic: Callable[[Dict[str, Any], Dict[str, Any]], float] = None
) -> Optional[List[str]]:
    """
    Find the shortest path between two nodes using A* algorithm.
    
    Args:
        graph: Graph to search
        start: Start node ID
        goal: Goal node ID
        heuristic: Heuristic function for A*
        
    Returns:
        Optional[List[str]]: Path from start to goal, or None if no path exists
    """
    # Check if start and goal exist
    if start not in graph or goal not in graph:
        return None
    
    # Default heuristic
    if heuristic is None:
        def heuristic(node1, node2):
            # Use embedding similarity if available
            if "embedding" in node1 and "embedding" in node2:
                # Calculate cosine similarity
                vec1 = np.array(node1["embedding"])
                vec2 = np.array(node2["embedding"])
                
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 1.0  # Maximum distance
                
                similarity = dot_product / (norm1 * norm2)
                return 1.0 - similarity  # Convert to distance
            
            # Use tag similarity if available
            elif "tags" in node1 and "tags" in node2:
                similarity = tag_similarity(node1["tags"], node2["tags"])
                return 1.0 - similarity  # Convert to distance
            
            # Default distance
            return 1.0
    
    # Initialize data structures
    open_set = []  # Priority queue
    closed_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(graph[start], graph[goal])}
    
    # Add start node to open set
    heapq.heappush(open_set, (f_score[start], start))
    
    while open_set:
        # Get node with lowest f_score
        _, current = heapq.heappop(open_set)
        
        # Check if goal reached
        if current == goal:
            return reconstruct_path(came_from, current)
        
        # Add to closed set
        closed_set.add(current)
        
        # Get neighbors
        neighbors = graph[current].get("links", [])
        
        for neighbor in neighbors:
            # Skip if neighbor not in graph
            if neighbor not in graph:
                continue
            
            # Skip if in closed set
            if neighbor in closed_set:
                continue
            
            # Calculate tentative g_score
            tentative_g_score = g_score[current] + 1  # Assuming uniform edge weights
            
            # Check if new path is better
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # Update path
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(graph[neighbor], graph[goal])
                
                # Add to open set if not already there
                if neighbor not in [node for _, node in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return None


def reconstruct_path(came_from: Dict[str, str], current: str) -> List[str]:
    """
    Reconstruct path from came_from dictionary.
    
    Args:
        came_from: Dictionary mapping each node to its predecessor
        current: Current node
        
    Returns:
        List[str]: Path from start to current
    """
    path = [current]
    
    while current in came_from:
        current = came_from[current]
        path.append(current)
    
    # Reverse path to get start to goal
    path.reverse()
    
    return path


def find_similar_nodes(
    graph: Dict[str, Dict[str, Any]],
    node_id: str,
    similarity_threshold: float = 0.7,
    max_results: int = 10
) -> List[Tuple[str, float]]:
    """
    Find nodes similar to the given node.
    
    Args:
        graph: Graph to search
        node_id: Node ID to find similar nodes for
        similarity_threshold: Minimum similarity score (0-1)
        max_results: Maximum number of results to return
        
    Returns:
        List[Tuple[str, float]]: List of (node_id, similarity) tuples
    """
    # Check if node exists
    if node_id not in graph:
        return []
    
    node = graph[node_id]
    results = []
    
    # Check if node has embedding
    if "embedding" in node:
        # Find similar nodes by embedding
        for other_id, other_node in graph.items():
            # Skip self
            if other_id == node_id:
                continue
            
            # Skip nodes without embedding
            if "embedding" not in other_node:
                continue
            
            # Calculate similarity
            vec1 = np.array(node["embedding"])
            vec2 = np.array(other_node["embedding"])
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                continue
            
            similarity = dot_product / (norm1 * norm2)
            
            # Add to results if above threshold
            if similarity >= similarity_threshold:
                results.append((other_id, similarity))
    
    # Check if node has tags
    elif "tags" in node:
        # Find similar nodes by tags
        for other_id, other_node in graph.items():
            # Skip self
            if other_id == node_id:
                continue
            
            # Skip nodes without tags
            if "tags" not in other_node:
                continue
            
            # Calculate similarity
            similarity = tag_similarity(node["tags"], other_node["tags"])
            
            # Add to results if above threshold
            if similarity >= similarity_threshold:
                results.append((other_id, similarity))
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Limit results
    return results[:max_results]

