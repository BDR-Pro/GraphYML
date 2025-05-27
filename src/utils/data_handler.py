"""
Data handler module for GraphYML.
Provides functions for loading and saving graph data.
"""
import os
import yaml
import json
import zipfile
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Set, Union, BinaryIO
from io import BytesIO


def validate_node_schema(node: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """
    Validate a node against a schema.
    
    Args:
        node: Node to validate
        
    Returns:
        Tuple[bool, Dict[str, str]]: (is_valid, error_messages)
    """
    errors = {}
    
    # Define schema
    schema = {
        "id": {"required": True, "type": "string"},
        "title": {"required": True, "type": "string"},
        "tags": {"required": False, "type": "array", "items": {"type": "string"}},
        "links": {"required": False, "type": "array", "items": {"type": "string"}},
        "genres": {"required": False, "type": "array", "items": {"type": "string"}},
        "embedding": {"required": False, "type": "array", "items": {"type": "number"}},
        "content": {"required": False, "type": "string"},
        "description": {"required": False, "type": "string"},
        "overview": {"required": False, "type": "string"},
        "metadata": {"required": False, "type": "object"}
    }
    
    # Check required fields
    for field, field_schema in schema.items():
        if field_schema.get("required", False) and field not in node:
            errors[field] = f"Missing required field: {field}"
        
        if field in node:
            # Check field type
            field_type = field_schema.get("type")
            
            if field_type == "string" and not isinstance(node[field], str):
                errors[field] = f"Field {field} must be a string"
            elif field_type == "number" and not isinstance(node[field], (int, float)):
                errors[field] = f"Field {field} must be a number"
            elif field_type == "boolean" and not isinstance(node[field], bool):
                errors[field] = f"Field {field} must be a boolean"
            elif field_type == "array" and not isinstance(node[field], list):
                errors[field] = f"Field {field} must be an array"
            elif field_type == "object" and not isinstance(node[field], dict):
                errors[field] = f"Field {field} must be an object"
            
            # Check array items
            if field_type == "array" and "items" in field_schema and node[field]:
                item_type = field_schema["items"].get("type")
                
                for i, item in enumerate(node[field]):
                    if item_type == "string" and not isinstance(item, str):
                        errors[f"{field}[{i}]"] = f"Items in {field} must be strings"
                    elif item_type == "number" and not isinstance(item, (int, float)):
                        errors[f"{field}[{i}]"] = f"Items in {field} must be numbers"
                    elif item_type == "boolean" and not isinstance(item, bool):
                        errors[f"{field}[{i}]"] = f"Items in {field} must be booleans"
                    elif item_type == "object" and not isinstance(item, dict):
                        errors[f"{field}[{i}]"] = f"Items in {field} must be objects"
    
    return len(errors) == 0, errors


def load_graph_from_folder(folder_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    Load graph data from a folder of YAML files.
    
    Args:
        folder_path: Path to folder containing YAML files
        
    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]: (graph, errors)
    """
    graph = {}
    errors = {}
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        return graph, errors
    
    # Load each YAML file
    for filename in os.listdir(folder_path):
        if filename.endswith(('.yaml', '.yml')):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    node = yaml.safe_load(f)
                
                # Skip invalid nodes
                if not isinstance(node, dict) or "id" not in node:
                    errors[filename] = "Invalid node format or missing ID"
                    continue
                
                # Add node to graph
                graph[node["id"]] = node
            except Exception as e:
                errors[filename] = f"Error loading file: {str(e)}"
    
    return graph, errors


def save_node_to_yaml(
    node: Dict[str, Any],
    folder_path: str,
    filename: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Save a node to a YAML file.
    
    Args:
        node: Node to save
        folder_path: Path to folder to save to
        filename: Optional filename (defaults to node_id.yaml)
        
    Returns:
        Tuple[bool, Optional[str]]: (success, error_message)
    """
    try:
        # Check if node has ID
        if "id" not in node:
            return False, "Node must have an ID"
        
        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Determine filename
        if filename is None:
            filename = f"{node['id']}.yaml"
        
        # Save node to file
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(node, f, default_flow_style=False, sort_keys=False)
        
        return True, None
    except Exception as e:
        return False, str(e)


def create_zip(folder_path: str) -> BytesIO:
    """
    Create a ZIP file from a folder.
    
    Args:
        folder_path: Path to folder to zip
        
    Returns:
        BytesIO: ZIP file as a BytesIO object
    """
    # Create a BytesIO object to store the ZIP file
    zip_buffer = BytesIO()
    
    # Create ZIP file
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add each file in folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Add file to ZIP with just the filename
                zipf.write(file_path, os.path.basename(file_path))
    
    # Reset buffer position
    zip_buffer.seek(0)
    
    return zip_buffer


def flatten_node(node: Dict[str, Any]) -> str:
    """
    Flatten a node by combining all values into a single string.
    
    Args:
        node: Node to flatten
        
    Returns:
        str: Flattened node as a string
    """
    # Create a list to store all values
    values = []
    
    # Helper function to extract values
    def extract_values(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                extract_values(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_values(item)
        else:
            values.append(str(obj))
    
    # Extract values
    extract_values(node)
    
    # Join values
    return " ".join(values)


def query_by_tag(graph: Dict[str, Dict[str, Any]], tag: str) -> Dict[str, Dict[str, Any]]:
    """
    Query graph by tag.
    
    Args:
        graph: Graph to query
        tag: Tag to query for
        
    Returns:
        Dict[str, Dict[str, Any]]: Matching nodes
    """
    results = {}
    
    for key, node in graph.items():
        # Check if node has tags
        if "tags" in node and isinstance(node["tags"], list):
            # Check if tag is in tags
            if tag in node["tags"]:
                results[key] = node
        
        # Check if node has genres
        if "genres" in node and isinstance(node["genres"], list):
            # Check if tag is in genres
            if tag in node["genres"]:
                results[key] = node
    
    return results
