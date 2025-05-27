"""
Data handling utilities for GraphYML.
Handles loading, saving, and processing YAML files.
"""
import os
import zipfile
import tempfile
from io import BytesIO
from pathlib import Path

import yaml
import cerberus

# Schema definition for YAML validation
NODE_SCHEMA = {
    "id": {"type": "string", "required": True},
    "title": {"type": "string", "required": True},
    "tags": {"type": "list", "schema": {"type": "string"}},
    "genres": {"type": "list", "schema": {"type": "string"}},
    "links": {"type": "list", "schema": {"type": "string"}},
    "embedding": {"type": "list"},
    # Additional fields that are optional
    "tagline": {"type": "string"},
    "director": {"type": "string"},
    "cast": {"type": "list", "schema": {"type": "string"}},
    "keywords": {"type": "list", "schema": {"type": "string"}},
    "overview": {"type": "string"},
    "runtime": {"type": "number"},
    "release_date": {"type": "string"},
    "vote_count": {"type": "integer"},
    "vote_average": {"type": "number"},
    "budget": {"type": "number"},
    "revenue": {"type": "number"},
    "budget_adj": {"type": "number"},
    "revenue_adj": {"type": "number"},
    "popularity": {"type": "number"},
    "production_companies": {"type": "list", "schema": {"type": "string"}},
    "year": {"type": "integer"},
    "script": {"type": "string"}
}


def validate_node_schema(node):
    """
    Validate a YAML node against the expected schema.
    
    Args:
        node (dict): Node data to validate
        
    Returns:
        tuple: (is_valid, errors)
    """
    validator = cerberus.Validator(NODE_SCHEMA)
    is_valid = validator.validate(node)
    return is_valid, validator.errors


def create_zip(folder):
    """
    Compress all YAML files in a folder into a zip archive.
    
    Args:
        folder (str): Path to the folder containing YAML files
        
    Returns:
        BytesIO: In-memory zip file buffer
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for file in Path(folder).glob("*.yaml"):
            zipf.write(file, arcname=file.name)
    zip_buffer.seek(0)
    return zip_buffer


def handle_zip_upload(uploaded_zip, save_dir):
    """
    Extract a ZIP file into the configured save directory.
    
    Args:
        uploaded_zip: Streamlit uploaded file object
        save_dir (str): Directory to save extracted files
        
    Returns:
        list: List of folders created
    """
    folders = set()
    
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
            folders.add(str(rel_folder))
    
    return list(folders)


def handle_yaml_uploads(uploaded_files, save_dir):
    """
    Save uploaded YAML files, using simulated folder paths from their filenames.
    
    Args:
        uploaded_files: List of Streamlit uploaded file objects
        save_dir (str): Directory to save files
        
    Returns:
        set: Set of folders created
    """
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
    """
    Load all YAML files in a given folder into a graph dictionary.
    
    Args:
        folder_path (str): Path to folder containing YAML files
        
    Returns:
        tuple: (graph_dict, error_list)
    """
    graph = {}
    errors = []
    
    for yml in Path(folder_path).glob("*.yaml"):
        try:
            with open(yml, encoding='utf-8') as f:
                node = yaml.safe_load(f)
                if node:
                    # Validate node against schema
                    is_valid, validation_errors = validate_node_schema(node)
                    if not is_valid:
                        errors.append((yml.name, validation_errors))
                        continue
                        
                    key = node.get("id") or node.get("title")
                    if key:
                        graph[key] = node
                    else:
                        errors.append((yml.name, "Missing id or title"))
        except Exception as e:
            errors.append((yml.name, str(e)))
    
    return graph, errors


def save_node_to_yaml(node, folder_path, filename=None):
    """
    Save a node to a YAML file.
    
    Args:
        node (dict): Node data to save
        folder_path (str): Directory to save the file
        filename (str, optional): Filename to use. If None, uses node id or title.
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        # Validate node
        is_valid, errors = validate_node_schema(node)
        if not is_valid:
            return False, f"Validation failed: {errors}"
        
        # Determine filename
        if not filename:
            key = node.get("id") or node.get("title")
            if not key:
                return False, "Node must have id or title"
            filename = f"{key}.yaml"
        
        # Ensure folder exists
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(folder / filename, "w", encoding="utf-8") as f:
            yaml.dump(node, f, sort_keys=False)
        
        return True, None
    except Exception as e:
        return False, str(e)


def flatten_node(d):
    """
    Flatten nested dict or list values to a flat list of strings.
    
    Args:
        d: Dictionary, list, or scalar value
        
    Returns:
        list: Flattened list of string values
    """
    values = []
    if isinstance(d, dict):
        for v in d.values():
            values += flatten_node(v)
    elif isinstance(d, list):
        for item in d:
            values += flatten_node(item)
    elif isinstance(d, (str, int, float)):
        values.append(str(d))
    return values


def query_by_tag(graph, tag):
    """
    Return keys where tag is found in any nested field.
    
    Args:
        graph (dict): Graph dictionary
        tag (str): Tag to search for
        
    Returns:
        list: List of matching node keys
    """
    return [key for key, node in graph.items() if tag.lower() in 
            [v.lower() for v in flatten_node(node) if isinstance(v, str)]]

