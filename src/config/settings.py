"""
Configuration management for the GraphYML application.
Handles loading, saving, and accessing application settings.
"""
import os
import json
from pathlib import Path

# Default configuration path
CONFIG_PATH = "graph_config.json"

# Default configuration values
DEFAULT_CONFIG = {
    "save_path": "saved_yamls",
    "ollama_url": "http://localhost:11434/api/embeddings",
    "ollama_model": "all-minilm-l6-v2",
    "edit_inline": True,
    "embedding_dimension": 384,  # Default for all-minilm-l6-v2
    "max_cluster_count": 4,
    "perplexity": 30,
    "node_distance": 200
}


def load_config(config_path=CONFIG_PATH):
    """
    Load configuration from disk or return defaults if not found.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**DEFAULT_CONFIG, **config}
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config: {e}")
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG


def save_config(config, config_path=CONFIG_PATH):
    """
    Save configuration to disk.
    
    Args:
        config (dict): Configuration to save
        config_path (str): Path to save the configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        return True
    except (IOError, TypeError) as e:
        print(f"Error saving config: {e}")
        return False


def ensure_directories(config):
    """
    Ensure all required directories exist.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Updated configuration with absolute paths
    """
    # Ensure save path exists
    save_path = Path(config["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Update config with absolute path
    config["save_path"] = str(save_path.absolute())
    
    return config

