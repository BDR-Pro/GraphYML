"""
Configuration management for the GraphYML application.
Handles loading, saving, and accessing application settings.
"""
import os
import json
from pathlib import Path
from src.utils.config_manager import ConfigManager

# Default configuration path
CONFIG_PATH = "graph_config.json"

# Create a global config manager instance
config_manager = ConfigManager(CONFIG_PATH)

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
    # Use the config manager to load config
    if config_path != CONFIG_PATH:
        # If a different path is specified, create a new manager
        return ConfigManager(config_path).config
    else:
        # Otherwise use the global instance
        return config_manager.config


def save_config(config, config_path=CONFIG_PATH):
    """
    Save configuration to disk.
    
    Args:
        config (dict): Configuration to save
        config_path (str): Path to save the configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    if config_path != CONFIG_PATH:
        # If a different path is specified, create a new manager
        manager = ConfigManager(config_path)
        manager.config = config
        return manager.save_config()
    else:
        # Otherwise use the global instance
        config_manager.config = config
        return config_manager.save_config()


def ensure_directories(config):
    """
    Ensure all required directories exist.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Updated configuration with absolute paths
    """
    # Create a temporary config manager with the provided config
    temp_manager = ConfigManager(CONFIG_PATH)
    temp_manager.config = config
    temp_manager.ensure_directories()
    
    return temp_manager.config
