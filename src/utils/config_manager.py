"""
Configuration manager for GraphYML.
Provides a centralized way to load, save, and access application settings.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for GraphYML.
    Handles loading, saving, and accessing application settings.
    """
    
    def __init__(self, config_path: str = "graph_config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from disk or return defaults if not found.
        
        Returns:
            dict: Configuration dictionary
        """
        # Default configuration values
        default_config = {
            "save_path": "saved_yamls",
            "ollama_url": "http://localhost:11434/api/embeddings",
            "ollama_model": "all-minilm-l6-v2",
            "edit_inline": True,
            "embedding_dimension": 384,  # Default for all-minilm-l6-v2
            "max_cluster_count": 4,
            "perplexity": 30,
            "node_distance": 200
        }
        
        # Check if config_path exists and is a file (not a directory)
        if os.path.exists(self.config_path) and os.path.isfile(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return {**default_config, **config}
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        elif os.path.exists(self.config_path) and not os.path.isfile(self.config_path):
            # If it exists but is not a file (e.g., it's a directory), use defaults
            logger.warning(f"{self.config_path} exists but is not a file. Using default configuration.")
            return default_config
        else:
            # If it doesn't exist, use defaults
            logger.info(f"Config file {self.config_path} not found. Using default configuration.")
            return default_config
    
    def save_config(self) -> bool:
        """
        Save configuration to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if config_path is a directory
        if os.path.exists(self.config_path) and not os.path.isfile(self.config_path):
            # If it's a directory, remove it and create a file
            try:
                import shutil
                shutil.rmtree(self.config_path)
            except (IOError, OSError) as e:
                logger.error(f"Error removing directory {self.config_path}: {e}")
                # Try an alternative path
                self.config_path = "graph_config_new.json"
                logger.info(f"Using alternative config path: {self.config_path}")
        
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            return True
        except (IOError, TypeError) as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def update(self, config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            config: New configuration values
        """
        self.config.update(config)
    
    def ensure_directories(self) -> None:
        """
        Ensure all required directories exist.
        """
        # Ensure save path exists
        save_path = Path(self.config["save_path"])
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Update config with absolute path
        self.config["save_path"] = str(save_path.absolute())
        
        # Check for other paths in config
        for key, value in self.config.items():
            if key.endswith("_path") or key.endswith("_dir"):
                path = Path(value)
                path.mkdir(parents=True, exist_ok=True)
                self.config[key] = str(path.absolute())

