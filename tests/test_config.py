"""
Unit tests for configuration module.
"""
import os
import json
import tempfile
import unittest
from pathlib import Path

from src.config.settings import load_config, save_config, ensure_directories, DEFAULT_CONFIG


class TestConfig(unittest.TestCase):
    """Test cases for configuration module."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.test_dir.name, "test_config.json")

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        self.test_dir.cleanup()

    def test_load_config_default(self):
        """Test loading default configuration when file doesn't exist."""
        config = load_config(self.config_path)
        self.assertEqual(config, DEFAULT_CONFIG)

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Create a test configuration
        test_config = {
            "save_path": "test_path",
            "ollama_url": "http://test.example.com",
            "ollama_model": "test-model",
            "edit_inline": False,
            "embedding_dimension": 512,
            "max_cluster_count": 8,
            "perplexity": 20,
            "node_distance": 300
        }

        # Save the configuration
        save_config(test_config, self.config_path)

        # Check if the file exists
        self.assertTrue(os.path.exists(self.config_path))

        # Load the configuration
        loaded_config = load_config(self.config_path)

        # Check if the loaded configuration matches the saved one
        self.assertEqual(loaded_config, test_config)

    def test_ensure_directories(self):
        """Test ensuring directories exist."""
        # Create a test configuration
        test_config = {
            "save_path": os.path.join(self.test_dir.name, "test_save_path")
        }

        # Ensure directories
        updated_config = ensure_directories(test_config)

        # Check if the directory exists
        self.assertTrue(os.path.exists(test_config["save_path"]))

        # Check if the path in the config is absolute
        self.assertTrue(os.path.isabs(updated_config["save_path"]))


if __name__ == "__main__":
    unittest.main()

