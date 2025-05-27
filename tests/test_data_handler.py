"""
Unit tests for data handler module.
"""
import os
import tempfile
import unittest
from pathlib import Path
import shutil
import yaml
from io import BytesIO

from src.utils.data_handler import (
    validate_node_schema, create_zip, load_graph_from_folder,
    save_node_to_yaml, flatten_node, query_by_tag
)


class TestDataHandler(unittest.TestCase):
    """Test cases for data handler module."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Create test nodes
        self.test_nodes = {
            "node1": {
                "id": "node1",
                "title": "Test Node 1",
                "tags": ["test", "node", "one"],
                "links": ["node2"]
            },
            "node2": {
                "id": "node2",
                "title": "Test Node 2",
                "tags": ["test", "node", "two"],
                "links": ["node1"]
            },
            "node3": {
                "id": "node3",
                "title": "Test Node 3",
                "tags": ["test", "node", "three"],
                "genres": ["action", "adventure"],
                "links": []
            }
        }

        # Create test files
        for key, node in self.test_nodes.items():
            node_path = self.test_path / f"{key}.yaml"
            with open(node_path, "w", encoding="utf-8") as f:
                yaml.dump(node, f)

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        self.test_dir.cleanup()

    def test_validate_node_schema(self):
        """Test node schema validation."""
        # Valid node
        valid_node = {
            "id": "test",
            "title": "Test Node",
            "tags": ["test", "node"],
            "links": []
        }
        is_valid, errors = validate_node_schema(valid_node)
        self.assertTrue(is_valid)
        self.assertEqual(errors, {})

        # Invalid node (missing required field)
        invalid_node = {
            "id": "test",
            "tags": ["test", "node"],
            "links": []
        }
        is_valid, errors = validate_node_schema(invalid_node)
        self.assertFalse(is_valid)
        self.assertIn("title", errors)

        # Invalid node (wrong type)
        invalid_node = {
            "id": "test",
            "title": "Test Node",
            "tags": "not a list",
            "links": []
        }
        is_valid, errors = validate_node_schema(invalid_node)
        self.assertFalse(is_valid)
        self.assertIn("tags", errors)

    def test_create_zip(self):
        """Test creating a ZIP archive."""
        # Create a ZIP archive
        zip_buffer = create_zip(self.test_path)

        # Check if the ZIP archive contains the expected files
        import zipfile
        with zipfile.ZipFile(zip_buffer) as zipf:
            file_list = zipf.namelist()
            self.assertEqual(len(file_list), 3)
            self.assertIn("node1.yaml", file_list)
            self.assertIn("node2.yaml", file_list)
            self.assertIn("node3.yaml", file_list)

    def test_load_graph_from_folder(self):
        """Test loading graph from folder."""
        # Load graph
        graph, errors = load_graph_from_folder(self.test_path)

        # Check if the graph contains the expected nodes
        self.assertEqual(len(graph), 3)
        self.assertIn("node1", graph)
        self.assertIn("node2", graph)
        self.assertIn("node3", graph)
        self.assertEqual(graph["node1"]["title"], "Test Node 1")
        self.assertEqual(graph["node2"]["title"], "Test Node 2")
        self.assertEqual(graph["node3"]["title"], "Test Node 3")

        # Check if there are no errors
        self.assertEqual(len(errors), 0)

        # Test with invalid YAML file
        invalid_path = self.test_path / "invalid.yaml"
        with open(invalid_path, "w", encoding="utf-8") as f:
            f.write("invalid: yaml: content")

        graph, errors = load_graph_from_folder(self.test_path)
        self.assertEqual(len(errors), 1)

    def test_save_node_to_yaml(self):
        """Test saving node to YAML file."""
        # Create a test node
        test_node = {
            "id": "test_save",
            "title": "Test Save Node",
            "tags": ["test", "save"],
            "links": []
        }

        # Save the node
        success, error = save_node_to_yaml(
            test_node,
            str(self.test_path),
            "test_save.yaml"
        )

        # Check if the save was successful
        self.assertTrue(success)
        self.assertIsNone(error)

        # Check if the file exists
        self.assertTrue((self.test_path / "test_save.yaml").exists())

        # Load the node and check if it matches
        with open(self.test_path / "test_save.yaml", "r", encoding="utf-8") as f:
            loaded_node = yaml.safe_load(f)
        
        self.assertEqual(loaded_node, test_node)

    def test_flatten_node(self):
        """Test flattening nested node data."""
        # Create a test node with nested data
        test_node = {
            "id": "test_flatten",
            "title": "Test Flatten Node",
            "tags": ["test", "flatten"],
            "metadata": {
                "created_by": "test_user",
                "stats": {
                    "views": 10,
                    "likes": 5
                }
            },
            "links": ["node1", "node2"]
        }

        # Flatten the node
        flattened = flatten_node(test_node)

        # Check if the flattened node contains all values
        self.assertIn("test_flatten", flattened)
        self.assertIn("Test Flatten Node", flattened)
        self.assertIn("test", flattened)
        self.assertIn("flatten", flattened)
        self.assertIn("test_user", flattened)
        self.assertIn("10", flattened)
        self.assertIn("5", flattened)
        self.assertIn("node1", flattened)
        self.assertIn("node2", flattened)

    def test_query_by_tag(self):
        """Test querying nodes by tag."""
        # Create a graph
        graph = {
            "node1": {
                "id": "node1",
                "title": "Test Node 1",
                "tags": ["test", "node", "one"],
                "links": ["node2"]
            },
            "node2": {
                "id": "node2",
                "title": "Test Node 2",
                "tags": ["test", "node", "two"],
                "links": ["node1"]
            },
            "node3": {
                "id": "node3",
                "title": "Test Node 3",
                "tags": ["test", "node", "three"],
                "genres": ["action", "adventure"],
                "links": []
            }
        }

        # Query by tag
        results = query_by_tag(graph, "one")
        self.assertEqual(len(results), 1)
        self.assertIn("node1", results)

        # Query by tag that appears in multiple nodes
        results = query_by_tag(graph, "test")
        self.assertEqual(len(results), 3)
        self.assertIn("node1", results)
        self.assertIn("node2", results)
        self.assertIn("node3", results)

        # Query by tag in nested field
        results = query_by_tag(graph, "action")
        self.assertEqual(len(results), 1)
        self.assertIn("node3", results)

        # Query by non-existent tag
        results = query_by_tag(graph, "nonexistent")
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()

