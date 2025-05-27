"""
Unit tests for graph operations module.
"""
import unittest
from unittest.mock import patch

from src.models.graph_ops import (
    auto_link_nodes, tag_similarity, a_star, reconstruct_path, find_similar_nodes
)


class TestGraphOps(unittest.TestCase):
    """Test cases for graph operations module."""

    def setUp(self):
        """Set up test environment."""
        # Create a test graph
        self.test_graph = {
            "node1": {
                "id": "node1",
                "title": "Test Node 1",
                "tags": ["test", "node", "one"],
                "genres": ["action", "adventure"],
                "links": [],
                "embedding": [0.1, 0.2, 0.3]
            },
            "node2": {
                "id": "node2",
                "title": "Test Node 2",
                "tags": ["test", "node", "two"],
                "genres": ["action", "comedy"],
                "links": [],
                "embedding": [0.2, 0.3, 0.4]
            },
            "node3": {
                "id": "node3",
                "title": "Test Node 3",
                "tags": ["test", "node", "three"],
                "genres": ["drama", "romance"],
                "links": [],
                "embedding": [0.3, 0.4, 0.5]
            },
            "node4": {
                "id": "node4",
                "title": "Test Node 4",
                "tags": ["test", "four"],
                "genres": ["comedy", "romance"],
                "links": [],
                "embedding": [0.4, 0.5, 0.6]
            }
        }

    def test_auto_link_nodes(self):
        """Test automatically linking nodes."""
        # Auto-link nodes
        linked_graph = auto_link_nodes(self.test_graph)

        # Check if nodes were linked correctly
        self.assertIn("node2", linked_graph["node1"]["links"])  # Shared genre: action
        self.assertIn("node1", linked_graph["node2"]["links"])  # Shared genre: action
        self.assertIn("node4", linked_graph["node2"]["links"])  # Shared genre: comedy
        self.assertIn("node2", linked_graph["node4"]["links"])  # Shared genre: comedy
        self.assertIn("node4", linked_graph["node3"]["links"])  # Shared genre: romance
        self.assertIn("node3", linked_graph["node4"]["links"])  # Shared genre: romance

        # Check if nodes with no shared genres/tags are not linked
        self.assertNotIn("node3", linked_graph["node1"]["links"])
        self.assertNotIn("node1", linked_graph["node3"]["links"])

    def test_tag_similarity(self):
        """Test tag similarity calculation."""
        # Create test tags
        tags1 = ["test", "node", "one"]
        tags2 = ["test", "node", "two"]
        tags3 = ["test", "three"]
        tags4 = ["four", "five"]
        tags5 = []

        # Calculate similarities
        sim1_2 = tag_similarity(tags1, tags2)
        sim1_3 = tag_similarity(tags1, tags3)
        sim1_4 = tag_similarity(tags1, tags4)
        sim1_5 = tag_similarity(tags1, tags5)
        sim5_5 = tag_similarity(tags5, tags5)

        # Check if similarities are correct
        self.assertAlmostEqual(sim1_2, 2/4)  # 2 shared out of 4 unique
        self.assertAlmostEqual(sim1_3, 1/4)  # 1 shared out of 4 unique
        self.assertAlmostEqual(sim1_4, 0.0)  # 0 shared
        self.assertAlmostEqual(sim1_5, 0.0)  # Empty set
        self.assertAlmostEqual(sim5_5, 0.0)  # Empty set

    @patch('src.models.embeddings.embedding_similarity')
    def test_a_star(self, mock_embedding_similarity):
        """Test A* pathfinding."""
        # Mock embedding similarity
        mock_embedding_similarity.return_value = 0.5

        # Create a graph with links
        graph = {
            "A": {
                "embedding": [0.1, 0.2, 0.3],
                "tags": ["tag1", "tag2"],
                "links": ["B", "C"]
            },
            "B": {
                "embedding": [0.2, 0.3, 0.4],
                "tags": ["tag2", "tag3"],
                "links": ["A", "D"]
            },
            "C": {
                "embedding": [0.3, 0.4, 0.5],
                "tags": ["tag1", "tag3"],
                "links": ["A", "D"]
            },
            "D": {
                "embedding": [0.4, 0.5, 0.6],
                "tags": ["tag3", "tag4"],
                "links": ["B", "C", "E"]
            },
            "E": {
                "embedding": [0.5, 0.6, 0.7],
                "tags": ["tag4", "tag5"],
                "links": ["D"]
            }
        }

        # Find path
        path = a_star(graph, "A", "E")

        # Check if path was found
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")

        # Check if path is valid
        for i in range(len(path) - 1):
            self.assertIn(path[i+1], graph[path[i]]["links"])

        # Test with non-existent nodes
        path = a_star(graph, "A", "F")
        self.assertIsNone(path)
        path = a_star(graph, "F", "E")
        self.assertIsNone(path)

        # Test with same start and goal
        path = a_star(graph, "A", "A")
        self.assertEqual(path, ["A"])

    def test_reconstruct_path(self):
        """Test path reconstruction."""
        # Create a came_from dictionary
        came_from = {
            "B": "A",
            "C": "B",
            "D": "C",
            "E": "D"
        }

        # Reconstruct path
        path = reconstruct_path(came_from, "E")

        # Check if path was reconstructed correctly
        self.assertEqual(path, ["A", "B", "C", "D", "E"])

        # Test with single node
        path = reconstruct_path({}, "A")
        self.assertEqual(path, ["A"])

    @patch('src.models.embeddings.embedding_similarity')
    def test_find_similar_nodes(self, mock_embedding_similarity):
        """Test finding similar nodes."""
        # Mock embedding similarity
        mock_embedding_similarity.side_effect = lambda a, b: 0.9 if "node1" in (a, b) else 0.5

        # Find similar nodes
        similar = find_similar_nodes(self.test_graph, "node1", top_n=2)

        # Check if similar nodes were found correctly
        self.assertEqual(len(similar), 2)
        self.assertEqual(similar[0][0], "node1")  # Most similar is self
        self.assertAlmostEqual(similar[0][1], 0.9)

        # Test with non-existent node
        similar = find_similar_nodes(self.test_graph, "nonexistent", top_n=2)
        self.assertEqual(similar, [])

        # Test with node without embedding
        graph_copy = self.test_graph.copy()
        del graph_copy["node1"]["embedding"]
        similar = find_similar_nodes(graph_copy, "node1", top_n=2)
        self.assertEqual(similar, [])


if __name__ == "__main__":
    unittest.main()

