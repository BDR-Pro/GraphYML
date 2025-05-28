"""
Extended unit tests for graph operations module.
"""
import unittest
from unittest.mock import patch, MagicMock

from src.models.graph_ops import (
    auto_link_nodes, tag_similarity, a_star, reconstruct_path, 
    find_similar_nodes, _calculate_node_similarities
)


class TestGraphOpsExtended(unittest.TestCase):
    """Extended test cases for graph operations module."""

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
            },
            "node5": {
                "id": "node5",
                "title": "Test Node 5",
                "tags": [],
                "genres": [],
                "links": [],
                "embedding": [0.5, 0.6, 0.7]
            }
        }

    @patch('src.models.graph_ops.embedding_similarity')
    def test_calculate_node_similarities(self, mock_embedding_similarity):
        """Test the _calculate_node_similarities function directly."""
        # Mock embedding similarity
        mock_embedding_similarity.side_effect = lambda a, b: 0.9 if a == [0.1, 0.2, 0.3] and b == [0.1, 0.2, 0.3] else 0.5
        
        # Test with source node1
        source_node = self.test_graph["node1"]
        similarities = _calculate_node_similarities(
            self.test_graph,
            source_node,
            "node1",
            0.4,  # Lower threshold to include all nodes
            10
        )
        
        # Should find all nodes with similarity >= 0.4
        self.assertEqual(len(similarities), 5)
        self.assertEqual(similarities[0][0], "node1")  # Most similar is self
        self.assertAlmostEqual(similarities[0][1], 0.9)
        
        # Test with higher threshold
        similarities = _calculate_node_similarities(
            self.test_graph,
            source_node,
            "node1",
            0.8,  # Higher threshold
            10
        )
        
        # Should only find node1 (self) with similarity >= 0.8
        self.assertEqual(len(similarities), 1)
        self.assertEqual(similarities[0][0], "node1")
        
        # Test with limited results
        similarities = _calculate_node_similarities(
            self.test_graph,
            source_node,
            "node1",
            0.4,  # Lower threshold
            2  # Limit to 2 results
        )
        
        # Should only return 2 results
        self.assertEqual(len(similarities), 2)
    
    def test_find_similar_nodes_edge_cases(self):
        """Test find_similar_nodes with various edge cases."""
        # Test with non-existent node
        similar = find_similar_nodes(self.test_graph, "nonexistent", top_n=2)
        self.assertEqual(similar, [])
        
        # Test with node without embedding
        graph_copy = self.test_graph.copy()
        del graph_copy["node1"]["embedding"]
        similar = find_similar_nodes(graph_copy, "node1", top_n=2)
        self.assertEqual(similar, [])
        
        # Test with empty graph
        similar = find_similar_nodes({}, "node1", top_n=2)
        self.assertEqual(similar, [])
        
        # Test with similarity_threshold parameter
        with patch('src.models.graph_ops._calculate_node_similarities') as mock_calc:
            mock_calc.return_value = [("node1", 0.9), ("node2", 0.8)]
            similar = find_similar_nodes(self.test_graph, "node3", similarity_threshold=0.75)
            mock_calc.assert_called_once()
            self.assertEqual(mock_calc.call_args[0][3], 0.75)  # Check threshold was passed
        
        # Test with max_results parameter
        with patch('src.models.graph_ops._calculate_node_similarities') as mock_calc:
            mock_calc.return_value = [("node1", 0.9), ("node2", 0.8)]
            similar = find_similar_nodes(self.test_graph, "node3", max_results=5)
            mock_calc.assert_called_once()
            self.assertEqual(mock_calc.call_args[0][4], 5)  # Check max_results was passed
    
    @patch('src.models.graph_ops.embedding_similarity')
    def test_auto_link_nodes_edge_cases(self, mock_embedding_similarity):
        """Test auto_link_nodes with various edge cases."""
        # Mock embedding similarity
        mock_embedding_similarity.return_value = 0.8
        
        # Test with empty graph
        result = auto_link_nodes({})
        self.assertEqual(result, {})
        
        # Test with graph without embeddings
        graph_without_embeddings = {
            "node1": {"id": "node1", "title": "Node 1", "links": []},
            "node2": {"id": "node2", "title": "Node 2", "links": []}
        }
        result = auto_link_nodes(graph_without_embeddings)
        self.assertEqual(len(result), 2)
        
        # Test with single node graph
        single_node = {"node1": {"id": "node1", "embedding": [0.1, 0.2], "links": []}}
        result = auto_link_nodes(single_node)
        self.assertEqual(len(result), 1)
        
        # Test with custom threshold
        graph = {
            "node1": {"id": "node1", "embedding": [0.1, 0.2], "links": []},
            "node2": {"id": "node2", "embedding": [0.2, 0.3], "links": []}
        }
        result = auto_link_nodes(graph, threshold=0.85)
        self.assertEqual(len(result), 2)
    
    def test_tag_similarity_edge_cases(self):
        """Test tag_similarity with various edge cases."""
        # Test with empty tags
        similarity = tag_similarity([], [])
        self.assertEqual(similarity, 0.0)
        
        # Test with one empty tag list
        similarity = tag_similarity(["tag1", "tag2"], [])
        self.assertEqual(similarity, 0.0)
        
        # Test with identical tags
        similarity = tag_similarity(["tag1", "tag2"], ["tag1", "tag2"])
        self.assertEqual(similarity, 1.0)
        
        # Test with case sensitivity (tags are case sensitive in the implementation)
        similarity = tag_similarity(["TAG1", "tag2"], ["tag1", "TAG2"])
        self.assertEqual(similarity, 0.0)  # Different because case matters
        
        # Test with partial overlap
        similarity = tag_similarity(["tag1", "tag2", "tag3"], ["tag1", "tag4"])
        self.assertAlmostEqual(similarity, 0.25)  # 1 common out of 4 unique
    
    def test_a_star_edge_cases(self):
        """Test a_star with various edge cases."""
        # Create a test graph for pathfinding
        graph = {
            "A": {"links": ["B", "C"]},
            "B": {"links": ["A", "D"]},
            "C": {"links": ["A", "D"]},
            "D": {"links": ["B", "C", "E"]},
            "E": {"links": ["D"]},
            "F": {"links": []}  # Isolated node
        }
        
        # Test with start == end
        path = a_star(graph, "A", "A")
        self.assertEqual(path, ["A"])
        
        # Test with unreachable end
        path = a_star(graph, "A", "F")
        self.assertIsNone(path)  # Returns None for unreachable nodes
        
        # Test with non-existent start
        path = a_star(graph, "Z", "E")
        self.assertIsNone(path)  # Returns None for non-existent nodes
        
        # Test with non-existent end
        path = a_star(graph, "A", "Z")
        self.assertIsNone(path)  # Returns None for non-existent nodes
    
    def test_reconstruct_path(self):
        """Test reconstruct_path function."""
        # Test normal case
        came_from = {"B": "A", "C": "B", "D": "C"}
        path = reconstruct_path(came_from, "D")
        self.assertEqual(path, ["A", "B", "C", "D"])
        
        # Test when start is not in came_from
        path = reconstruct_path({"B": "A", "C": "B", "D": "C"}, "A")
        self.assertEqual(path, ["A"])
        
        # Test with empty came_from
        path = reconstruct_path({}, "A")
        self.assertEqual(path, ["A"])


if __name__ == "__main__":
    unittest.main()
