"""
Unit tests for ORM module.
"""
import unittest
from unittest.mock import patch, MagicMock

from src.models.orm import GraphORM


class TestORM(unittest.TestCase):
    """Test cases for ORM module."""

    def setUp(self):
        """Set up test environment."""
        # Create a mock database
        self.mock_db = MagicMock()
        
        # Create a test graph
        self.test_graph = {
            "node1": {
                "id": "node1",
                "title": "Inception",
                "director": "Christopher Nolan",
                "year": 2010,
                "rating": 8.8,
                "tags": ["sci-fi", "action", "mind-bending"],
                "genres": ["Sci-Fi", "Action", "Thriller"],
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "runtime": 148,
                    "language": "English"
                }
            },
            "node2": {
                "id": "node2",
                "title": "The Matrix",
                "director": "Lana Wachowski",
                "year": 1999,
                "rating": 8.7,
                "tags": ["sci-fi", "action", "cyberpunk"],
                "genres": ["Sci-Fi", "Action"],
                "embedding": [0.2, 0.3, 0.4]
            },
            "node3": {
                "id": "node3",
                "title": "Interstellar",
                "director": "Christopher Nolan",
                "year": 2014,
                "rating": 8.6,
                "tags": ["sci-fi", "space", "time-travel"],
                "genres": ["Sci-Fi", "Adventure", "Drama"],
                "embedding": [0.3, 0.4, 0.5]
            }
        }
        
        # Set up mock database
        self.mock_db.graph = self.test_graph
        self.mock_db.query_engine = MagicMock()
        
        # Create ORM
        self.orm = GraphORM(self.mock_db)

    def test_find_by_id(self):
        """Test finding a node by ID."""
        # Set up mock
        self.mock_db.get_node.return_value = (self.test_graph["node1"], None)
        
        # Find node
        node = self.orm.find_by_id("node1", None)
        
        # Check if the correct node was found
        self.assertEqual(node, self.test_graph["node1"])
        
        # Check if the get_node method was called with the correct parameters
        self.mock_db.get_node.assert_called_once_with("node1", None)

    def test_find_by_title(self):
        """Test finding nodes by title."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node1"]
        
        # Find nodes
        results = self.orm.find_by_title("Inception", None)
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.test_graph["node1"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once_with('title = "Inception"')

    def test_find_by_field(self):
        """Test finding nodes by field value."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node1", "node3"]
        
        # Find nodes
        results = self.orm.find_by_field("director", "Christopher Nolan", "=", None)
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], self.test_graph["node1"])
        self.assertEqual(results[1], self.test_graph["node3"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once_with('director = "Christopher Nolan"')

    def test_find_by_tag(self):
        """Test finding nodes by tag."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node2"]
        
        # Find nodes
        results = self.orm.find_by_tag("cyberpunk", None)
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.test_graph["node2"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once_with('tags contains "cyberpunk"')

    def test_find_by_genre(self):
        """Test finding nodes by genre."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node3"]
        
        # Find nodes
        results = self.orm.find_by_genre("Drama", None)
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.test_graph["node3"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once_with('genres contains "Drama"')

    def test_find_by_year_range(self):
        """Test finding nodes by year range."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node1", "node3"]
        
        # Find nodes
        results = self.orm.find_by_year_range(2010, 2020, None)
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], self.test_graph["node1"])
        self.assertEqual(results[1], self.test_graph["node3"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once_with('year >= 2010 AND year <= 2020')

    def test_find_by_rating(self):
        """Test finding nodes by minimum rating."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node1"]
        
        # Find nodes
        results = self.orm.find_by_rating(8.8, None)
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.test_graph["node1"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once_with('rating >= 8.8')

    def test_find_by_text_search(self):
        """Test finding nodes by text search."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node1"]
        
        # Find nodes
        results = self.orm.find_by_text_search("inception", ["title", "overview"], None)
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.test_graph["node1"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once_with('title contains "inception" OR overview contains "inception"')

    @patch('src.models.orm.GraphORM.find_by_similarity')
    def test_find_by_combined_criteria(self, mock_find_by_similarity):
        """Test finding nodes by combined criteria."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node1"]
        mock_find_by_similarity.return_value = [(self.test_graph["node1"], 0.9)]
        
        # Find nodes
        results = self.orm.find_by_combined_criteria(
            genres=["Sci-Fi"],
            director="Christopher Nolan",
            min_rating=8.5,
            similar_to="node2",
            similarity_threshold=0.7,
            user=None
        )
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.test_graph["node1"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once()
        query = self.mock_db.query_engine.execute_query.call_args[0][0]
        self.assertIn('genres contains "Sci-Fi"', query)
        self.assertIn('director = "Christopher Nolan"', query)
        self.assertIn('rating >= 8.5', query)
        
        # Check if find_by_similarity was called with the correct parameters
        mock_find_by_similarity.assert_called_once_with(
            node_id="node2",
            threshold=0.7,
            limit=len(self.test_graph),
            user=None
        )

    def test_group_by_field(self):
        """Test grouping nodes by field value."""
        # Group by director
        grouped = self.orm.group_by_field("director", None)
        
        # Check if the nodes were grouped correctly
        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(grouped["Christopher Nolan"]), 2)
        self.assertEqual(len(grouped["Lana Wachowski"]), 1)
        self.assertEqual(grouped["Christopher Nolan"][0], self.test_graph["node1"])
        self.assertEqual(grouped["Christopher Nolan"][1], self.test_graph["node3"])
        self.assertEqual(grouped["Lana Wachowski"][0], self.test_graph["node2"])

    def test_aggregate_by_field(self):
        """Test aggregating nodes by field value."""
        # Aggregate by director
        aggregated = self.orm.aggregate_by_field("director", "count", None)
        
        # Check if the nodes were aggregated correctly
        self.assertEqual(len(aggregated), 2)
        self.assertEqual(aggregated["Christopher Nolan"], 2)
        self.assertEqual(aggregated["Lana Wachowski"], 1)
        
        # Aggregate by director with avg rating
        aggregated = self.orm.aggregate_by_field("director", "avg", None)
        
        # Check if the nodes were aggregated correctly
        self.assertEqual(len(aggregated), 2)
        self.assertAlmostEqual(aggregated["Christopher Nolan"], (8.8 + 8.6) / 2)
        self.assertAlmostEqual(aggregated["Lana Wachowski"], 8.7)

    def test_find_by_field_contains(self):
        """Test finding nodes where a field contains a value."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node2"]
        
        # Find nodes
        results = self.orm.find_by_field_contains("tags", "cyberpunk", None)
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.test_graph["node2"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once_with('tags contains "cyberpunk"')

    def test_find_by_field_range(self):
        """Test finding nodes by field range."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node1", "node3"]
        
        # Find nodes
        results = self.orm.find_by_field_range("year", 2010, 2020, None)
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], self.test_graph["node1"])
        self.assertEqual(results[1], self.test_graph["node3"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once_with('year >= 2010 AND year <= 2020')

    def test_find_by_field_min(self):
        """Test finding nodes by minimum field value."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node1"]
        
        # Find nodes
        results = self.orm.find_by_field_min("rating", 8.8, None)
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.test_graph["node1"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once_with('rating >= 8.8')

    @patch('src.models.orm.GraphORM.find_by_similarity')
    def test_find_by_criteria(self, mock_find_by_similarity):
        """Test finding nodes by combined criteria."""
        # Set up mock
        self.mock_db.query_engine.execute_query.return_value = ["node1"]
        mock_find_by_similarity.return_value = [(self.test_graph["node1"], 0.9)]
        
        # Find nodes
        results = self.orm.find_by_criteria(
            criteria={
                "director": {"=": "Christopher Nolan"},
                "rating": {">=": 8.5}
            },
            text_search={
                ("title", "overview"): "inception"
            },
            similarity={
                "node_id": "node2",
                "threshold": 0.7
            },
            user=None
        )
        
        # Check if the correct nodes were found
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.test_graph["node1"])
        
        # Check if the execute_query method was called with the correct parameters
        self.mock_db.query_engine.execute_query.assert_called_once()
        query = self.mock_db.query_engine.execute_query.call_args[0][0]
        self.assertIn('director = "Christopher Nolan"', query)
        self.assertIn('rating >= 8.5', query)
        self.assertIn('title contains "inception" OR overview contains "inception"', query)
        
        # Check if find_by_similarity was called with the correct parameters
        mock_find_by_similarity.assert_called_once_with(
            node_id="node2",
            threshold=0.7,
            limit=len(self.test_graph),
            user=None
        )


if __name__ == "__main__":
    unittest.main()
