"""
Unit tests for query engine module.
"""
import unittest
from unittest.mock import patch

from src.models.query_engine import (
    QueryCondition, Query, QueryParser, QueryEngine, OPERATORS
)


class TestQueryEngine(unittest.TestCase):
    """Test cases for query engine module."""

    def setUp(self):
        """Set up test environment."""
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

    def test_query_condition(self):
        """Test query condition evaluation."""
        # Create a test node
        node = self.test_graph["node1"]

        # Test equality operator
        condition = QueryCondition("title", "=", "Inception")
        self.assertTrue(condition.evaluate(node))

        # Test inequality operator
        condition = QueryCondition("year", ">", 2000)
        self.assertTrue(condition.evaluate(node))
        condition = QueryCondition("year", "<", 2000)
        self.assertFalse(condition.evaluate(node))

        # Test contains operator
        condition = QueryCondition("tags", "contains", "sci-fi")
        self.assertTrue(condition.evaluate(node))
        condition = QueryCondition("tags", "contains", "horror")
        self.assertFalse(condition.evaluate(node))

        # Test in operator
        condition = QueryCondition("director", "in", ["Christopher Nolan", "Steven Spielberg"])
        self.assertTrue(condition.evaluate(node))
        condition = QueryCondition("director", "in", ["James Cameron", "Steven Spielberg"])
        self.assertFalse(condition.evaluate(node))

        # Test string operators
        condition = QueryCondition("title", "startswith", "Inc")
        self.assertTrue(condition.evaluate(node))
        condition = QueryCondition("title", "endswith", "tion")
        self.assertTrue(condition.evaluate(node))
        condition = QueryCondition("title", "matches", "Inc.*")
        self.assertTrue(condition.evaluate(node))

        # Test nested fields
        condition = QueryCondition("metadata.runtime", ">", 120)
        self.assertTrue(condition.evaluate(node))
        condition = QueryCondition("metadata.language", "=", "English")
        self.assertTrue(condition.evaluate(node))

        # Test negation
        condition = QueryCondition("year", "<", 2015, negate=True)
        self.assertFalse(condition.evaluate(node))

        # Test non-existent field
        condition = QueryCondition("nonexistent", "=", "value")
        self.assertFalse(condition.evaluate(node))

    def test_query(self):
        """Test query evaluation."""
        # Create a test node
        node = self.test_graph["node1"]

        # Test single condition
        query = Query()
        query.add_condition("title", "=", "Inception")
        self.assertTrue(query.evaluate(node))

        # Test multiple conditions with AND
        query = Query()
        query.add_condition("title", "=", "Inception")
        query.add_condition("year", ">", 2000, "AND")
        self.assertTrue(query.evaluate(node))

        # Test multiple conditions with OR
        query = Query()
        query.add_condition("title", "=", "The Matrix")
        query.add_condition("director", "=", "Christopher Nolan", "OR")
        self.assertTrue(query.evaluate(node))

        # Test complex query
        query = Query()
        query.add_condition("year", ">", 2000)
        query.add_condition("rating", ">=", 8.5, "AND")
        query.add_condition("genres", "contains", "Horror", "OR")
        self.assertTrue(query.evaluate(node))

        # Test query that doesn't match
        query = Query()
        query.add_condition("year", "<", 2000)
        query.add_condition("rating", "<", 8.0, "OR")
        self.assertFalse(query.evaluate(node))

    def test_query_parser(self):
        """Test query parser."""
        # Test simple query
        query_str = "title = \"Inception\""
        query = QueryParser.parse(query_str)
        self.assertEqual(len(query.conditions), 1)
        self.assertEqual(query.conditions[0].field, "title")
        self.assertEqual(query.conditions[0].operator_str, "=")
        self.assertEqual(query.conditions[0].value, "Inception")

        # Test query with multiple conditions
        query_str = "year > 2000 AND rating >= 8.5"
        query = QueryParser.parse(query_str)
        self.assertEqual(len(query.conditions), 2)
        self.assertEqual(query.conditions[0].field, "year")
        self.assertEqual(query.conditions[0].operator_str, ">")
        self.assertEqual(query.conditions[0].value, 2000)
        self.assertEqual(query.conditions[1].field, "rating")
        self.assertEqual(query.conditions[1].operator_str, ">=")
        self.assertEqual(query.conditions[1].value, 8.5)
        self.assertEqual(query.operators[0], "AND")

        # Test query with OR
        query_str = "title = \"Inception\" OR title = \"The Matrix\""
        query = QueryParser.parse(query_str)
        self.assertEqual(len(query.conditions), 2)
        self.assertEqual(query.operators[0], "OR")

        # Test query with NOT
        query_str = "NOT year < 2000"
        query = QueryParser.parse(query_str)
        self.assertEqual(len(query.conditions), 1)
        self.assertTrue(query.conditions[0].negate)

        # Test complex query
        query_str = "director = \"Christopher Nolan\" AND (year > 2010 OR rating > 9.0)"
        query = QueryParser.parse(query_str)
        self.assertEqual(len(query.conditions), 3)
        self.assertEqual(query.conditions[0].field, "director")
        self.assertEqual(query.conditions[0].value, "Christopher Nolan")
        self.assertEqual(query.operators[0], "AND")
        self.assertEqual(query.operators[1], "OR")

        # Test value conversion
        query_str = "year = 2010 AND rating = 8.8 AND featured = true"
        query = QueryParser.parse(query_str)
        self.assertEqual(query.conditions[0].value, 2010)
        self.assertEqual(query.conditions[1].value, 8.8)
        self.assertEqual(query.conditions[2].value, True)

    def test_query_engine(self):
        """Test query engine."""
        # Create a query engine
        engine = QueryEngine(self.test_graph)

        # Test simple query
        results = engine.execute_query("title = \"Inception\"")
        self.assertEqual(len(results), 1)
        self.assertIn("node1", results)

        # Test query with multiple conditions
        results = engine.execute_query("director = \"Christopher Nolan\" AND year > 2010")
        self.assertEqual(len(results), 1)
        self.assertIn("node3", results)

        # Test query with OR
        results = engine.execute_query("title = \"Inception\" OR title = \"The Matrix\"")
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node2", results)

        # Test query with contains
        results = engine.execute_query("tags contains \"cyberpunk\"")
        self.assertEqual(len(results), 1)
        self.assertIn("node2", results)

        # Test query with multiple genres
        results = engine.execute_query("genres contains \"Sci-Fi\" AND genres contains \"Drama\"")
        self.assertEqual(len(results), 1)
        self.assertIn("node3", results)

        # Test query with no results
        results = engine.execute_query("year < 1990")
        self.assertEqual(len(results), 0)

        # Test find_by_field
        results = engine.find_by_field("director", "Christopher Nolan")
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node3", results)

    @patch('src.models.embeddings.embedding_similarity')
    def test_find_by_embedding_similarity(self, mock_embedding_similarity):
        """Test finding nodes by embedding similarity."""
        # Mock embedding similarity
        mock_embedding_similarity.side_effect = lambda a, b: 0.9 if a[0] == 0.1 and b[0] == 0.1 else 0.5

        # Create a query engine
        engine = QueryEngine(self.test_graph)

        # Find similar nodes
        results = engine.find_by_embedding_similarity([0.1, 0.2, 0.3], 0.7, 2)

        # Check if similar nodes were found correctly
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node1")
        self.assertAlmostEqual(results[0][1], 0.9)

        # Test with lower threshold
        results = engine.find_by_embedding_similarity([0.1, 0.2, 0.3], 0.4, 3)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()

