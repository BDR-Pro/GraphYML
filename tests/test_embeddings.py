"""
Unit tests for embeddings module.
"""
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.embeddings import (
    EmbeddingGenerator, embedding_similarity, batch_generate_embeddings
)


class TestEmbeddings(unittest.TestCase):
    """Test cases for embeddings module."""

    def setUp(self):
        """Set up test environment."""
        # Create a test configuration
        self.test_config = {
            "ollama_url": "http://test.example.com",
            "ollama_model": "test-model",
            "embedding_dimension": 384,
            "allow_fallback": True
        }

        # Create a test embedding generator
        self.embedding_generator = EmbeddingGenerator(self.test_config)

    def test_embedding_similarity(self):
        """Test embedding similarity calculation."""
        # Create test embeddings
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        embedding3 = [1.0, 1.0, 0.0]

        # Calculate similarities
        sim1_2 = embedding_similarity(embedding1, embedding2)
        sim1_3 = embedding_similarity(embedding1, embedding3)
        sim2_3 = embedding_similarity(embedding2, embedding3)
        sim1_1 = embedding_similarity(embedding1, embedding1)

        # Check if similarities are correct
        self.assertAlmostEqual(sim1_2, 0.0)
        self.assertAlmostEqual(sim1_3, 0.7071067811865475)
        self.assertAlmostEqual(sim2_3, 0.7071067811865475)
        self.assertAlmostEqual(sim1_1, 1.0)

        # Test with empty embeddings
        self.assertEqual(embedding_similarity([], []), 0.0)
        self.assertEqual(embedding_similarity(None, []), 0.0)
        self.assertEqual(embedding_similarity([], None), 0.0)
        self.assertEqual(embedding_similarity(None, None), 0.0)

    @patch('requests.post')
    def test_generate_ollama_embedding(self, mock_post):
        """Test generating embeddings using Ollama API."""
        # Mock the response from Ollama API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3]
        }
        mock_post.return_value = mock_response

        # Generate embedding
        embedding, error = self.embedding_generator._generate_ollama_embedding("test text")

        # Check if the embedding was generated correctly
        self.assertIsNotNone(embedding)
        self.assertIsNone(error)
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

        # Check if the API was called with the correct parameters
        mock_post.assert_called_once_with(
            self.test_config["ollama_url"],
            json={"model": self.test_config["ollama_model"], "prompt": "test text"},
            timeout=30
        )

        # Test error handling
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        embedding, error = self.embedding_generator._generate_ollama_embedding("test text")
        self.assertIsNone(embedding)
        self.assertIsNotNone(error)
        self.assertIn("Ollama API error", error)

    def test_generate_fallback_embedding(self):
        """Test generating fallback embeddings."""
        # Generate fallback embedding
        embedding, error = self.embedding_generator._generate_fallback_embedding("test text")

        # Check if the embedding was generated correctly
        self.assertIsNotNone(embedding)
        self.assertIsNotNone(error)
        self.assertIn("fallback", error)
        self.assertEqual(len(embedding), self.test_config["embedding_dimension"])

        # Check if the embedding is normalized
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=6)

        # Check if different texts produce different embeddings
        embedding2, _ = self.embedding_generator._generate_fallback_embedding("different text")
        self.assertNotEqual(embedding, embedding2)

    @patch('src.models.embeddings.EmbeddingGenerator.generate_embedding')
    def test_batch_generate_embeddings(self, mock_generate):
        """Test batch generating embeddings."""
        # Mock the generate_embedding method
        mock_generate.side_effect = lambda text: ([0.1, 0.2, 0.3], None)

        # Create a test graph
        test_graph = {
            "node1": {
                "id": "node1",
                "title": "Test Node 1",
                "overview": "This is a test node.",
                "tagline": "Test tagline"
            },
            "node2": {
                "id": "node2",
                "title": "Test Node 2",
                "overview": "This is another test node.",
                "embedding": [0.4, 0.5, 0.6]
            }
        }

        # Batch generate embeddings
        updated_graph, errors = batch_generate_embeddings(
            test_graph,
            self.embedding_generator
        )

        # Check if embeddings were generated correctly
        self.assertEqual(len(errors), 0)
        self.assertIn("embedding", updated_graph["node1"])
        self.assertEqual(updated_graph["node1"]["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(updated_graph["node2"]["embedding"], [0.4, 0.5, 0.6])

        # Check if the generate_embedding method was called with the correct text
        mock_generate.assert_called_once_with(
            "Test Node 1 This is a test node. Test tagline"
        )


if __name__ == "__main__":
    unittest.main()

