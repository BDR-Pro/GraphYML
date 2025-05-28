"""
Unit tests for the modular indexing system.
"""
import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

from src.models.modular import (
    BaseIndex, HashIndex, BTreeIndex, FullTextIndex, VectorIndex, IndexManager, IndexType
)
from src.models.embeddings import embedding_similarity


class TestModularIndexing(unittest.TestCase):
    """Test cases for the modular indexing system."""

    def setUp(self):
        """Set up test environment."""
        # Create a test graph
        self.test_graph = {
            "node1": {
                "id": "node1",
                "title": "Test Node 1",
                "content": "This is test node one with some content",
                "tags": ["test", "node", "one"],
                "embedding": [0.1, 0.2, 0.3]
            },
            "node2": {
                "id": "node2",
                "title": "Test Node 2",
                "content": "This is test node two with different content",
                "tags": ["test", "node", "two"],
                "embedding": [0.2, 0.3, 0.4]
            },
            "node3": {
                "id": "node3",
                "title": "Test Node 3",
                "content": "This is test node three with unique content",
                "tags": ["test", "node", "three"],
                "embedding": [0.3, 0.4, 0.5]
            }
        }
        
        # Create a temporary directory for index files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_hash_index_basic(self):
        """Test basic HashIndex functionality."""
        # Create a hash index
        index = HashIndex("test_hash", "tags")
        
        # Build the index
        index.build(self.test_graph)
        self.assertTrue(index.is_built)
        
        # Test search
        results = index.search("test")
        self.assertEqual(len(results), 3)  # All nodes have the "test" tag
        
        results = index.search("one")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "node1")
        
        # Test update
        index.update("node4", {"id": "node4", "tags": ["test", "four"]})
        results = index.search("four")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "node4")
        
        # Test delete
        index.update("node4", {}, is_delete=True)
        results = index.search("four")
        self.assertEqual(len(results), 0)
    
    def test_btree_index_basic(self):
        """Test basic BTreeIndex functionality."""
        # Create a btree index
        index = BTreeIndex("test_btree", "title")
        
        # Build the index
        index.build(self.test_graph)
        self.assertTrue(index.is_built)
        
        # Test search with prefix
        results = index.search("Test", prefix=True)
        self.assertEqual(len(results), 3)  # All nodes have titles starting with "Test"
        
        # Test search with exact match
        results = index.search("Test Node 1", prefix=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "node1")
        
        # Test update
        index.update("node4", {"id": "node4", "title": "Test Node 4"})
        results = index.search("Test Node 4")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "node4")
        
        # Test delete
        index.update("node4", {}, is_delete=True)
        results = index.search("Test Node 4")
        self.assertEqual(len(results), 0)
    
    def test_fulltext_index_basic(self):
        """Test basic FullTextIndex functionality."""
        # Create a fulltext index
        index = FullTextIndex("test_fulltext", "content")
        
        # Build the index
        index.build(self.test_graph)
        self.assertTrue(index.is_built)
        
        # Test search with multiple terms
        results = index.search("test node")
        self.assertEqual(len(results), 3)  # All nodes have "test" and "node"
        
        # Test search with specific term
        results = index.search("unique")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node3")
        
        # Test search with phrase
        results = index.search("\"test node three\"")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node3")
        
        # Test update
        index.update("node4", {"id": "node4", "content": "This is test node four with special content"})
        results = index.search("special")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node4")
        
        # Test delete
        index.update("node4", {}, is_delete=True)
        results = index.search("special")
        self.assertEqual(len(results), 0)
    
    @patch('src.models.embeddings.embedding_similarity')
    def test_vector_index_basic(self, mock_similarity):
        """Test basic VectorIndex functionality."""
        # Create a vector index
        index = VectorIndex("test_vector", "embedding")
        
        # Create a test graph with controlled embeddings
        test_graph = {
            "node1": {"embedding": [1.0, 0.0, 0.0]},  # Orthogonal to node2 and node3
            "node2": {"embedding": [0.0, 1.0, 0.0]},  # Orthogonal to node1 and node3
            "node3": {"embedding": [0.0, 0.0, 1.0]}   # Orthogonal to node1 and node2
        }
        
        # Build the index
        index.build(test_graph)
        self.assertTrue(index.is_built)
        
        # Test search with exact match (cosine similarity = 1.0)
        results = index.search([1.0, 0.0, 0.0], threshold=0.9)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node1")
        
        # Test search with no matches (all orthogonal vectors)
        results = index.search([0.0, 1.0, 0.0], threshold=0.9)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node2")
        
        # Test update
        index.update("node4", {"embedding": [0.5, 0.5, 0.0]})  # Similar to both node1 and node2
        
        # Test search with lower threshold to get multiple results
        results = index.search([0.7, 0.7, 0.0], threshold=0.5)
        self.assertTrue(len(results) >= 2)  # Should match at least node1, node2, and node4
        
        # Test delete
        index.update("node4", {}, is_delete=True)
        results = index.search([0.5, 0.5, 0.0], threshold=0.9)
        self.assertEqual(len(results), 0)  # node4 should be gone
    
    def test_index_manager_basic(self):
        """Test basic IndexManager functionality."""
        # Create an index manager
        manager = IndexManager()
        
        # Create indexes
        hash_index = manager.create_index("test_hash", "tags", IndexType.HASH)
        btree_index = manager.create_index("test_btree", "title", IndexType.BTREE)
        fulltext_index = manager.create_index("test_fulltext", "content", IndexType.FULLTEXT)
        vector_index = manager.create_index("test_vector", "embedding", IndexType.VECTOR)
        
        # Check indexes were created
        self.assertIsInstance(hash_index, HashIndex)
        self.assertIsInstance(btree_index, BTreeIndex)
        self.assertIsInstance(fulltext_index, FullTextIndex)
        self.assertIsInstance(vector_index, VectorIndex)
        
        # Build indexes
        manager.rebuild_indexes(self.test_graph)
        
        # Test get_index
        retrieved_hash_index = manager.get_index("test_hash")
        self.assertIsInstance(retrieved_hash_index, HashIndex)
        self.assertEqual(retrieved_hash_index.name, "test_hash")
        
        # Test search
        results = manager.search("test_hash", "test")
        self.assertEqual(len(results), 3)  # All nodes have the "test" tag
        
        # Test drop_index
        manager.drop_index("test_hash")
        with self.assertRaises(ValueError):
            manager.get_index("test_hash")
    
    def test_save_load_operations(self):
        """Test save and load operations."""
        # Create an index manager with the index directory
        manager = IndexManager(index_dir=self.temp_dir)
        
        # Create indexes
        manager.create_index("test_hash", "tags", IndexType.HASH)
        manager.create_index("test_btree", "title", IndexType.BTREE)
        manager.create_index("test_fulltext", "content", IndexType.FULLTEXT)
        manager.create_index("test_vector", "embedding", IndexType.VECTOR)
        
        # Build indexes
        manager.rebuild_indexes(self.test_graph)
        
        # Save indexes
        success = manager.save_indexes()
        self.assertTrue(success)
        
        # Check files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_hash.idx")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_btree.idx")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_fulltext.idx")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_vector.idx")))
        
        # Create a new manager and load indexes
        new_manager = IndexManager(index_dir=self.temp_dir)
        success = new_manager.load_indexes()
        self.assertTrue(success)
        
        # Check indexes were loaded
        self.assertIn("test_hash", new_manager.indexes)
        self.assertIn("test_btree", new_manager.indexes)
        self.assertIn("test_fulltext", new_manager.indexes)
        self.assertIn("test_vector", new_manager.indexes)
        
        # Test search with loaded indexes
        results = new_manager.search("test_hash", "test")
        self.assertEqual(len(results), 3)  # All nodes have the "test" tag


if __name__ == "__main__":
    unittest.main()
