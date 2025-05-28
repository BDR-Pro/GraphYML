"""
Extended unit tests for indexing module.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import json

from src.models.indexing import (
    BaseIndex, HashIndex, BTreeIndex, FullTextIndex, VectorIndex, IndexManager, IndexType
)
from src.models.embeddings import embedding_similarity


class TestIndexingExtended(unittest.TestCase):
    """Extended test cases for indexing module."""

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
        # Remove temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_hash_index_edge_cases(self):
        """Test HashIndex with edge cases."""
        # Create a hash index
        index = HashIndex("test_hash", "tags")
        
        # Test with empty graph
        index.build({})
        self.assertTrue(index.is_built)  # build() sets is_built to True even for empty graphs
        
        # Test search with empty index
        results = index.search("test")
        self.assertEqual(results, [])
        
        # Test with node missing the field
        graph = {"node1": {"id": "node1", "title": "Test"}}  # No tags field
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test update with missing field
        index.build(self.test_graph)  # Rebuild with valid data
        index.update("node4", {"id": "node4", "title": "Test"})  # No tags field
        
        # Test delete of non-existent key
        index.update("nonexistent", {}, is_delete=True)
        
        # Test save and load
        index_file = os.path.join(self.temp_dir, "hash_index.json")
        index.save(index_file)
        
        # Create a new index and load
        new_index = HashIndex("test_hash", "tags")
        new_index.load(index_file)
        
        # Verify loaded index
        self.assertEqual(new_index.name, index.name)
        self.assertEqual(new_index.field, index.field)
        self.assertEqual(new_index.index, index.index)
    
    def test_btree_index_edge_cases(self):
        """Test BTreeIndex with edge cases."""
        # Create a btree index
        index = BTreeIndex("test_btree", "title")
        
        # Test with empty graph
        index.build({})
        self.assertTrue(index.is_built)  # build() sets is_built to True even for empty graphs
        
        # Test search with empty index
        results = index.search("Test")
        self.assertEqual(results, [])
        
        # Test with node missing the field
        graph = {"node1": {"id": "node1", "tags": ["test"]}}  # No title field
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test update with missing field
        index.build(self.test_graph)  # Rebuild with valid data
        index.update("node4", {"id": "node4", "tags": ["test"]})  # No title field
        
        # Test delete of non-existent key
        index.update("nonexistent", {}, is_delete=True)
        
        # Test search with prefix
        results = index.search("Test", prefix=True)
        self.assertEqual(len(results), 3)  # All nodes have titles starting with "Test"
        
        # Test search with exact match
        results = index.search("Test Node 1", prefix=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "node1")
    
    def test_fulltext_index_edge_cases(self):
        """Test FullTextIndex with edge cases."""
        # Create a fulltext index
        index = FullTextIndex("test_fulltext", "content")
        
        # Test with empty graph
        index.build({})
        self.assertTrue(index.is_built)  # build() sets is_built to True even for empty graphs
        
        # Test search with empty index
        results = index.search("test")
        self.assertEqual(results, [])
        
        # Test with node missing the field
        graph = {"node1": {"id": "node1", "title": "Test"}}  # No content field
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test update with missing field
        index.build(self.test_graph)  # Rebuild with valid data
        index.update("node4", {"id": "node4", "title": "Test"})  # No content field
        
        # Test delete of non-existent key
        index.update("nonexistent", {}, is_delete=True)
        
        # Test search with multiple terms
        results = index.search("test node unique")
        self.assertEqual(len(results), 3)  # All nodes have "test" and "node", one has "unique"
        
        # Test search with exact phrase
        results = index.search("\"test node three\"")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node3")
    
    def test_vector_index_edge_cases(self):
        """Test VectorIndex with edge cases."""
        # Create a vector index
        index = VectorIndex("test_vector", "embedding")
        
        # Test with empty graph
        index.build({})
        self.assertTrue(index.is_built)  # build() sets is_built to True even for empty graphs
        
        # Test search with empty index
        results = index.search([0.1, 0.2, 0.3])
        self.assertEqual(results, [])
        
        # Test with node missing the field
        graph = {"node1": {"id": "node1", "title": "Test"}}  # No embedding field
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test update with missing field
        index.build(self.test_graph)  # Rebuild with valid data
        index.update("node4", {"id": "node4", "title": "Test"})  # No embedding field
        
        # Test delete of non-existent key
        index.update("nonexistent", {}, is_delete=True)
        
        # Test search with different thresholds
        with patch('src.models.embeddings.embedding_similarity') as mock_sim:
            mock_sim.side_effect = lambda a, b: 0.95 if a == b else 0.5
            
            # High threshold
            results = index.search([0.1, 0.2, 0.3], threshold=0.9)
            self.assertEqual(len(results), 1)  # Only exact match
            
            # Low threshold
            results = index.search([0.1, 0.2, 0.3], threshold=0.4)
            self.assertEqual(len(results), 3)  # All nodes
    
    def test_index_manager_edge_cases(self):
        """Test IndexManager with edge cases."""
        # Create an index manager
        manager = IndexManager()
        
        # Test get_index with non-existent index
        with self.assertRaises(ValueError):
            manager.get_index("nonexistent")
        
        # Test drop_index with non-existent index
        with self.assertRaises(ValueError):
            manager.drop_index("nonexistent")
        
        # Test create_index with invalid type
        with self.assertRaises(ValueError):
            # Need to pass a string that's not a valid IndexType enum value
            manager.create_index("invalid_type", "test_invalid", "field")
        
        # Test search with non-existent index
        results = manager.search("nonexistent", "query")
        self.assertEqual(results, [])
        
        # Test update_indexes with empty graph
        manager.update_indexes("node1", {}, is_delete=True)
        
        # Test rebuild_indexes with empty graph
        manager.rebuild_indexes({})
    
    def test_index_manager_save_load(self):
        """Test IndexManager save and load functionality."""
        # Create an index manager and add indexes
        manager = IndexManager()
        manager.create_index(IndexType.HASH, "test_hash", "tags")
        manager.create_index(IndexType.BTREE, "test_btree", "title")
        manager.create_index(IndexType.FULLTEXT, "test_fulltext", "content")
        manager.create_index(IndexType.VECTOR, "test_vector", "embedding")
        
        # Build indexes
        manager.rebuild_indexes(self.test_graph)
        
        # Save indexes
        index_dir = self.temp_dir
        success = manager.save_indexes(index_dir)
        self.assertTrue(success)
        
        # Create a new manager and load indexes
        new_manager = IndexManager()
        success = new_manager.load_indexes(index_dir)
        self.assertTrue(success)
        
        # Verify loaded indexes
        self.assertEqual(len(new_manager.indexes), 4)
        self.assertIn("test_hash", new_manager.indexes)
        self.assertIn("test_btree", new_manager.indexes)
        self.assertIn("test_fulltext", new_manager.indexes)
        self.assertIn("test_vector", new_manager.indexes)
        
        # Test search with loaded indexes
        results = new_manager.search("test_hash", "test")
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
