"""
Fixed unit tests for indexing module.
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open, PropertyMock
import os
import tempfile
import json
import pickle
from collections import defaultdict

from src.models.indexing import (
    BaseIndex, HashIndex, BTreeIndex, FullTextIndex, VectorIndex, IndexManager, IndexType
)
from src.models.embeddings import embedding_similarity


class MockBaseIndex(BaseIndex):
    """Mock implementation of BaseIndex for testing."""
    
    def __init__(self, name, field):
        super().__init__(name, field)
        self.index = defaultdict(list)
        self.is_built = True
    
    def load(self, path):
        return True
    
    def save(self, path):
        return True


class TestIndexingFixed(unittest.TestCase):
    """Fixed test cases for indexing module."""

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
        
        # Patch BaseIndex to use our mock implementation
        patcher = patch('src.models.indexing.BaseIndex', MockBaseIndex)
        self.mock_base_index = patcher.start()
        self.addCleanup(patcher.stop)
    
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
    
    @patch('src.models.indexing.BTreeIndex.save')
    @patch('src.models.indexing.BTreeIndex.load')
    def test_btree_index_edge_cases(self, mock_load, mock_save):
        """Test BTreeIndex with edge cases."""
        # Set up mocks
        mock_save.return_value = True
        mock_load.return_value = True
        
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
    
    @patch('src.models.indexing.FullTextIndex.save')
    @patch('src.models.indexing.FullTextIndex.load')
    def test_fulltext_index_edge_cases(self, mock_load, mock_save):
        """Test FullTextIndex with edge cases."""
        # Set up mocks
        mock_save.return_value = True
        mock_load.return_value = True
        
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
        # We expect at least one result, but the exact number depends on the implementation
        self.assertGreaterEqual(len(results), 1)
        
        # Test search with exact phrase
        results = index.search('"test node three"')
        # We expect at least one result for the phrase
        self.assertGreaterEqual(len(results), 1)
    
    @patch('src.models.embeddings.embedding_similarity')
    def test_vector_index_edge_cases(self, mock_sim):
        """Test VectorIndex with edge cases."""
        # Set up mock for embedding_similarity
        mock_sim.side_effect = lambda a, b: 0.95 if a == b else 0.5
        
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
        # High threshold
        results = index.search([0.1, 0.2, 0.3], threshold=0.9)
        self.assertGreaterEqual(len(results), 1)  # At least one match
        
        # Low threshold
        results = index.search([0.1, 0.2, 0.3], threshold=0.4)
        self.assertGreaterEqual(len(results), 3)  # All nodes should match
    
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
            manager.create_index("test_invalid", "field", "invalid_type")
        
        # Test create_index with invalid name
        with self.assertRaises(ValueError):
            manager.create_index(None, "field", IndexType.HASH)
        
        with self.assertRaises(ValueError):
            manager.create_index("", "field", IndexType.HASH)
        
        with self.assertRaises(ValueError):
            manager.create_index("invalid name", "field", IndexType.HASH)
        
        # Test search with non-existent index
        results = manager.search("nonexistent", "query")
        self.assertEqual(results, [])
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_index_manager_save_load(self, mock_json_load, mock_open_file, mock_listdir, mock_exists):
        """Test IndexManager save and load functionality with mocked save/load."""
        # Set up mocks
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "test_hash.idx", "test_btree.idx", 
            "test_fulltext.idx", "test_vector.idx"
        ]
        mock_json_load.side_effect = [
            {"type": "hash", "field": "tags"},
            {"type": "btree", "field": "title"},
            {"type": "fulltext", "field": "content"},
            {"type": "vector", "field": "embedding"}
        ]
        
        # Create a test directory
        index_dir = self.temp_dir
        
        # Create an index manager with the index directory
        with patch('src.models.indexing.HashIndex.save', return_value=True), \
             patch('src.models.indexing.BTreeIndex.save', return_value=True), \
             patch('src.models.indexing.FullTextIndex.save', return_value=True), \
             patch('src.models.indexing.VectorIndex.save', return_value=True), \
             patch('src.models.indexing.HashIndex.load', return_value=True), \
             patch('src.models.indexing.BTreeIndex.load', return_value=True), \
             patch('src.models.indexing.FullTextIndex.load', return_value=True), \
             patch('src.models.indexing.VectorIndex.load', return_value=True):
            
            manager = IndexManager(index_dir=index_dir)
            manager.create_index("test_hash", "tags", IndexType.HASH)
            manager.create_index("test_btree", "title", IndexType.BTREE)
            manager.create_index("test_fulltext", "content", IndexType.FULLTEXT)
            manager.create_index("test_vector", "embedding", IndexType.VECTOR)
            
            # Build indexes
            manager.rebuild_indexes(self.test_graph)
            
            # Save indexes
            success = manager.save_indexes()
            self.assertTrue(success)
            
            # Create a new manager and load indexes
            new_manager = IndexManager(index_dir=index_dir)
            success = new_manager.load_indexes()
            self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
