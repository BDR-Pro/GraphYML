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
            # Pass a string that's not a valid enum value
            manager.create_index("test_invalid", "field", "invalid_type")
        
        # Test search with non-existent index
        results = manager.search("nonexistent", "query")
        self.assertEqual(results, [])
    
    @patch('src.models.indexing.BaseIndex.save')
    @patch('src.models.indexing.BaseIndex.load')
    def test_index_manager_save_load(self, mock_load, mock_save):
        """Test IndexManager save and load functionality with mocked save/load."""
        # Set up mocks
        mock_save.return_value = True
        mock_load.return_value = True
        
        # Create a test directory
        index_dir = self.temp_dir
        
        # Create an index manager with the index directory
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
        self.assertEqual(mock_save.call_count, 4)  # Called for each index
        
        # Create a new manager and load indexes
        new_manager = IndexManager(index_dir=index_dir)
        
        # Mock the index creation during load
        with patch('src.models.indexing.HashIndex') as mock_hash_index, \
             patch('src.models.indexing.BTreeIndex') as mock_btree_index, \
             patch('src.models.indexing.FullTextIndex') as mock_fulltext_index, \
             patch('src.models.indexing.VectorIndex') as mock_vector_index:
            
            # Set up mock indexes
            mock_indexes = {
                "test_hash": MagicMock(),
                "test_btree": MagicMock(),
                "test_fulltext": MagicMock(),
                "test_vector": MagicMock()
            }
            
            # Set up mock index constructors
            mock_hash_index.return_value = mock_indexes["test_hash"]
            mock_btree_index.return_value = mock_indexes["test_btree"]
            mock_fulltext_index.return_value = mock_indexes["test_fulltext"]
            mock_vector_index.return_value = mock_indexes["test_vector"]
            
            # Mock the index metadata
            with patch('os.path.exists', return_value=True), \
                 patch('builtins.open', mock_open(read_data='{"type": "hash", "field": "tags"}')), \
                 patch('json.load', return_value={"type": "hash", "field": "tags"}):
                
                success = new_manager.load_indexes()
                self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
