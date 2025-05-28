"""
Fixed unit tests for indexing module.
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import json

from src.models.indexing import (
    BaseIndex, HashIndex, BTreeIndex, FullTextIndex, VectorIndex, IndexManager, IndexType
)
from src.models.embeddings import embedding_similarity


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
    
    @patch('src.models.indexing.IndexType')
    def test_index_manager_edge_cases(self, mock_index_type):
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
        mock_index_type.__eq__.return_value = False
        with self.assertRaises(ValueError):
            manager.create_index("test_invalid", "field", "invalid_type")
        
        # Test search with non-existent index
        results = manager.search("nonexistent", "query")
        self.assertEqual(results, [])
    
    @patch('src.models.indexing.HashIndex.save')
    
    
    @patch('src.models.indexing.VectorIndex.save')
    @patch('src.models.indexing.HashIndex.load')
    
    
    @patch('src.models.indexing.VectorIndex.load')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_index_manager_save_load(self, mock_json_load, mock_open_file, mock_listdir, 
                                    mock_exists, mock_vector_load, mock_fulltext_load, 
                                    mock_btree_load, mock_hash_load, mock_vector_save, 
                                    mock_fulltext_save, mock_btree_save, mock_hash_save):
        """Test IndexManager save and load functionality with mocked save/load."""
        # Set up mocks
        mock_hash_save.return_value = True
        mock_btree_save.return_value = True
        mock_fulltext_save.return_value = True
        mock_vector_save.return_value = True
        mock_hash_load.return_value = True
        mock_btree_load.return_value = True
        mock_fulltext_load.return_value = True
        mock_vector_load.return_value = True
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
        
        # Verify mock calls
        self.assertEqual(mock_hash_save.call_count, 1)
        self.assertEqual(mock_btree_save.call_count, 1)
        self.assertEqual(mock_fulltext_save.call_count, 1)
        self.assertEqual(mock_vector_save.call_count, 1)
        self.assertEqual(mock_hash_load.call_count, 1)
        self.assertEqual(mock_btree_load.call_count, 1)
        self.assertEqual(mock_fulltext_load.call_count, 1)
        self.assertEqual(mock_vector_load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
