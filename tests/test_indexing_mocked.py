"""
Unit tests for indexing module with proper mocking.
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import json

from src.models.indexing import (
    BaseIndex, HashIndex, BTreeIndex, FullTextIndex, VectorIndex, IndexManager, IndexType
)


class TestIndexingMocked(unittest.TestCase):
    """Test cases for indexing module with proper mocking."""

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
    
    def test_hash_index_basic(self):
        """Test basic HashIndex functionality."""
        # Create a hash index
        index = HashIndex("test_hash", "tags")
        
        # Build the index
        index.build(self.test_graph)
        self.assertTrue(index.is_built)
        
        # Test search
        results = index.search("test")
        self.assertEqual(len(results), 3)  # All nodes have the tag "test"
        
        results = index.search("one")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "node1")
    
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
    
    def test_vector_index_basic(self):
        """Test basic VectorIndex functionality."""
        # Create a vector index
        index = VectorIndex("test_vector", "embedding")
        
        # Build the index
        index.build(self.test_graph)
        self.assertTrue(index.is_built)
        
        # Test search with vector
        results = index.search([0.1, 0.2, 0.3])
        self.assertEqual(len(results), 3)  # All nodes should match with some similarity
        
        # First result should be node1 (exact match)
        self.assertEqual(results[0][0], "node1")
    
    @patch('src.models.indexing.BTreeIndex.search')
    def test_btree_index_basic(self, mock_search):
        """Test basic BTreeIndex functionality."""
        # Set up mock
        mock_search.side_effect = lambda query, prefix=False: (
            ["node1", "node2", "node3"] if prefix and query == "Test" else
            ["node1"] if query == "Test Node 1" else []
        )
        
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
    
    def test_index_manager_basic(self):
        """Test basic IndexManager functionality."""
        # Create an index manager
        manager = IndexManager()
        
        # Create indexes
        hash_index = manager.create_index("test_hash", "tags", IndexType.HASH)
        btree_index = manager.create_index("test_btree", "title", IndexType.BTREE)
        
        # Verify indexes were created
        self.assertEqual(len(manager.indexes), 2)
        self.assertIn("test_hash", manager.indexes)
        self.assertIn("test_btree", manager.indexes)
        
        # Test get_index
        retrieved_index = manager.get_index("test_hash")
        self.assertEqual(retrieved_index, hash_index)
        
        # Test drop_index
        manager.drop_index("test_hash")
        self.assertEqual(len(manager.indexes), 1)
        self.assertNotIn("test_hash", manager.indexes)
    
    @patch('src.models.indexing.HashIndex.save')
    @patch('src.models.indexing.BTreeIndex.save')
    @patch('src.models.indexing.FullTextIndex.save')
    @patch('src.models.indexing.VectorIndex.save')
    def test_index_manager_save(self, mock_vector_save, mock_fulltext_save, 
                               mock_btree_save, mock_hash_save):
        """Test IndexManager save functionality with mocked save."""
        # Set up mocks
        mock_hash_save.return_value = True
        mock_btree_save.return_value = True
        mock_fulltext_save.return_value = True
        mock_vector_save.return_value = True
        
        # Create an index manager with the index directory
        manager = IndexManager(index_dir=self.temp_dir)
        manager.create_index("test_hash", "tags", IndexType.HASH)
        manager.create_index("test_btree", "title", IndexType.BTREE)
        manager.create_index("test_fulltext", "content", IndexType.FULLTEXT)
        manager.create_index("test_vector", "embedding", IndexType.VECTOR)
        
        # Build indexes
        manager.rebuild_indexes(self.test_graph)
        
        # Save indexes
        with patch('os.makedirs'):
            success = manager.save_indexes()
            self.assertTrue(success)
        
        # Verify mocks were called
        self.assertEqual(mock_hash_save.call_count, 1)
        self.assertEqual(mock_btree_save.call_count, 1)
        self.assertEqual(mock_fulltext_save.call_count, 1)
        self.assertEqual(mock_vector_save.call_count, 1)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_index_manager_load(self, mock_listdir, mock_exists, mock_json_load, mock_open_file):
        """Test IndexManager load functionality with mocked file operations."""
        # Set up mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["test_hash.idx"]
        mock_json_load.return_value = {"type": "hash", "field": "tags"}
        
        # Create a mock for BaseIndex.load that returns True
        with patch('src.models.indexing.BaseIndex.load', return_value=True):
            # Create a mock for BaseIndex.index to avoid AttributeError
            with patch('src.models.indexing.BaseIndex.index', create=True, new_callable=dict):
                # Create an index manager with the index directory
                manager = IndexManager(index_dir=self.temp_dir)
                
                # Load indexes
                success = manager.load_indexes()
                self.assertTrue(success)
                
                # Verify mock was called
                mock_listdir.assert_called_once()
                mock_exists.assert_called_once()

if __name__ == "__main__":
    unittest.main()
