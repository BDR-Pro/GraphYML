"""
Unit tests for the modular indexing system.
"""
import unittest
import tempfile
import os
import shutil
import pickle
import json
from unittest.mock import patch, MagicMock, mock_open

from src.models.modular import (
    BaseIndex, HashIndex, BTreeIndex, FullTextIndex, VectorIndex, IndexManager, IndexType
)
from src.models.embeddings import embedding_similarity


class MockBaseIndex(BaseIndex):
    """Mock implementation of BaseIndex for testing abstract methods."""
    
    def build(self, graph):
        self.is_built = True
    
    def search(self, query, **kwargs):
        return []
    
    def update(self, node_id, node_data, is_delete=False):
        pass


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
    
    def test_base_index_abstract_methods(self):
        """Test BaseIndex abstract methods."""
        # Create a base index directly (should raise TypeError)
        with self.assertRaises(TypeError):
            index = BaseIndex("test_base", "field")
        
        # Create a mock index
        index = MockBaseIndex("test_base", "field")
        
        # Test abstract methods
        self.assertEqual(index.search("query"), [])
        index.update("node_id", {"field": "value"})
        index.build({})
        self.assertTrue(index.is_built)
    
    def test_base_index_error_handling(self):
        """Test BaseIndex error handling."""
        # Create a mock index
        index = MockBaseIndex("test_base", "field")
        
        # Test save method with invalid path
        with patch('builtins.open', side_effect=Exception("Test error")):
            result = index.save("/invalid/path")
            self.assertFalse(result)
        
        # Test load method with non-existent file
        result = index.load("/non/existent/file")
        self.assertFalse(result)
        
        # Test load method with invalid data
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=Exception("Test error")):
                result = index.load("/invalid/data")
                self.assertFalse(result)
    
    def test_base_index_methods(self):
        """Test BaseIndex methods."""
        # Create a mock index
        index = MockBaseIndex("test_base", "field")
        
        # Test save method
        save_path = os.path.join(self.temp_dir, "test_base.idx")
        result = index.save(save_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(save_path + ".meta"))
        
        # Test load method
        new_index = MockBaseIndex("new_base", "new_field")
        result = new_index.load(save_path)
        self.assertTrue(result)
        self.assertEqual(new_index.name, "test_base")
        self.assertEqual(new_index.field, "field")
        
        # Test _get_field_value method
        node = {"field": "value", "other_field": "other_value"}
        value = index._get_field_value(node)
        self.assertEqual(value, "value")
        
        # Test _get_field_value with missing field
        node = {"other_field": "other_value"}
        value = index._get_field_value(node)
        self.assertIsNone(value)
        
        # Test serialization methods
        serialized = index._get_serializable_index()
        self.assertIsNone(serialized)
        
        index._set_index_from_serialized("test_data")
        self.assertEqual(index.index, "test_data")
    
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
        self.assertEqual(len(results), 1)  # Only node1 has the "one" tag
        self.assertEqual(results[0], "node1")
        
        # Test update
        index.update("node4", {"id": "node4", "tags": ["test", "node", "four"]})
        results = index.search("four")
        self.assertEqual(len(results), 1)  # node4 has the "four" tag
        self.assertEqual(results[0], "node4")
        
        # Test delete
        index.update("node4", {}, is_delete=True)
        results = index.search("four")
        self.assertEqual(len(results), 0)
    
    def test_hash_index_edge_cases(self):
        """Test HashIndex edge cases."""
        # Create a hash index
        index = HashIndex("test_hash", "tags")
        
        # Test with empty graph
        index.build({})
        self.assertTrue(index.is_built)
        
        # Test search with empty index
        results = index.search("test")
        self.assertEqual(results, [])
        
        # Test with node missing the field
        graph = {"node1": {"id": "node1", "title": "Test Node 1"}}  # No tags field
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test update with missing field
        index.update("node2", {"id": "node2", "title": "Test Node 2"})  # No tags field
        
        # Test delete of non-existent node
        index.update("non_existent", {}, is_delete=True)
        
        # Test with non-list field value
        graph = {"node1": {"id": "node1", "tags": "not_a_list"}}
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test search with non-existent value
        results = index.search("non_existent")
        self.assertEqual(results, [])
    
    def test_btree_index_basic(self):
        """Test basic BTreeIndex functionality."""
        # Create a btree index
        index = BTreeIndex("test_btree", "title")
        
        # Build the index
        index.build(self.test_graph)
        self.assertTrue(index.is_built)
        
        # Test search with exact match
        results = index.search("Test Node 1")
        self.assertEqual(len(results), 1)  # Only node1 has "Test Node 1" in title
        self.assertEqual(results[0], "node1")
        
        # Test update
        index.update("node4", {"id": "node4", "title": "Test Node 4"})
        results = index.search("Test Node 4")
        self.assertEqual(len(results), 1)  # node4 has "Test Node 4" in title
        self.assertEqual(results[0], "node4")
        
        # Test delete
        index.update("node4", {}, is_delete=True)
        results = index.search("Test Node 4")
        self.assertEqual(len(results), 0)
    
    def test_btree_index_edge_cases(self):
        """Test BTreeIndex edge cases."""
        # Create a btree index
        index = BTreeIndex("test_btree", "title")
        
        # Test with empty graph
        index.build({})
        self.assertTrue(index.is_built)
        
        # Test search with empty index
        results = index.search("Test")
        self.assertEqual(results, [])
        
        # Test with node missing the field
        graph = {"node1": {"id": "node1", "tags": ["test"]}}  # No title field
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test update with missing field
        index.update("node2", {"id": "node2", "tags": ["test"]})  # No title field
        
        # Test delete of non-existent node
        index.update("non_existent", {}, is_delete=True)
        
        # Test serialization methods
        serialized = index._get_serializable_index()
        self.assertIsInstance(serialized, dict)
        self.assertIn("index", serialized)
        self.assertIn("sorted_keys", serialized)
        self.assertIn("value_to_nodes", serialized)
        
        # Test deserialization
        new_index = BTreeIndex("new_btree", "title")
        new_index._set_index_from_serialized(serialized)
        self.assertEqual(new_index.index, serialized["index"])
        self.assertEqual(new_index.sorted_keys, serialized["sorted_keys"])
        self.assertEqual(new_index.value_to_nodes, serialized["value_to_nodes"])
    
    def test_fulltext_index_basic(self):
        """Test basic FullTextIndex functionality."""
        # Create a fulltext index
        index = FullTextIndex("test_fulltext", "content")
        
        # Build the index
        index.build(self.test_graph)
        self.assertTrue(index.is_built)
        
        # Test search
        results = index.search("test")
        self.assertEqual(len(results), 3)  # All nodes have "test" in content
        
        results = index.search("one")
        self.assertEqual(len(results), 1)  # Only node1 has "one" in content
        self.assertEqual(results[0][0], "node1")
        
        # Test update
        index.update("node4", {"id": "node4", "content": "This is test node four with special content"})
        results = index.search("four")
        self.assertEqual(len(results), 1)  # node4 has "four" in content
        self.assertEqual(results[0][0], "node4")
        
        # Test delete
        index.update("node4", {}, is_delete=True)
        results = index.search("special")
        self.assertEqual(len(results), 0)
    
    def test_fulltext_index_edge_cases(self):
        """Test FullTextIndex edge cases."""
        # Create a fulltext index
        index = FullTextIndex("test_fulltext", "content")
        
        # Test with empty graph
        index.build({})
        self.assertTrue(index.is_built)
        
        # Test search with empty index
        results = index.search("test")
        self.assertEqual(results, [])
        
        # Test with node missing the field
        graph = {"node1": {"id": "node1", "title": "Test Node 1"}}  # No content field
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test with non-string field value
        graph = {"node1": {"id": "node1", "content": 123}}  # Not a string
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test update with missing field
        index.update("node2", {"id": "node2", "title": "Test Node 2"})  # No content field
        
        # Test delete of non-existent node
        index.update("non_existent", {}, is_delete=True)
        
        # Test search with empty query
        results = index.search("")
        self.assertEqual(results, [])
        
        # Test serialization methods
        serialized = index._get_serializable_index()
        self.assertIsInstance(serialized, dict)
        self.assertIn("inverted_index", serialized)
        self.assertIn("node_terms", serialized)
        
        # Test deserialization
        new_index = FullTextIndex("new_fulltext", "content")
        new_index._set_index_from_serialized(serialized)
        self.assertEqual(dict(new_index.inverted_index), serialized["inverted_index"])
        self.assertEqual(new_index.node_terms, serialized["node_terms"])
    
    def test_fulltext_index_tokenize(self):
        """Test FullTextIndex tokenize method."""
        # Create a fulltext index
        index = FullTextIndex("test_fulltext", "content")
        
        # Test tokenize with simple text
        tokens = index._tokenize("Hello, world!")
        self.assertEqual(tokens, ["hello", "world"])
        
        # Test tokenize with punctuation
        tokens = index._tokenize("Hello, world! This is a test.")
        self.assertEqual(tokens, ["hello", "world", "this", "is", "a", "test"])
        
        # Test tokenize with numbers
        tokens = index._tokenize("Test 123")
        self.assertEqual(tokens, ["test", "123"])
        
        # Test tokenize with empty string
        tokens = index._tokenize("")
        self.assertEqual(tokens, [])
    
    def test_vector_index_basic(self):
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
    
    def test_vector_index_edge_cases(self):
        """Test VectorIndex edge cases."""
        # Create a vector index
        index = VectorIndex("test_vector", "embedding")
        
        # Test with empty graph
        index.build({})
        self.assertTrue(index.is_built)
        
        # Test search with empty index
        results = index.search([0.1, 0.2, 0.3])
        self.assertEqual(results, [])
        
        # Test with node missing the field
        graph = {"node1": {"id": "node1", "title": "Test Node 1"}}  # No embedding field
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test with non-list field value
        graph = {"node1": {"id": "node1", "embedding": "not_a_list"}}  # Not a list
        index.build(graph)
        self.assertTrue(index.is_built)
        
        # Test update with missing field
        index.update("node2", {"id": "node2", "title": "Test Node 2"})  # No embedding field
        
        # Test delete of non-existent node
        index.update("non_existent", {}, is_delete=True)
    
    def test_index_manager_basic(self):
        """Test basic IndexManager functionality."""
        # Create an index manager
        manager = IndexManager()
        
        # Create indexes
        hash_index = manager.create_index("test_hash", "tags", IndexType.HASH)
        btree_index = manager.create_index("test_btree", "title", IndexType.BTREE)
        
        # Build indexes
        hash_index.build(self.test_graph)
        btree_index.build(self.test_graph)
        
        # Test get_index
        retrieved_hash_index = manager.get_index("test_hash")
        self.assertEqual(retrieved_hash_index, hash_index)
        
        # Test search
        results = manager.search("test_hash", "test")
        self.assertEqual(len(results), 3)  # All nodes have the "test" tag
        
        # Test drop_index
        manager.drop_index("test_hash")
        with self.assertRaises(ValueError):
            manager.get_index("test_hash")
    
    def test_save_load_operations(self):
        """Test save and load operations."""
        # Create an index manager
        manager = IndexManager(index_dir=self.temp_dir)
        
        # Create and build indexes
        hash_index = manager.create_index("test_hash", "tags", IndexType.HASH)
        hash_index.build(self.test_graph)
        
        # Save indexes
        result = manager.save_indexes()
        self.assertTrue(result)
        
        # Verify files were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_hash.idx")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_hash.idx.meta")))
        
        # Create a new manager and load indexes
        new_manager = IndexManager(index_dir=self.temp_dir)
        result = new_manager.load_indexes()
        self.assertTrue(result)
        
        # Verify indexes were loaded
        self.assertIn("test_hash", new_manager.indexes)
        self.assertIsInstance(new_manager.get_index("test_hash"), HashIndex)
        
        # Test search with loaded indexes
        results = new_manager.search("test_hash", "test")
        self.assertEqual(len(results), 3)  # All nodes have the "test" tag
    
    def test_index_manager_error_handling(self):
        """Test IndexManager error handling."""
        # Create an index manager
        manager = IndexManager()
        
        # Test get_index with non-existent index
        with self.assertRaises(ValueError):
            manager.get_index("non_existent")
        
        # Test drop_index with non-existent index
        with self.assertRaises(ValueError):
            manager.drop_index("non_existent")
        
        # Test create_index with invalid type
        with self.assertRaises(ValueError):
            manager.create_index("test_invalid", "field", "invalid_type")
        
        # Test search with non-existent index
        results = manager.search("non_existent", "query")
        self.assertEqual(results, [])
    
    def test_index_manager_save_load_errors(self):
        """Test IndexManager save/load error handling."""
        # Create an index manager with no index directory
        manager = IndexManager()
        
        # Test save_indexes with no index directory
        result = manager.save_indexes()
        self.assertFalse(result)
        
        # Test load_indexes with no index directory
        result = manager.load_indexes()
        self.assertFalse(result)
        
        # Create an index manager with non-existent directory
        manager = IndexManager(index_dir="/non/existent/dir")
        
        # Test load_indexes with non-existent directory
        result = manager.load_indexes()
        self.assertFalse(result)
        
        # Create an index manager with valid directory but invalid metadata
        manager = IndexManager(index_dir=self.temp_dir)
        
        # Create a dummy index file without metadata
        with open(os.path.join(self.temp_dir, "test_invalid.idx"), "wb") as f:
            pickle.dump({}, f)
        
        # Test load_indexes with invalid metadata
        with patch('os.path.exists', return_value=False):  # Make metadata file not exist
            result = manager.load_indexes()
            self.assertFalse(result)
    
    def test_index_manager_load_with_valid_metadata(self):
        """Test IndexManager load with valid metadata."""
        # Create an index manager with valid directory
        manager = IndexManager(index_dir=self.temp_dir)
        
        # Create a dummy index file with metadata
        index_path = os.path.join(self.temp_dir, "test_hash.idx")
        with open(index_path, "wb") as f:
            pickle.dump({
                "name": "test_hash",
                "field": "tags",
                "is_built": True,
                "index": {}
            }, f)
        
        # Create metadata file
        metadata_path = index_path + ".meta"
        with open(metadata_path, "w") as f:
            json.dump({
                "name": "test_hash",
                "field": "tags",
                "type": "hash"
            }, f)
        
        # Test load_indexes
        result = manager.load_indexes()
        self.assertTrue(result)
        self.assertIn("test_hash", manager.indexes)
        self.assertIsInstance(manager.indexes["test_hash"], HashIndex)
        
        # Test with invalid index type
        with open(os.path.join(self.temp_dir, "test_invalid.idx"), "wb") as f:
            pickle.dump({
                "name": "test_invalid",
                "field": "field",
                "is_built": True,
                "index": {}
            }, f)
        
        # Create metadata file with invalid type
        with open(os.path.join(self.temp_dir, "test_invalid.idx.meta"), "w") as f:
            json.dump({
                "name": "test_invalid",
                "field": "field",
                "type": "invalid"
            }, f)
        
        # Test load_indexes with invalid index type
        with patch.object(IndexManager, 'create_index', side_effect=ValueError("Invalid type")):
            result = manager.load_indexes()
            self.assertFalse(result)
    
    def test_index_manager_create_all_index_types(self):
        """Test IndexManager creating all index types."""
        # Create an index manager
        manager = IndexManager()
        
        # Create all index types
        manager.create_index("test_hash", "tags", IndexType.HASH)
        manager.create_index("test_btree", "title", IndexType.BTREE)
        manager.create_index("test_fulltext", "content", IndexType.FULLTEXT)
        manager.create_index("test_vector", "embedding", IndexType.VECTOR)
        
        # Verify all indexes were created
        self.assertEqual(len(manager.indexes), 4)
        self.assertIsInstance(manager.get_index("test_hash"), HashIndex)
        self.assertIsInstance(manager.get_index("test_btree"), BTreeIndex)
        self.assertIsInstance(manager.get_index("test_fulltext"), FullTextIndex)
        self.assertIsInstance(manager.get_index("test_vector"), VectorIndex)
        
        # Test drop_index
        manager.drop_index("test_hash")
        self.assertEqual(len(manager.indexes), 3)
        with self.assertRaises(ValueError):
            manager.get_index("test_hash")
    

if __name__ == "__main__":
    unittest.main()
