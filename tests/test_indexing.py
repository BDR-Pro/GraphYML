"""
Unit tests for indexing module.
"""
import unittest
import tempfile
import os
import shutil
from src.models.indexing import (
    IndexType, BaseIndex, HashIndex, BTreeIndex, 
    FullTextIndex, VectorIndex, IndexManager
)


class TestBaseIndex(unittest.TestCase):
    """Test cases for BaseIndex class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        # Create a base index
        index = BaseIndex("test", "field")
        
        # Test build method
        with self.assertRaises(NotImplementedError) as context:
            index.build({})
        self.assertEqual(str(context.exception), "Subclasses must implement build()")
        
        # Test update method
        with self.assertRaises(NotImplementedError) as context:
            index.update("key", {})
        self.assertEqual(str(context.exception), "Subclasses must implement update()")
        
        # Test search method
        with self.assertRaises(NotImplementedError) as context:
            index.search("query")
        self.assertEqual(str(context.exception), "Subclasses must implement search()")
        
        # Test save method
        with self.assertRaises(NotImplementedError) as context:
            index.save("path")
        self.assertEqual(str(context.exception), "Subclasses must implement save()")
        
        # Test load method
        with self.assertRaises(NotImplementedError) as context:
            index.load("path")
        self.assertEqual(str(context.exception), "Subclasses must implement load()")


class TestHashIndex(unittest.TestCase):
    """Test cases for HashIndex class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test graph
        self.test_graph = {
            "node1": {
                "id": "node1",
                "title": "Test Node 1",
                "tags": ["tag1", "tag2"],
                "category": "test"
            },
            "node2": {
                "id": "node2",
                "title": "Test Node 2",
                "tags": ["tag2", "tag3"],
                "category": "test"
            },
            "node3": {
                "id": "node3",
                "title": "Test Node 3",
                "tags": ["tag1", "tag3"],
                "category": "other"
            }
        }
        
        # Create a temporary directory for index files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_build_and_search(self):
        """Test building and searching a hash index."""
        # Create a hash index for tags
        index = HashIndex("tags_index", "tags")
        
        # Build the index
        index.build(self.test_graph)
        
        # Search for tag1
        results = index.search("tag1")
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node3", results)
        
        # Search for tag2
        results = index.search("tag2")
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node2", results)
        
        # Search for tag3
        results = index.search("tag3")
        self.assertEqual(len(results), 2)
        self.assertIn("node2", results)
        self.assertIn("node3", results)
        
        # Search for non-existent tag
        results = index.search("tag4")
        self.assertEqual(len(results), 0)
    
    def test_update(self):
        """Test updating a hash index."""
        # Create a hash index for category
        index = HashIndex("category_index", "category")
        
        # Build the index
        index.build(self.test_graph)
        
        # Search for test category
        results = index.search("test")
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node2", results)
        
        # Update node1 to change category
        index.update("node1", {"category": "updated"})
        
        # Search for test category again
        results = index.search("test")
        self.assertEqual(len(results), 1)
        self.assertIn("node2", results)
        
        # Search for updated category
        results = index.search("updated")
        self.assertEqual(len(results), 1)
        self.assertIn("node1", results)
        
        # Delete node2
        index.update("node2", {}, is_delete=True)
        
        # Search for test category again
        results = index.search("test")
        self.assertEqual(len(results), 0)
    
    def test_save_and_load(self):
        """Test saving and loading a hash index."""
        # Create a hash index for tags
        index = HashIndex("tags_index", "tags")
        
        # Build the index
        index.build(self.test_graph)
        
        # Save the index
        index_path = os.path.join(self.temp_dir, "tags_index.idx")
        success = index.save(index_path)
        self.assertTrue(success)
        
        # Create a new index
        new_index = HashIndex("tags_index", "tags")
        
        # Load the index
        success = new_index.load(index_path)
        self.assertTrue(success)
        
        # Search for tag1
        results = new_index.search("tag1")
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node3", results)


class TestBTreeIndex(unittest.TestCase):
    """Test cases for BTreeIndex class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test graph
        self.test_graph = {
            "node1": {
                "id": "node1",
                "title": "Test Node 1",
                "year": 2020,
                "rating": 4.5
            },
            "node2": {
                "id": "node2",
                "title": "Test Node 2",
                "year": 2021,
                "rating": 3.8
            },
            "node3": {
                "id": "node3",
                "title": "Test Node 3",
                "year": 2019,
                "rating": 4.2
            }
        }
        
        # Create a temporary directory for index files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_build_and_search(self):
        """Test building and searching a B-tree index."""
        # Create a B-tree index for year
        index = BTreeIndex("year_index", "year")
        
        # Build the index
        index.build(self.test_graph)
        
        # Search for year range
        results = index.search({"min": 2019, "max": 2020})
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node3", results)
        
        # Search for min year
        results = index.search({"min": 2020})
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node2", results)
        
        # Search for max year
        results = index.search({"max": 2020})
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node3", results)
    
    def test_update(self):
        """Test updating a B-tree index."""
        # Create a B-tree index for rating
        index = BTreeIndex("rating_index", "rating")
        
        # Build the index
        index.build(self.test_graph)
        
        # Search for rating range
        results = index.search({"min": 4.0, "max": 5.0})
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node3", results)
        
        # Update node3 to change rating
        index.update("node3", {"rating": 3.5})
        
        # Search for rating range again
        results = index.search({"min": 4.0, "max": 5.0})
        self.assertEqual(len(results), 1)
        self.assertIn("node1", results)
        
        # Search for lower rating range
        results = index.search({"min": 3.0, "max": 4.0})
        self.assertEqual(len(results), 2)
        self.assertIn("node2", results)
        self.assertIn("node3", results)


class TestFullTextIndex(unittest.TestCase):
    """Test cases for FullTextIndex class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test graph
        self.test_graph = {
            "node1": {
                "id": "node1",
                "title": "Machine Learning Basics",
                "description": "Introduction to machine learning concepts and algorithms."
            },
            "node2": {
                "id": "node2",
                "title": "Deep Learning Fundamentals",
                "description": "Exploring neural networks and deep learning architectures."
            },
            "node3": {
                "id": "node3",
                "title": "Natural Language Processing",
                "description": "Processing and analyzing human language with machine learning."
            }
        }
        
        # Create a temporary directory for index files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_build_and_search(self):
        """Test building and searching a full-text index."""
        # Create a full-text index for title
        index = FullTextIndex("title_index", "title")
        
        # Build the index
        index.build(self.test_graph)
        
        # Search for "learning"
        results = index.search("learning")
        self.assertEqual(len(results), 2)
        self.assertTrue(any(key == "node1" for key, _ in results))
        self.assertTrue(any(key == "node2" for key, _ in results))
        
        # Search for "machine"
        results = index.search("machine")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node1")
        
        # Search for "natural language"
        results = index.search("natural language")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node3")
    
    def test_update(self):
        """Test updating a full-text index."""
        # Create a full-text index for description
        index = FullTextIndex("description_index", "description")
        
        # Build the index
        index.build(self.test_graph)
        
        # Search for "machine"
        results = index.search("machine")
        self.assertEqual(len(results), 2)
        self.assertTrue(any(key == "node1" for key, _ in results))
        self.assertTrue(any(key == "node3" for key, _ in results))
        
        # Update node1 to change description
        index.update("node1", {"description": "Introduction to data science concepts."})
        
        # Search for "machine" again
        results = index.search("machine")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node3")
        
        # Search for "data science"
        results = index.search("data science")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node1")


class TestVectorIndex(unittest.TestCase):
    """Test cases for VectorIndex class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test graph
        self.test_graph = {
            "node1": {
                "id": "node1",
                "title": "Test Node 1",
                "embedding": [0.1, 0.2, 0.3]
            },
            "node2": {
                "id": "node2",
                "title": "Test Node 2",
                "embedding": [0.2, 0.3, 0.4]
            },
            "node3": {
                "id": "node3",
                "title": "Test Node 3",
                "embedding": [0.5, 0.6, 0.7]
            }
        }
        
        # Create a temporary directory for index files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_build_and_search(self):
        """Test building and searching a vector index."""
        # Create a vector index for embedding
        index = VectorIndex("embedding_index", "embedding")
        
        # Build the index
        index.build(self.test_graph)
        
        # Search for similar embeddings
        results = index.search([0.1, 0.2, 0.3], threshold=0.9)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node1")
        
        # Search with lower threshold
        results = index.search([0.1, 0.2, 0.3], threshold=0.8)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "node1")
        self.assertEqual(results[1][0], "node2")
    
    def test_update(self):
        """Test updating a vector index."""
        # Create a vector index for embedding
        index = VectorIndex("embedding_index", "embedding")
        
        # Build the index
        index.build(self.test_graph)
        
        # Search for similar embeddings
        results = index.search([0.1, 0.2, 0.3], threshold=0.9)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node1")
        
        # Update node1 to change embedding
        index.update("node1", {"embedding": [0.5, 0.6, 0.7]})
        
        # Search for original embedding again
        results = index.search([0.1, 0.2, 0.3], threshold=0.9)
        self.assertEqual(len(results), 0)
        
        # Search for new embedding
        results = index.search([0.5, 0.6, 0.7], threshold=0.9)
        self.assertEqual(len(results), 2)
        self.assertTrue(any(key == "node1" for key, _ in results))
        self.assertTrue(any(key == "node3" for key, _ in results))


class TestIndexManager(unittest.TestCase):
    """Test cases for IndexManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test graph
        self.test_graph = {
            "node1": {
                "id": "node1",
                "title": "Machine Learning Basics",
                "tags": ["ml", "basics"],
                "year": 2020,
                "embedding": [0.1, 0.2, 0.3]
            },
            "node2": {
                "id": "node2",
                "title": "Deep Learning Fundamentals",
                "tags": ["dl", "basics"],
                "year": 2021,
                "embedding": [0.2, 0.3, 0.4]
            },
            "node3": {
                "id": "node3",
                "title": "Natural Language Processing",
                "tags": ["nlp", "ml"],
                "year": 2019,
                "embedding": [0.5, 0.6, 0.7]
            }
        }
        
        # Create a temporary directory for index files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_create_and_get_index(self):
        """Test creating and getting indexes."""
        # Create an index manager
        manager = IndexManager(self.test_graph, self.temp_dir)
        
        # Create indexes
        manager.create_index("tags_index", "tags", IndexType.HASH)
        manager.create_index("year_index", "year", IndexType.BTREE)
        manager.create_index("title_index", "title", IndexType.FULLTEXT)
        manager.create_index("embedding_index", "embedding", IndexType.VECTOR)
        
        # Get indexes
        indexes = manager.get_indexes()
        self.assertEqual(len(indexes), 4)
        
        # Get specific indexes
        tags_index = manager.get_index("tags_index")
        self.assertIsInstance(tags_index, HashIndex)
        
        year_index = manager.get_index("year_index")
        self.assertIsInstance(year_index, BTreeIndex)
        
        title_index = manager.get_index("title_index")
        self.assertIsInstance(title_index, FullTextIndex)
        
        embedding_index = manager.get_index("embedding_index")
        self.assertIsInstance(embedding_index, VectorIndex)
    
    def test_drop_index(self):
        """Test dropping an index."""
        # Create an index manager
        manager = IndexManager(self.test_graph, self.temp_dir)
        
        # Create an index
        manager.create_index("tags_index", "tags", IndexType.HASH)
        
        # Verify index exists
        self.assertIsNotNone(manager.get_index("tags_index"))
        
        # Drop the index
        success = manager.drop_index("tags_index")
        self.assertTrue(success)
        
        # Verify index is gone
        self.assertIsNone(manager.get_index("tags_index"))
        
        # Try to drop non-existent index
        success = manager.drop_index("nonexistent_index")
        self.assertFalse(success)
    
    def test_update_indexes(self):
        """Test updating all indexes."""
        # Create an index manager
        manager = IndexManager(self.test_graph, self.temp_dir)
        
        # Create indexes
        manager.create_index("tags_index", "tags", IndexType.HASH)
        manager.create_index("year_index", "year", IndexType.BTREE)
        
        # Update a node
        new_node = {
            "id": "node1",
            "title": "Updated Machine Learning",
            "tags": ["ml", "advanced"],
            "year": 2022
        }
        manager.update_indexes("node1", new_node)
        
        # Search tags index
        tags_index = manager.get_index("tags_index")
        results = tags_index.search("advanced")
        self.assertEqual(len(results), 1)
        self.assertIn("node1", results)
        
        # Search year index
        year_index = manager.get_index("year_index")
        results = year_index.search({"min": 2022})
        self.assertEqual(len(results), 1)
        self.assertIn("node1", results)
    
    def test_rebuild_indexes(self):
        """Test rebuilding all indexes."""
        # Create an index manager
        manager = IndexManager(self.test_graph, self.temp_dir)
        
        # Create indexes
        manager.create_index("tags_index", "tags", IndexType.HASH)
        manager.create_index("year_index", "year", IndexType.BTREE)
        
        # Modify the graph directly
        self.test_graph["node1"]["tags"] = ["ml", "advanced"]
        self.test_graph["node1"]["year"] = 2022
        
        # Rebuild indexes
        manager.rebuild_indexes()
        
        # Search tags index
        tags_index = manager.get_index("tags_index")
        results = tags_index.search("advanced")
        self.assertEqual(len(results), 1)
        self.assertIn("node1", results)
        
        # Search year index
        year_index = manager.get_index("year_index")
        results = year_index.search({"min": 2022})
        self.assertEqual(len(results), 1)
        self.assertIn("node1", results)
    
    def test_search(self):
        """Test searching through the index manager."""
        # Create an index manager
        manager = IndexManager(self.test_graph, self.temp_dir)
        
        # Create indexes
        manager.create_index("tags_index", "tags", IndexType.HASH)
        manager.create_index("title_index", "title", IndexType.FULLTEXT)
        manager.create_index("embedding_index", "embedding", IndexType.VECTOR)
        
        # Search hash index
        results = manager.search("tags_index", "ml")
        self.assertEqual(len(results), 2)
        self.assertIn("node1", results)
        self.assertIn("node3", results)
        
        # Search full-text index
        results = manager.search("title_index", "learning")
        self.assertEqual(len(results), 2)
        self.assertTrue(any(key == "node1" for key, _ in results))
        self.assertTrue(any(key == "node2" for key, _ in results))
        
        # Search vector index
        results = manager.search("embedding_index", [0.1, 0.2, 0.3], threshold=0.9)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "node1")
        
        # Try to search non-existent index
        with self.assertRaises(ValueError):
            manager.search("nonexistent_index", "query")


if __name__ == "__main__":
    unittest.main()

