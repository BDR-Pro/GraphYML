"""
Database manager for GraphYML.
Integrates all database components into a unified interface.
"""
import os
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path

import yaml

from src.config.settings import load_config, ensure_directories
from src.utils.data_handler import (
    load_graph_from_folder, save_node_to_yaml, validate_node_schema
)
from src.models.transaction import TransactionManager, Transaction
from src.models.indexing import IndexManager, IndexType
from src.models.query_engine import QueryEngine
from src.models.auth import AuthManager, User, Permission, Role
from src.models.embeddings import EmbeddingGenerator
from src.models.orm import GraphORM


class Database:
    """
    Main database class that integrates all components.
    Provides a unified interface for database operations.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the database.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        self.config = ensure_directories(self.config)
        
        # Set up paths
        self.base_dir = Path(self.config["save_path"])
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.index_dir = self.base_dir / "indexes"
        self.index_dir.mkdir(exist_ok=True)
        
        self.tx_dir = self.base_dir / "transactions"
        self.tx_dir.mkdir(exist_ok=True)
        
        self.auth_dir = self.base_dir / "auth"
        self.auth_dir.mkdir(exist_ok=True)
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize components
        self.graph = {}
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.auth_manager = AuthManager(str(self.auth_dir))
        
        # Load graph data
        self._load_data()
        
        # Initialize managers after data is loaded
        self.tx_manager = TransactionManager(
            self.graph, 
            str(self.tx_dir),
            self._save_node
        )
        
        self.index_manager = IndexManager(
            self.graph,
            str(self.index_dir)
        )
        
        self.query_engine = QueryEngine(self.graph)
        
        # Initialize ORM interface
        self.orm = GraphORM(self)
        
        # Create default indexes if they don't exist
        self._create_default_indexes()
        
        print(f"Database initialized with {len(self.graph)} nodes")
    
    def _load_data(self):
        """Load all data from disk."""
        with self.lock:
            # Load all folders
            for folder in self.data_dir.iterdir():
                if folder.is_dir():
                    graph_data, errors = load_graph_from_folder(folder)
                    
                    # Add to main graph
                    self.graph.update(graph_data)
                    
                    if errors:
                        print(f"Errors loading {folder.name}: {len(errors)} errors")
    
    def _save_node(self, key: str, node: Dict[str, Any], filename: str) -> bool:
        """
        Save a node to disk.
        
        Args:
            key: Node key
            node: Node data
            filename: Filename to save as
            
        Returns:
            bool: Success flag
        """
        with self.lock:
            # Determine folder based on node type or category
            folder = node.get("category", "default")
            folder_path = self.data_dir / folder
            folder_path.mkdir(exist_ok=True)
            
            # Save node
            success, error = save_node_to_yaml(node, str(folder_path), filename)
            
            if not success:
                print(f"Error saving node {key}: {error}")
            
            return success
    
    def _create_default_indexes(self):
        """Create default indexes if they don't exist."""
        with self.lock:
            # Check if we have any indexes
            if self.index_manager.get_indexes():
                return
            
            # Create default indexes
            try:
                # Hash index for id field
                self.index_manager.create_index(
                    "id_index",
                    "id",
                    IndexType.HASH
                )
                
                # Hash index for tags
                self.index_manager.create_index(
                    "tags_index",
                    "tags",
                    IndexType.HASH
                )
                
                # Full-text index for title
                self.index_manager.create_index(
                    "title_index",
                    "title",
                    IndexType.FULLTEXT
                )
                
                # Vector index for embeddings
                self.index_manager.create_index(
                    "embedding_index",
                    "embedding",
                    IndexType.VECTOR
                )
                
                print("Created default indexes")
            
            except Exception as e:
                print(f"Error creating default indexes: {e}")
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Optional[User]: User object if authentication succeeds, None otherwise
        """
        return self.auth_manager.authenticate(username, password)
    
    def begin_transaction(self, user: User) -> Optional[Transaction]:
        """
        Begin a new transaction.
        
        Args:
            user: User initiating the transaction
            
        Returns:
            Optional[Transaction]: Transaction object or None if not authorized
        """
        if not user.has_permission(Permission.WRITE):
            return None
        
        return self.tx_manager.begin_transaction()
    
    def commit_transaction(self, tx_id: str, user: User) -> bool:
        """
        Commit a transaction.
        
        Args:
            tx_id: Transaction ID
            user: User committing the transaction
            
        Returns:
            bool: Success flag
        """
        if not user.has_permission(Permission.WRITE):
            return False
        
        success = self.tx_manager.commit_transaction(tx_id)
        
        if success:
            # Update indexes
            self.index_manager.rebuild_indexes()
            
            # Update query engine
            self.query_engine = QueryEngine(self.graph)
        
        return success
    
    def abort_transaction(self, tx_id: str, user: User) -> bool:
        """
        Abort a transaction.
        
        Args:
            tx_id: Transaction ID
            user: User aborting the transaction
            
        Returns:
            bool: Success flag
        """
        if not user.has_permission(Permission.WRITE):
            return False
        
        return self.tx_manager.abort_transaction(tx_id)
    
    def create_node(
        self,
        node_data: Dict[str, Any],
        user: User,
        tx: Optional[Transaction] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Create a new node.
        
        Args:
            node_data: Node data
            user: User creating the node
            tx: Optional transaction
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: 
                (success, node_key, error_message)
        """
        if not user.has_permission(Permission.WRITE):
            return False, None, "Permission denied"
        
        # Validate node
        is_valid, errors = validate_node_schema(node_data)
        if not is_valid:
            return False, None, f"Validation failed: {errors}"
        
        # Get node key
        node_key = node_data.get("id") or node_data.get("title")
        if not node_key:
            return False, None, "Node must have id or title"
        
        # Check if node already exists
        if node_key in self.graph and tx is None:
            return False, None, f"Node {node_key} already exists"
        
        # Use transaction if provided
        if tx is not None:
            success = tx.create_node(node_key, node_data)
            return success, node_key if success else None, None
        
        # Create node directly (without transaction)
        with self.lock:
            # Add to graph
            self.graph[node_key] = node_data
            
            # Save to disk
            success = self._save_node(node_key, node_data, f"{node_key}.yaml")
            
            if success:
                # Update indexes
                self.index_manager.update_indexes(node_key, node_data)
            
            return success, node_key if success else None, None
    
    def update_node(
        self,
        node_key: str,
        node_data: Dict[str, Any],
        user: User,
        tx: Optional[Transaction] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Update an existing node.
        
        Args:
            node_key: Key of the node to update
            node_data: New node data
            user: User updating the node
            tx: Optional transaction
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        if not user.has_permission(Permission.WRITE):
            return False, "Permission denied"
        
        # Validate node
        is_valid, errors = validate_node_schema(node_data)
        if not is_valid:
            return False, f"Validation failed: {errors}"
        
        # Check if node exists
        if node_key not in self.graph:
            return False, f"Node {node_key} does not exist"
        
        # Use transaction if provided
        if tx is not None:
            success = tx.update_node(node_key, node_data)
            return success, None if success else "Transaction update failed"
        
        # Update node directly (without transaction)
        with self.lock:
            # Update graph
            self.graph[node_key] = node_data
            
            # Save to disk
            success = self._save_node(node_key, node_data, f"{node_key}.yaml")
            
            if success:
                # Update indexes
                self.index_manager.update_indexes(node_key, node_data)
            
            return success, None if success else "Failed to save node"
    
    def delete_node(
        self,
        node_key: str,
        user: User,
        tx: Optional[Transaction] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Delete a node.
        
        Args:
            node_key: Key of the node to delete
            user: User deleting the node
            tx: Optional transaction
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        if not user.has_permission(Permission.DELETE):
            return False, "Permission denied"
        
        # Check if node exists
        if node_key not in self.graph:
            return False, f"Node {node_key} does not exist"
        
        # Use transaction if provided
        if tx is not None:
            success = tx.delete_node(node_key)
            return success, None if success else "Transaction delete failed"
        
        # Delete node directly (without transaction)
        with self.lock:
            # Get node data for index updates
            node_data = self.graph[node_key]
            
            # Remove from graph
            del self.graph[node_key]
            
            # Determine folder based on node type or category
            folder = node_data.get("category", "default")
            folder_path = self.data_dir / folder
            
            # Remove file
            file_path = folder_path / f"{node_key}.yaml"
            if file_path.exists():
                try:
                    file_path.unlink()
                    
                    # Update indexes
                    self.index_manager.update_indexes(node_key, node_data, is_delete=True)
                    
                    return True, None
                except Exception as e:
                    # Restore node in graph
                    self.graph[node_key] = node_data
                    return False, f"Failed to delete node file: {e}"
            
            return True, None
    
    def get_node(
        self,
        node_key: str,
        user: User
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Get a node by key.
        
        Args:
            node_key: Node key
            user: User requesting the node
            
        Returns:
            Tuple[Optional[Dict[str, Any]], Optional[str]]: 
                (node_data, error_message)
        """
        if not user.has_permission(Permission.READ):
            return None, "Permission denied"
        
        with self.lock:
            if node_key not in self.graph:
                return None, f"Node {node_key} does not exist"
            
            return self.graph[node_key], None
    
    def query(
        self,
        query_str: str,
        user: User
    ) -> Tuple[List[str], Optional[str]]:
        """
        Query the database.
        
        Args:
            query_str: Query string
            user: User executing the query
            
        Returns:
            Tuple[List[str], Optional[str]]: (result_keys, error_message)
        """
        if not user.has_permission(Permission.READ):
            return [], "Permission denied"
        
        try:
            results = self.query_engine.execute_query(query_str)
            return results, None
        except Exception as e:
            return [], f"Query error: {str(e)}"
    
    def search_by_embedding(
        self,
        text_or_embedding: Union[str, List[float]],
        threshold: float = 0.7,
        limit: int = 10,
        user: User
    ) -> Tuple[List[Tuple[str, float]], Optional[str]]:
        """
        Search by text similarity using embeddings.
        
        Args:
            text_or_embedding: Text or embedding vector to search for
            threshold: Similarity threshold
            limit: Maximum number of results
            user: User executing the search
            
        Returns:
            Tuple[List[Tuple[str, float]], Optional[str]]: 
                (result_keys_with_scores, error_message)
        """
        if not user.has_permission(Permission.READ):
            return [], "Permission denied"
        
        # Generate embedding if text is provided
        if isinstance(text_or_embedding, str):
            embedding, error = self.embedding_generator.generate_embedding(text_or_embedding)
            
            if embedding is None:
                return [], f"Failed to generate embedding: {error}"
        else:
            embedding = text_or_embedding
        
        # Get vector index
        vector_index = self.index_manager.get_index("embedding_index")
        
        if vector_index is None:
            # Fallback to query engine
            return self.query_engine.find_by_embedding_similarity(
                embedding, threshold, limit
            ), None
        
        # Use vector index for search
        results = vector_index.search(embedding, limit, threshold)
        return results, None
    
    def create_index(
        self,
        name: str,
        field_path: str,
        index_type: IndexType,
        user: User
    ) -> Tuple[bool, Optional[str]]:
        """
        Create a new index.
        
        Args:
            name: Index name
            field_path: Path to the field to index
            index_type: Type of index
            user: User creating the index
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        if not user.has_permission(Permission.ADMIN):
            return False, "Permission denied"
        
        try:
            self.index_manager.create_index(name, field_path, index_type)
            return True, None
        except Exception as e:
            return False, f"Failed to create index: {str(e)}"
    
    def drop_index(
        self,
        name: str,
        user: User
    ) -> Tuple[bool, Optional[str]]:
        """
        Drop an index.
        
        Args:
            name: Index name
            user: User dropping the index
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        if not user.has_permission(Permission.ADMIN):
            return False, "Permission denied"
        
        success = self.index_manager.drop_index(name)
        
        if not success:
            return False, f"Index {name} does not exist"
        
        return True, None
    
    def get_indexes(
        self,
        user: User
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Get information about all indexes.
        
        Args:
            user: User requesting index information
            
        Returns:
            Tuple[List[Dict[str, Any]], Optional[str]]: 
                (index_info, error_message)
        """
        if not user.has_permission(Permission.READ):
            return [], "Permission denied"
        
        return self.index_manager.get_indexes(), None
    
    def get_statistics(
        self,
        user: User
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Get database statistics.
        
        Args:
            user: User requesting statistics
            
        Returns:
            Tuple[Dict[str, Any], Optional[str]]: (statistics, error_message)
        """
        if not user.has_permission(Permission.READ):
            return {}, "Permission denied"
        
        with self.lock:
            # Count nodes by category
            categories = {}
            for node in self.graph.values():
                category = node.get("category", "default")
                categories[category] = categories.get(category, 0) + 1
            
            # Count nodes with embeddings
            with_embeddings = sum(
                1 for node in self.graph.values()
                if "embedding" in node
            )
            
            # Count links
            total_links = sum(
                len(node.get("links", []))
                for node in self.graph.values()
            )
            
            # Get active transactions
            active_tx_count = self.tx_manager.get_active_transaction_count()
            
            # Get index count
            index_count = len(self.index_manager.get_indexes())
            
            return {
                "node_count": len(self.graph),
                "categories": categories,
                "with_embeddings": with_embeddings,
                "without_embeddings": len(self.graph) - with_embeddings,
                "total_links": total_links,
                "active_transactions": active_tx_count,
                "index_count": index_count
            }, None
    
    def backup(
        self,
        backup_dir: str,
        user: User
    ) -> Tuple[bool, Optional[str]]:
        """
        Create a backup of the database.
        
        Args:
            backup_dir: Directory to store backup
            user: User initiating the backup
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        if not user.has_permission(Permission.ADMIN):
            return False, "Permission denied"
        
        try:
            import shutil
            from datetime import datetime
            
            # Create backup directory
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped backup directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = backup_path / f"backup_{timestamp}"
            backup_dir.mkdir()
            
            # Copy data directory
            shutil.copytree(
                str(self.data_dir),
                str(backup_dir / "data")
            )
            
            # Copy index directory
            shutil.copytree(
                str(self.index_dir),
                str(backup_dir / "indexes")
            )
            
            # Copy transaction directory
            shutil.copytree(
                str(self.tx_dir),
                str(backup_dir / "transactions")
            )
            
            # Copy auth directory
            shutil.copytree(
                str(self.auth_dir),
                str(backup_dir / "auth")
            )
            
            return True, None
        
        except Exception as e:
            return False, f"Backup failed: {str(e)}"
    
    def restore(
        self,
        backup_dir: str,
        user: User
    ) -> Tuple[bool, Optional[str]]:
        """
        Restore the database from a backup.
        
        Args:
            backup_dir: Directory containing the backup
            user: User initiating the restore
            
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        if not user.has_permission(Permission.ADMIN):
            return False, "Permission denied"
        
        try:
            import shutil
            
            backup_path = Path(backup_dir)
            
            # Check if backup directory exists
            if not backup_path.exists() or not backup_path.is_dir():
                return False, f"Backup directory {backup_dir} does not exist"
            
            # Check if required directories exist
            if not (backup_path / "data").exists():
                return False, "Backup directory does not contain data directory"
            
            # Stop all active transactions
            with self.lock:
                # Abort all active transactions
                for tx_id in self.tx_manager.active_transactions.keys():
                    self.tx_manager.abort_transaction(tx_id)
                
                # Clear graph
                self.graph.clear()
                
                # Remove existing directories
                shutil.rmtree(str(self.data_dir))
                shutil.rmtree(str(self.index_dir))
                shutil.rmtree(str(self.tx_dir))
                shutil.rmtree(str(self.auth_dir))
                
                # Restore from backup
                shutil.copytree(
                    str(backup_path / "data"),
                    str(self.data_dir)
                )
                
                shutil.copytree(
                    str(backup_path / "indexes"),
                    str(self.index_dir)
                )
                
                shutil.copytree(
                    str(backup_path / "transactions"),
                    str(self.tx_dir)
                )
                
                shutil.copytree(
                    str(backup_path / "auth"),
                    str(self.auth_dir)
                )
                
                # Reload data
                self._load_data()
                
                # Reinitialize components
                self.tx_manager = TransactionManager(
                    self.graph, 
                    str(self.tx_dir),
                    self._save_node
                )
                
                self.index_manager = IndexManager(
                    self.graph,
                    str(self.index_dir)
                )
                
                self.query_engine = QueryEngine(self.graph)
                
                self.auth_manager = AuthManager(str(self.auth_dir))
                
                # Reinitialize ORM interface
                self.orm = GraphORM(self)
                
                return True, None
        
        except Exception as e:
            return False, f"Restore failed: {str(e)}"
