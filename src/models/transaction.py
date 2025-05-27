"""
Transaction support for GraphYML.
Provides basic ACID properties for database operations.
"""
import os
import json
import time
import uuid
import shutil
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from pathlib import Path

import yaml


class TransactionStatus(Enum):
    """Possible states of a transaction."""
    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTED = "aborted"


class TransactionOperation(Enum):
    """Types of operations in a transaction."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class TransactionLog:
    """Log of transaction operations for recovery."""
    
    def __init__(self, log_dir: str):
        """
        Initialize the transaction log.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory if it doesn't exist
        self.logs_path = self.log_dir / "logs"
        self.logs_path.mkdir(exist_ok=True)
        
        # Create active transactions directory
        self.active_path = self.log_dir / "active"
        self.active_path.mkdir(exist_ok=True)
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def log_operation(
        self, 
        tx_id: str, 
        operation: TransactionOperation, 
        node_key: str, 
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None
    ):
        """
        Log a transaction operation.
        
        Args:
            tx_id: Transaction ID
            operation: Operation type
            node_key: Key of the affected node
            old_value: Previous node value (for UPDATE/DELETE)
            new_value: New node value (for CREATE/UPDATE)
        """
        with self.lock:
            tx_log_path = self.active_path / f"{tx_id}.json"
            
            # Create or load existing log
            if tx_log_path.exists():
                with open(tx_log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            else:
                log_data = {
                    "tx_id": tx_id,
                    "status": TransactionStatus.ACTIVE.value,
                    "start_time": time.time(),
                    "operations": []
                }
            
            # Add operation to log
            op_data = {
                "type": operation.value,
                "node_key": node_key,
                "timestamp": time.time()
            }
            
            if old_value is not None:
                op_data["old_value"] = old_value
            
            if new_value is not None:
                op_data["new_value"] = new_value
            
            log_data["operations"].append(op_data)
            
            # Write updated log
            with open(tx_log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)
    
    def commit_transaction(self, tx_id: str):
        """
        Mark a transaction as committed.
        
        Args:
            tx_id: Transaction ID
        """
        with self.lock:
            tx_log_path = self.active_path / f"{tx_id}.json"
            
            if not tx_log_path.exists():
                raise ValueError(f"Transaction {tx_id} not found")
            
            # Load log
            with open(tx_log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            
            # Update status
            log_data["status"] = TransactionStatus.COMMITTED.value
            log_data["commit_time"] = time.time()
            
            # Write updated log
            with open(tx_log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)
            
            # Move to committed logs
            shutil.move(
                str(tx_log_path),
                str(self.logs_path / f"{tx_id}.json")
            )
    
    def abort_transaction(self, tx_id: str):
        """
        Mark a transaction as aborted.
        
        Args:
            tx_id: Transaction ID
        """
        with self.lock:
            tx_log_path = self.active_path / f"{tx_id}.json"
            
            if not tx_log_path.exists():
                raise ValueError(f"Transaction {tx_id} not found")
            
            # Load log
            with open(tx_log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            
            # Update status
            log_data["status"] = TransactionStatus.ABORTED.value
            log_data["abort_time"] = time.time()
            
            # Write updated log
            with open(tx_log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)
            
            # Move to committed logs
            shutil.move(
                str(tx_log_path),
                str(self.logs_path / f"{tx_id}.json")
            )
    
    def get_active_transactions(self) -> List[str]:
        """
        Get list of active transaction IDs.
        
        Returns:
            List[str]: List of active transaction IDs
        """
        with self.lock:
            return [
                f.stem for f in self.active_path.glob("*.json")
            ]
    
    def get_transaction_log(self, tx_id: str) -> Dict[str, Any]:
        """
        Get log data for a transaction.
        
        Args:
            tx_id: Transaction ID
            
        Returns:
            Dict[str, Any]: Transaction log data
        """
        with self.lock:
            # Check active transactions first
            tx_log_path = self.active_path / f"{tx_id}.json"
            
            if not tx_log_path.exists():
                # Check committed/aborted transactions
                tx_log_path = self.logs_path / f"{tx_id}.json"
            
            if not tx_log_path.exists():
                raise ValueError(f"Transaction {tx_id} not found")
            
            with open(tx_log_path, "r", encoding="utf-8") as f:
                return json.load(f)


class Transaction:
    """Represents a database transaction with ACID properties."""
    
    def __init__(
        self, 
        tx_log: TransactionLog,
        graph: Dict[str, Dict[str, Any]],
        save_func: Callable[[str, Dict[str, Any], str], bool]
    ):
        """
        Initialize a transaction.
        
        Args:
            tx_log: Transaction log
            graph: Graph to operate on
            save_func: Function to save a node to disk
        """
        self.tx_id = str(uuid.uuid4())
        self.tx_log = tx_log
        self.graph = graph
        self.save_func = save_func
        self.status = TransactionStatus.ACTIVE
        self.modified_nodes = set()
    
    def create_node(self, node_key: str, node_data: Dict[str, Any]) -> bool:
        """
        Create a new node in the graph.
        
        Args:
            node_key: Key for the new node
            node_data: Node data
            
        Returns:
            bool: Success flag
        """
        if self.status != TransactionStatus.ACTIVE:
            raise ValueError(f"Transaction is not active: {self.status}")
        
        if node_key in self.graph:
            return False
        
        # Log operation
        self.tx_log.log_operation(
            self.tx_id,
            TransactionOperation.CREATE,
            node_key,
            new_value=node_data
        )
        
        # Update in-memory graph
        self.graph[node_key] = node_data
        self.modified_nodes.add(node_key)
        
        return True
    
    def update_node(self, node_key: str, node_data: Dict[str, Any]) -> bool:
        """
        Update an existing node in the graph.
        
        Args:
            node_key: Key of the node to update
            node_data: New node data
            
        Returns:
            bool: Success flag
        """
        if self.status != TransactionStatus.ACTIVE:
            raise ValueError(f"Transaction is not active: {self.status}")
        
        if node_key not in self.graph:
            return False
        
        # Log operation
        self.tx_log.log_operation(
            self.tx_id,
            TransactionOperation.UPDATE,
            node_key,
            old_value=self.graph[node_key],
            new_value=node_data
        )
        
        # Update in-memory graph
        self.graph[node_key] = node_data
        self.modified_nodes.add(node_key)
        
        return True
    
    def delete_node(self, node_key: str) -> bool:
        """
        Delete a node from the graph.
        
        Args:
            node_key: Key of the node to delete
            
        Returns:
            bool: Success flag
        """
        if self.status != TransactionStatus.ACTIVE:
            raise ValueError(f"Transaction is not active: {self.status}")
        
        if node_key not in self.graph:
            return False
        
        # Log operation
        self.tx_log.log_operation(
            self.tx_id,
            TransactionOperation.DELETE,
            node_key,
            old_value=self.graph[node_key]
        )
        
        # Update in-memory graph
        del self.graph[node_key]
        self.modified_nodes.add(node_key)
        
        return True
    
    def commit(self) -> bool:
        """
        Commit the transaction.
        
        Returns:
            bool: Success flag
        """
        if self.status != TransactionStatus.ACTIVE:
            raise ValueError(f"Transaction is not active: {self.status}")
        
        try:
            # Save modified nodes to disk
            for node_key in self.modified_nodes:
                if node_key in self.graph:
                    # Node was created or updated
                    self.save_func(node_key, self.graph[node_key], f"{node_key}.yaml")
                else:
                    # Node was deleted - remove file
                    # This would be handled by the save_func
                    pass
            
            # Mark transaction as committed
            self.tx_log.commit_transaction(self.tx_id)
            self.status = TransactionStatus.COMMITTED
            
            return True
        except Exception as e:
            print(f"Error committing transaction: {e}")
            self.abort()
            return False
    
    def abort(self):
        """Abort the transaction and roll back changes."""
        if self.status != TransactionStatus.ACTIVE:
            return
        
        try:
            # Get transaction log
            tx_data = self.tx_log.get_transaction_log(self.tx_id)
            
            # Rollback operations in reverse order
            for op in reversed(tx_data["operations"]):
                node_key = op["node_key"]
                op_type = op["type"]
                
                if op_type == TransactionOperation.CREATE.value:
                    # Rollback creation by removing the node
                    if node_key in self.graph:
                        del self.graph[node_key]
                
                elif op_type == TransactionOperation.UPDATE.value:
                    # Rollback update by restoring old value
                    if "old_value" in op and node_key in self.graph:
                        self.graph[node_key] = op["old_value"]
                
                elif op_type == TransactionOperation.DELETE.value:
                    # Rollback deletion by restoring the node
                    if "old_value" in op and node_key not in self.graph:
                        self.graph[node_key] = op["old_value"]
            
            # Mark transaction as aborted
            self.tx_log.abort_transaction(self.tx_id)
            self.status = TransactionStatus.ABORTED
        
        except Exception as e:
            print(f"Error aborting transaction: {e}")
            # Force status change even if there was an error
            self.status = TransactionStatus.ABORTED


class TransactionManager:
    """Manages database transactions."""
    
    def __init__(
        self, 
        graph: Dict[str, Dict[str, Any]],
        save_dir: str,
        save_func: Callable[[str, Dict[str, Any], str], bool]
    ):
        """
        Initialize the transaction manager.
        
        Args:
            graph: Graph to operate on
            save_dir: Directory to save data
            save_func: Function to save a node to disk
        """
        self.graph = graph
        self.save_dir = Path(save_dir)
        self.save_func = save_func
        
        # Create transaction log directory
        tx_log_dir = self.save_dir / "tx_logs"
        self.tx_log = TransactionLog(str(tx_log_dir))
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Active transactions
        self.active_transactions = {}
        
        # Recover from any crashed transactions
        self._recover()
    
    def begin_transaction(self) -> Transaction:
        """
        Begin a new transaction.
        
        Returns:
            Transaction: New transaction object
        """
        with self.lock:
            tx = Transaction(self.tx_log, self.graph, self.save_func)
            self.active_transactions[tx.tx_id] = tx
            return tx
    
    def commit_transaction(self, tx_id: str) -> bool:
        """
        Commit a transaction.
        
        Args:
            tx_id: Transaction ID
            
        Returns:
            bool: Success flag
        """
        with self.lock:
            if tx_id not in self.active_transactions:
                return False
            
            tx = self.active_transactions[tx_id]
            success = tx.commit()
            
            if success:
                del self.active_transactions[tx_id]
            
            return success
    
    def abort_transaction(self, tx_id: str) -> bool:
        """
        Abort a transaction.
        
        Args:
            tx_id: Transaction ID
            
        Returns:
            bool: Success flag
        """
        with self.lock:
            if tx_id not in self.active_transactions:
                return False
            
            tx = self.active_transactions[tx_id]
            tx.abort()
            
            del self.active_transactions[tx_id]
            return True
    
    def get_transaction(self, tx_id: str) -> Optional[Transaction]:
        """
        Get a transaction by ID.
        
        Args:
            tx_id: Transaction ID
            
        Returns:
            Optional[Transaction]: Transaction object or None
        """
        with self.lock:
            return self.active_transactions.get(tx_id)
    
    def _recover(self):
        """Recover from crashed transactions."""
        with self.lock:
            active_tx_ids = self.tx_log.get_active_transactions()
            
            for tx_id in active_tx_ids:
                try:
                    tx_data = self.tx_log.get_transaction_log(tx_id)
                    
                    # Create a transaction object
                    tx = Transaction(self.tx_log, self.graph, self.save_func)
                    tx.tx_id = tx_id
                    
                    # Abort the transaction
                    tx.abort()
                    
                    print(f"Recovered from crashed transaction: {tx_id}")
                except Exception as e:
                    print(f"Error recovering transaction {tx_id}: {e}")
    
    def get_active_transaction_count(self) -> int:
        """
        Get the number of active transactions.
        
        Returns:
            int: Number of active transactions
        """
        with self.lock:
            return len(self.active_transactions)

