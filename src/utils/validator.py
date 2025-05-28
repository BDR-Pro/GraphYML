"""
Data validation utilities for GraphYML.
Provides functions for validating data structures and input.
"""
import logging
import cerberus
from typing import Dict, Any, Tuple, List, Optional, Union

from src.utils.error_handler import ValidationError

# Set up logging
logger = logging.getLogger(__name__)

# Default schema for graph nodes
DEFAULT_NODE_SCHEMA = {
    "id": {"type": "string", "required": True},
    "title": {"type": "string", "required": True},
    "tags": {"type": "list", "schema": {"type": "string"}},
    "genres": {"type": "list", "schema": {"type": "string"}},
    "links": {"type": "list", "schema": {"type": "string"}},
    "embedding": {"type": "list", "schema": {"type": "float"}},
    "content": {"type": "string"},
    "description": {"type": "string"},
    "overview": {"type": "string"},
    "metadata": {"type": "dict"}
}


class GraphValidator:
    """
    Validator for graph data structures.
    """
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the validator.
        
        Args:
            schema: Schema for validation (defaults to DEFAULT_NODE_SCHEMA)
        """
        self.schema = schema or DEFAULT_NODE_SCHEMA
        self.validator = cerberus.Validator(self.schema)
    
    def validate_node(self, node: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a node against the schema.
        
        Args:
            node: Node to validate
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_valid, errors)
        """
        is_valid = self.validator.validate(node)
        return is_valid, self.validator.errors
    
    def validate_graph(self, graph: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Validate all nodes in a graph.
        
        Args:
            graph: Graph to validate
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: List of (node_id, errors) for invalid nodes
        """
        errors = []
        
        for node_id, node in graph.items():
            is_valid, node_errors = self.validate_node(node)
            
            if not is_valid:
                errors.append((node_id, node_errors))
        
        return errors
    
    def validate_and_raise(self, node: Dict[str, Any], node_id: Optional[str] = None) -> None:
        """
        Validate a node and raise an exception if invalid.
        
        Args:
            node: Node to validate
            node_id: Optional node ID for error message
            
        Raises:
            ValidationError: If node is invalid
        """
        is_valid, errors = self.validate_node(node)
        
        if not is_valid:
            error_message = f"Invalid node"
            
            if node_id:
                error_message += f" (ID: {node_id})"
            
            raise ValidationError(error_message, {"errors": errors})


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate configuration has required keys.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, missing_keys)
    """
    missing_keys = [key for key in required_keys if key not in config]
    return len(missing_keys) == 0, missing_keys


def validate_query_params(params: Dict[str, Any], allowed_params: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate query parameters.
    
    Args:
        params: Parameters to validate
        allowed_params: List of allowed parameter names
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, invalid_params)
    """
    invalid_params = [param for param in params if param not in allowed_params]
    return len(invalid_params) == 0, invalid_params

