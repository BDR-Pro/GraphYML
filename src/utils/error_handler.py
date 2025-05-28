"""
Error handling utilities for GraphYML.
Provides consistent error handling and logging across the application.
"""
import logging
import traceback
from typing import Tuple, Any, Optional, Dict, List, Callable, TypeVar

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar('T')

class GraphYMLError(Exception):
    """Base exception class for GraphYML errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ConfigError(GraphYMLError):
    """Exception raised for configuration errors."""
    pass


class DatabaseError(GraphYMLError):
    """Exception raised for database errors."""
    pass


class AuthError(GraphYMLError):
    """Exception raised for authentication errors."""
    pass


class EmbeddingError(GraphYMLError):
    """Exception raised for embedding errors."""
    pass


class ValidationError(GraphYMLError):
    """Exception raised for validation errors."""
    pass


def safe_execute(func: Callable[..., T], *args, **kwargs) -> Tuple[Optional[T], Optional[str]]:
    """
    Execute a function safely and return result or error message.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple[Optional[T], Optional[str]]: Result and error message (if any)
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error executing {func.__name__}: {error_message}")
        logger.debug(traceback.format_exc())
        return None, error_message


def log_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to log errors from a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    return wrapper


def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    Format an exception as a response dictionary.
    
    Args:
        error: Exception to format
        
    Returns:
        Dict[str, Any]: Formatted error response
    """
    if isinstance(error, GraphYMLError):
        response = {
            "error": error.message,
            "error_type": error.__class__.__name__,
        }
        
        if error.details:
            response["details"] = error.details
        
        return response
    else:
        return {
            "error": str(error),
            "error_type": error.__class__.__name__
        }


def collect_errors(results: List[Tuple[Any, Optional[str]]]) -> List[str]:
    """
    Collect error messages from a list of results.
    
    Args:
        results: List of (result, error) tuples
        
    Returns:
        List[str]: List of error messages
    """
    return [error for _, error in results if error is not None]

