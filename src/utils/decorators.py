"""
Decorator utilities for GraphYML.
Provides decorators for common patterns like logging, error handling, and timing.
"""
import time
import logging
import functools
import traceback
from typing import Callable, Any, TypeVar, cast, Optional

from src.utils.error_handler import GraphYMLError

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar('T')


def log_function_call(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args} kwargs={kwargs}")
        result = func(*args, **kwargs)
        logger.debug(f"Result of {func.__name__}: {result}")
        return result
    
    return cast(Callable[..., T], wrapper)


def log_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to log errors from a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    return cast(Callable[..., T], wrapper)


def timing(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    
    return cast(Callable[..., T], wrapper)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: tuple = (Exception,)) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    
                    if attempts >= max_attempts:
                        logger.error(f"Failed after {attempts} attempts: {str(e)}")
                        raise
                    
                    logger.warning(f"Attempt {attempts} failed: {str(e)}. Retrying in {current_delay:.2f} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return cast(Callable[..., T], wrapper)
    
    return decorator


def validate_args(validator: Callable[..., bool]) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to validate function arguments.
    
    Args:
        validator: Function to validate arguments
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise ValueError(f"Invalid arguments for {func.__name__}")
            
            return func(*args, **kwargs)
        
        return cast(Callable[..., T], wrapper)
    
    return decorator


def require_auth(permission: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to require authentication.
    
    Args:
        permission: Required permission
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if user is in kwargs
            user = kwargs.get("user")
            
            if not user:
                raise GraphYMLError("Authentication required")
            
            # Check permission if specified
            if permission and not user.has_permission(permission):
                raise GraphYMLError(f"Permission denied: {permission}")
            
            return func(*args, **kwargs)
        
        return cast(Callable[..., T], wrapper)
    
    return decorator
