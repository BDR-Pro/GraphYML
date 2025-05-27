"""
Decorators module for GraphYML.
Provides decorators for common functionality.
"""
import time
import functools
import logging
from typing import Callable, Any, Optional

# Get logger
logger = logging.getLogger(__name__)


def log_function(level: int = logging.DEBUG) -> Callable:
    """
    Decorator to log function calls and results.
    
    Args:
        level: Log level
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function name
            func_name = func.__name__
            
            # Log function call
            logger.log(level, f"Calling {func_name} with args={args} kwargs={kwargs}")
            
            # Call function
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # Log function result
                logger.log(level, f"Result of {func_name}: {result}")
                logger.log(level, f"Execution time of {func_name}: {end_time - start_time:.4f} seconds")
                
                return result
            except Exception as e:
                # Log error
                logger.error(f"Error in {func_name}: {str(e)}", exc_info=True)
                raise
        
        return wrapper
    
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)) -> Callable:
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
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function name
            func_name = func.__name__
            
            # Try function
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # Log error
                    logger.warning(f"Attempt {attempt}/{max_attempts} for {func_name} failed: {str(e)}")
                    
                    # Check if this is the last attempt
                    if attempt == max_attempts:
                        logger.error(f"All {max_attempts} attempts for {func_name} failed")
                        raise
                    
                    # Wait before retrying
                    logger.info(f"Retrying {func_name} in {current_delay:.2f} seconds")
                    time.sleep(current_delay)
                    
                    # Increase delay
                    current_delay *= backoff
                    
                    # Increment attempt counter
                    attempt += 1
        
        return wrapper
    
    return decorator


def cache_result(ttl: Optional[float] = None) -> Callable:
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Create cache
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(kwargs)
            
            # Check if result is in cache
            if key in cache:
                result, timestamp = cache[key]
                
                # Check if result has expired
                if ttl is None or time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache[key] = (result, time.time())
            
            return result
        
        # Add clear_cache method
        def clear_cache():
            cache.clear()
            logger.debug(f"Cache cleared for {func.__name__}")
        
        wrapper.clear_cache = clear_cache
        
        return wrapper
    
    return decorator


def validate_args(**validators) -> Callable:
    """
    Decorator to validate function arguments.
    
    Args:
        **validators: Validator functions for arguments
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            func_name = func.__name__
            
            # Combine args and kwargs
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            all_args = dict(zip(arg_names, args))
            all_args.update(kwargs)
            
            # Validate arguments
            for arg_name, validator in validators.items():
                if arg_name in all_args:
                    arg_value = all_args[arg_name]
                    
                    # Run validator
                    if not validator(arg_value):
                        error_msg = f"Invalid value for {arg_name}: {arg_value}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
            
            # Call function
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

