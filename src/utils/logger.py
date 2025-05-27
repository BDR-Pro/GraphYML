"""
Logger module for GraphYML.
Provides functions for setting up and using logging.
"""
import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Default log directory
DEFAULT_LOG_DIR = "logs"


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    log_level: int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    console_output: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Log file path (optional)
        log_level: Log level
        log_format: Log format
        console_output: Whether to output to console
        max_bytes: Maximum log file size
        backup_count: Number of backup files
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add file handler if log_file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if console_output is True
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(
    name: str,
    config: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Get a logger with configuration.
    
    Args:
        name: Logger name
        config: Logger configuration
        
    Returns:
        logging.Logger: Configured logger
    """
    # Default configuration
    config = config or {}
    
    # Get log level
    log_level_str = config.get("log_level", "INFO")
    log_level = getattr(logging, log_level_str, DEFAULT_LOG_LEVEL)
    
    # Get log directory
    log_dir = config.get("log_dir", DEFAULT_LOG_DIR)
    
    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Set up logger
    return setup_logger(
        name,
        log_file=log_file,
        log_level=log_level,
        log_format=config.get("log_format", DEFAULT_LOG_FORMAT),
        console_output=config.get("console_output", True),
        max_bytes=config.get("max_bytes", 10485760),
        backup_count=config.get("backup_count", 5)
    )


def log_function_call(logger: logging.Logger, func_name: str, args: tuple, kwargs: dict):
    """
    Log a function call.
    
    Args:
        logger: Logger to use
        func_name: Function name
        args: Function arguments
        kwargs: Function keyword arguments
    """
    logger.debug(f"Calling {func_name} with args={args} kwargs={kwargs}")


def log_function_result(logger: logging.Logger, func_name: str, result: Any):
    """
    Log a function result.
    
    Args:
        logger: Logger to use
        func_name: Function name
        result: Function result
    """
    logger.debug(f"Result of {func_name}: {result}")


def log_error(logger: logging.Logger, func_name: str, error: Exception):
    """
    Log an error.
    
    Args:
        logger: Logger to use
        func_name: Function name
        error: Error
    """
    logger.error(f"Error in {func_name}: {str(error)}", exc_info=True)

