"""
Logging configuration for GraphYML.
Sets up logging with appropriate handlers and formatters.
"""
import os
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Default log directory
DEFAULT_LOG_DIR = "logs"


def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Set up logging with configuration.
    
    Args:
        config: Logging configuration
            - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            - log_dir: Directory for log files
            - log_format: Format string for log messages
            - console_output: Whether to output logs to console
            - max_bytes: Maximum size of log files before rotation
            - backup_count: Number of backup log files to keep
    
    Returns:
        logging.Logger: Configured logger
    """
    # Default configuration
    if config is None:
        config = {}
    
    log_level = config.get("log_level", "INFO")
    log_dir = config.get("log_dir", "logs")
    log_format = config.get("log_format", DEFAULT_LOG_FORMAT)
    console_output = config.get("console_output", True)
    max_bytes = config.get("max_bytes", 10 * 1024 * 1024)  # 10 MB
    backup_count = config.get("backup_count", 5)
    
    # Create logger
    logger = logging.getLogger("graphyml")
    
    # Set level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler if enabled
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    try:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "graphyml.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError) as e:
        logger.warning(f"Could not set up file logging: {e}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(f"graphyml.{name}")
