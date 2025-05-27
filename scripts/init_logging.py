#!/usr/bin/env python
"""
Script to initialize the logging system.
"""
import os
import sys
import json
import logging
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import logger module
from src.utils.logger import setup_logger, DEFAULT_LOG_DIR


def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return {}


def init_logging(config: Dict[str, Any]) -> None:
    """
    Initialize logging system.
    
    Args:
        config: Configuration dictionary
    """
    # Get logging configuration
    logging_config = config.get('logging', {})
    
    # Get log level
    log_level_str = logging_config.get('log_level', 'INFO')
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Get log directory
    log_dir = logging_config.get('log_dir', DEFAULT_LOG_DIR)
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up root logger
    root_logger = setup_logger(
        'root',
        log_file=os.path.join(log_dir, 'graphyml.log'),
        log_level=log_level,
        log_format=logging_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        console_output=logging_config.get('console_output', True),
        max_bytes=logging_config.get('max_bytes', 10485760),
        backup_count=logging_config.get('backup_count', 5)
    )
    
    # Set up module loggers
    modules = [
        'src.models.embeddings',
        'src.models.graph_ops',
        'src.models.indexing',
        'src.models.query_engine',
        'src.utils.data_handler',
        'src.utils.decorators'
    ]
    
    for module in modules:
        setup_logger(
            module,
            log_file=os.path.join(log_dir, f"{module.split('.')[-1]}.log"),
            log_level=log_level,
            log_format=logging_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            console_output=False,
            max_bytes=logging_config.get('max_bytes', 10485760),
            backup_count=logging_config.get('backup_count', 5)
        )
    
    # Log initialization
    root_logger.info("Logging system initialized")
    root_logger.info(f"Log level: {log_level_str}")
    root_logger.info(f"Log directory: {log_dir}")


def main():
    """
    Main function.
    """
    # Load configuration
    config = load_config()
    
    # Initialize logging
    init_logging(config)
    
    print("Logging system initialized")


if __name__ == '__main__':
    main()

