#!/usr/bin/env python
"""
Script to view and analyze logs.
"""
import os
import sys
import argparse
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import logger module
from src.utils.logger import DEFAULT_LOG_DIR


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a log line.
    
    Args:
        line: Log line to parse
        
    Returns:
        Optional[Dict[str, Any]]: Parsed log line or None if line is invalid
    """
    # Define regex pattern for log line
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)'
    
    # Match pattern
    match = re.match(pattern, line)
    
    if not match:
        return None
    
    # Extract groups
    timestamp_str, logger_name, level, message = match.groups()
    
    # Parse timestamp
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
    except ValueError:
        timestamp = None
    
    # Create log entry
    return {
        'timestamp': timestamp,
        'logger_name': logger_name,
        'level': level,
        'message': message
    }


def filter_logs(
    logs: List[Dict[str, Any]],
    level: Optional[str] = None,
    logger_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    message_pattern: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter logs.
    
    Args:
        logs: Logs to filter
        level: Log level to filter by
        logger_name: Logger name to filter by
        start_time: Start time to filter by
        end_time: End time to filter by
        message_pattern: Message pattern to filter by
        
    Returns:
        List[Dict[str, Any]]: Filtered logs
    """
    filtered_logs = []
    
    for log in logs:
        # Filter by level
        if level and log['level'] != level:
            continue
        
        # Filter by logger name
        if logger_name and log['logger_name'] != logger_name:
            continue
        
        # Filter by start time
        if start_time and log['timestamp'] and log['timestamp'] < start_time:
            continue
        
        # Filter by end time
        if end_time and log['timestamp'] and log['timestamp'] > end_time:
            continue
        
        # Filter by message pattern
        if message_pattern and not re.search(message_pattern, log['message']):
            continue
        
        filtered_logs.append(log)
    
    return filtered_logs


def format_log(log: Dict[str, Any], format_str: str) -> str:
    """
    Format a log entry.
    
    Args:
        log: Log entry to format
        format_str: Format string
        
    Returns:
        str: Formatted log entry
    """
    # Define format placeholders
    placeholders = {
        '%t': log['timestamp'].strftime('%Y-%m-%d %H:%M:%S,%f')[:-3] if log['timestamp'] else 'N/A',
        '%n': log['logger_name'],
        '%l': log['level'],
        '%m': log['message']
    }
    
    # Replace placeholders
    formatted = format_str
    
    for placeholder, value in placeholders.items():
        formatted = formatted.replace(placeholder, value)
    
    return formatted


def get_log_files(log_dir: str, pattern: Optional[str] = None) -> List[str]:
    """
    Get log files in a directory.
    
    Args:
        log_dir: Log directory
        pattern: File pattern to match
        
    Returns:
        List[str]: List of log file paths
    """
    log_files = []
    
    # Check if directory exists
    if not os.path.exists(log_dir):
        return log_files
    
    # Get all files in directory
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        
        # Check if file is a log file
        if os.path.isfile(file_path) and filename.endswith('.log'):
            # Check if file matches pattern
            if pattern and not re.search(pattern, filename):
                continue
            
            log_files.append(file_path)
    
    return log_files


def parse_log_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a log file.
    
    Args:
        file_path: Log file path
        
    Returns:
        List[Dict[str, Any]]: Parsed log entries
    """
    logs = []
    
    # Open log file
    with open(file_path, 'r') as f:
        # Parse each line
        for line in f:
            log = parse_log_line(line.strip())
            
            if log:
                logs.append(log)
    
    return logs


def main():
    """
    Main function.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='View and analyze logs')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR, help='Log directory')
    parser.add_argument('--file-pattern', type=str, help='Log file pattern')
    parser.add_argument('--level', type=str, help='Log level to filter by')
    parser.add_argument('--logger', type=str, help='Logger name to filter by')
    parser.add_argument('--start-time', type=str, help='Start time to filter by (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end-time', type=str, help='End time to filter by (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--message', type=str, help='Message pattern to filter by')
    parser.add_argument('--format', type=str, default='%t - %n - %l - %m', help='Output format')
    parser.add_argument('--tail', type=int, help='Show only the last N lines')
    parser.add_argument('--follow', action='store_true', help='Follow log file (like tail -f)')
    args = parser.parse_args()
    
    # Parse start and end times
    start_time = None
    end_time = None
    
    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"Invalid start time format: {args.start_time}")
            return
    
    if args.end_time:
        try:
            end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"Invalid end time format: {args.end_time}")
            return
    
    # Get log files
    log_files = get_log_files(args.log_dir, args.file_pattern)
    
    if not log_files:
        print(f"No log files found in {args.log_dir}")
        return
    
    # Parse log files
    all_logs = []
    
    for file_path in log_files:
        logs = parse_log_file(file_path)
        all_logs.extend(logs)
    
    # Sort logs by timestamp
    all_logs.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min)
    
    # Filter logs
    filtered_logs = filter_logs(
        all_logs,
        level=args.level,
        logger_name=args.logger,
        start_time=start_time,
        end_time=end_time,
        message_pattern=args.message
    )
    
    # Apply tail
    if args.tail and args.tail > 0:
        filtered_logs = filtered_logs[-args.tail:]
    
    # Print logs
    for log in filtered_logs:
        print(format_log(log, args.format))
    
    # Follow log file
    if args.follow:
        import time
        
        # Get the most recent log file
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if not log_files:
            print("No log files to follow")
            return
        
        latest_log_file = log_files[0]
        
        # Get the current file size
        file_size = os.path.getsize(latest_log_file)
        
        print(f"Following {latest_log_file}...")
        
        try:
            while True:
                # Check if file size has changed
                current_size = os.path.getsize(latest_log_file)
                
                if current_size > file_size:
                    # Open file and seek to the previous position
                    with open(latest_log_file, 'r') as f:
                        f.seek(file_size)
                        
                        # Read new lines
                        for line in f:
                            log = parse_log_line(line.strip())
                            
                            if log:
                                # Filter log
                                if filter_logs([log], args.level, args.logger, start_time, end_time, args.message):
                                    print(format_log(log, args.format))
                    
                    # Update file size
                    file_size = current_size
                
                # Wait before checking again
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopped following log file")


if __name__ == '__main__':
    main()

