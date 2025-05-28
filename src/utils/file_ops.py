"""
File operation utilities for GraphYML.
Provides functions for working with files and directories.
"""
import os
import json
import yaml
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, BinaryIO

from src.utils.error_handler import GraphYMLError


class FileError(GraphYMLError):
    """Exception raised for file operation errors."""
    pass


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path: Absolute path to directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj.absolute()


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dict[str, Any]: Loaded YAML data
        
    Raises:
        FileError: If file cannot be loaded
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, IOError) as e:
        raise FileError(f"Error loading YAML file {file_path}: {str(e)}")


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Data to save
        file_path: Path to save to
        
    Raises:
        FileError: If file cannot be saved
    """
    try:
        # Ensure directory exists
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False)
    except (yaml.YAMLError, IOError) as e:
        raise FileError(f"Error saving YAML file {file_path}: {str(e)}")


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dict[str, Any]: Loaded JSON data
        
    Raises:
        FileError: If file cannot be loaded
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise FileError(f"Error loading JSON file {file_path}: {str(e)}")


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save to
        
    Raises:
        FileError: If file cannot be saved
    """
    try:
        # Ensure directory exists
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except (TypeError, IOError) as e:
        raise FileError(f"Error saving JSON file {file_path}: {str(e)}")


def create_zip(folder_path: Union[str, Path], include_pattern: str = "*.yaml") -> BytesIO:
    """
    Create a ZIP archive of files in a folder.
    
    Args:
        folder_path: Path to folder
        include_pattern: Pattern for files to include
        
    Returns:
        BytesIO: ZIP archive as bytes
        
    Raises:
        FileError: If ZIP cannot be created
    """
    try:
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for file in Path(folder_path).glob(include_pattern):
                zipf.write(file, arcname=file.name)
        
        zip_buffer.seek(0)
        return zip_buffer
    except (zipfile.BadZipFile, IOError) as e:
        raise FileError(f"Error creating ZIP from {folder_path}: {str(e)}")


def extract_zip(zip_file: BinaryIO, extract_path: Union[str, Path]) -> List[Path]:
    """
    Extract a ZIP archive.
    
    Args:
        zip_file: ZIP file object
        extract_path: Path to extract to
        
    Returns:
        List[Path]: List of extracted files
        
    Raises:
        FileError: If ZIP cannot be extracted
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(zip_file.read())
            temp_path = temp_file.name
        
        # Extract ZIP
        extracted_files = []
        
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
            # Get list of extracted files
            for file_info in zip_ref.infolist():
                if not file_info.is_dir():
                    extracted_files.append(Path(extract_path) / file_info.filename)
        
        # Clean up
        os.unlink(temp_path)
        
        return extracted_files
    except (zipfile.BadZipFile, IOError) as e:
        raise FileError(f"Error extracting ZIP: {str(e)}")


def load_yaml_folder(folder_path: Union[str, Path]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Load all YAML files in a folder.
    
    Args:
        folder_path: Path to folder
        
    Returns:
        Tuple[Dict[str, Dict[str, Any]], List[str]]: (loaded_data, errors)
    """
    loaded_data = {}
    errors = []
    
    for yml in Path(folder_path).glob("*.yaml"):
        try:
            node = load_yaml(yml)
            
            if node:
                key = node.get("id") or node.get("title")
                
                if key:
                    loaded_data[key] = node
                else:
                    errors.append(f"Missing ID or title in {yml.name}")
        except Exception as e:
            errors.append(f"Failed to load {yml.name}: {str(e)}")
    
    return loaded_data, errors


def save_yaml_folder(data: Dict[str, Dict[str, Any]], folder_path: Union[str, Path]) -> List[str]:
    """
    Save data to YAML files in a folder.
    
    Args:
        data: Data to save
        folder_path: Path to folder
        
    Returns:
        List[str]: List of errors
    """
    errors = []
    folder = ensure_directory(folder_path)
    
    for key, node in data.items():
        try:
            file_path = folder / f"{key}.yaml"
            save_yaml(node, file_path)
        except Exception as e:
            errors.append(f"Failed to save {key}: {str(e)}")
    
    return errors


def backup_folder(source_path: Union[str, Path], backup_path: Union[str, Path]) -> None:
    """
    Create a backup of a folder.
    
    Args:
        source_path: Path to source folder
        backup_path: Path to backup folder
        
    Raises:
        FileError: If backup cannot be created
    """
    try:
        # Ensure backup directory exists
        backup_dir = ensure_directory(backup_path)
        
        # Copy files
        for file in Path(source_path).glob("*"):
            if file.is_file():
                shutil.copy2(file, backup_dir / file.name)
    except (IOError, OSError) as e:
        raise FileError(f"Error creating backup: {str(e)}")


def restore_backup(backup_path: Union[str, Path], restore_path: Union[str, Path]) -> None:
    """
    Restore a backup.
    
    Args:
        backup_path: Path to backup folder
        restore_path: Path to restore to
        
    Raises:
        FileError: If backup cannot be restored
    """
    try:
        # Ensure restore directory exists
        restore_dir = ensure_directory(restore_path)
        
        # Copy files
        for file in Path(backup_path).glob("*"):
            if file.is_file():
                shutil.copy2(file, restore_dir / file.name)
    except (IOError, OSError) as e:
        raise FileError(f"Error restoring backup: {str(e)}")

