"""
Utility functions for indexing in GraphYML.
Provides helper functions for working with indexes.
"""
import json
from typing import Any, Dict, List, Tuple, Union, Optional, Set, Hashable


def make_hashable(value: Any) -> Hashable:
    """
    Convert a value to a hashable type.
    
    Args:
        value: Value to convert
        
    Returns:
        Hashable: Hashable version of the value
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return tuple(make_hashable(item) for item in value)
    elif isinstance(value, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
    else:
        # Convert to string as a last resort
        return str(value)


def normalize_query(query: Any) -> Any:
    """
    Normalize a query value for consistent comparison.
    
    Args:
        query: Query value
        
    Returns:
        Any: Normalized query value
    """
    if isinstance(query, (list, tuple)):
        return tuple(normalize_query(item) for item in query)
    elif isinstance(query, dict):
        return {k: normalize_query(v) for k, v in query.items()}
    elif isinstance(query, str):
        return query.lower()
    else:
        return query


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List[str]: List of tokens
    """
    if not text or not isinstance(text, str):
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace punctuation with spaces
    for char in '.,;:!?()[]{}"\'':
        text = text.replace(char, ' ')
    
    # Split on whitespace
    tokens = text.split()
    
    # Remove empty tokens
    tokens = [token for token in tokens if token]
    
    return tokens


def extract_phrases(text: str) -> List[str]:
    """
    Extract quoted phrases from text.
    
    Args:
        text: Text to extract phrases from
        
    Returns:
        List[str]: List of phrases
    """
    if not text or not isinstance(text, str):
        return []
    
    phrases = []
    in_quote = False
    current_phrase = []
    
    for char in text:
        if char == '"':
            if in_quote:
                # End of phrase
                phrases.append(''.join(current_phrase).lower())
                current_phrase = []
            in_quote = not in_quote
        elif in_quote:
            current_phrase.append(char)
    
    return phrases


def serialize_complex_value(value: Any) -> str:
    """
    Serialize a complex value to a string.
    
    Args:
        value: Value to serialize
        
    Returns:
        str: Serialized value
    """
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)

