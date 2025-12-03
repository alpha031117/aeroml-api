"""Utility functions for JSON data cleaning and handling."""

import math
from typing import Any, Dict, List, Union


def clean_json_for_storage(data: Any) -> Any:
    """
    Clean JSON-compatible data by replacing NaN, Infinity, and -Infinity with None.
    
    PostgreSQL JSON type doesn't accept NaN, Infinity, or -Infinity values.
    This function recursively converts these invalid JSON values to None (null).
    
    Parameters
    ----------
    data : Any
        The data to clean (dict, list, or primitive value)
    
    Returns
    -------
    Any
        Cleaned data with NaN/Infinity replaced by None
    """
    if isinstance(data, dict):
        return {key: clean_json_for_storage(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_json_for_storage(item) for item in data]
    elif isinstance(data, float):
        # Check for NaN, Infinity, or -Infinity
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        # For other types (int, str, bool, None), return as-is
        return data


def clean_nan_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean NaN, Infinity, and -Infinity values from a dictionary.
    
    This is a convenience wrapper around clean_json_for_storage for dictionaries.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary that may contain NaN/Infinity values
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with NaN/Infinity replaced by None
    """
    return clean_json_for_storage(data)

