import numpy as np
from typing import Any

def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert NumPy and other non-serializable objects to JSON serializable types.
    
    Parameters
    ----------
    obj : Any
        Object to convert
        
    Returns
    -------
    Any
        JSON serializable object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj

def safe_divide(numerator, denominator):
    """
    Perform division safely by handling division by zero.
    
    Parameters
    ----------
    numerator : float
        The numerator value.
    denominator : float
        The denominator value.
    
    Returns
    -------
    float
        Result of division, or 0 if denominator is 0.
    """
    return numerator / denominator if denominator != 0 else 0