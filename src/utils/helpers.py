"""
Utility helper functions for the diabetes prediction pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the pipeline.
    
    Args:
        level: Logging level
        log_file: Optional file path for logging
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )
    
    return logging.getLogger("diabetes_prediction")


def save_results(results: Dict[str, Any], path: str) -> None:
    """
    Save evaluation results to JSON.
    
    Args:
        results: Dictionary of results
        path: File path to save to
    """
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert_types(results), f, indent=2)


def load_results(path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    import json
    with open(path, 'r') as f:
        return json.load(f)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    raise_on_missing: bool = True
) -> bool:
    """
    Validate that dataframe has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        raise_on_missing: Whether to raise exception or return False
        
    Returns:
        True if valid, False otherwise (if raise_on_missing is False)
        
    Raises:
        ValueError: If required columns are missing and raise_on_missing is True
    """
    missing = set(required_columns) - set(df.columns)
    
    if missing:
        msg = f"Missing required columns: {missing}"
        if raise_on_missing:
            raise ValueError(msg)
        return False
    return True
