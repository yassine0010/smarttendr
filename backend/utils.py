"""
SmartTender AI - Shared Utilities
==================================
Common utility functions used across modules.
"""

import json
import os
from pathlib import Path
from typing import Any, List, Dict

# Data directory path
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Output directory path
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_json(filename: str) -> List[Dict]:
    """
    Load a JSON file from the data directory.

    Args:
        filename: Name of the JSON file

    Returns:
        Parsed JSON data (list of dictionaries)
    """
    filepath = DATA_DIR / filename

    if not filepath.exists():
        print(f"[Warning] File not found: {filepath}")
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[Error] Invalid JSON in {filename}: {e}")
        return []


def save_json(data: Any, filename: str) -> bool:
    """
    Save data to a JSON file in the data directory.

    Args:
        data: Data to save (must be JSON-serializable)
        filename: Name of the output file

    Returns:
        True if successful, False otherwise
    """
    filepath = DATA_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[Error] Failed to save {filename}: {e}")
        return False


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safe division to avoid ZeroDivisionError.

    Args:
        a: Numerator
        b: Denominator
        default: Value to return if denominator is zero

    Returns:
        Result of division or default value
    """
    return a / b if b != 0 else default


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
