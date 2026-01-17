"""I/O utilities for MindCube data processing.

Provides consistent file reading/writing interfaces across the project.
"""

import json
import os
from typing import Any, Dict, List


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries from the JSONL file

    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file.

    Args:
        data: List of dictionaries to save
        file_path: Output file path

    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary from the JSON file

    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """Save data to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Output file path
        indent: JSON indentation level

    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def ensure_dir(dir_path: str) -> None:
    """Ensure directory exists, create if it doesn't.

    Args:
        dir_path: Directory path to create

    """
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
