"""Text processing utilities for MindCube.

Handles text cleaning, JSON extraction, and format normalization.
"""

import json
import re
from typing import Any, Dict, Optional


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text using multiple strategies.

    Args:
        text: Input text that may contain JSON

    Returns:
        Extracted JSON dictionary or None if not found

    """
    if not text:
        return None

    # Strategy 1: Look for JSON between <CogMap> tags
    cogmap_match = re.search(r"<CogMap>\s*(.*?)\s*<", text, re.DOTALL | re.IGNORECASE)
    if cogmap_match:
        json_text = cogmap_match.group(1).strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # Strategy 2: Look for JSON in code blocks
    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_block_match:
        json_text = code_block_match.group(1).strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Look for any JSON-like structure
    json_matches = re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    for match in json_matches:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    return None


def clean_text(text: str) -> str:
    """Clean and normalize text.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text

    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters that might interfere with processing
    text = text.strip()

    return text


def normalize_direction(direction: str) -> str:
    """Normalize direction strings to standard format.

    Args:
        direction: Input direction string

    Returns:
        Normalized direction (up, down, left, right, inner, outer)

    """
    if not direction:
        return ""

    direction = direction.lower().strip()

    # Handle common variations
    direction_map = {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "inner": "inner",
        "outer": "outer",
        "north": "up",
        "south": "down",
        "west": "left",
        "east": "right",
        "forward": "inner",
        "backward": "outer",
        "front": "inner",
        "back": "outer",
    }

    return direction_map.get(direction, direction)
