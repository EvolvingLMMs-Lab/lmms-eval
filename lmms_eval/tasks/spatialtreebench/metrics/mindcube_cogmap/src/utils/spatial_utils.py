"""Spatial processing utilities for MindCube.

Handles coordinate calculations and spatial relationships.
"""

import math
from typing import List, Tuple


def calculate_position_similarity(pos1: List[float], pos2: List[float]) -> float:
    """Calculate similarity between two positions using Euclidean distance.

    Args:
        pos1: First position [x, y]
        pos2: Second position [x, y]

    Returns:
        Similarity score (1.0 for identical positions, lower for farther apart)

    """
    if not pos1 or not pos2 or len(pos1) != 2 or len(pos2) != 2:
        return 0.0

    distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    # Convert distance to similarity (higher similarity for smaller distance)
    # Using exponential decay: similarity = exp(-distance/scale)
    scale = 2.0  # Adjust this to control how quickly similarity drops
    similarity = math.exp(-distance / scale)

    return similarity


def normalize_coordinates(coordinates: List[List[float]], grid_size: Tuple[int, int] = (10, 10)) -> List[List[float]]:
    """Normalize coordinates to fit within a specified grid.

    Args:
        coordinates: List of [x, y] coordinates
        grid_size: Target grid size (width, height)

    Returns:
        Normalized coordinates

    """
    if not coordinates:
        return []

    # Find min/max values
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Calculate scaling factors
    x_range = max_x - min_x if max_x != min_x else 1
    y_range = max_y - min_y if max_y != min_y else 1

    x_scale = (grid_size[0] - 1) / x_range
    y_scale = (grid_size[1] - 1) / y_range

    # Normalize coordinates
    normalized = []
    for coord in coordinates:
        norm_x = (coord[0] - min_x) * x_scale
        norm_y = (coord[1] - min_y) * y_scale
        normalized.append([norm_x, norm_y])

    return normalized


def get_relative_position(pos1: List[float], pos2: List[float]) -> str:
    """Get relative position of pos2 with respect to pos1.

    Args:
        pos1: Reference position [x, y]
        pos2: Target position [x, y]

    Returns:
        Relative position string (up, down, left, right, etc.)

    """
    if not pos1 or not pos2 or len(pos1) != 2 or len(pos2) != 2:
        return "unknown"

    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]

    # Handle exact positions
    if abs(dx) < 0.1 and abs(dy) < 0.1:
        return "same"

    # Determine primary direction
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


def calculate_center_of_mass(positions: List[List[float]]) -> List[float]:
    """Calculate the center of mass for a set of positions.

    Args:
        positions: List of [x, y] positions

    Returns:
        Center of mass [x, y]

    """
    if not positions:
        return [0.0, 0.0]

    total_x = sum(pos[0] for pos in positions)
    total_y = sum(pos[1] for pos in positions)

    center_x = total_x / len(positions)
    center_y = total_y / len(positions)

    return [center_x, center_y]
