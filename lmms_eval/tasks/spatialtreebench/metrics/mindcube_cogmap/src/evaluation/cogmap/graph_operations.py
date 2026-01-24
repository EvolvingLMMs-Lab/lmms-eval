"""Graph operations module for cognitive maps.

This module provides functions to:
1. Create a graph representation from a cognitive map
2. Apply rotations to graphs
3. Calculate relative directions and positions
4. Normalize facing directions
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


def create_graph_from_cogmap(cogmap: Dict) -> nx.DiGraph:
    """Create a graph representation from a cognitive map.

    Args:
        cogmap: The cognitive map dictionary

    Returns:
        A directed graph (DiGraph) representation of the cognitive map

    """
    G = nx.DiGraph()

    try:
        # Handle complex format with objects and views
        if "objects" in cogmap and "views" in cogmap:
            # Add all objects as nodes
            for obj in cogmap["objects"]:
                # Skip objects with missing required data
                if "name" not in obj:
                    continue

                name = obj["name"]

                # Handle position data
                position = extract_position(obj.get("position"))
                facing = normalize_facing(obj.get("facing", None))
                G.add_node(name, position=position, facing=facing, type="object")

            # Add all views as nodes
            for view in cogmap["views"]:
                # Skip views with missing required data
                if "name" not in view:
                    continue

                name = view["name"]
                position = extract_position(view.get("position"))
                facing = normalize_facing(view.get("facing", None))
                G.add_node(name, position=position, facing=facing, type="view")

            # Add edges based on relative positions
            _add_relative_position_edges(G)

        # Handle simple format (object categories with positions)
        else:
            for obj_name, obj_data in cogmap.items():
                # Skip if obj_data is None or not a dict
                if obj_data is None or not isinstance(obj_data, dict):
                    continue

                position = extract_position(obj_data.get("position"))
                facing = normalize_facing(obj_data.get("facing", None))
                G.add_node(obj_name, position=position, facing=facing, type="object")

            # Add edges based on relative positions
            _add_relative_position_edges(G)

    except Exception as e:
        print(f"Error creating graph: {e}")
        # Return empty graph on error
        return nx.DiGraph()

    return G


def _add_relative_position_edges(G: nx.DiGraph) -> None:
    """Add edges to the graph based on relative positions between nodes.

    Args:
        G: The directed graph to add edges to

    """
    nodes = list(G.nodes(data=True))
    for i, (node1, data1) in enumerate(nodes):
        pos1 = data1.get("position", (0, 0))
        for node2, data2 in nodes[i + 1 :]:
            pos2 = data2.get("position", (0, 0))
            # Calculate relative direction
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]

            # Add edges with relative positions
            if dx != 0 or dy != 0:  # Don't add edges for same position
                angle = np.arctan2(dy, dx) * 180 / np.pi
                distance = np.sqrt(dx**2 + dy**2)
                G.add_edge(node1, node2, angle=angle, distance=distance)
                G.add_edge(node2, node1, angle=(angle + 180) % 360, distance=distance)


def extract_position(position: Any) -> Tuple[float, float]:
    """Extract and validate position data.

    Args:
        position: The position value

    Returns:
        A tuple of (x, y) coordinates

    """
    if position is None:
        return (0.0, 0.0)

    # Check if position is a list with at least 2 elements
    if isinstance(position, list) and len(position) >= 2:
        # Ensure position values are numeric
        try:
            return (float(position[0]), float(position[1]))
        except (ValueError, TypeError):
            return (0.0, 0.0)
    else:
        return (0.0, 0.0)


def normalize_facing(facing: Any) -> Optional[str]:
    """Normalize facing field to standard direction.

    Args:
        facing: The facing value to normalize

    Returns:
        Normalized facing direction or None

    """
    if facing is None:
        return None

    # Handle empty strings
    if isinstance(facing, str) and not facing.strip():
        return None

    # Handle list format
    if isinstance(facing, list):
        if not facing:
            return None
        # Use the first element if it's a list
        facing = facing[0]

    # If it's not a string at this point, return None
    if not isinstance(facing, str):
        return None

    # Normalize the string
    facing = facing.lower().strip()

    # If empty after stripping, return None
    if not facing:
        return None

    # Map common variations to standard names
    facing_map = {
        "top": "up",
        "bottom": "down",
        "north": "up",
        "south": "down",
        "east": "right",
        "west": "left",
        "front": "inner",
        "back": "outer",
        "into": "inner",
        "out": "outer",
        "inside": "inner",
        "outside": "outer",
        "forward": "inner",
        "backward": "outer",
    }

    return facing_map.get(facing, facing)


def extract_objects_with_extended_info(cog_map: Dict) -> Dict:
    """Extract objects with position and facing information.
    Handles both complex and simple formats.

    Args:
        cog_map: The cognitive map

    Returns:
        Dictionary of objects with position and facing information

    """
    objects_info = {}

    # Check if cog_map has objects array directly (complex format)
    if isinstance(cog_map, dict) and "objects" in cog_map:
        for obj in cog_map.get("objects", []):
            if "name" not in obj:
                continue

            name = obj["name"]

            # Safely extract position
            try:
                if "position" in obj and obj["position"] is not None:
                    if isinstance(obj["position"], list) and len(obj["position"]) >= 2:
                        # Convert position values to float
                        position = [
                            float(obj["position"][0]),
                            float(obj["position"][1]),
                        ]
                    else:
                        position = [0.0, 0.0]
                else:
                    position = [0.0, 0.0]
            except (ValueError, TypeError):
                position = [0.0, 0.0]

            facing = normalize_facing(obj.get("facing", None))

            objects_info[name] = {
                "position": np.array(position, dtype=float),
                "facing": facing,
            }

        # Add view info if available (views are important too)
        for view in cog_map.get("views", []):
            if "name" not in view:
                continue

            name = view["name"]

            try:
                if "position" in view and view["position"] is not None:
                    if isinstance(view["position"], list) and len(view["position"]) >= 2:
                        position = [
                            float(view["position"][0]),
                            float(view["position"][1]),
                        ]
                    else:
                        position = [0.0, 0.0]
                else:
                    position = [0.0, 0.0]
            except (ValueError, TypeError):
                position = [0.0, 0.0]

            facing = normalize_facing(view.get("facing", None))

            objects_info[name] = {
                "position": np.array(position, dtype=float),
                "facing": facing,
            }

    # Handle simple format (object categories with positions)
    elif isinstance(cog_map, dict):
        for obj_name, obj_data in cog_map.items():
            # Skip if obj_data is None or not a dict
            if obj_data is None or not isinstance(obj_data, dict):
                continue

            # Safely extract position
            try:
                if "position" in obj_data and obj_data["position"] is not None:
                    if isinstance(obj_data["position"], list) and len(obj_data["position"]) >= 2:
                        # Convert position values to float
                        position = [
                            float(obj_data["position"][0]),
                            float(obj_data["position"][1]),
                        ]
                    else:
                        position = [0.0, 0.0]
                else:
                    position = [0.0, 0.0]
            except (ValueError, TypeError):
                position = [0.0, 0.0]

            facing = normalize_facing(obj_data.get("facing", None))

            objects_info[obj_name] = {
                "position": np.array(position, dtype=float),
                "facing": facing,
            }

    return objects_info


def get_rotation_matrices() -> List[Dict]:
    """Generate rotation matrices for 6 main orientations in 3D space.

    Returns:
        List of rotation definitions with matrices and transformers

    """
    rotations = []

    # Identity (no rotation)
    rotations.append(
        {
            "name": "identity",
            "matrix": np.eye(3),
            "facing_transform": lambda x: x,  # No change to facings
        }
    )

    # 90 degree rotations around Z-axis (top view rotations)
    for angle in [90, 180, 270]:
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        rotations.append(
            {
                "name": f"z_rot_{angle}",
                "matrix": matrix,
                "facing_transform": lambda x, angle=angle: rotate_facing_z(x, angle),
            }
        )

    # 90 degree rotations around X and Y axes (viewing from different sides)
    rotations.append(
        {
            "name": "x_rot_90",
            "matrix": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            "facing_transform": lambda x: rotate_facing_x(x),
        }
    )

    rotations.append(
        {
            "name": "y_rot_90",
            "matrix": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            "facing_transform": lambda x: rotate_facing_y(x),
        }
    )

    return rotations


def apply_rotation_to_map(objects_data: Dict, rotation: Dict) -> Dict:
    """Apply 3D rotation to all objects in the map.

    Args:
        objects_data: Dictionary of objects with position and facing information
        rotation: Rotation definition with matrix and facing transform

    Returns:
        Dictionary of rotated objects

    """
    rotated_data = {}

    for name, obj_data in objects_data.items():
        # Get position and ensure it's a numpy array with exactly 2 elements
        pos = obj_data["position"]

        # Ensure position is a numpy array of 2 elements
        if not isinstance(pos, np.ndarray):
            pos = np.array(pos, dtype=float)

        # If position has fewer than 2 elements, pad with zeros
        if pos.size < 2:
            pos = np.pad(pos, (0, 2 - pos.size), "constant")
        # If more than 2 elements, take only the first 2
        elif pos.size > 2:
            pos = pos[:2]

        # Extend 2D position to 3D and apply rotation
        pos_3d = np.append(pos, 0.0)  # Add z=0
        rotation_matrix = rotation["matrix"]

        # Ensure pos_3d is the right shape for matrix multiplication
        # It should be a column vector (3,) for @ operator
        pos_3d = pos_3d.reshape(3)

        # Apply rotation
        rotated_pos_3d = rotation_matrix @ pos_3d

        # Project back to 2D
        rotated_pos_2d = rotated_pos_3d[:2]

        # Transform facing direction
        rotated_facing = rotation["facing_transform"](obj_data["facing"])

        rotated_data[name] = {"position": rotated_pos_2d, "facing": rotated_facing}

    return rotated_data


def rotate_facing_z(facing: Optional[str], angle: float) -> Optional[str]:
    """Rotate facing direction around Z-axis.

    Args:
        facing: The facing direction
        angle: Rotation angle in degrees

    Returns:
        Rotated facing direction

    """
    if not facing:
        return facing

    # Handle case where facing is a list
    if isinstance(facing, list):
        if len(facing) == 0:
            return facing
        # Use the first element if it's a list
        facing = facing[0]

    # Convert to string if it's still not a string
    if not isinstance(facing, str):
        return facing

    direction_map = {
        "up": ["up", "right", "down", "left"],
        "right": ["right", "down", "left", "up"],
        "down": ["down", "left", "up", "right"],
        "left": ["left", "up", "right", "down"],
        "inner": "inner",  # Inner/outer unchanged by z-rotation
        "outer": "outer",
    }

    if facing in ["inner", "outer"]:
        return facing

    if facing in direction_map:
        idx = angle // 90
        return direction_map[facing][idx % 4]

    return facing


def rotate_facing_x(facing: Optional[str]) -> Optional[str]:
    """Rotate facing direction around X-axis.

    Args:
        facing: The facing direction

    Returns:
        Rotated facing direction

    """
    if not facing:
        return facing

    # Handle case where facing is a list
    if isinstance(facing, list):
        if len(facing) == 0:
            return facing
        # Use the first element if it's a list
        facing = facing[0]

    # Convert to string if it's still not a string
    if not isinstance(facing, str):
        return facing

    # This is a simplified mapping
    direction_map = {
        "up": "inner",
        "down": "outer",
        "inner": "down",
        "outer": "up",
        "left": "left",
        "right": "right",
    }

    return direction_map.get(facing, facing)


def rotate_facing_y(facing: Optional[str]) -> Optional[str]:
    """Rotate facing direction around Y-axis.

    Args:
        facing: The facing direction

    Returns:
        Rotated facing direction

    """
    if not facing:
        return facing

    # Handle case where facing is a list
    if isinstance(facing, list):
        if len(facing) == 0:
            return facing
        # Use the first element if it's a list
        facing = facing[0]

    # Convert to string if it's still not a string
    if not isinstance(facing, str):
        return facing

    # This is a simplified mapping
    direction_map = {
        "left": "inner",
        "right": "outer",
        "inner": "right",
        "outer": "left",
        "up": "up",
        "down": "down",
    }

    return direction_map.get(facing, facing)


def get_extended_direction(
    pos1: np.ndarray,
    pos2: np.ndarray,
    facing1: Optional[str] = None,
    facing2: Optional[str] = None,
) -> Optional[str]:
    """Determine the extended direction from pos1 to pos2.
    Includes: up, right, down, left, inner, outer.

    Args:
        pos1: Position of first object
        pos2: Position of second object
        facing1: Facing direction of first object
        facing2: Facing direction of second object

    Returns:
        Direction string or None

    """
    # Ensure positions are valid arrays with non-zero elements
    if pos1 is None or pos2 is None:
        return None

    # Convert to numpy arrays if they aren't already
    pos1 = np.array(pos1, dtype=float)
    pos2 = np.array(pos2, dtype=float)

    # Check if arrays are valid
    if pos1.size == 0 or pos2.size == 0:
        return None

    # Ensure positions have at least 2 elements
    if pos1.size < 2:
        pos1 = np.pad(pos1, (0, 2 - pos1.size), "constant")
    if pos2.size < 2:
        pos2 = np.pad(pos2, (0, 2 - pos2.size), "constant")

    # Extract first two elements for 2D calculations
    pos1_2d = pos1[:2]
    pos2_2d = pos2[:2]

    dx = pos2_2d[0] - pos1_2d[0]  # right is positive
    dy = pos2_2d[1] - pos1_2d[1]  # down is positive

    # Calculate distance for inner/outer detection
    distance = np.sqrt(dx**2 + dy**2)

    # Check if objects are close enough for inner/outer relationship
    if distance < 0.5:  # Threshold for considering inner/outer
        return determine_inner_outer_relationship(pos1_2d, pos2_2d, facing1, facing2)

    # Regular 2D directions
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


def determine_inner_outer_relationship(pos1: np.ndarray, pos2: np.ndarray, facing1: Optional[str], facing2: Optional[str]) -> Optional[str]:
    """Determine if relationship is inner/outer based on positions and facings.

    Args:
        pos1: Position of first object
        pos2: Position of second object
        facing1: Facing direction of first object
        facing2: Facing direction of second object

    Returns:
        'inner', 'outer', or None

    """
    # Check if either facing directly indicates inner/outer
    if facing1 == "inner" or facing2 == "outer":
        return "inner"
    elif facing1 == "outer" or facing2 == "inner":
        return "outer"

    # Default for very close objects
    if np.linalg.norm(pos1 - pos2) < 0.2:
        return "inner"

    return None


def build_comprehensive_relation_matrix(objects_data: Dict, object_names: List[str]) -> Dict:
    """Build a relationship matrix including inner/outer relationships.

    Args:
        objects_data: Dictionary of objects with position and facing information
        object_names: List of object names to include in the matrix

    Returns:
        Dictionary of relationships between objects

    """
    # Initialize with None instead of requiring specific None values
    relations = {obj1: {} for obj1 in object_names}

    for obj1 in object_names:
        for obj2 in object_names:
            if obj1 != obj2:
                if obj1 in objects_data and obj2 in objects_data:
                    pos1 = objects_data[obj1]["position"]
                    pos2 = objects_data[obj2]["position"]
                    facing1 = objects_data[obj1]["facing"]
                    facing2 = objects_data[obj2]["facing"]

                    # Get extended direction including inner/outer
                    relations[obj1][obj2] = get_extended_direction(pos1, pos2, facing1, facing2)
                else:
                    # If either object is missing, set relation to None
                    relations[obj1][obj2] = None

    return relations


# ==============================
# --------- TESTS ---------
# ==============================


def test_create_graph_from_cogmap():
    print("========== Testing create_graph_from_cogmap ==========")
    # first is complex format
    json_obj = {
        "objects": [
            {"name": "object1", "position": [1, 2], "facing": "up"},
            {"name": "object2", "position": [3, 4], "facing": "down"},
            {"name": "object3", "position": [5, 6]},
        ],
        "views": [
            {"name": "view1", "position": [1, 2], "facing": "up"},
            {"name": "view2", "position": [3, 4], "facing": "down"},
        ],
    }
    map = create_graph_from_cogmap(json_obj)
    print(map)

    # second is simple format
    json_obj = {
        "object1": {"position": [1, 2], "facing": "up"},
        "object2": {"position": [3, 4], "facing": "down"},
        "object3": {"position": [5, 6]},
    }
    print(create_graph_from_cogmap(json_obj))

    # third is simple format
    json_obj = {
        "object1": {"position": [1, 2], "facing": "up"},
        "object2": {"position": [3, 4], "facing": "down"},
        "object3": [],
    }
    print(create_graph_from_cogmap(json_obj))


if __name__ == "__main__":
    test_create_graph_from_cogmap()
