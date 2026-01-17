"""Cognitive map evaluation metrics.

This module provides functions to calculate similarity between generated and
ground truth cognitive maps, including:
1. Rotation-invariant isomorphism
2. Position and facing similarity
3. Coverage and validity assessment
"""

from typing import Any, Dict, List, Tuple, Union

from .graph_operations import (
    apply_rotation_to_map,
    build_comprehensive_relation_matrix,
    extract_objects_with_extended_info,
    get_rotation_matrices,
)


def is_complex_format(cogmap: Dict) -> bool:
    """Determine if the cognitive map uses complex format (with objects/views arrays)
    or simple key-value format.

    Args:
        cogmap: The cognitive map JSON

    Returns:
        True if complex format, False if simple format

    """
    if not isinstance(cogmap, dict):
        return False

    return "objects" in cogmap and isinstance(cogmap.get("objects"), list)


def is_valid_position(position: Any) -> bool:
    """Check if a position value is valid (a list of 2 numeric values).

    Args:
        position: The position value to check

    Returns:
        True if valid, False otherwise

    """
    if not isinstance(position, list):
        return False

    if len(position) < 2:
        return False

    try:
        # Check if first two elements are numeric
        float(position[0])
        float(position[1])
        return True
    except (ValueError, TypeError):
        return False


def is_valid_facing(facing: Any) -> bool:
    """Check if a facing value is valid (one of: up, down, left, right, inner, outer).

    Args:
        facing: The facing value to check

    Returns:
        True if valid, False otherwise

    """
    if facing is None:
        return True  # Facing is optional

    if isinstance(facing, list):
        if not facing:
            return True
        facing = facing[0]

    if not isinstance(facing, str):
        return False

    # Normalize
    facing = facing.lower().strip()

    valid_facings = {
        "up",
        "down",
        "left",
        "right",
        "inner",
        "outer",
        "top",
        "bottom",
        "north",
        "south",
        "east",
        "west",
        "front",
        "back",
        "into",
        "out",
        "inside",
        "outside",
        "forward",
        "backward",
    }

    return facing in valid_facings


def validate_cogmap_format(cogmap: Dict) -> Tuple[bool, List[str]]:
    """Validate if a cognitive map has the correct format.

    Args:
        cogmap: The cognitive map to validate

    Returns:
        Tuple of (is_valid, error_messages)

    """
    errors = []

    # Check if cogmap is a dictionary
    if not isinstance(cogmap, dict):
        errors.append("Cognitive map must be a dictionary")
        return False, errors

    # Check format type
    if is_complex_format(cogmap):
        # Validate complex format
        if "objects" not in cogmap:
            errors.append("Complex format must have 'objects' key")
        else:
            objects = cogmap["objects"]
            if not isinstance(objects, list):
                errors.append("'objects' must be a list")
            else:
                for i, obj in enumerate(objects):
                    if not isinstance(obj, dict):
                        errors.append(f"Object {i} must be a dictionary")
                        continue

                    if "name" not in obj:
                        errors.append(f"Object {i} missing 'name' field")

                    if "position" not in obj:
                        errors.append(f"Object {i} missing 'position' field")
                    elif not is_valid_position(obj["position"]):
                        errors.append(f"Object {i} has invalid position: {obj['position']}")

                    if "facing" in obj and not is_valid_facing(obj["facing"]):
                        errors.append(f"Object {i} has invalid facing: {obj['facing']}")

        # Check views if present
        if "views" in cogmap:
            views = cogmap["views"]
            if not isinstance(views, list):
                errors.append("'views' must be a list")
            else:
                for i, view in enumerate(views):
                    if not isinstance(view, dict):
                        errors.append(f"View {i} must be a dictionary")
                        continue

                    if "name" not in view:
                        errors.append(f"View {i} missing 'name' field")

                    if "position" not in view:
                        errors.append(f"View {i} missing 'position' field")
                    elif not is_valid_position(view["position"]):
                        errors.append(f"View {i} has invalid position: {view['position']}")

                    if "facing" in view and not is_valid_facing(view["facing"]):
                        errors.append(f"View {i} has invalid facing: {view['facing']}")
    else:
        # Validate simple format (key-value pairs)
        for key, value in cogmap.items():
            if not isinstance(value, dict):
                errors.append(f"Object '{key}' must be a dictionary")
                continue

            if "position" not in value:
                errors.append(f"Object '{key}' missing 'position' field")
            elif not is_valid_position(value["position"]):
                errors.append(f"Object '{key}' has invalid position: {value['position']}")

            if "facing" in value and not is_valid_facing(value["facing"]):
                errors.append(f"Object '{key}' has invalid facing: {value['facing']}")

    return len(errors) == 0, errors


def truncate_position_list_into_one(positions: Union[List[Dict], Dict]) -> Dict:
    """Truncate a list of positions into a single position."""
    if isinstance(positions, list):
        return positions[0]
    return positions


def trucate_object_position(raw_cogmap: Dict) -> Dict:
    """Truncate a list of positions into a single position."""
    if not isinstance(raw_cogmap, dict):
        return {}

    return {k: truncate_position_list_into_one(v) for k, v in raw_cogmap.items()}


def calculate_cogmap_similarity(generated_map: Dict, grounded_map: Dict) -> Dict:
    """Calculate similarity between generated and grounded cognitive maps.
    Supports inner/outer relationships and 3D rotation invariance.

    Args:
        generated_map: Generated cognitive map
        grounded_map: Ground truth cognitive map

    Returns:
        Dictionary of similarity metrics

    """
    if not generated_map or not grounded_map:
        return _empty_similarity_result()

    # Call the extended evaluation function
    extended_result = calculate_extended_cogmap_similarity(generated_map, grounded_map)

    # Map results to the original metric names for backward compatibility
    result = {
        "isomorphic": extended_result["rotation_invariant_isomorphic"],
        "rotation_invariant_isomorphic": extended_result["rotation_invariant_isomorphic"],
        "position_similarity": extended_result["directional_similarity"],
        "facing_similarity": extended_result["facing_similarity"],
        "directional_similarity": extended_result["directional_similarity"],
        "relative_position_accuracy": extended_result["directional_similarity"],
        "overall_similarity": extended_result["overall_similarity"],
        "valid_graph": extended_result["valid_graph"],
        "parsable_json": extended_result.get("parsable_json", True),
        "valid_format": extended_result.get("valid_format", False),
        "coverage": extended_result["coverage"],
        "best_rotation": extended_result["best_rotation"],
    }

    return result


def _empty_similarity_result() -> Dict:
    """Returns an empty similarity result with default values.

    Returns:
        Dictionary with default metrics

    """
    return {
        "isomorphic": False,
        "rotation_invariant_isomorphic": False,
        "position_similarity": 0.0,
        "facing_similarity": 0.0,
        "directional_similarity": 0.0,
        "relative_position_accuracy": 0.0,
        "overall_similarity": 0.0,
        "valid_graph": False,
        "parsable_json": False,
        "valid_format": False,
        "coverage": 0.0,
        "best_rotation": None,
    }


def calculate_extended_cogmap_similarity(generated_map: Dict, grounded_map: Dict) -> Dict:
    """Calculate similarity between generated and grounded cognitive maps.
    Supports inner/outer relationships and 3D rotation invariance.
    Handles both simple format (only objects) and complex format (objects and views).

    Args:
        generated_map: Generated cognitive map
        grounded_map: Ground truth cognitive map

    Returns:
        Dictionary of extended similarity metrics

    """
    # Create an empty result structure to build upon
    result = _empty_extended_similarity_result()

    # Check if inputs are None or empty
    if not generated_map or not grounded_map:
        result["parsable_json"] = False
        return result

    # Ensure the inputs are dictionaries
    if not isinstance(generated_map, dict) or not isinstance(grounded_map, dict):
        result["parsable_json"] = False
        return result

    # Mark as parsable JSON since we've confirmed they are dictionaries
    result["parsable_json"] = True

    # Apply truncation to simple format maps if needed
    if not is_complex_format(generated_map):
        generated_map = trucate_object_position(generated_map)

    # Validate format correctness
    gen_valid_format, gen_errors = validate_cogmap_format(generated_map)
    ground_valid_format, ground_errors = validate_cogmap_format(grounded_map)

    # Record validation result
    result["valid_format"] = gen_valid_format and ground_valid_format

    # If format is invalid, return early
    if not result["valid_format"]:
        return result

    # Extract objects and positions
    gen_data = extract_objects_with_extended_info(generated_map)
    ground_data = extract_objects_with_extended_info(grounded_map)

    # If either map has no objects with valid positions, return failed result
    if not gen_data or not ground_data:
        return result

    # Mark as valid graph since we have extracted objects
    result["valid_graph"] = True

    # Determine if generated map is simple or complex format
    is_gen_complex = "views" in generated_map if isinstance(generated_map, dict) else False

    # For complex format grounded map, filter objects based on generated map format
    ground_objects_set = set(ground_data.keys())
    gen_objects_set = set(gen_data.keys())

    # If generated map is simple format, only consider objects in grounded map (exclude views)
    if not is_gen_complex and isinstance(grounded_map, dict) and "objects" in grounded_map:
        ground_object_names = {obj["name"] for obj in grounded_map.get("objects", []) if "name" in obj}
        ground_objects_set = ground_object_names

    # Calculate coverage of ground truth objects in generated map
    common_objects = ground_objects_set & gen_objects_set
    coverage = len(common_objects) / len(ground_objects_set) if ground_objects_set else 0
    result["coverage"] = coverage
    result["common_objects"] = list(common_objects)

    # If no common objects, maps cannot be compared
    if not common_objects:
        return result

    # Build ground truth relation matrix
    ground_relations = build_comprehensive_relation_matrix(ground_data, list(ground_objects_set))

    # Try different rotations to find best match
    best_similarity = 0.0
    best_rotation = None
    best_directional_sim = 0.0
    best_facing_sim = 0.0
    rotation_invariant_isomorphic = False

    # Get rotation matrices
    rotations = get_rotation_matrices()

    # Test rotations
    for rotation in rotations:
        try:
            # Apply rotation to generated map
            rotated_gen_data = apply_rotation_to_map(gen_data, rotation)

            # Build relation matrix for rotated data using all gen objects
            gen_relations = build_comprehensive_relation_matrix(rotated_gen_data, list(gen_objects_set))

            # Check isomorphism - generated map must contain all ground truth relations
            is_isomorphic = check_rotation_invariant_isomorphism(gen_relations, ground_relations)

            # Calculate directional similarity - how many ground truth relations are correctly represented in generated map
            total_ground_relations = 0
            matching_relations = 0

            for obj1 in ground_objects_set:
                if obj1 not in gen_objects_set:
                    continue

                for obj2 in ground_objects_set:
                    if obj2 not in gen_objects_set or obj1 == obj2:
                        continue

                    ground_rel = ground_relations.get(obj1, {}).get(obj2)
                    gen_rel = gen_relations.get(obj1, {}).get(obj2)

                    if ground_rel is not None:
                        total_ground_relations += 1
                        if gen_rel == ground_rel:
                            matching_relations += 1

            directional_sim = matching_relations / total_ground_relations if total_ground_relations > 0 else 0

            # Calculate facing similarity (use same logic as old version)
            total_facings = 0
            matching_facings = 0

            for obj in ground_objects_set:
                if obj not in gen_objects_set:
                    continue

                ground_facing = ground_data[obj]["facing"]
                gen_facing = rotated_gen_data[obj]["facing"]

                if ground_facing:
                    total_facings += 1
                    if gen_facing == ground_facing:
                        matching_facings += 1

            facing_sim = matching_facings / total_facings if total_facings > 0 else 1.0

            # Overall similarity is weighted average
            overall_sim = 0.7 * directional_sim + 0.3 * facing_sim

            # Update best match
            if overall_sim > best_similarity:
                best_similarity = overall_sim
                best_rotation = rotation
                best_directional_sim = directional_sim
                best_facing_sim = facing_sim
                rotation_invariant_isomorphic = is_isomorphic

        except Exception:
            # Skip this rotation if error occurs
            continue

    # Update result with best match
    result["rotation_invariant_isomorphic"] = rotation_invariant_isomorphic
    result["directional_similarity"] = best_directional_sim
    result["facing_similarity"] = best_facing_sim
    result["overall_similarity"] = best_similarity
    result["best_rotation"] = best_rotation

    return result


def _empty_extended_similarity_result() -> Dict:
    """Returns an empty extended similarity result with default values.

    Returns:
        Dictionary with default extended metrics

    """
    return {
        "rotation_invariant_isomorphic": False,
        "directional_similarity": 0.0,
        "facing_similarity": 0.0,
        "overall_similarity": 0.0,
        "valid_graph": False,
        "parsable_json": False,
        "valid_format": False,
        "coverage": 0.0,
        "common_objects": [],
        "best_rotation": None,
    }


def check_rotation_invariant_isomorphism(gen_relations: Dict, ground_relations: Dict) -> bool:
    """Check if generated relations are isomorphic to ground truth relations.

    Args:
        gen_relations: Generated relation matrix
        ground_relations: Ground truth relation matrix

    Returns:
        True if isomorphic, False otherwise

    """
    # For each relation in ground truth, check if it exists in generated
    for obj1, relations in ground_relations.items():
        if obj1 not in gen_relations:
            return False

        for obj2, relation in relations.items():
            if obj2 not in gen_relations[obj1]:
                return False

            if gen_relations[obj1][obj2] != relation:
                return False

    return True
