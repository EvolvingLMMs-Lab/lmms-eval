from .parse_output import (
    _extract_cognitive_map,
    _extract_grounded_cogmap,
    extract_answer,
    get_setting_from_id,
)
from .result_init import _initialize_cogmap_results_structure
from .src.evaluation.cogmap.cogmap_metrics import calculate_cogmap_similarity


def mindcube_process_results(result: dict) -> dict:
    """Process the results of the MindCube cognitive map evaluation.

    Args:
        result (dict): The raw results of the evaluation.

    Returns:
        dict: The processed results.

    """
    extracted_answer = extract_answer(result["prediction"])
    grounded_cogmap = _extract_grounded_cogmap(result["cogmap"])
    generated_cogmap = _extract_cognitive_map(result["prediction"])
    cogmap_results = _initialize_cogmap_results_structure()
    setting = get_setting_from_id(result["id_type"])
    cogmap_results["settings"][setting]["total"] += 1
    include_in_overall = cogmap_results["settings"][setting].get("include_in_overall", True)

    if not extracted_answer:
        is_correct = False
    else:
        is_correct = extracted_answer == result["ground_truth"]
    correct = 1 if is_correct else 0

    similarity = calculate_cogmap_similarity(generated_cogmap, grounded_cogmap)

    return_result = {
        "answer_correct": correct,
        "parsable_json": similarity["parsable_json"],
        "valid_format": similarity["valid_format"],
        "valid_graph": similarity["valid_graph"],
        "coverage": similarity["coverage"],
        "overall_similarity": similarity["overall_similarity"],
        "best_rotation": similarity["best_rotation"],
        "rotation_invariant_isomorphic": similarity["rotation_invariant_isomorphic"],
        "position_similarity": similarity["position_similarity"],  # no
        "facing_similarity": similarity["facing_similarity"],  # no
        "include_in_overall": include_in_overall,
    }
    return return_result
