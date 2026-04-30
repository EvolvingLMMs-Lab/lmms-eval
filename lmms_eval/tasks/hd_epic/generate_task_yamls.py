#!/usr/bin/env python3
"""
generate_task_yamls.py
======================
Generate one YAML per HD-EPIC question prototype + 7 category-level group
YAMLs + 1 master group YAML, all inheriting from `_hd_epic_base.yaml`.

Source of truth: the official annotation directory at
    https://github.com/hd-epic/hd-epic-annotations/tree/main/vqa-benchmark
which lists exactly 30 prototype JSONs grouped into 7 categories
(matching Figure 10 of the HD-EPIC paper).

Usage:
    python generate_task_yamls.py [--output-dir /path/to/tasks/hd_epic]
"""

import argparse
import os
import textwrap

# ---------------------------------------------------------------------------
# 30 official prototypes, keyed by category. The string is the JSON filename
# stem from `hd-epic-annotations/vqa-benchmark/`. Each becomes a subtask:
#       hd_epic_<prototype>
# ---------------------------------------------------------------------------
CATEGORIES = {
    "recipe": [
        "recipe_recipe_recognition",
        "recipe_multi_recipe_recognition",
        "recipe_multi_step_localization",
        "recipe_step_localization",
        "recipe_prep_localization",
        "recipe_step_recognition",
        "recipe_rough_step_localization",
        "recipe_following_activity_recognition",
    ],
    "ingredient": [
        "ingredient_ingredient_retrieval",
        "ingredient_ingredient_weight",
        "ingredient_ingredients_order",
        "ingredient_ingredient_adding_localization",
        "ingredient_ingredient_recognition",
        "ingredient_exact_ingredient_recognition",
    ],
    "nutrition": [
        "nutrition_image_nutrition_estimation",
        "nutrition_nutrition_change",
        "nutrition_video_nutrition_estimation",
    ],
    "fine_grained": [
        "fine_grained_action_recognition",
        "fine_grained_how_recognition",
        "fine_grained_why_recognition",
        "fine_grained_action_localization",
    ],
    "3d_perception": [
        "3d_perception_fixture_location",
        "3d_perception_object_location",
        "3d_perception_object_contents_retrieval",
        "3d_perception_fixture_interaction_counting",
    ],
    "object_motion": [
        "object_motion_object_movement_itinerary",
        "object_motion_object_movement_counting",
        "object_motion_stationary_object_localization",
    ],
    "gaze": [
        "gaze_gaze_estimation",
        "gaze_interaction_anticipation",
    ],
}

# Flat list of all 30 prototype names (preserves declared order)
TASK_TYPES = [t for tasks in CATEGORIES.values() for t in tasks]

# Human-readable per-prototype descriptions, derived from the actual
# question text in each JSON file (not guessed).
DESCRIPTIONS = {
    # Recipe (8)
    "recipe_recipe_recognition": '"Which of these recipes were carried out by the participant?" (multi-video, identify the recipe)',
    "recipe_multi_recipe_recognition": '"Which of these recipes were carried out in this video?" (single-video, may contain multiple recipes)',
    "recipe_multi_step_localization": '"In this video, when did the participant perform each of the following steps: A, B, C?" (localise multiple steps simultaneously)',
    "recipe_step_localization": '"When did the participant perform step X from recipe Y?" (precise temporal localisation, often multi-video)',
    "recipe_prep_localization": '"When did the participant perform prep for X from recipe Y?" (find prep moments, often multi-video)',
    "recipe_step_recognition": '"What step did the participant do between TIME and TIME?" (given a window, identify the step)',
    "recipe_rough_step_localization": '"Which of these time segments belongs to recipe X step Y?" (pick the correct coarse window)',
    "recipe_following_activity_recognition": '"Which high-level activity did the participant do while completing recipe step X?"',
    # Ingredient (6)
    "ingredient_ingredient_retrieval": '"Between TIME and TIME, which of these ingredients were added to the dish?"',
    "ingredient_ingredient_weight": '"How much did the participant weigh of X in this video?" (gram amounts)',
    "ingredient_ingredients_order": '"What is the order of ingredients added to the dish in this video?" (choices are ordered lists)',
    "ingredient_ingredient_adding_localization": '"When was ingredient X added to recipe Y?" (pick the correct time range)',
    "ingredient_ingredient_recognition": '"Which of these ingredients is NOT used in recipe X?" (multi-video, NOT pattern)',
    "ingredient_exact_ingredient_recognition": '"What was the exact quantity of X used in Y?" (multi-video, exact tbsp/g/etc.)',
    # Nutrition (3)
    "nutrition_image_nutrition_estimation": '"Which of the ingredients in these images showcase higher carbs/fat/etc.?" (5 image inputs)',
    "nutrition_nutrition_change": '"From TIME to TIME, what changed in the nutrition values of the dish?"',
    "nutrition_video_nutrition_estimation": '"What is the ingredient with highest carbs/fat/etc. in this recipe?"',
    # Fine-grained Actions (4)
    "fine_grained_action_recognition": '"Which of these sentences best describe the ongoing action(s) in the video?"',
    "fine_grained_how_recognition": '"What is the best description for HOW the person carried out the action <X>?"',
    "fine_grained_why_recognition": '"What is the best description for WHY the person performed the action <X>?"',
    "fine_grained_action_localization": '"When did the action <X> happen in the video?" (pick from 5 time-range choices)',
    # 3D Perception (4)
    "3d_perception_fixture_location": "\"Given the direction I am looking at TIME, where is the X located?\" (clock-face directions: 1 o'clock ... 12 o'clock)",
    "3d_perception_object_location": '"Where did I put the object identified by <BBOX> at TIME after taking it at TIME?"',
    "3d_perception_object_contents_retrieval": '"Which of these objects did the person put in/on the item indicated by <BBOX> in TIME?"',
    "3d_perception_fixture_interaction_counting": '"How many times did I close the item indicated by <BBOX> in TIME?"',
    # Object Motion (3)
    "object_motion_object_movement_itinerary": '"Where was the object <BBOX> seen at TIME moved from/to throughout the video?" (multi-hop trajectory)',
    "object_motion_object_movement_counting": '"How many times did the object <BBOX> seen at TIME change locations in the video?"',
    "object_motion_stationary_object_localization": '"After the object <BBOX> seen at TIME is first moved, from which starting time does it remain static for >150s?"',
    # Gaze (2)
    "gaze_gaze_estimation": '"What is the person looking at in this video segment?"',
    "gaze_interaction_anticipation": '"What object will the person interact with next, ignoring ongoing interactions?"',
}


def task_name(task_type: str) -> str:
    return f"hd_epic_{task_type}"


def yaml_for_task(task_type: str) -> str:
    desc = DESCRIPTIONS.get(task_type, task_type.replace("_", " "))
    # Escape inner double quotes so the YAML string remains valid.
    # Descriptions deliberately quote the literal question text, e.g.
    #   "How many times did I close the item indicated by <BBOX>..."
    desc_escaped = desc.replace("\\", "\\\\").replace('"', '\\"')
    return textwrap.dedent(
        f"""\
        # HD-EPIC subtask: {task_type}
        # {desc}
        include: _hd_epic_base.yaml

        task: {task_name(task_type)}

        # Filter the combined JSONL down to this prototype's rows.
        process_docs: !function utils.filter_{task_type}

        metadata:
          version: 1.0
          task_type: {task_type}
          description: "{desc_escaped}"
        """
    )


def category_group_yaml(category: str, prototypes: list) -> str:
    """One YAML per high-level category (recipe, ingredient, nutrition, ...)."""
    subtasks = "\n".join(f"  - {task_name(t)}" for t in prototypes)
    header = textwrap.dedent(
        f"""\
        # HD-EPIC '{category}' category -- bundles its {len(prototypes)} prototypes.
        # Run all of them with --tasks hd_epic_{category}
        group: hd_epic_{category}
        task:
        """
    )
    return header + subtasks + "\nmetadata:\n  version: 1.0\n"


def master_group_yaml() -> str:
    """The top-level group: union of all 7 category groups."""
    cats = "\n".join(f"  - hd_epic_{c}" for c in CATEGORIES.keys())
    header = textwrap.dedent(
        """\
        # HD-EPIC -- top-level group covering all 30 prototypes.
        # Use --tasks hd_epic to evaluate the full benchmark.
        # The 7 category sub-groups are also runnable individually,
        # e.g. --tasks hd_epic_recipe, --tasks hd_epic_gaze.
        group: hd_epic
        task:
        """
    )
    return header + cats + "\nmetadata:\n  version: 1.0\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Per-prototype YAMLs (30 files)
    for tt in TASK_TYPES:
        fn = os.path.join(args.output_dir, f"hd_epic_{tt}.yaml")
        with open(fn, "w") as f:
            f.write(yaml_for_task(tt))

    # Per-category group YAMLs (7 files)
    for cat, protos in CATEGORIES.items():
        fn = os.path.join(args.output_dir, f"_group_hd_epic_{cat}.yaml")
        with open(fn, "w") as f:
            f.write(category_group_yaml(cat, protos))

    # Master group YAML (1 file)
    fn = os.path.join(args.output_dir, "_group_hd_epic.yaml")
    with open(fn, "w") as f:
        f.write(master_group_yaml())

    print(f"Generated:\n" f"  - {len(TASK_TYPES)} per-prototype YAMLs\n" f"  - {len(CATEGORIES)} category-group YAMLs\n" f"  - 1 master group YAML (hd_epic)")


if __name__ == "__main__":
    main()
