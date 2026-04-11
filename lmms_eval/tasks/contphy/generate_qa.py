"""Generate ContPhy QA JSON from raw dataset zips.

Downloads (or reads from local cache) the ContPhy zip files from HuggingFace
and produces a single JSONL file with multiple-choice QA pairs suitable for
lmms-eval ingestion.

Usage:
    python -m lmms_eval.tasks.contphy.generate_qa --output /path/to/contphy_qa.json

Set CONTPHY_DATA_DIR to skip download if you already have the zips extracted.
"""

import argparse
import json
import os
import random
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Scenario -> zip name mapping
# ---------------------------------------------------------------------------
SCENARIO_ZIPS = {
    "fluid": "fluid_full.zip",
    "rope": "rope_full.zip",
    "cloth": "cloth_full.zip",
    "ball": "ball_full.zip",
}

SCENARIO_DIRS = {
    "fluid": "fluid_slides",
    "rope": "pulley_group",
    "cloth": "cloth_collision",
    "ball": "soft_body",
}

HF_BASE = "https://huggingface.co/datasets/zzcnewly/ContPhy_Dataset/resolve/main"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _color_name(name: str) -> str:
    """Extract color from object name like 'Blue Fluid' -> 'blue'."""
    parts = name.split()
    if len(parts) >= 2:
        return parts[0].lower()
    return name.lower()


def _compare_word(greater: bool) -> str:
    return "greater" if greater else "less"


# ---------------------------------------------------------------------------
# Question generators per scenario
# ---------------------------------------------------------------------------
def generate_fluid_questions(data: dict, video_id: str) -> list[dict]:
    """Generate MC questions for the fluid scenario."""
    questions = []
    fluids = data.get("metaSamplingData", {}).get("fluids", [])
    sticks = data.get("metaSamplingData", {}).get("sticks", [])
    tracking = data.get("trackingData", {})
    receptor_stats = tracking.get("perReceptorFluidStat", {})
    cf_annotations = data.get("CounterFactualAnnotations", {})

    # Filter to fluids without 'Later Emitted' prefix for cleaner questions
    named_fluids = [f for f in fluids if not f["name"].startswith("Later")]

    # --- Property: Density comparison ---
    if len(named_fluids) >= 2:
        for i in range(len(named_fluids)):
            for j in range(i + 1, len(named_fluids)):
                f1, f2 = named_fluids[i], named_fluids[j]
                gt_greater = f1["density"] > f2["density"]
                q = f"Is the density of the {f1['name'].lower()} " f"{_compare_word(True)} than that of the {f2['name'].lower()}?"
                answer = "Yes" if gt_greater else "No"
                questions.append(
                    {
                        "question": q,
                        "options": ["Yes", "No", "Cannot Judge"],
                        "answer": answer,
                        "question_type": "property",
                        "question_class": "density",
                        "scenario": "fluid",
                        "video_id": video_id,
                    }
                )

    # --- Property: Stick count ---
    if sticks:
        q = "How many sticks are there in the video?"
        n = len(sticks)
        options = sorted(set([str(n), str(max(1, n - 1)), str(n + 1), str(n + 2)]))
        questions.append(
            {
                "question": q,
                "options": options,
                "answer": str(n),
                "question_type": "property",
                "question_class": "stick_number",
                "scenario": "fluid",
                "video_id": video_id,
            }
        )

    # --- Predictive: Which container will fluid flow into ---
    for container_name, container_fluids in receptor_stats.items():
        for fluid_name, amount in container_fluids.items():
            if fluid_name.startswith("Later"):
                continue
            if amount > 0:
                other_containers = [c for c in receptor_stats if c != container_name]
                if other_containers:
                    q = f"Which container will {fluid_name.lower()} flow into?"
                    options = [container_name] + other_containers[:2]
                    if len(options) < 3:
                        options.append("None of the above")
                    random.shuffle(options)
                    questions.append(
                        {
                            "question": q,
                            "options": options,
                            "answer": container_name,
                            "question_type": "predictive",
                            "question_class": "container",
                            "scenario": "fluid",
                            "video_id": video_id,
                        }
                    )
                    break  # One per container
        else:
            continue
        break  # One total

    # --- Counterfactual: If stick removed, where would fluid go ---
    for cf_key, cf_data in list(cf_annotations.items())[:1]:
        removed = cf_data.get("removedStickName", "")
        if not removed:
            continue
        cf_receptors = cf_data.get("trackingData", {}).get("perReceptorFluidStat", {})
        # Find a fluid that changed destination
        for fluid_name in [f["name"] for f in named_fluids]:
            orig_containers = {c for c, fs in receptor_stats.items() if fluid_name in fs and fs[fluid_name] > 0}
            cf_containers = {c for c, fs in cf_receptors.items() if fluid_name in fs and fs[fluid_name] > 0}
            new_containers = cf_containers - orig_containers
            if new_containers:
                answer = list(new_containers)[0]
                all_containers = list(set(list(receptor_stats.keys()) + list(cf_receptors.keys())))
                options = [answer] + [c for c in all_containers if c != answer][:2]
                if len(options) < 3:
                    options.append("None of the above")
                random.shuffle(options)
                q = f"If the {removed.lower()} were removed, which container would " f"{fluid_name.lower()} flow into?"
                questions.append(
                    {
                        "question": q,
                        "options": options,
                        "answer": answer,
                        "question_type": "counterfactual",
                        "question_class": "container",
                        "scenario": "fluid",
                        "video_id": video_id,
                    }
                )
                break

    return questions


def generate_rope_questions(data: dict, video_id: str) -> list[dict]:
    """Generate MC questions for the rope scenario."""
    questions = []
    masses = data.get("outputMass", {})
    rotations = data.get("ResultRotation", {})
    tension_avg = data.get("ResultTensionAvg", {})
    meta = data.get("metaSamplingData", {})
    cf_annotations = data.get("CounterFactualAnnotations", {})

    objects_with_mass = list(masses.keys())

    # --- Property: Mass comparison ---
    if len(objects_with_mass) >= 2:
        for i in range(len(objects_with_mass)):
            for j in range(i + 1, len(objects_with_mass)):
                o1, o2 = objects_with_mass[i], objects_with_mass[j]
                m1, m2 = masses[o1], masses[o2]
                gt_greater = m1 > m2 * 0.5
                q = f"Is the mass of the {o1.lower()} greater than " f"half that of the {o2.lower()}?"
                answer = "Yes" if gt_greater else "No"
                questions.append(
                    {
                        "question": q,
                        "options": ["Yes", "No", "Cannot Judge"],
                        "answer": answer,
                        "question_type": "property",
                        "question_class": "mass",
                        "scenario": "rope",
                        "video_id": video_id,
                    }
                )

    # --- Property: Tension comparison ---
    rope_names = list(tension_avg.keys())
    if len(rope_names) >= 2:
        r1, r2 = rope_names[0], rope_names[1]
        t1, t2 = abs(tension_avg[r1]), abs(tension_avg[r2])
        gt_greater = t1 > t2
        q = f"Is the tension of the {r1.lower()} greater than " f"that of the {r2.lower()}?"
        answer = "Yes" if gt_greater else "No"
        questions.append(
            {
                "question": q,
                "options": ["Yes", "No", "Cannot Judge"],
                "answer": answer,
                "question_type": "property",
                "question_class": "tension",
                "scenario": "rope",
                "video_id": video_id,
            }
        )

    # --- Counterfactual: If object were heavier, which direction ---
    for cf_key, cf_data in list(cf_annotations.items())[:1]:
        cf_rotations = cf_data.get("ResultRotation", {})
        # Find a pulley that changed rotation direction
        for pulley_name, orig_rot in rotations.items():
            cf_rot = cf_rotations.get(pulley_name, orig_rot)
            if cf_rot != orig_rot and cf_rot != 0:
                direction = "clockwise" if cf_rot > 0 else "anti-clockwise"
                # Pick a changed mass object
                changed_obj = cf_data.get("changedMassObjectName", "")
                if not changed_obj and objects_with_mass:
                    changed_obj = objects_with_mass[0]
                q = f"If the {changed_obj.lower()} were far much heavier, " f"which direction would the {pulley_name.lower()} rotate?"
                questions.append(
                    {
                        "question": q,
                        "options": ["Clockwise", "Anti-clockwise", "No rotation"],
                        "answer": direction.capitalize(),
                        "question_type": "counterfactual",
                        "question_class": "rotation",
                        "scenario": "rope",
                        "video_id": video_id,
                    }
                )
                break

    # --- Property: Object counting (shapes/colors) ---
    name2pos = meta.get("name2position", {})
    if name2pos:
        # Count objects by type
        type_counts = {}
        for obj_name in name2pos:
            for obj_type in ["Cube", "Sphere", "Pulley"]:
                if obj_type in obj_name:
                    type_counts[obj_type.lower()] = type_counts.get(obj_type.lower(), 0) + 1
        for obj_type, count in type_counts.items():
            plural = obj_type + "s" if count != 1 else obj_type
            q = f"How many {plural} are there in the video?"
            options = sorted(set([str(count), str(max(1, count - 1)), str(count + 1), str(count + 2)]))
            questions.append(
                {
                    "question": q,
                    "options": options,
                    "answer": str(count),
                    "question_type": "property",
                    "question_class": "shape",
                    "scenario": "rope",
                    "video_id": video_id,
                }
            )

    return questions


def generate_cloth_questions(data: dict, video_id: str) -> list[dict]:
    """Generate MC questions for the cloth scenario."""
    questions = []
    cloth_left = data.get("clothLeft", {})
    cloth_right = data.get("clothRight", {})
    ofa = data.get("objectFullAnnotation", {})

    # --- Property: Elasticity (stretching compliance) ---
    sc_l = cloth_left.get("stretchingCompliance", 0)
    sc_r = cloth_right.get("stretchingCompliance", 0)
    # Higher compliance = easier to stretch = more elastic
    gt_easier = sc_l > sc_r
    q = "Is the left cloth much easier to stretch than the other?"
    answer = "Yes" if gt_easier else "No"
    questions.append(
        {
            "question": q,
            "options": ["Yes", "No"],
            "answer": answer,
            "question_type": "property",
            "question_class": "elasticity",
            "scenario": "cloth",
            "video_id": video_id,
        }
    )

    # --- Property: Bending ---
    bc_l = cloth_left.get("bendingCompliance", 0)
    bc_r = cloth_right.get("bendingCompliance", 0)
    # Lower compliance = harder to bend
    gt_harder = bc_l < bc_r
    q = "Is the left cloth much harder to bend or have wrinkles " "than the other?"
    answer = "Yes" if gt_harder else "No"
    questions.append(
        {
            "question": q,
            "options": ["Yes", "No"],
            "answer": answer,
            "question_type": "property",
            "question_class": "bending",
            "scenario": "cloth",
            "video_id": video_id,
        }
    )

    # --- Predictive: Object fall over ---
    for side_key, side_name in [("leftAll", "left"), ("rightAll", "right")]:
        side_objs = ofa.get(side_key, {})
        isolated = side_objs.get("isolatedObjects", {})
        for obj_name, obj_data in isolated.items():
            pose = obj_data.get("endPoseDescription", "")
            if pose and pose != "Upright":
                q = f"Does the {obj_name.lower()} fall over?"
                answer = "Yes"
            else:
                q = f"Does the {obj_name.lower()} fall over?"
                answer = "No"
            questions.append(
                {
                    "question": q,
                    "options": ["Yes", "No"],
                    "answer": answer,
                    "question_type": "predictive",
                    "question_class": "fall_over",
                    "scenario": "cloth",
                    "video_id": video_id,
                }
            )
            break  # One per side
        else:
            continue
        break  # One total

    # --- Predictive: Final pose ---
    for side_key in ["leftAll", "rightAll"]:
        side_objs = ofa.get(side_key, {})
        isolated = side_objs.get("isolatedObjects", {})
        for obj_name, obj_data in isolated.items():
            pose = obj_data.get("endPoseDescription", "")
            if pose:
                q = f"Which phrase below can best describe the final pose " f"of the {obj_name.lower()}?"
                pose_options = ["Standing upright", "Leaning", "Lying horizontally"]
                if pose == "Upright":
                    answer = "Standing upright"
                elif "lean" in pose.lower() or "tilt" in pose.lower():
                    answer = "Leaning"
                else:
                    answer = "Lying horizontally"
                questions.append(
                    {
                        "question": q,
                        "options": pose_options,
                        "answer": answer,
                        "question_type": "predictive",
                        "question_class": "pose",
                        "scenario": "cloth",
                        "video_id": video_id,
                    }
                )
                break
        else:
            continue
        break

    return questions


def generate_ball_questions(data: dict, video_id: str) -> list[dict]:
    """Generate MC questions for the ball scenario."""
    questions = []
    tracking = data.get("trackingData", {})
    meta = data.get("metaSamplingData", {})
    balls = meta.get("balls", [])
    holes = meta.get("holesCenterXValue", [])
    cf_annotations = data.get("CounterFactualAnnotations", {})

    # --- Property: Elasticity comparison ---
    if len(balls) >= 2:
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                b1, b2 = balls[i], balls[j]
                e1 = b1.get("elasticityType", "")
                e2 = b2.get("elasticityType", "")
                # Elastic > Plastic > Rigid in terms of elasticity
                elasticity_rank = {"Elastic": 3, "Plastic": 2, "Rigid": 1}
                r1 = elasticity_rank.get(e1, 0)
                r2 = elasticity_rank.get(e2, 0)
                if r1 != r2:
                    gt_greater = r1 > r2
                    q = f"Is the elasticity (deformability) of the " f"{b1['name'].lower()} much greater than " f"the {b2['name'].lower()}?"
                    answer = "Yes" if gt_greater else "No"
                    questions.append(
                        {
                            "question": q,
                            "options": ["Yes", "No"],
                            "answer": answer,
                            "question_type": "property",
                            "question_class": "elasticity",
                            "scenario": "ball",
                            "video_id": video_id,
                        }
                    )

    # --- Property: Plasticity comparison ---
    if len(balls) >= 2:
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                b1, b2 = balls[i], balls[j]
                e1 = b1.get("elasticityType", "")
                e2 = b2.get("elasticityType", "")
                # Plastic has highest plasticity
                plasticity_rank = {"Plastic": 3, "Rigid": 2, "Elastic": 1}
                r1 = plasticity_rank.get(e1, 0)
                r2 = plasticity_rank.get(e2, 0)
                if r1 != r2:
                    gt_greater = r1 > r2
                    q = f"Is the plasticity of the " f"{b1['name'].lower()} much greater than " f"the {b2['name'].lower()}?"
                    answer = "Yes" if gt_greater else "No"
                    questions.append(
                        {
                            "question": q,
                            "options": ["Yes", "No"],
                            "answer": answer,
                            "question_type": "property",
                            "question_class": "plasticity",
                            "scenario": "ball",
                            "video_id": video_id,
                        }
                    )
                    break
            else:
                continue
            break

    # --- Predictive: Final drop (which pit) ---
    for ball_name, ball_tracking in tracking.items():
        pit_result = ball_tracking.get("pitResult", "")
        if "Left" in pit_result:
            answer = "The left pit"
        elif "Right" in pit_result:
            answer = "The right pit"
        elif "No Pit" in pit_result:
            answer = "None of the above"
        else:
            continue

        q = f"Will the {ball_name.lower()} finally drop into the left pit or the right pit?"
        options = ["The left pit", "The right pit", "None of the above"]
        questions.append(
            {
                "question": q,
                "options": options,
                "answer": answer,
                "question_type": "predictive",
                "question_class": "final_drop",
                "scenario": "ball",
                "video_id": video_id,
            }
        )
        break  # One per video

    # --- Counterfactual: Remove wall, which pit ---
    for cf_key, cf_data in list(cf_annotations.items())[:1]:
        removed_wall = cf_data.get("removedWall", "")
        ball_name = cf_data.get("ball", "")
        cf_tracking = cf_data.get("trackingData", {})
        if not removed_wall or not ball_name:
            continue
        ball_track = cf_tracking.get(ball_name, {})
        pit_result = ball_track.get("pitResult", "")
        if "Left" in pit_result:
            answer = "The left pit"
        elif "Right" in pit_result:
            answer = "The right pit"
        else:
            answer = "None of the above"

        q = f"If we removed the {removed_wall.lower()} and other balls, " f"which pit would the {ball_name.lower()} drop into?"
        options = ["The left pit", "The right pit", "None of the above"]
        questions.append(
            {
                "question": q,
                "options": options,
                "answer": answer,
                "question_type": "counterfactual",
                "question_class": "remove",
                "scenario": "ball",
                "video_id": video_id,
            }
        )

    return questions


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------
GENERATORS = {
    "fluid": generate_fluid_questions,
    "rope": generate_rope_questions,
    "cloth": generate_cloth_questions,
    "ball": generate_ball_questions,
}


def process_scenario(scenario: str, data_dir: str) -> list[dict]:
    """Process all trials in a scenario directory."""
    scenario_dir_name = SCENARIO_DIRS[scenario]
    scenario_path = Path(data_dir) / scenario_dir_name

    if not scenario_path.exists():
        print(f"  Skipping {scenario}: {scenario_path} not found")
        return []

    generator = GENERATORS[scenario]
    all_questions = []

    trial_dirs = sorted(
        [d for d in scenario_path.iterdir() if d.is_dir()],
        key=lambda d: int(d.name) if d.name.isdigit() else 0,
    )

    for trial_dir in trial_dirs:
        outputs_path = trial_dir / "outputs.json"
        if not outputs_path.exists():
            continue

        with open(outputs_path) as f:
            data = json.load(f)

        if not data.get("validity", False):
            continue

        video_file = trial_dir / "output_Full.mp4"
        if not video_file.exists():
            continue

        video_id = f"{scenario_dir_name}/{trial_dir.name}"
        questions = generator(data, video_id)
        all_questions.extend(questions)

    return all_questions


def download_and_extract(cache_dir: str, use_mini: bool = False) -> str:
    """Download ContPhy zips from HuggingFace and extract them."""
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)

    if use_mini:
        zip_name = "contphy_mini.zip"
        url = f"{HF_BASE}/{zip_name}"
        zip_path = os.path.join(cache_dir, zip_name)
        extract_dir = os.path.join(cache_dir, "contphy_data")

        if not os.path.exists(zip_path):
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, zip_path)

        if not os.path.exists(extract_dir):
            print(f"Extracting {zip_path} ...")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

        return extract_dir
    else:
        extract_dir = os.path.join(cache_dir, "contphy_data")
        os.makedirs(extract_dir, exist_ok=True)

        for scenario, zip_name in SCENARIO_ZIPS.items():
            zip_path = os.path.join(cache_dir, zip_name)
            scenario_dir = os.path.join(extract_dir, SCENARIO_DIRS[scenario])

            if os.path.exists(scenario_dir):
                print(f"  {scenario}: already extracted")
                continue

            if not os.path.exists(zip_path):
                url = f"{HF_BASE}/{zip_name}"
                print(f"  Downloading {url} ...")
                urllib.request.urlretrieve(url, zip_path)

            print(f"  Extracting {zip_path} ...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

        return extract_dir


def main():
    parser = argparse.ArgumentParser(description="Generate ContPhy QA JSON")
    parser.add_argument("--data-dir", type=str, default="", help="Path to extracted ContPhy data. If empty, downloads from HF.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset (20 videos per scenario) instead of full")
    parser.add_argument("--cache-dir", type=str, default="", help="Cache directory for downloads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for option shuffling")
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = args.data_dir
    if not data_dir:
        cache_dir = args.cache_dir or os.path.join(
            os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")),
            "contphy",
        )
        data_dir = download_and_extract(cache_dir, use_mini=args.mini)

    all_questions = []
    for scenario in ["fluid", "rope", "cloth", "ball"]:
        print(f"Processing {scenario}...")
        questions = process_scenario(scenario, data_dir)
        print(f"  Generated {len(questions)} questions")
        all_questions.extend(questions)

    # Add index
    for i, q in enumerate(all_questions):
        q["idx"] = i

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\nTotal: {len(all_questions)} questions written to {args.output}")

    # Stats
    by_scenario = {}
    by_type = {}
    for q in all_questions:
        s = q["scenario"]
        t = q["question_type"]
        by_scenario[s] = by_scenario.get(s, 0) + 1
        by_type[t] = by_type.get(t, 0) + 1
    print("\nBy scenario:", by_scenario)
    print("By type:", by_type)


if __name__ == "__main__":
    main()
