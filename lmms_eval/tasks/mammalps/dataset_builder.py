#!/usr/bin/env python3
"""
A configurable script to create datasets from _cot.jsonl files for any video dataset.
Supports multiple tasks and flexible configuration.
"""

import json
import re
import ast
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset creation"""

    name: str
    base_dir: str
    tasks: List[str]
    output_prefix: str
    include_conversations: bool = False
    file_pattern: str = "{task}_recognition_cot.jsonl"
    distribute_clips: bool = True  # Whether to copy video clips to output folders

    def get_task_file(self, split: str, task: str) -> Path:
        """Get the file path for a specific task and split"""
        filename = self.file_pattern.format(task=task)
        return Path(self.base_dir) / split / filename

    def get_clips_dir(self) -> Path:
        """Get the source clips directory"""
        return Path(self.base_dir) / "clips"


# Predefined configurations for known datasets
DATASET_CONFIGS = {
    "animalkingdom": DatasetConfig(name="AnimalKingdom", base_dir="AnimalKingdom_videos_annotations", tasks=["action", "activity", "animal"], output_prefix="animalkingdom", include_conversations=False, distribute_clips=True),
    "mammalnet": DatasetConfig(name="MammalNet", base_dir="MammalNet_videos_annotations", tasks=["action", "animal"], output_prefix="mammalnet", include_conversations=False, distribute_clips=False),
    "mammalps": DatasetConfig(name="MammAlps", base_dir="MammAlps_videos_annotations", tasks=["animal", "action", "activity"], output_prefix="mammalps", include_conversations=False, distribute_clips=True),
}


def extract_entities_from_gpt_response(response: str) -> List[str]:
    """
    Extract ground truth labels from GPT response.
    Looks for 'Final answer: [...]' pattern and parses the list.
    """
    # Look for "Final answer:" followed by a list
    final_answer_pattern = r"Final answer:\s*(\[.*?\])"
    match = re.search(final_answer_pattern, response, re.IGNORECASE | re.DOTALL)

    if match:
        try:
            # Use ast.literal_eval for safe evaluation of Python list literals
            list_str = match.group(1)
            entities = ast.literal_eval(list_str)

            # Ensure it's a list of strings
            if isinstance(entities, list):
                return [str(entity).strip() for entity in entities if entity]
            else:
                return [str(entities)] if entities else []

        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse list from response: {list_str}")
            return []

    # Fallback: try eval for backwards compatibility
    list_pattern = r"\[([^\]]+)\]"
    matches = re.findall(list_pattern, response)

    for match in matches:
        try:
            if "'" in match or '"' in match:
                entities = eval(f"[{match}]")
                if isinstance(entities, list):
                    return [str(entity).strip() for entity in entities]
        except:
            continue

    return []


def extract_prompts_from_conversations(conversations: List[Dict]) -> str:
    """Extract the human prompt from conversations."""
    for conv in conversations:
        if conv.get("from") == "human":
            return conv.get("value", "")
    return ""


def add_task_example_to_prompt(prompt: str, task_type: str) -> str:
    """Add the required example format to the prompt based on task type."""

    # Task-specific examples to append
    task_examples = {
        "action": "\n\nYour answer should follow the example below:\nstep 1\nactions = recognize(entity_type='action')\noutput:List[str]: ['eating', 'attending']\n\nstep 2\nreturn actions\noutput:Final answer: ['eating', 'attending']",
        "activity": "\n\nYour answer should follow the example below:\nstep 1\nactivities = recognize(entity_type='activity')\noutput:List[str]: ['foraging']\n\nstep 2\nreturn activities\noutput:Final answer: ['foraging']",
        "animal": "\n\nYour answer should follow the example below:\nstep 1\nanimals = recognize(entity_type='animal')\noutput:List[str]: ['common crane']\n\nstep 2\nreturn animals\noutput:Final answer: ['common crane']",
    }

    # Append the example at the very end of the prompt
    modified_prompt = prompt + task_examples.get(task_type, "")

    return modified_prompt


def distribute_video_clips(dataset: List[Dict], config: DatasetConfig, split: str, output_dir: Path):
    """Distribute video clips from source to output clips directory."""
    if not config.distribute_clips:
        return

    source_clips_dir = config.get_clips_dir()
    if not source_clips_dir.exists():
        print(f"Warning: Source clips directory not found: {source_clips_dir}")
        return

    target_clips_dir = output_dir / "clips"
    target_clips_dir.mkdir(exist_ok=True)

    copied_count = 0
    missing_count = 0

    for record in dataset:
        clip_path = record["clip"]

        if config.name == "MammAlps":
            # For MammalAlps, the clip path is already the full relative path
            # e.g., "clips/S1_C3_E154_V0066_ID1_T1/S1_C3_E154_V0066_ID1_T1_c0.mp4"
            source_file = config.get_clips_dir().parent / clip_path
        else:
            # For other datasets, try different approaches
            video_id = record["video_id"]
            video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
            source_file = None

            # First, try direct file in clips directory
            for ext in video_extensions:
                potential_source = source_clips_dir / f"{video_id}{ext}"
                if potential_source.exists():
                    source_file = potential_source
                    break

            # If not found, try looking in subdirectories
            if not source_file:
                for subdir in source_clips_dir.iterdir():
                    if subdir.is_dir() and video_id in subdir.name:
                        for ext in video_extensions:
                            potential_files = list(subdir.glob(f"*{ext}"))
                            if potential_files:
                                source_file = potential_files[0]
                                break
                        if source_file:
                            break

            # If still not found, try a broader search
            if not source_file:
                for ext in video_extensions:
                    potential_files = list(source_clips_dir.rglob(f"*{video_id}*{ext}"))
                    if potential_files:
                        source_file = potential_files[0]
                        break

        if source_file and source_file.exists():
            # For MammalAlps, preserve directory structure
            if config.name == "MammAlps":
                # Extract the relative path from clips/ onwards
                rel_path = clip_path
                if rel_path.startswith("clips/"):
                    rel_path = rel_path[6:]  # Remove "clips/" prefix

                target_file = target_clips_dir / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                target_file = target_clips_dir / source_file.name

            if not target_file.exists():
                try:
                    shutil.copy2(source_file, target_file)
                    copied_count += 1
                except Exception as e:
                    print(f"Warning: Could not copy {source_file} to {target_file}: {e}")
        else:
            missing_count += 1

    if copied_count > 0:
        print(f"Copied {copied_count} video clips to {target_clips_dir}")
    if missing_count > 0:
        print(f"Warning: {missing_count} video clips not found in source directory")


def process_jsonl_file(file_path: Path) -> List[Dict]:
    """Process a JSONL file and return records with extracted data."""
    records = []

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return records

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Extract basic info
                record = {"id": data.get("id"), "video": data.get("video", ""), "conversations": data.get("conversations", [])}

                # Extract prompt and answer
                prompt = extract_prompts_from_conversations(record["conversations"])

                # Extract answer from GPT response
                gpt_response = ""
                for conv in record["conversations"]:
                    if conv.get("from") == "gpt":
                        gpt_response = conv.get("value", "")
                        break

                answer = extract_entities_from_gpt_response(gpt_response)

                record["prompt"] = prompt
                record["answer"] = answer
                records.append(record)

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num} in {file_path}: {e}")
                continue

    return records


def create_dataset(config: DatasetConfig, split: str = "test") -> List[Dict]:
    """Create dataset for specified configuration and split."""

    print(f"Creating {config.name} {split} dataset...")
    print(f"Tasks: {config.tasks}")

    # Process each task file
    all_records = {}

    for task_name in config.tasks:
        file_path = config.get_task_file(split, task_name)
        print(f"Processing {task_name} task from {file_path}")

        records = process_jsonl_file(file_path)

        for record in records:
            video_id = record["id"]
            video_path = record["video"]

            # Handle different video path formats
            if config.name == "MammAlps":
                # For MammalAlps, video path is already relative to base directory
                # e.g., "benchmark_1/clips/S1_C3_E154_V0066_ID1_T1/S1_C3_E154_V0066_ID1_T1_c0.mp4"
                # We want to extract just the clips part: "clips/S1_C3_E154_V0066_ID1_T1/S1_C3_E154_V0066_ID1_T1_c0.mp4"
                if "clips/" in video_path:
                    clip_path = video_path[video_path.find("clips/") :]
                    video_filename = Path(clip_path).name
                    video_id_from_path = Path(video_filename).stem
                else:
                    clip_path = f"clips/{Path(video_path).name}"
                    video_filename = Path(video_path).name
                    video_id_from_path = Path(video_filename).stem
            else:
                # For other datasets, extract filename only
                video_filename = Path(video_path).name
                clip_path = f"clips/{video_filename}"
                video_id_from_path = Path(video_filename).stem

            if video_id not in all_records:
                all_records[video_id] = {
                    "id": video_id,
                    "clip": clip_path,
                    "video_id": video_id_from_path,
                }

                # Include conversations if specified in config
                if config.include_conversations:
                    all_records[video_id]["conversations"] = record["conversations"]

            # Add task-specific data
            all_records[video_id][task_name] = {"prompt": add_task_example_to_prompt(record["prompt"], task_name), "answer": record["answer"]}

    # Convert to list and filter complete records
    dataset = []
    for video_id, record in all_records.items():
        # Check which tasks are present
        present_tasks = [task for task in config.tasks if task in record]

        if present_tasks:  # Include record if it has at least one task
            dataset.append(record)
        else:
            print(f"Warning: Record {video_id} has no valid tasks")

    # Sort by ID for consistency
    dataset.sort(key=lambda x: x["id"])

    print(f"Created {split} dataset with {len(dataset)} records")

    # Verify task coverage
    task_counts = defaultdict(int)
    for record in dataset:
        for task in config.tasks:
            if task in record:
                task_counts[task] += 1

    print(f"Task coverage: {dict(task_counts)}")

    return dataset


def save_dataset(dataset: List[Dict], config: DatasetConfig, split: str):
    """Save dataset with README and directory structure."""

    # Create output directory
    output_dir = Path(f"{config.name}_HF_Dataset_{split.title()}")
    output_dir.mkdir(exist_ok=True)

    # Create clips directory
    (output_dir / "clips").mkdir(exist_ok=True)

    # Distribute video clips if enabled
    distribute_video_clips(dataset, config, split, output_dir)

    # Save dataset
    output_file = output_dir / f"{config.output_prefix}_{split}_dataset.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"{split.title()} dataset saved to {output_file}")

    # Create README
    readme_content = generate_readme(dataset, config, split)
    readme_file = output_dir / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    print(f"README saved to {readme_file}")


def save_unified_dataset(test_dataset: List[Dict], train_dataset: List[Dict], config: DatasetConfig):
    """Save unified dataset with both train and test splits in one directory."""

    # Create unified output directory
    output_dir = Path(f"{config.name}_HF_Dataset_Unified")
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories for each split
    test_dir = output_dir / "test"
    train_dir = output_dir / "train"
    test_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)

    # Create clips directories
    (test_dir / "clips").mkdir(exist_ok=True)
    (train_dir / "clips").mkdir(exist_ok=True)

    # Distribute video clips for each split
    distribute_video_clips(test_dataset, config, "test", test_dir)
    distribute_video_clips(train_dataset, config, "train", train_dir)

    # Save datasets
    test_file = test_dir / f"{config.output_prefix}_test_dataset.json"
    train_file = train_dir / f"{config.output_prefix}_train_dataset.json"

    with open(test_file, "w") as f:
        json.dump(test_dataset, f, indent=2)

    with open(train_file, "w") as f:
        json.dump(train_dataset, f, indent=2)

    print(f"Test dataset saved to {test_file}")
    print(f"Train dataset saved to {train_file}")

    # Create unified README
    readme_content = generate_unified_readme(test_dataset, train_dataset, config)
    readme_file = output_dir / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    print(f"Unified README saved to {readme_file}")
    print(f"Unified dataset created at {output_dir}")


def generate_unified_readme(test_dataset: List[Dict], train_dataset: List[Dict], config: DatasetConfig) -> str:
    """Generate README content for unified dataset."""

    # Calculate task statistics for both splits
    test_task_counts = defaultdict(int)
    train_task_counts = defaultdict(int)

    for record in test_dataset:
        for task in config.tasks:
            if task in record:
                test_task_counts[task] += 1

    for record in train_dataset:
        for task in config.tasks:
            if task in record:
                train_task_counts[task] += 1

    total_count = len(test_dataset) + len(train_dataset)
    task_list = ", ".join([f"{task} recognition" for task in config.tasks])

    return f"""# {config.name} Dataset

## Dataset Description

This dataset contains video clips and annotations for {task_list} tasks in animal videos.

## Dataset Statistics

- **Total Records**: {total_count:,}
- **Train Split**: {len(train_dataset):,} records
- **Test Split**: {len(test_dataset):,} records
- **Tasks**: {task_list}
- **Format**: Task-centric with original prompts and List[str] answers

## Dataset Structure

```
{config.output_prefix}/
├── train/
│   ├── {config.output_prefix}_train_dataset.json
│   └── clips/
│       ├── video1.mp4
│       └── ...
└── test/
    ├── {config.output_prefix}_test_dataset.json
    └── clips/
        ├── video2.mp4
        └── ...
```

## Task Coverage

### Test Split
{chr(10).join([f"- {task}: {test_task_counts[task]} records" for task in config.tasks])}

### Train Split
{chr(10).join([f"- {task}: {train_task_counts[task]} records" for task in config.tasks])}

## Record Format

```json
{{
  "id": "original_id",
  "clip": "clips/video.mp4",
  "video_id": "video_identifier",
  {chr(10).join([f'  "{task}": {{"prompt": "...", "answer": ["..."]}}' for task in config.tasks])}
}}
```

## Tasks

{chr(10).join([f"- **{task.title()} Recognition**: Identify {task}s in animal videos" for task in config.tasks])}

"""


def save_unified_dataset_single_clips(test_dataset: List[Dict], train_dataset: List[Dict], config: DatasetConfig):
    """Save unified dataset with both train and test splits using a single clips folder."""

    # Create unified output directory
    output_dir = Path(f"{config.name}_HF_Dataset_Unified")
    output_dir.mkdir(exist_ok=True)

    # Create single clips directory for all videos
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    # Track copied clips to avoid duplicates
    copied_clips = set()

    # Get existing dataset directories to copy clips from
    test_source_dir = Path(f"{config.name}_HF_Dataset_Test")
    train_source_dir = Path(f"{config.name}_HF_Dataset_Train")

    # Check if separate datasets exist, if not copy directly from source
    if test_source_dir.exists() and train_source_dir.exists():
        print("Using clips from existing separate datasets...")

        # Copy clips from test dataset
        copied_count = 0
        if test_dataset:
            test_clips_source = test_source_dir / "clips"
            if test_clips_source.exists():
                for clip_file in test_clips_source.rglob("*"):
                    if clip_file.is_file() and clip_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                        # Preserve directory structure relative to clips directory
                        relative_path = clip_file.relative_to(test_clips_source)
                        dest_file = clips_dir / relative_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)

                        # Copy if not already copied
                        if str(relative_path) not in copied_clips:
                            shutil.copy2(clip_file, dest_file)
                            copied_clips.add(str(relative_path))
                            copied_count += 1
                print(f"Copied {len(list(test_clips_source.rglob('*.mp4')))} test clips")

        # Copy clips from train dataset (skip duplicates)
        if train_dataset:
            train_clips_source = train_source_dir / "clips"
            if train_clips_source.exists():
                train_count = 0
                for clip_file in train_clips_source.rglob("*"):
                    if clip_file.is_file() and clip_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                        # Preserve directory structure relative to clips directory
                        relative_path = clip_file.relative_to(train_clips_source)
                        dest_file = clips_dir / relative_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)

                        # Copy if not already copied
                        if str(relative_path) not in copied_clips:
                            shutil.copy2(clip_file, dest_file)
                            copied_clips.add(str(relative_path))
                            train_count += 1
                print(f"Copied {train_count} new train clips")
    else:
        print("Separate datasets not found, copying directly from source...")

        # Copy clips for each dataset separately to maintain proper structure
        copied_count = 0

        if test_dataset:
            print("Copying test clips...")
            distribute_video_clips(test_dataset, config, "test", output_dir)
            copied_count += len(test_dataset)

        if train_dataset:
            print("Copying train clips...")
            distribute_video_clips(train_dataset, config, "train", output_dir)
            copied_count += len(train_dataset)

        # Count actual copied clips
        actual_clips = len(list(clips_dir.rglob("*.mp4")))
        print(f"Successfully copied {actual_clips} clips directly from source")

    print(f"Total clips in unified dataset: {len(list(clips_dir.rglob('*.mp4')))}")

    # The datasets already have correct clip paths pointing to clips/ directory
    # No need to update paths since they're already correct

    # Save JSON files at root level
    if test_dataset:
        test_file = output_dir / f"{config.output_prefix}_test_dataset.json"
        with open(test_file, "w") as f:
            json.dump(test_dataset, f, indent=2)
        print(f"Test dataset saved to {test_file}")

    if train_dataset:
        train_file = output_dir / f"{config.output_prefix}_train_dataset.json"
        with open(train_file, "w") as f:
            json.dump(train_dataset, f, indent=2)
        print(f"Train dataset saved to {train_file}")

    # Create unified README
    readme_content = generate_unified_single_clips_readme(test_dataset, train_dataset, config)
    readme_file = output_dir / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    print(f"Unified README saved to {readme_file}")
    print(f"Unified dataset with single clips folder created at {output_dir}")


def generate_unified_single_clips_readme(test_dataset: List[Dict], train_dataset: List[Dict], config: DatasetConfig) -> str:
    """Generate README content for unified dataset with single clips folder."""

    # Calculate task statistics for both splits
    test_task_counts = defaultdict(int)
    train_task_counts = defaultdict(int)

    if test_dataset:
        for record in test_dataset:
            for task in config.tasks:
                if task in record:
                    test_task_counts[task] += 1

    if train_dataset:
        for record in train_dataset:
            for task in config.tasks:
                if task in record:
                    train_task_counts[task] += 1

    total_count = len(test_dataset or []) + len(train_dataset or [])
    task_list = ", ".join([f"{task} recognition" for task in config.tasks])

    return f"""# {config.name} Dataset

## Dataset Description

This dataset contains video clips and annotations for {task_list} tasks in animal videos.

## Dataset Statistics

- **Total Records**: {total_count:,}
- **Train Split**: {len(train_dataset or []):,} records
- **Test Split**: {len(test_dataset or []):,} records
- **Tasks**: {task_list}
- **Format**: Task-centric with original prompts and List[str] answers

## Dataset Structure

```
{config.output_prefix}/
├── {config.output_prefix}_train_dataset.json
├── {config.output_prefix}_test_dataset.json
├── clips/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── README.md
```

## Task Coverage

### Test Split
{chr(10).join([f"- {task}: {test_task_counts[task]} records" for task in config.tasks]) if test_dataset else "No test data"}

### Train Split
{chr(10).join([f"- {task}: {train_task_counts[task]} records" for task in config.tasks]) if train_dataset else "No train data"}

## Record Format

```json
{{
  "id": "original_id",
  "clip": "clips/video.mp4",
  "video_id": "video_identifier",
  {chr(10).join([f'  "{task}": {{"prompt": "...", "answer": ["..."]}}' for task in config.tasks])}
}}
```

## Tasks

{chr(10).join([f"- **{task.title()} Recognition**: Identify {task}s in animal videos" for task in config.tasks])}

"""


def _get_size_category(count: int) -> str:
    """Get size category for dataset."""
    if count < 1000:
        return "n<1K"
    elif count < 10000:
        return "1K<n<10K"
    elif count < 100000:
        return "10K<n<100K"
    elif count < 1000000:
        return "100K<n<1M"
    else:
        return "n>1M"


def generate_readme(dataset: List[Dict], config: DatasetConfig, split: str) -> str:
    """Generate README content for the dataset."""

    # Calculate task statistics
    task_counts = defaultdict(int)
    for record in dataset:
        for task in config.tasks:
            if task in record:
                task_counts[task] += 1

    # Task list for documentation
    task_list = ", ".join([f"{task} recognition" for task in config.tasks])

    return f"""# {config.name} {split.title()} Dataset

## Dataset Information
- Records: {len(dataset)}
- Tasks: {task_list}
- Format: Task-centric with original prompts and List[str] answers

## Task Coverage
{chr(10).join([f"- {task}: {task_counts[task]} records" for task in config.tasks])}

## Structure
Each record contains:
- id: Original ID from source data
- clip: Path to video clip (to be populated)
- video_id: Video identifier
{chr(10).join([f"- {task}: {task.title()} recognition task with prompt and answer" for task in config.tasks])}
{"- conversations: Original conversation data for evaluation compatibility" if config.include_conversations else ""}

## Source
Created from {config.name} conversation data (_cot.jsonl files) with ground truth labels extracted from GPT responses.
"""


def main():
    """Main function with command-line argument support."""

    parser = argparse.ArgumentParser(description="Universal Dataset Builder")
    parser.add_argument("--dataset", "-d", choices=list(DATASET_CONFIGS.keys()) + ["all"], default="all", help="Dataset to build (default: all)")
    parser.add_argument("--split", "-s", choices=["test", "train", "both"], default="both", help="Split to build (default: both)")
    parser.add_argument("--unified", "-u", action="store_true", help="Create unified dataset with both splits in one directory")

    args = parser.parse_args()

    print("Universal Dataset Builder")
    print("=" * 50)

    # Determine which datasets to build
    if args.dataset == "all":
        datasets_to_build = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_build = [args.dataset]

    # List available datasets
    print("Available datasets:")
    for key, config in DATASET_CONFIGS.items():
        print(f"  {key}: {config.name} ({', '.join(config.tasks)})")

    print(f"\nBuilding datasets: {datasets_to_build}")
    print(f"Splits: {args.split}")
    print(f"Unified: {args.unified}")

    for dataset_key in datasets_to_build:
        config = DATASET_CONFIGS[dataset_key]
        print(f"\n{'=' * 60}")
        print(f"Building {config.name} Dataset")
        print(f"{'=' * 60}")

        test_dataset = None
        train_dataset = None

        # Create datasets based on split argument
        if args.split in ["test", "both"]:
            print(f"\n--- {config.name} Test Dataset ---")
            test_dataset = create_dataset(config, "test")

        if args.split in ["train", "both"]:
            print(f"\n--- {config.name} Train Dataset ---")
            train_dataset = create_dataset(config, "train")

        # Save datasets
        if args.unified and test_dataset is not None and train_dataset is not None:
            # Save unified dataset with single clips folder
            save_unified_dataset_single_clips(test_dataset, train_dataset, config)
        else:
            # Save separate datasets
            if test_dataset is not None:
                save_dataset(test_dataset, config, "test")
            if train_dataset is not None:
                save_dataset(train_dataset, config, "train")

        # Print summary
        print(f"\n{config.name} datasets complete!")
        if test_dataset:
            print(f"   Test: {len(test_dataset)} records")
        if train_dataset:
            print(f"   Train: {len(train_dataset)} records")


def build_custom_dataset(name: str, base_dir: str, tasks: List[str], include_conversations: bool = False, output_prefix: Optional[str] = None, distribute_clips: bool = True):
    """Build a custom dataset with specified parameters."""

    config = DatasetConfig(name=name, base_dir=base_dir, tasks=tasks, output_prefix=output_prefix or name.lower(), include_conversations=include_conversations, distribute_clips=distribute_clips)

    print(f"Building custom dataset: {name}")

    # Create test dataset
    test_dataset = create_dataset(config, "test")
    save_dataset(test_dataset, config, "test")

    # Create train dataset
    train_dataset = create_dataset(config, "train")
    save_dataset(train_dataset, config, "train")

    return test_dataset, train_dataset


if __name__ == "__main__":
    main()
