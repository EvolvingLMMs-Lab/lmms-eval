#!/usr/bin/env python3
"""
Prepare ImgEdit dataset for lmms-eval

This script converts the ImgEdit singleturn.json format to a Hugging Face datasets
compatible format that can be loaded by lmms-eval.

Usage:
    python prepare_dataset.py \
        --json_file /path/to/singleturn.json \
        --img_root /path/to/singleturn/ \
        --output_dir /path/to/output/dataset

The output dataset will be saved in Arrow format and can be loaded with:
    datasets.load_from_disk(output_dir)
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
from tqdm import tqdm


def load_singleturn_json(json_path: str) -> dict:
    """Load the singleturn.json file"""
    with open(json_path, "r") as f:
        return json.load(f)


def convert_to_dataset_format(
    singleturn_data: dict,
    img_root: str,
    verify_images: bool = True,
) -> list:
    """
    Convert singleturn.json format to dataset format.

    Args:
        singleturn_data: Dict loaded from singleturn.json
        img_root: Root directory containing images (e.g., .../singleturn/)
        verify_images: Whether to verify that images exist

    Returns:
        List of dicts suitable for creating a HF Dataset
    """
    records = []
    missing_images = []

    for key, item in tqdm(singleturn_data.items(), desc="Converting"):
        image_id = item.get("id", "")  # e.g., "animal/000342021.jpg"
        prompt = item.get("prompt", "")
        edit_type = item.get("edit_type", "adjust")

        # Full path to original image
        image_path = os.path.join(img_root, image_id)

        if verify_images and not os.path.exists(image_path):
            missing_images.append(image_path)
            continue

        record = {
            "key": key,
            "id": image_id,
            "prompt": prompt,
            "edit_type": edit_type,
            "image_path": image_path,
        }

        # Try to load image if verification is enabled
        if verify_images:
            try:
                img = PILImage.open(image_path).convert("RGB")
                record["input_image"] = img
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")
                missing_images.append(image_path)
                continue

        records.append(record)

    if missing_images:
        print(f"\nWarning: {len(missing_images)} images not found:")
        for path in missing_images[:10]:
            print(f"  - {path}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")

    return records


def create_dataset(records: list, include_images: bool = True) -> Dataset:
    """Create a Hugging Face Dataset from records"""
    if include_images:
        # Dataset with embedded images
        features = Features(
            {
                "key": Value("string"),
                "id": Value("string"),
                "prompt": Value("string"),
                "edit_type": Value("string"),
                "image_path": Value("string"),
                "input_image": Image(),
            }
        )
    else:
        # Dataset with only paths (images loaded at runtime)
        # Remove input_image from records if present
        for record in records:
            if "input_image" in record:
                del record["input_image"]

        features = Features(
            {
                "key": Value("string"),
                "id": Value("string"),
                "prompt": Value("string"),
                "edit_type": Value("string"),
                "image_path": Value("string"),
            }
        )

    return Dataset.from_list(records, features=features)


def main():
    parser = argparse.ArgumentParser(description="Prepare ImgEdit dataset for lmms-eval")
    parser.add_argument(
        "--json_file",
        type=str,
        default="ImgEdit/Benchmark/singleturn/singleturn.json",
        help="Path to singleturn.json file",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="ImgEdit/Benchmark/singleturn",
        help="Root directory containing images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ImgEdit/imgedit_dataset",
        help="Output directory for the prepared dataset",
    )
    parser.add_argument(
        "--embed_images",
        action="store_true",
        help="Embed images in the dataset (larger file size but faster loading)",
    )
    parser.add_argument(
        "--skip_verification",
        action="store_true",
        help="Skip image verification (useful if images are not yet downloaded)",
    )
    args = parser.parse_args()

    print(f"Loading singleturn.json from: {args.json_file}")
    singleturn_data = load_singleturn_json(args.json_file)
    print(f"Found {len(singleturn_data)} entries")

    print(f"\nConverting to dataset format...")
    print(f"Image root: {args.img_root}")
    records = convert_to_dataset_format(
        singleturn_data,
        args.img_root,
        verify_images=not args.skip_verification,
    )
    print(f"Converted {len(records)} valid entries")

    print(f"\nCreating HuggingFace Dataset...")
    dataset = create_dataset(records, include_images=args.embed_images)

    print(f"\nSaving dataset to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)

    print(f"\nDone! Dataset saved with {len(dataset)} samples")
    print(f"\nTo use with lmms-eval, update imgedit.yaml:")
    print(f"  dataset_path: {args.output_dir}")
    print(f"  dataset_kwargs:")
    print(f"    load_from_disk: True")
    print(f"  test_split: train")

    # Print edit type distribution
    edit_types = {}
    for record in records:
        edit_type = record.get("edit_type", "unknown")
        edit_types[edit_type] = edit_types.get(edit_type, 0) + 1

    print(f"\nEdit type distribution:")
    for edit_type, count in sorted(edit_types.items()):
        print(f"  {edit_type}: {count}")


if __name__ == "__main__":
    main()
