"""
Script to convert IllusionBench from its original format to lmms-eval format.

Usage:
    python convert_dataset.py --output_dir ./illusionbench_converted

This will:
1. Download Image_properties.json from HuggingFace
2. Download and extract images from IllusionDataset.zip
3. Create a HuggingFace dataset with the proper format

Required dataset columns:
- image: PIL Image
- question: str
- answer: str
- question_type: str ("TF" or "select")
- category: str ("A", "B", or "C")
- question_id: str (unique identifier)
- image_name: str (original image filename)
- difficulty_level: int
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import requests
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
from tqdm import tqdm


def download_json(url: str) -> list:
    response = requests.get(url)
    response.raise_for_status()
    return json.loads(response.text)


def download_and_extract_images(url: str, output_dir: Path) -> dict[str, PILImage.Image]:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    images = {}
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for name in tqdm(zf.namelist(), desc="Extracting images"):
            if name.endswith((".png", ".jpg", ".jpeg")):
                with zf.open(name) as f:
                    img = PILImage.open(f).convert("RGB")
                    img_name = Path(name).name
                    images[img_name] = img.copy()
    return images


def convert_to_dataset(json_data: list, images: dict) -> Dataset:
    records = []
    question_id = 0

    for item in tqdm(json_data, desc="Processing QA pairs"):
        img_props = item["image_property"]
        image_name = img_props["image_name"]

        if image_name not in images:
            continue

        image = images[image_name]
        category = img_props.get("Category", "Unknown")
        difficulty = img_props.get("Difficult Level", 0)

        for qa in item["qa_data"]:
            records.append(
                {
                    "image": image,
                    "question": qa["Question"],
                    "answer": qa["Correct Answer"],
                    "question_type": qa["Question Type"],
                    "category": category,
                    "question_id": f"{image_name}_{question_id}",
                    "image_name": image_name,
                    "difficulty_level": difficulty,
                }
            )
            question_id += 1

    features = Features(
        {
            "image": Image(),
            "question": Value("string"),
            "answer": Value("string"),
            "question_type": Value("string"),
            "category": Value("string"),
            "question_id": Value("string"),
            "image_name": Value("string"),
            "difficulty_level": Value("int32"),
        }
    )

    return Dataset.from_list(records, features=features)


def main():
    parser = argparse.ArgumentParser(description="Convert IllusionBench dataset")
    parser.add_argument("--output_dir", type=str, default="./illusionbench_converted")
    parser.add_argument(
        "--json_url",
        type=str,
        default="https://huggingface.co/datasets/MingZhangSJTU/IllusionBench/resolve/main/Image_properties.json",
    )
    parser.add_argument(
        "--images_url",
        type=str,
        default="https://huggingface.co/datasets/MingZhangSJTU/IllusionBench/resolve/main/IllusionDataset.zip",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading JSON metadata...")
    json_data = download_json(args.json_url)
    print(f"Found {len(json_data)} images with metadata")

    print("Downloading and extracting images...")
    images = download_and_extract_images(args.images_url, output_dir)
    print(f"Extracted {len(images)} images")

    print("Converting to HuggingFace dataset...")
    dataset = convert_to_dataset(json_data, images)
    print(f"Created dataset with {len(dataset)} QA pairs")

    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir / "dataset"))

    print("Done! Dataset statistics:")
    print(f"  Total QA pairs: {len(dataset)}")
    print(f"  Categories: {set(dataset['category'])}")
    print(f"  Question types: {set(dataset['question_type'])}")


if __name__ == "__main__":
    main()
