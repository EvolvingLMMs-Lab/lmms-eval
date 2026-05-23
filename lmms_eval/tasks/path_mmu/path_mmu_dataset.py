"""Build script for PathMMU dataset.

PathMMU (https://huggingface.co/datasets/jamessyx/PathMMU) is a pathology
multimodal benchmark with 5 categories: PubMed, EduContent, PathCLS, Atlas,
and SocialPath.  Images come from multiple sources:

- PubMed, EduContent, PatchCamelyon: included in the gated HF repo images.zip
- SocialPath: downloaded from Twitter/X via gallery-dl
- PathCLS (8 sub-datasets) and Atlas: from local pre-downloaded datasets

This script downloads data.json + images.zip from HF, resolves images from
local storage, and builds a HuggingFace DatasetDict with ``validation`` and
``test`` splits (test_tiny merged into test to match the paper).

Usage:
    python -m lmms_eval.tasks.path_mmu.path_mmu_dataset [--output_dir DIR]
"""

import ast
import json
import os
import shutil
import zipfile
from typing import Optional

import datasets
from huggingface_hub import hf_hub_download
from loguru import logger

_REPO_ID = "jamessyx/PathMMU"

# Map from data.json split names to HF split names.
_SPLIT_MAP = {
    "val": "validation",
    "test_tiny": "test_tiny",
    "test": "test",
}

# Default cache location for the built dataset
_DEFAULT_OUTPUT = os.path.join(
    os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    "path_mmu_dataset",
)

# ---------------------------------------------------------------------------
# Local image resolution
# ---------------------------------------------------------------------------

# Root directory with pre-downloaded histopathology datasets.
# Override with PATHMMU_HISTO_DIR env var.
_LOCAL_HISTO_DIR = os.getenv(
    "PATHMMU_HISTO_DIR",
    "/capstor/store/cscs/swissai/infra01/vision-datasets/eval/histo",
)

# Mapping: source_img prefix in data.json -> subdirectory under _LOCAL_HISTO_DIR.
# Only needed when the local directory name differs from the source_img prefix.
_LOCAL_PATH_MAP = {
    "CRC100K/": "crc-100k/",
    "SICAPv2/": "SICAPv2/SICAPv2/",
    "SkinCancer/": "SkinCancer/SkinCancer/",
    "MHIST/": "mhist/",
    "WSSSLUAD/": "wsss4luad/",
    "ICIAR2018_BACH_Challenge/": "bach/ICIAR2018_BACH_Challenge/",
}

# LC25000 has different subdirectory nesting locally.
_LC25000_SUBDIR_MAP = {
    "colon_aca": "colon_image_sets/colon_aca",
    "colon_n": "colon_image_sets/colon_n",
    "lung_aca": "lung_image_sets/lung_aca",
    "lung_n": "lung_image_sets/lung_n",
    "lung_scc": "lung_image_sets/lung_scc",
}


def _resolve_local_source(source_img: str) -> Optional[str]:
    """Resolve a source_img path to a file in _LOCAL_HISTO_DIR."""
    if not os.path.isdir(_LOCAL_HISTO_DIR):
        return None

    # Try mapped prefix
    for prefix, local_prefix in _LOCAL_PATH_MAP.items():
        if source_img.startswith(prefix):
            path = os.path.join(
                _LOCAL_HISTO_DIR,
                source_img.replace(prefix, local_prefix, 1),
            )
            if os.path.isfile(path):
                return path

    # Try as-is (e.g. Osteo/)
    path = os.path.join(_LOCAL_HISTO_DIR, source_img)
    if os.path.isfile(path):
        return path

    # LC25000: "LC25000/colon_n/file" -> "lc25000/lung_colon_image_set/colon_image_sets/colon_n/file"
    if source_img.startswith("LC25000/"):
        rest = source_img[len("LC25000/") :]
        subdir = rest.split("/")[0]
        if subdir in _LC25000_SUBDIR_MAP:
            path = os.path.join(
                _LOCAL_HISTO_DIR,
                "lc25000/lung_colon_image_set",
                _LC25000_SUBDIR_MAP[subdir],
                *rest.split("/")[1:],
            )
            if os.path.isfile(path):
                return path

    # Atlas: "books_set/images/uuid.png" -> "arch/books_set/images/uuid.png"
    if source_img.startswith("books_set/"):
        path = os.path.join(_LOCAL_HISTO_DIR, "arch", source_img)
        if os.path.isfile(path):
            return path

    return None


# ---------------------------------------------------------------------------
# Image setup: copy from local sources into the HF cache images/ dir
# ---------------------------------------------------------------------------


def _copy_missing_images(data: dict, images_dir: str) -> None:
    """Copy images from local datasets into images_dir for samples
    that have a source_img but are missing from images.zip."""
    from PIL import Image

    copied = 0
    missing = 0
    for category in data:
        if not isinstance(data[category], dict):
            continue
        for split_samples in data[category].values():
            if not isinstance(split_samples, list):
                continue
            for sample in split_samples:
                source_img = sample.get("source_img", "")
                if not source_img:
                    continue
                target = os.path.join(images_dir, sample.get("img", ""))
                if os.path.isfile(target):
                    continue

                local_path = _resolve_local_source(source_img)
                if not local_path:
                    missing += 1
                    continue

                ext = os.path.splitext(local_path)[1].lower()
                if ext in (".tif", ".tiff"):
                    Image.open(local_path).convert("RGB").save(target, "JPEG")
                else:
                    shutil.copy2(local_path, target)
                copied += 1

    if copied:
        logger.info(f"Copied {copied} images from local datasets")
    if missing:
        logger.info(f"{missing} images not found locally (likely deleted SocialPath tweets)")


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------


def _make_record(sample: dict, category: str, images_dir: str) -> dict:
    """Convert a raw data.json sample into a flat record."""
    img_filename = sample.get("img", sample.get("img_path", ""))
    options = sample.get("options", [])
    if isinstance(options, str):
        options = ast.literal_eval(options)

    # Derive subcategory for PathCLS sub-datasets
    source_img = sample.get("source_img", "")
    if category == "PathCLS":
        subcategory = source_img.split("/")[0] if source_img else "PatchCamelyon"
    else:
        subcategory = category

    return {
        "id": str(sample.get("No", "")),
        "question": sample.get("question", ""),
        "options": options,
        "answer": sample.get("answer", ""),
        "explanation": sample.get("explanation", ""),
        "category": category,
        "subcategory": subcategory,
        "image_path": os.path.join(images_dir, img_filename),
    }


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

_FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "options": datasets.Sequence(datasets.Value("string")),
        "answer": datasets.Value("string"),
        "explanation": datasets.Value("string"),
        "category": datasets.Value("string"),
        "subcategory": datasets.Value("string"),
        "image_path": datasets.Value("string"),
    }
)


def get_dataset_path() -> str:
    """Return the on-disk path where the built dataset is stored."""
    return os.getenv("PATH_MMU_DATASET_DIR", _DEFAULT_OUTPUT)


def build_dataset(output_dir: Optional[str] = None) -> str:
    """Download, flatten, and save PathMMU as a HF DatasetDict.

    Returns the path to the saved dataset directory.
    """
    output_dir = output_dir or get_dataset_path()

    # Skip if already built
    dict_path = os.path.join(output_dir, "dataset_dict.json")
    if os.path.isfile(dict_path):
        with open(dict_path) as f:
            info = json.load(f)
        if info.get("splits"):
            return output_dir

    # Download data.json and images.zip from the gated HF repo
    json_path = hf_hub_download(
        repo_id=_REPO_ID,
        filename="data.json",
        repo_type="dataset",
    )
    zip_path = hf_hub_download(
        repo_id=_REPO_ID,
        filename="images.zip",
        repo_type="dataset",
    )

    # Extract images.zip (PubMed, EduContent, PatchCamelyon, SocialPath)
    cache_dir = os.path.dirname(zip_path)
    images_dir = os.path.join(cache_dir, "images")
    if not os.path.isdir(images_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)

    # Load data.json
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Copy missing images from local datasets (PathCLS, Atlas)
    _copy_missing_images(raw_data, images_dir)

    # Flatten into split records, keeping only samples with images
    split_records: dict[str, list[dict]] = {s: [] for s in set(_SPLIT_MAP.values())}
    total = 0
    available = 0
    for category, splits in raw_data.items():
        if not isinstance(splits, dict):
            continue
        for src_split, hf_split in _SPLIT_MAP.items():
            if src_split not in splits:
                continue
            for sample in splits[src_split]:
                total += 1
                record = _make_record(sample, category, images_dir)
                if os.path.isfile(record["image_path"]):
                    split_records[hf_split].append(record)
                    available += 1

    logger.info(f"PathMMU: {available}/{total} samples with images " f"({total - available} missing)")

    # Save as HF DatasetDict
    ds_dict = {}
    for hf_split, records in split_records.items():
        if records:
            ds_dict[hf_split] = datasets.Dataset.from_list(records, features=_FEATURES)

    dataset = datasets.DatasetDict(ds_dict)
    dataset.save_to_disk(output_dir)
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PathMMU dataset for lmms-eval")
    parser.add_argument(
        "--output_dir",
        default=None,
        help=f"Output directory (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    path = build_dataset(args.output_dir)
    print(f"PathMMU dataset saved to: {path}")
