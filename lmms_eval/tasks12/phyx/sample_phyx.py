#!/usr/bin/env python3
"""
Sample fixed subsets from PhyX dataset by category.

Usage:
    python sample_phyx.py --category Mechanics --num_samples 100 --seed 42 --output_path /path/to/output.parquet
"""

import argparse
import random

from datasets import load_dataset
from loguru import logger


def sample_phyx_by_category(
    category: str,
    num_samples: int = 100,
    seed: int = 42,
    output_path: str = None,
    split: str = "phyx_mc",
):
    """
    Sample a fixed subset from PhyX dataset filtered by category.

    Args:
        category: Category to filter (e.g., "Mechanics", "Optics")
        num_samples: Number of samples to select
        seed: Random seed for reproducibility
        output_path: Path to save the parquet file
        split: Dataset split to use
    """
    logger.info(f"Loading PhyX dataset (split: {split})...")
    dataset = load_dataset("Cloudriver/PhyX", "data_for_llms_eval", split=split)

    logger.info(f"Total samples in {split}: {len(dataset)}")

    # Filter by category
    filtered_dataset = dataset.filter(lambda x: x["category"] == category)
    total_in_category = len(filtered_dataset)
    logger.info(f"Samples in category '{category}': {total_in_category}")

    if num_samples > total_in_category:
        logger.warning(
            f"Requested {num_samples} samples but only {total_in_category} available. "
            f"Using all {total_in_category} samples."
        )
        num_samples = total_in_category

    # Set random seed for reproducibility
    random.seed(seed)

    # Sample indices
    all_indices = list(range(total_in_category))
    sampled_indices = sorted(random.sample(all_indices, num_samples))

    logger.info(f"Sampled {num_samples} indices with seed {seed}")
    logger.info(f"First 10 indices: {sampled_indices[:10]}")

    # Select samples
    sampled_dataset = filtered_dataset.select(sampled_indices)

    # Save as parquet
    if output_path:
        sampled_dataset.to_parquet(output_path)
        logger.info(f"Saved sampled dataset to: {output_path}")

    # Print sample statistics
    if "subfield" in sampled_dataset.column_names:
        from collections import Counter

        subfields = Counter(sampled_dataset["subfield"])
        logger.info("Subfield distribution:")
        for sf, count in sorted(subfields.items()):
            logger.info(f"  {sf}: {count} ({100*count/num_samples:.1f}%)")

    if "reasoning_type" in sampled_dataset.column_names:
        from collections import Counter

        reasoning_types = Counter(sampled_dataset["reasoning_type"])
        logger.info("Reasoning type distribution:")
        for rt, count in sorted(reasoning_types.items()):
            logger.info(f"  {rt}: {count} ({100*count/num_samples:.1f}%)")

    return output_path, sampled_indices


def main():
    parser = argparse.ArgumentParser(description="Sample PhyX by category")
    parser.add_argument("--category", type=str, required=True, help="Category to filter")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_path", type=str, required=True, help="Output parquet path")
    parser.add_argument("--split", type=str, default="phyx_mc", help="Dataset split")

    args = parser.parse_args()

    sample_phyx_by_category(
        category=args.category,
        num_samples=args.num_samples,
        seed=args.seed,
        output_path=args.output_path,
        split=args.split,
    )


if __name__ == "__main__":
    main()
