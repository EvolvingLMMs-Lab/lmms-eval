#!/usr/bin/env python3
"""
Sample a fixed subset from ChartQA dataset.

Usage:
    python sample_chartqa.py --num_samples 100 --seed 42 --output_path /path/to/output.parquet
"""

import argparse
import random

from datasets import load_dataset
from loguru import logger


def sample_dataset(
    num_samples: int = 100,
    seed: int = 42,
    output_path: str = "./chartqa100.parquet",
    split: str = "test",
):
    """
    Sample a fixed subset from ChartQA dataset.
    """
    logger.info(f"Loading ChartQA dataset (split: {split})...")
    dataset = load_dataset("lmms-lab/ChartQA", split=split)

    total_samples = len(dataset)
    logger.info(f"Total samples in {split} split: {total_samples}")

    if num_samples > total_samples:
        logger.warning(
            f"Requested {num_samples} samples but only {total_samples} available. "
            f"Using all {total_samples} samples."
        )
        num_samples = total_samples

    # Set random seed for reproducibility
    random.seed(seed)

    # Sample indices
    all_indices = list(range(total_samples))
    sampled_indices = sorted(random.sample(all_indices, num_samples))

    logger.info(f"Sampled {num_samples} indices with seed {seed}")
    logger.info(f"First 10 indices: {sampled_indices[:10]}")

    # Select samples
    sampled_dataset = dataset.select(sampled_indices)

    # Save as parquet
    sampled_dataset.to_parquet(output_path)
    logger.info(f"Saved sampled dataset to: {output_path}")

    # Print sample statistics
    if "type" in sampled_dataset.column_names:
        type_counts = {}
        for item in sampled_dataset:
            t = item.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info("Sample distribution:")
        for t, count in sorted(type_counts.items()):
            logger.info(f"  {t}: {count} ({100*count/num_samples:.1f}%)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Sample a fixed subset from ChartQA")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_path", type=str, required=True, help="Output parquet path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")

    args = parser.parse_args()

    sample_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        output_path=args.output_path,
        split=args.split,
    )


if __name__ == "__main__":
    main()
