"""
Test Bagel model on EASI spatial reasoning benchmarks

Usage:
    # Test all benchmarks
    python test_bagel_easi.py --model_path /path/to/BAGEL-7B-MoT

    # Quick test with 10 samples per task
    python test_bagel_easi.py --model_path /path/to/BAGEL-7B-MoT --limit 10

    # Test specific benchmark
    python test_bagel_easi.py --model_path /path/to/BAGEL-7B-MoT --tasks mmsibench

    # Specify output directory
    python test_bagel_easi.py --model_path /path/to/BAGEL-7B-MoT --output_dir ./my_results
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# EASI spatial reasoning benchmarks
EASI_BENCHMARKS = {
    "mmsibench": "MMSI - Multi-modal Spatial Intelligence",
    "embspatialbench": "OmniSpatial - Embodied Spatial Reasoning",
    "mindcubebench": "MindCube - 3D Spatial Reasoning",
    "spatial457": "SpatialViz - 7-level Hierarchical Spatial Understanding",
}


def run_evaluation(
    model_name: str,
    model_path: str,
    task: str,
    batch_size: int,
    device: str,
    output_dir: str,
    limit: int = None,
) -> bool:
    """Run evaluation for a single task"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{task}_{timestamp}.log")

    print("=" * 60)
    print(f"Task: {task}")
    print(f"Description: {EASI_BENCHMARKS[task]}")
    print(f"Log: {log_file}")
    print("=" * 60)

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "lmms_eval",
        "--model",
        model_name,
        "--model_args",
        f"pretrained={model_path}",
        "--tasks",
        task,
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--output_path",
        output_dir,
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"Command: {' '.join(cmd)}")
    print()

    # Run evaluation
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end="")
                f.write(line)
                f.flush()

            process.wait()

            if process.returncode == 0:
                print(f"✓ {task} completed successfully\n")
                return True
            else:
                print(f"✗ {task} failed with return code {process.returncode}\n")
                return False

    except Exception as e:
        print(f"✗ {task} failed with exception: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Bagel model on EASI spatial reasoning benchmarks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to BAGEL model (e.g., /path/to/BAGEL-7B-MoT)",
    )
    parser.add_argument(
        "--model_name", type=str, default="bagel", help="Model name (default: bagel)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        choices=list(EASI_BENCHMARKS.keys()),
        help="Specific tasks to run (default: all)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device (default: cuda:0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs/bagel_easi_results",
        help="Output directory (default: ./logs/bagel_easi_results)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (default: None, test all)",
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine tasks to run
    tasks = args.tasks if args.tasks else list(EASI_BENCHMARKS.keys())

    # Print configuration
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Model Path: {args.model_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Output Directory: {args.output_dir}")
    if args.limit:
        print(f"Limit: {args.limit} samples per task")
    print("=" * 60)
    print()

    # Run evaluations
    results = {}
    for task in tasks:
        success = run_evaluation(
            model_name=args.model_name,
            model_path=args.model_path,
            task=task,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
            limit=args.limit,
        )
        results[task] = success

    # Print summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total tasks: {len(tasks)}")
    print(f"Completed: {sum(results.values())}")
    print(f"Failed: {len(tasks) - sum(results.values())}")
    print()

    if all(results.values()):
        print("✓ All tasks completed successfully!")
        print()
        print(f"Results saved to: {args.output_dir}")
        sys.exit(0)
    else:
        print("Failed tasks:")
        for task, success in results.items():
            if not success:
                print(f"  - {task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
