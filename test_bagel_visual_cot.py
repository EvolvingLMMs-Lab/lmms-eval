"""
Test Bagel Visual Chain-of-Thought model on MathVista

Usage:
    # Quick test with 5 samples
    python test_bagel_visual_cot.py --model_path /path/to/BAGEL-7B-MoT --limit 5

    # Full test
    python test_bagel_visual_cot.py --model_path /path/to/BAGEL-7B-MoT

    # Custom output directory
    python test_bagel_visual_cot.py --model_path /path/to/BAGEL-7B-MoT --output_dir ./my_results
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def run_test(
    model_path: str,
    task: str = "mathvista_visual_cot",
    batch_size: int = 1,
    device: str = "cuda:0",
    output_dir: str = "./logs/bagel_visual_cot_test",
    limit: int = None,
) -> bool:
    """Run BagelVisualCoT test"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"test_{timestamp}.log")

    print("=" * 60)
    print("Testing Bagel Visual Chain-of-Thought")
    print("=" * 60)
    print(f"Model: bagel_visual_cot")
    print(f"Model Path: {model_path}")
    print(f"Task: {task}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"Output Directory: {output_dir}")
    if limit:
        print(f"Limit: {limit} samples")
    print(f"Log: {log_file}")
    print("=" * 60)
    print()

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "lmms_eval",
        "--model",
        "bagel_visual_cot",
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

    # Run test
    try:
        os.makedirs(output_dir, exist_ok=True)

        with open(log_file, "w", encoding="utf-8") as f:
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
                print(f"\n✓ Test completed successfully\n")
                print(f"Results saved to: {output_dir}")
                print(f"Log saved to: {log_file}")
                return True
            else:
                print(f"\n✗ Test failed with return code {process.returncode}\n")
                return False

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Bagel Visual Chain-of-Thought model on MathVista"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to BAGEL model (e.g., /path/to/BAGEL-7B-MoT)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mathvista_visual_cot",
        help="Task to run (default: mathvista_visual_cot)",
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
        default="./logs/bagel_visual_cot_test",
        help="Output directory (default: ./logs/bagel_visual_cot_test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (default: None, test all)",
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Run test
    success = run_test(
        model_path=args.model_path,
        task=args.task,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        limit=args.limit,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
