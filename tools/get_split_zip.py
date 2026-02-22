#!/usr/bin/env python3
"""Split large ZIP files into smaller parts.

Useful for splitting datasets into chunks that comply with hosting limits
(e.g., Hugging Face's 5GB per-file limit).

Usage:
    python get_split_zip.py input.zip output_dir/
    python get_split_zip.py input.zip output_dir/ --max-size 2GB
"""

import argparse
import os
import zipfile


def parse_size(size_str: str) -> int:
    """Parse human-readable size string to bytes.

    Args:
        size_str: Size string like "5GB", "500MB", "1024" (bytes)

    Returns:
        Size in bytes
    """
    size_str = size_str.strip().upper()
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            return int(float(size_str[: -len(unit)]) * multiplier)

    return int(size_str)


def split_zip(input_zip: str, output_dir: str, max_size: int = 5 * 1024**3) -> int:
    """Split a ZIP file into multiple smaller ZIP files.

    Args:
        input_zip: Path to the input ZIP file
        output_dir: Directory to write the split ZIP files
        max_size: Maximum size per output file in bytes (default: 5GB)

    Returns:
        Number of parts created
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    part = 1
    current_size = 0
    prefix_name = os.path.splitext(os.path.basename(input_zip))[0]
    output_zip = zipfile.ZipFile(
        os.path.join(output_dir, f"{prefix_name}_part_{part}.zip"),
        "w",
        zipfile.ZIP_DEFLATED,
    )

    with zipfile.ZipFile(input_zip, "r") as zip_ref:
        for file in zip_ref.namelist():
            file_data = zip_ref.read(file)
            file_size = len(file_data)

            if current_size + file_size > max_size:
                output_zip.close()
                part += 1
                current_size = 0
                output_zip = zipfile.ZipFile(
                    os.path.join(output_dir, f"{prefix_name}_part_{part}.zip"),
                    "w",
                    zipfile.ZIP_DEFLATED,
                )

            output_zip.writestr(file, file_data)
            current_size += file_size

    output_zip.close()
    return part


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split large ZIP files into smaller parts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s dataset.zip ./split_output/
    %(prog)s dataset.zip ./split_output/ --max-size 2GB
    %(prog)s large_archive.zip ./parts/ --max-size 500MB
        """,
    )
    parser.add_argument("input_zip", help="Path to the input ZIP file")
    parser.add_argument("output_dir", help="Directory to write split ZIP files")
    parser.add_argument(
        "--max-size",
        default="5GB",
        help="Maximum size per output file (default: 5GB). " "Supports units: B, KB, MB, GB, TB",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_zip):
        parser.error(f"Input file not found: {args.input_zip}")

    max_bytes = parse_size(args.max_size)
    num_parts = split_zip(args.input_zip, args.output_dir, max_bytes)
    print(f"Split into {num_parts} part(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
