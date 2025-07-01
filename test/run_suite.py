#!/usr/bin/env python3
"""
Test suite runner for lmms-eval CI/CD

This script runs different test suites based on the provided argument:
- unit: Run unit tests only
- integration: Run integration tests only
- all: Run all tests
- throughput: Run throughput-specific tests only
"""
import argparse
import subprocess
import sys
import unittest
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def run_unit_tests():
    """Run unit tests"""
    test_files = [
        "test_throughput_metrics_unit.py",
        "test_api_components.py",
    ]

    success = True
    for test_file in test_files:
        if not run_command(["python", "-m", "pytest", test_file, "-v"], f"Unit tests: {test_file}"):
            success = False

    return success


def run_integration_tests():
    """Run integration tests"""
    test_files = [
        "test_chat_models.py",
    ]

    success = True
    for test_file in test_files:
        if not run_command(["python", "-m", "pytest", test_file, "-v"], f"Integration tests: {test_file}"):
            success = False

    return success


def run_throughput_tests():
    """Run throughput-specific tests"""
    test_files = [
        "test_throughput_metrics_unit.py",
        "test_throughput_metrics.py",
    ]

    success = True
    for test_file in test_files:
        if Path(test_file).exists():
            if not run_command(["python", test_file], f"Throughput tests: {test_file}"):
                success = False

    return success


def run_linting():
    """Run code linting"""
    commands = [
        (["python", "-m", "black", "--check", ".", "--line-length", "240"], "Black formatting check"),
        (["python", "-m", "isort", "--check-only", "."], "Import sorting check"),
    ]

    success = True
    for cmd, description in commands:
        if not run_command(cmd, description):
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(description="Run lmms-eval test suite")
    parser.add_argument("suite", choices=["unit", "integration", "all", "throughput", "lint"], help="Test suite to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Change to test directory
    test_dir = Path(__file__).parent
    import os

    os.chdir(test_dir)

    print(f"Running {args.suite} test suite...")
    print(f"Working directory: {test_dir}")

    success = True

    if args.suite == "unit":
        success = run_unit_tests()
    elif args.suite == "integration":
        success = run_integration_tests()
    elif args.suite == "throughput":
        success = run_throughput_tests()
    elif args.suite == "lint":
        success = run_linting()
    elif args.suite == "all":
        print("Running all test suites...")
        success &= run_linting()
        success &= run_unit_tests()
        success &= run_integration_tests()
        success &= run_throughput_tests()

    if success:
        print(f"\n✅ All {args.suite} tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ Some {args.suite} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
