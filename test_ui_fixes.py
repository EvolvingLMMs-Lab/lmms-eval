#!/usr/bin/env python3
"""Test script to verify UI fixes."""

import os
import time

from lmms_eval.ui import create_dashboard


def test_dashboard_without_blocking():
    """Test that dashboard doesn't block on final results."""
    print("Testing dashboard without blocking...")

    # Test with UI enabled
    os.environ["ENABLE_UI"] = "true"
    dashboard = create_dashboard()

    # Test initialization
    dashboard.start_evaluation("test_model", "test_args", ["task1"])

    # Test task operations
    dashboard.start_task("task1", 10)

    # Simulate progress updates with metrics
    for i in range(1, 11):
        dashboard.update_task_progress("task1", i)
        # Add metrics periodically
        if i % 3 == 0:
            metrics = {"accuracy": 0.8 + i * 0.01, "score": 80 + i}
            dashboard.add_task_metrics("task1", metrics)
        time.sleep(0.1)

    # End task
    dashboard.end_task("task1")

    # Test final results - should not block
    results = {"results": {"task1": {"accuracy": 0.85}}, "task1": {"accuracy": 0.85}}

    print("Showing final results...")
    start_time = time.time()
    dashboard.show_final_results(results)
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Final results display took {elapsed:.2f} seconds")

    # Should not block indefinitely
    if elapsed < 5:
        print("✓ Dashboard doesn't block on final results")
        return True
    else:
        print("✗ Dashboard still blocks on final results")
        return False


def test_minimal_dashboard():
    """Test minimal dashboard functionality."""
    print("\nTesting minimal dashboard...")

    # Test with UI disabled
    os.environ["ENABLE_UI"] = "false"
    dashboard = create_dashboard()

    # Test all methods work without errors
    dashboard.start_evaluation("test_model", "test_args", ["task1"])
    dashboard.start_task("task1", 10)
    dashboard.update_task_progress("task1", 5)
    dashboard.add_task_metrics("task1", {"accuracy": 0.85})
    dashboard.end_task("task1")
    dashboard.show_final_results({"results": {"task1": {"accuracy": 0.85}}})

    print("✓ Minimal dashboard works correctly")
    return True


def main():
    """Run UI fixes tests."""
    print("=== Testing UI Fixes ===\n")

    try:
        success = True

        # Test Rich dashboard
        success &= test_dashboard_without_blocking()

        # Test minimal dashboard
        success &= test_minimal_dashboard()

        if success:
            print("\n✅ All UI fixes working correctly!")
            return 0
        else:
            print("\n❌ Some UI fixes failed!")
            return 1

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
