#!/usr/bin/env python3
"""Simple test script to verify UI integration works."""

import os
import time

from lmms_eval.ui import create_dashboard


def test_ui_creation():
    """Test that dashboard can be created."""
    print("Testing UI dashboard creation...")

    # Test with UI enabled
    os.environ["ENABLE_UI"] = "true"
    dashboard = create_dashboard()
    print(f"Dashboard created: {type(dashboard).__name__}")

    # Test with UI disabled
    os.environ["ENABLE_UI"] = "false"
    dashboard_minimal = create_dashboard()
    print(f"Minimal dashboard created: {type(dashboard_minimal).__name__}")

    return dashboard, dashboard_minimal


def test_dashboard_functionality(dashboard):
    """Test basic dashboard functionality."""
    print("Testing dashboard functionality...")

    # Test initialization
    dashboard.start_evaluation("test_model", "test_args", ["task1", "task2"])
    print("✓ Dashboard initialized")

    # Test task operations
    dashboard.start_task("task1", 100)
    print("✓ Task started")

    # Simulate progress updates
    for i in range(0, 101, 20):
        dashboard.update_task_progress("task1", i)
        time.sleep(0.1)  # Small delay for demo
    print("✓ Task progress updated")

    # Test metrics
    metrics = {"accuracy": 0.85, "score": 92.5}
    dashboard.add_task_metrics("task1", metrics)
    print("✓ Task metrics added")

    # End task
    dashboard.end_task("task1")
    print("✓ Task completed")

    # Test final results
    results = {"results": {"task1": {"accuracy": 0.85}}, "task1": {"accuracy": 0.85}}
    dashboard.show_final_results(results)
    print("✓ Final results displayed")


def main():
    """Run UI integration tests."""
    print("=== LMMS-Eval UI Integration Test ===\n")

    try:
        # Test dashboard creation
        dashboard, dashboard_minimal = test_ui_creation()
        print()

        # Test with Rich dashboard (if available)
        if hasattr(dashboard, "live"):
            print("Testing Rich dashboard functionality...")
            # Quick test without live context for safety
            test_dashboard_functionality(dashboard)
        else:
            print("Rich dashboard not available, testing minimal dashboard...")
            test_dashboard_functionality(dashboard_minimal)

        print("\n✅ All tests passed! UI integration is working.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
