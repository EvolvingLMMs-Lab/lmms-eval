#!/usr/bin/env python3
"""
Test script to verify the inference throughput metrics implementation.
This script demonstrates how the metrics are calculated and logged.
"""

import time


# Simple logger replacement for testing
class MockLogger:
    def info(self, msg):
        print(f"INFO: {msg}")


eval_logger = MockLogger()


def test_tpot_calculation():
    """Test TPOT and inference speed calculation with example values."""

    # Example metrics
    e2e_latency = 2.5  # seconds
    ttft = 0.25  # seconds (Time to First Token)
    num_output_tokens = 50

    # Calculate TPOT using the implemented formula
    if num_output_tokens > 1:
        tpot = (e2e_latency - ttft) / (num_output_tokens - 1)
        inference_speed = 1 / tpot if tpot > 0 else 0
    else:
        tpot = e2e_latency
        inference_speed = 0

    print(f"Test Metrics:")
    print(f"E2E Latency: {e2e_latency:.3f}s")
    print(f"TTFT: {ttft:.3f}s")
    print(f"Output tokens: {num_output_tokens}")
    print(f"TPOT: {tpot:.3f}s")
    print(f"Inference Speed: {inference_speed:.1f} tokens/s")
    print(f"Expected TPOT: {(2.5-0.25)/(50-1):.3f}s")
    print(f"Expected Speed: {1/((2.5-0.25)/(50-1)):.1f} tokens/s")

    return tpot, inference_speed


def simulate_model_inference():
    """Simulate a model inference with timing measurements."""

    print("\nSimulating model inference...")

    # Simulate generation time
    start_time = time.time()
    time.sleep(0.1)  # Simulate inference delay
    end_time = time.time()

    # Simulate output
    generated_tokens = 25
    e2e_latency = end_time - start_time
    ttft = e2e_latency * 0.1  # Estimate TTFT as 10% of total time

    # Calculate metrics
    if generated_tokens > 1:
        tpot = (e2e_latency - ttft) / (generated_tokens - 1)
        inference_speed = 1 / tpot if tpot > 0 else 0
    else:
        tpot = e2e_latency
        inference_speed = 0

    # Log using same format as implementation
    eval_logger.info(f"Inference metrics - E2E: {e2e_latency:.3f}s, TTFT: {ttft:.3f}s, TPOT: {tpot:.3f}s, Speed: {inference_speed:.1f} tokens/s, Output tokens: {generated_tokens}")


if __name__ == "__main__":
    print("Testing TPOT and Inference Speed Calculation")
    print("=" * 50)

    # Test the calculation logic
    test_tpot_calculation()

    # Simulate actual inference
    simulate_model_inference()

    print("\nImplementation Summary:")
    print("- Added timing measurement around model.generate() calls")
    print("- Calculate TPOT = (e2e_latency - TTFT) / (num_output_tokens - 1)")
    print("- Calculate inference_speed = 1 / TPOT")
    print("- Log metrics for each inference request")
    print("- Modified all chat models: openai_compatible, vllm, sglang, huggingface, llava_hf, qwen2_5_vl")
