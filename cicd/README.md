# CICD Evaluation Server Tests

This directory contains scripts for running Continuous Integration/Continuous Deployment (CICD) tests for the lmms-eval HTTP evaluation server.

## Overview

The CICD system provides an automated way to test the evaluation server across multiple models. It handles:
- Server lifecycle management (start/stop)
- GPU management
- Job submission and result verification
- Cleanup of temporary resources

## Directory Structure

```
cicd/
├── README.md              # This file
└── run_evalcicd.sh        # Main entry point for running CICD tests

test/eval/                 # Python test implementations (separate folder)
├── run_cicd.py           # Python unittest launcher
├── utils.py              # Utility functions for server tests
├── __init__.py           # Package init
└── qwen2_5_vl/           # Qwen 2.5 VL model tests
    ├── __init__.py
    └── test_qwen2_5_vl.py
```

## Quick Start

### Basic Usage

Run all evaluation server tests with default settings (8 GPUs):

```bash
./cicd/run_evalcicd.sh
```

### Test Specific Model

Test a specific model (e.g., qwen2_5_vl):

```bash
./cicd/run_evalcicd.sh --model-name qwen2_5_vl
```

### Custom GPU Count

Specify the number of GPUs to use:

```bash
./cicd/run_evalcicd.sh --gpu-count 4
```

## Command Line Options

### `run_evalcicd.sh` Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-name NAME` | Model to test (e.g., qwen2_5_vl) | (empty - tests all models) |
| `--gpu-count NUM` | Number of GPUs to use | 8 |
| `--no-verbose` | Disable verbose output | (verbose enabled) |
| `--help` | Show help message | - |

### Available Models

- `qwen2_5_vl` - Qwen 2.5 Vision-Language model

## Python Test Runner

The bash script wraps around a Python-based test runner located at `test/eval/run_cicd.py`. This Python script uses the unittest framework to discover and run tests.

### Direct Python Usage

You can also run tests directly with Python:

```bash
# Run all tests
python test/eval/run_cicd.py --verbose

# Run tests for a specific model
python test/eval/run_cicd.py --model-name qwen2_5_vl --verbose

# Specify GPU count
python test/eval/run_cicd.py --gpu-count 4 --verbose

# Stop on first failure
python test/eval/run_cicd.py --failfast
```

### Python Runner Options

| Option | Description |
|--------|-------------|
| `--test-pattern PATTERN` | Pattern to match test files (default: test_*.py) |
| `--verbose`, `-v` | Run tests in verbose mode |
| `--failfast` | Stop on first failure |
| `--gpu-count NUM` | Override GPU count for testing |
| `--model-name NAME` | Optional model name to test |

## Test Utilities

The `test/eval/utils.py` file provides helper functions for evaluation server tests:

- **`with_temp_dir`**: Decorator that creates a temporary directory for tests
- **`get_available_gpus`**: Returns the number of available GPUs
- **`find_free_port`**: Finds an available port for the server
- **`wait_for_server`**: Waits for server to become available
- **`ServerProcess`**: Class to manage server lifecycle
- **`managed_server`**: Context manager for running a server during tests
- **`with_server`**: Decorator that starts/stops server for each test
- **`run_evaluation_test`**: Helper to run evaluation tests

## How It Works

1. **Server Startup**: Each test starts a fresh evaluation server on a random available port
2. **Job Submission**: Tests submit evaluation jobs via the HTTP API
3. **Result Verification**: Tests wait for job completion and verify results
4. **Cleanup**: Server is stopped and temporary files are removed

## Examples

### Example 1: Quick Test of All Models

```bash
cd /path/to/lmms-eval
./cicd/run_evalcicd.sh
```

### Example 2: Test Single Model with 4 GPUs

```bash
./cicd/run_evalcicd.sh --model-name qwen2_5_vl --gpu-count 4
```

### Example 3: Run Tests Without Verbose Output

```bash
./cicd/run_evalcicd.sh --no-verbose
```

### Example 4: Direct Python Execution with Failfast

```bash
python test/eval/run_cicd.py --model-name qwen2_5_vl --verbose --failfast
```

## Adding New Model Tests

When adding tests for a new model:

1. Create a new directory under `test/eval/` with the model name (e.g., `test/eval/llava/`)
2. Add `__init__.py` to the new directory
3. Add test files with the pattern `test_*.py`
4. Use the utilities from `utils.py` for consistency:

```python
import unittest
from unittest import TestCase
from utils import run_evaluation_test, with_server, with_temp_dir


class TestNewModel(TestCase):
    @with_temp_dir
    @with_server
    def test_new_model_task(self, temp_dir, server):
        result = run_evaluation_test(
            server_url=server.url,
            model="new_model",
            tasks=["some_task"],
            model_args={"pretrained": "model/path"},
            batch_size=4,
            limit=4,
            num_gpus=8,
            timeout=600,
        )
        self.assertTrue(result.success, f"Evaluation failed: {result.error}")


if __name__ == "__main__":
    unittest.main()
```

5. Update this README with the new model name

## Environment Variables

The test runner sets `CUDA_VISIBLE_DEVICES` based on the `--gpu-count` parameter to control GPU visibility for the tests.

## Troubleshooting

### Server Won't Start

If the server fails to start:
1. Check if the port is already in use
2. Verify that lmms-eval is properly installed
3. Check the logs for error messages

### GPU Issues

If tests fail due to GPU problems:
1. Check available GPUs: `nvidia-smi`
2. Adjust `--gpu-count` to match your available GPUs
3. Verify CUDA is properly installed

### Test Failures

If tests fail:
1. Run with `--verbose` flag for detailed output
2. Use `--model-name` to isolate which model is failing
3. Check the server logs for errors
4. Verify model weights are accessible

## Related Files

- `test/eval/run_cicd.py` - Python unittest test runner
- `test/eval/utils.py` - Shared test utilities
- `test/eval/*/` - Individual model test directories
- `lmms_eval/launch_server.py` - Server launcher
- `lmms_eval/entrypoints/http_server.py` - HTTP server implementation
- `lmms_eval/entrypoints/client.py` - Python client for server

