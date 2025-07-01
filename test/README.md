# Testing Framework for lmms-eval

This directory contains the test suite for lmms-eval, designed for CI/CD integration and comprehensive testing of the codebase.

## Structure

```
test/
├── __init__.py                     # Test package initialization
├── conftest.py                     # pytest fixtures and configuration
├── requirements-test.txt           # Testing dependencies
├── run_suite.py                    # Test suite runner
├── test_api_components.py          # Core API component tests
├── test_chat_models.py             # Chat model integration tests
├── test_throughput_metrics.py      # Original throughput demo script
└── test_throughput_metrics_unit.py # Unit tests for throughput metrics
```

## Test Categories

### Unit Tests
- **test_throughput_metrics_unit.py**: Tests for TPOT and inference speed calculations
- **test_api_components.py**: Tests for core API components (Instance, registries, metrics)

### Integration Tests
- **test_chat_models.py**: Integration tests for chat models with throughput metrics

### Throughput Tests
- **test_throughput_metrics.py**: Demo script showing throughput calculations
- **test_throughput_metrics_unit.py**: Comprehensive unit tests for timing logic

## Running Tests

### Using the Test Runner
```bash
# Run all tests
python test/run_suite.py all

# Run specific test suites
python test/run_suite.py unit
python test/run_suite.py integration
python test/run_suite.py throughput
python test/run_suite.py lint
```

### Using pytest Directly
```bash
# Install test dependencies
pip install -r test/requirements-test.txt

# Run all tests
pytest test/

# Run specific test files
pytest test/test_throughput_metrics_unit.py -v

# Run with coverage
pytest test/ --cov=lmms_eval --cov-report=html
```

### Using unittest
```bash
# Run individual test files
python test/test_throughput_metrics_unit.py
python test/test_api_components.py
```

## CI/CD Integration

### GitHub Actions
The test suite is integrated with GitHub Actions through `.github/workflows/test.yml`:

- **Lint Check**: Runs black and isort formatting checks
- **Unit Tests**: Runs on Python 3.9, 3.10, 3.11
- **Integration Tests**: Tests model integration with mocks
- **Throughput Tests**: Validates throughput metric calculations
- **Coverage**: Generates test coverage reports

### Pre-commit Hooks
Tests are automatically run through pre-commit hooks:
```bash
pre-commit install
pre-commit run --all-files
```

## Test Design Principles

### 1. Fast Unit Tests
- Mock external dependencies (models, APIs)
- Test core logic without heavy I/O
- Focus on edge cases and error handling

### 2. Comprehensive Integration Tests
- Test real component interactions
- Use minimal mocking for integration points
- Validate end-to-end workflows

### 3. Throughput-Specific Tests
- Validate TPOT formula: `(e2e_latency - TTFT) / (num_output_tokens - 1)`
- Test inference speed calculation: `1 / TPOT`
- Verify timing measurement accuracy
- Test batch processing scenarios

### 4. Maintainable Test Code
- Use fixtures for common test data
- Clear test names describing what's being tested
- Comprehensive error message assertions
- Clean separation between test categories

## Adding New Tests

### For New Features
1. Add unit tests in appropriate `test_*.py` file
2. Add integration tests if feature involves multiple components
3. Update `run_suite.py` if new test categories are needed
4. Update CI workflow if special setup is required

### For Throughput Metrics
1. Add calculation tests to `test_throughput_metrics_unit.py`
2. Add integration tests to `test_chat_models.py`
3. Ensure timing accuracy tests cover edge cases

### Test Naming Convention
- Test files: `test_<component>.py`
- Test classes: `Test<Component>`
- Test methods: `test_<specific_behavior>`

## Dependencies

### Core Testing
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Mocking utilities

### Code Quality
- `black`: Code formatting
- `isort`: Import sorting
- `coverage`: Coverage analysis

### Optional
- `torch`: For model-related tests
- `transformers`: For HuggingFace model tests
- `openai`: For API model tests

## Best Practices

### Writing Tests
- Keep tests focused on single behaviors
- Use descriptive assertions with clear error messages
- Mock external dependencies appropriately
- Test both success and failure cases

### Performance Testing
- Use timing measurements for throughput validation
- Allow reasonable variance in timing tests
- Test edge cases (zero tokens, single token, large batches)

### CI/CD Considerations
- Tests should be deterministic and reliable
- Avoid network dependencies in CI
- Use matrix testing for multiple Python versions
- Generate coverage reports for code quality tracking

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure lmms-eval is installed with `pip install -e .`
2. **Missing Dependencies**: Install test requirements with `pip install -r test/requirements-test.txt`
3. **Timing Test Failures**: Check system load; timing tests may be sensitive to CPU usage

### Debug Mode
```bash
# Run tests with detailed output
pytest test/ -v -s

# Run specific test with pdb debugging
pytest test/test_throughput_metrics_unit.py::TestThroughputMetrics::test_tpot_calculation -v -s --pdb
```