# Terminal UI Integration for LMMS-Eval

This document describes the terminal UI integration that brings a real-time dashboard interface to LMMS-Eval, inspired by and adapted from the genai-bench project.

## Overview

The terminal UI provides a rich, real-time dashboard that displays:
- **Overall evaluation progress** across all tasks
- **Current task progress** with detailed metrics
- **Request processing status** for different request types
- **Real-time performance metrics** (accuracy, speed, etc.)
- **Task list** with current task highlighting
- **Live logs** for debugging and monitoring

## Features

### Rich Terminal Dashboard
- **Progress Bars**: Visual progress tracking for overall evaluation and individual tasks
- **Metrics Panels**: Real-time display of evaluation and performance metrics
- **Task Management**: Dynamic task list with current task highlighting
- **Live Updates**: Refreshes at 4Hz for smooth real-time monitoring

### Minimal Dashboard Mode
- **Non-intrusive**: Falls back to minimal mode when UI is disabled
- **Metrics Collection**: Still collects metrics for analysis
- **Compatibility**: Works in environments where rich UI is not suitable

## Usage

The UI is **disabled by default**. You can control it in several ways:

### 1. Command Line Flag
```bash
# Enable UI (default)
lmms-eval --model llava_hf --tasks mme --enable_ui --model_args pretrained=llava-hf/llava-onevision-qwen2-7b-ov-hf,device_map=cuda:0 --verbosity=DEBUG

# Disable UI
lmms-eval --model llava_hf --tasks mme --no-enable_ui --model_args pretrained=llava-hf/llava-onevision-qwen2-7b-ov-hf,device_map=cuda:0 --verbosity=DEBUG
```

### 2. Environment Variable
```bash
# Disable UI via environment variable
export ENABLE_UI=false
lmms-eval --model llava_hf --tasks mme --model_args pretrained=llava-hf/llava-onevision-qwen2-7b-ov-hf,device_map=cuda:0 --verbosity=DEBUG

# Enable UI via environment variable (default)
export ENABLE_UI=true
lmms-eval --model llava_hf --tasks mme --model_args pretrained=llava-hf/llava-onevision-qwen2-7b-ov-hf,device_map=cuda:0 --verbosity=DEBUG
```

## Dashboard Layout

The terminal UI is organized into several sections:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LMMS-Eval Dashboard                          │
│                Model: llava-onevision-qwen2-7b-ov-hf            │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ Overall Progress    ████████████░░░░░░░░  60%  [2m:30s < 1m:40s]│
│ Task Progress       ██████████████████  100%    [500/500]       │
│ Request Processing  ████████████░░░░░░░░░  75%  [750/1000]      │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────┬───────────────────────────────┐
│ Evaluation Metrics              │ Performance Metrics           │
│ Tasks Completed: 2/4            │ Elapsed Time: 2m:30s          │
│ Total Samples: 1,250            │ Current Speed: 3.2 samples/s  │
│ Avg Accuracy: 0.847             │                               │
└─────────────────────────────────┴───────────────────────────────┘
┌─────────────────┬───────────────────────────────────────────────┐
│ Task List       │ Current Task: mme                             │
│ ▶ mme        │ Progress: 500/500 (100%)                        │
│   vqav2        │ Speed: 3.2 samples/s                          │
│   scienceqa    │ Elapsed: 2m:15s                               │
│   gqa          │                                               │
│                 │ Metrics:                                        │
│                 │   accuracy: 0.847                              │
│                 │   score: 84.7                                   │
└─────────────────┴─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ Logs                                                            │
│ 14:32:15 Starting evaluation with 4 tasks                      │
│ 14:32:16 Started task: mme (500 samples)                     │
│ 14:34:31 Completed task: mme                                 │
│ 14:34:32 Started task: vqav2 (1000 samples)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### Architecture

The UI integration is built on several key components:

#### 1. Dashboard Classes (`lmms_eval/ui/dashboard.py`)
- **`RichDashboard`**: Full-featured terminal UI using the Rich library
- **`MinimalDashboard`**: Lightweight fallback with no visual output
- **`create_dashboard()`**: Factory function that selects appropriate dashboard

#### 2. Layout Management (`lmms_eval/ui/layout.py`)
- **Dynamic layouts**: Automatically adjusts to terminal size
- **Progress bars**: Multiple progress tracking with time estimates
- **Panels**: Organized display of metrics and information

#### 3. Visualization (`lmms_eval/ui/plots.py`)
- **Bar charts**: For metrics comparison
- **Histograms**: For distribution visualization
- **Sparklines**: For trend indication

#### 4. Metrics Collection (`lmms_eval/ui/metrics.py`)
- **Real-time metrics**: Collection and aggregation during evaluation
- **Performance tracking**: Speed, accuracy, and other KPIs
- **Format conversion**: Adapts metrics for display

### Integration Points

The UI is integrated at key points in the evaluation process:

1. **Evaluation Start**: Initialize dashboard with model info and task list
2. **Task Start**: Begin tracking for each task
3. **Request Processing**: Update progress during model inference
4. **Task Progress**: Real-time updates during postprocessing
5. **Task Completion**: Mark tasks as done and update metrics
6. **Final Results**: Display comprehensive results

## Dependencies

The UI integration adds one new dependency:
- **`rich>=10.0.0`**: For terminal UI components and formatting

## Configuration

### Environment Variables

- **`ENABLE_UI`**: Controls UI activation (default: `true`)
  - `true`: Enable rich terminal UI
  - `false`: Use minimal dashboard mode

### Dashboard Refresh Rate

The dashboard updates at 4Hz (4 times per second) by default, providing smooth real-time monitoring without excessive CPU usage.

## Compatibility

### Multi-GPU Support
- The UI only runs on the main process (rank 0) in distributed setups
- Other processes use minimal dashboard mode automatically
- Progress aggregation works correctly across multiple GPUs

### Terminal Compatibility
- Works with most modern terminals that support ANSI escape sequences
- Gracefully falls back to minimal mode on unsupported terminals
- Terminal resizing is handled automatically

### Performance Impact
- Minimal overhead when UI is enabled (~1-2% CPU usage)
- Zero overhead when UI is disabled
- Does not affect evaluation accuracy or results

## Troubleshooting

### Common Issues

1. **UI not displaying properly**
   ```bash
   # Try disabling UI
   export ENABLE_UI=false
   lmms-eval --model llava_hf --tasks mme --model_args pretrained=llava-hf/llava-onevision-qwen2-7b-ov-hf,device_map=cuda:0 --verbosity=DEBUG
   ```

2. **Terminal appears corrupted after interruption**
   ```bash
   # Reset terminal
   reset
   ```

3. **UI conflicts with logging**
   - The UI automatically disables tqdm progress bars when enabled
   - Log messages are captured and displayed in the logs panel

### Debug Mode

To debug UI issues, you can run the test script:

```bash
python test_ui_integration.py
```

This will verify that all UI components are working correctly.

## Comparison with genai-bench

This integration is inspired by and adapted from the genai-bench project's terminal UI. Key adaptations for LMMS-Eval include:

### Similarities
- Rich library-based terminal interface
- Real-time progress tracking and metrics display
- Modular dashboard architecture
- Environment variable configuration

### Key Differences
- **Task-oriented**: Focused on evaluation tasks rather than benchmarking requests
- **Multi-modal support**: Adapted for vision-language model evaluation
- **Distributed evaluation**: Handles multi-GPU setups appropriately
- **Evaluation metrics**: Displays accuracy, scores, and task-specific metrics
- **Integration depth**: Embedded throughout the evaluation pipeline

## Future Enhancements

Potential improvements for future versions:

1. **Interactive features**: Task selection, pause/resume functionality
2. **Export capabilities**: Save dashboard screenshots or metrics
3. **Custom layouts**: User-configurable dashboard arrangements
4. **Advanced visualizations**: More chart types and data representations
5. **Remote monitoring**: Web-based dashboard for remote access

## Contributing

When contributing to the UI integration:

1. **Test thoroughly**: Run `test_ui_integration.py` after changes
2. **Maintain compatibility**: Ensure minimal dashboard mode always works
3. **Performance**: Keep UI overhead minimal
4. **Documentation**: Update this README for significant changes

## License

This UI integration follows the same license as LMMS-Eval (MIT License).