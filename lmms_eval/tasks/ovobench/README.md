# OVO-Bench Integration for lmms-eval

Original benchmark repository: [https://github.com/JoeLeelyf/OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench)

## Usage

Before using this implementation of OVO-Bench with `lmms-eval`, you must download and unpack the video chunks by following the instructions at:  
[https://github.com/JoeLeelyf/OVO-Bench#data-preparation](https://github.com/JoeLeelyf/OVO-Bench#data-preparation)

In all task configuration files (`ovo_backward.yaml`, `ovo_realtime.yaml`, `ovo_forward.yaml`), set the path to the unpacked video chunks directory in:  
`lmms_eval_specific_kwargs.default.data_dir`

## Limitations

This directory contains the `ovobench_data` subfolder with three tasks: `backward`, `realtime`, and `forward`. Each task includes JSON annotation files derived from the original file:  
[https://github.com/JoeLeelyf/OVO-Bench/blob/main/data/ovo_bench_new.json](https://github.com/JoeLeelyf/OVO-Bench/blob/main/data/ovo_bench_new.json)

A key limitation is that model implementations must support the `generate_until_multi_round` logic to correctly evaluate the `realtime` task.

## 📊 Evaluation Results

Below are the evaluation results on OVO-Bench

| Model         | n_frames | EPM   | HLD   | ASI   | Backward Avg. | STU   | OJR   | ATR   | FPD   | ACR   | OCR   | Realtime Avg. | REC   | CRR   | SSR   | Forward Avg. |
|---------------|----------|-------|-------|-------|---------------|-------|-------|-------|-------|-------|-------|---------------|-------|-------|-------|--------------|
| Qwen2-VL-7B   | 64       | 48.48 | 37.10 | 58.11 | 47.90         | 47.75 | 53.26 | 61.21 | 63.37 | 46.79 | 59.73 | 55.35         | 32.66 | 50.83 | 66.14 | 49.88        |
| Qwen2.5-VL-7B | 64       | 43.77 | 24.19 | 56.76 | 41.57         | 51.69 | 53.80 | 67.24 | 62.38 | 56.88 | 65.10 | 59.51         | 33.09 | 41.25 | 47.54 | 40.63        |

> **Note**: All values are accuracy percentages (%). Rounded to two decimal places for readability.