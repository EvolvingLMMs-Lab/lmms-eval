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