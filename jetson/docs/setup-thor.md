# Running the benchmarks on Jetson Thor

Tested on a Jetson AGX Thor Developer Kit (L4T R38.4 / JetPack 7.1, CUDA 13.0, 120 W). Results: [../results/thor/SUMMARY.md](../results/thor/SUMMARY.md); what happened during setup and the runs: [../results/thor/NOTES.md](../results/thor/NOTES.md).

| | Orin | Thor |
|---|---|---|
| JetPack / L4T | 6.2 / R36.4 | 7.1 / R38.4 |
| CUDA / GPU arch | 12.6 / sm_87 | 13.0 / sm_110 |
| Memory | 32 GB unified | 128 GB unified |
| Base image | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` |
| Versions in the image | torch 2.10, transformers 4.57.3, vLLM 0.19, flash-attn | same, vLLM 0.19.0+cu130, flash-attn 2.8.4 |
| TensorRT Edge-LLM | v0.6.0, runtime only (export on x86) | v0.10.1, export and runtime on the device; FP8 and NVFP4 |

## 1. Prepare the board

```bash
sudo nvpmodel -m 0 && nvpmodel -q          # MAXN, as on Orin (the Thor results were taken at 120 W: no sudo)
docker info | grep -i runtime               # needs the nvidia runtime
git clone -b jetson-benchmark https://github.com/michalszczepanski91/lmms-eval.git && cd lmms-eval
export HF_CACHE=/opt/hf-cache               # shared model/dataset cache, about 100 GB
export SHARED_GROUP=mlusers                 # optional: group added to the containers so the cache stays shared
```

Results go to `jetson/results/thor` automatically (board detected from `/proc/device-tree/model`).

## 2. Build the images

```bash
docker build --build-arg BASE_IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
  -f jetson/docker/Dockerfile -t lmms-eval-jetson:latest .
docker build --build-arg BASE_IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor --build-arg CUDA_ARCH=110 \
  -f jetson/frameworks/llamacpp/Dockerfile -t llamacpp-jetson:b11135 jetson/frameworks/llamacpp
```

If the board clock is not synchronized, apt rejects the Ubuntu Release files as "not valid yet": add `--build-arg APT_OPTS="-o Acquire::Check-Date=false"`. Check the versions:

```bash
docker run --rm --runtime nvidia lmms-eval-jetson:latest python -c "
import torch, transformers, vllm, flash_attn; print(torch.__version__, torch.cuda.get_device_capability(), transformers.__version__, vllm.__version__, flash_attn.__version__)"
```

Without `flash_attn`, run HF with `ATTN=sdpa`. If `pip install` fails on a version pin, loosen that line in `jetson/docker/requirements-jetson.txt`; don't replace the image's torch or vLLM.

## 3. Download models and data

```bash
for spec in "hf 3b" "hf 7b" "vllm 3b-awq" "vllm 7b-awq" "llamacpp 3b-q8_0" "llamacpp 3b-q4_k_m" "llamacpp 7b-q8_0" "llamacpp 7b-q4_k_m"; do
  jetson/scripts/download_assets.sh $spec
done
jetson/scripts/download_assets.sh dataset:lmms-lab-encoder/GQA:testdev_balanced_instructions \
  dataset:lmms-lab-encoder/GQA:testdev_balanced_images dataset:lmms-lab-encoder/LMMs-Eval-Lite:coco2017_cap_val
```

## 4. Run

```bash
SMOKE=1 jetson/experiments/qwen2_5_vl/mme.sh     # 8 samples per framework/precision: all should finish with sensible answers
jetson/experiments/qwen2_5_vl/mme.sh             # full MME (a full run takes 30-120 min)
jetson/experiments/qwen2_5_vl/gqa.sh             # full GQA (35 min to 1.5 h per run)
jetson/experiments/qwen2_5_vl/followups.sh       # transport, COCO captioning, resolution sweep
```

TensorRT Edge-LLM needs its engines first: [tensorrt-edgellm.md](tensorrt-edgellm.md). Thor has 4× Orin's memory; the vLLM memory budget is a fixed number of GB (13.8, or 21.0 for 7B bf16), so the KV cache is the same size as on Orin.

Watch long runs: if a run's `run.log` stops growing for 30 min while it is still running, it is stuck (a llama.cpp MME run once hung for 27 h on one request). Kill that `run_eval.sh`; its exit handler stops llama-server.

## 5. Summarize and commit

```bash
python3 jetson/scripts/summarize.py
git add jetson/results/thor && git commit -m "results(jetson): ..." && git push
```

`run_eval.sh` already slims each run dir for git (see [methodology.md](methodology.md#result-files)). Compare like with like: the same framework, precision, image transport (`+pil` / `+png1`) and power mode.
