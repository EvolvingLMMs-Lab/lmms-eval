#!/usr/bin/env python3
"""Analytical FLOPs of Qwen2.5-VL for every sample of a run_eval.sh run -> <run dir>/flops.json.

Runs inside the eval image (jetson/compute_flops.sh wraps docker): it reloads the task docs to get each
image's size, applies the same resize as every framework (min_pixels 200704 / max_pixels 1605632 =
256..2048 visual tokens) to get the patch grid, and takes the prompt / output token counts recorded by the
backend (TensorRT Edge-LLM records none: those are recounted with the tokenizer).

Counted: matrix multiplies at 2 FLOPs per multiply-accumulate, i.e. every linear layer (incl. patch embedding,
patch merger and LM head) plus attention scores and attention-weighted values. Norms, activations, rotary
embeddings and softmax are left out (< 1%). Batch size 1, no prefix / encoder caching (as run_eval.sh runs):
  vision  - ViT patch embedding + 32 blocks (window attention over 8x8-patch windows except the full-attention
            blocks) + 2x2 patch merger MLP into the LLM width
  prefill - LLM forward over the whole prompt (text + image tokens, causal attention) + LM head for the 1st token
  decode  - one LLM forward per further output token (attending to the KV cache) + LM head
The FLOPs are the same for every framework and precision (quantization changes the cost per FLOP, not the count).

Usage: python jetson/flops.py <run dir> [<run dir> ...]
  run dir = <results dir>/<model>/<tasks>/<framework>-<precision>[+tag]/<run_id> (as written by run_eval.sh)
"""

import glob
import json
import os
import re
import sys
from types import SimpleNamespace

MIN_PIXELS, MAX_PIXELS = 200704, 1605632


def vision_flops(v, grid_h, grid_w):
    """Vision encoder + patch merger for one image of grid_h x grid_w patches (grid_t = 1 for images)."""
    d, merge = v.hidden_size, v.spatial_merge_size
    n = grid_h * grid_w
    patch_embed = 2 * n * (v.in_chans * v.temporal_patch_size * v.patch_size**2) * d
    linear = 2 * n * v.depth * (4 * d * d + 3 * d * v.intermediate_size)  # qkv + proj, SwiGLU gate/up/down
    # Windows are window_size px = 4x4 merged units (8x8 patches); edge windows are smaller.
    win = v.window_size // merge // v.patch_size
    mh, mw = grid_h // merge, grid_w // merge
    window_sq = sum((min(win, mh - i) * min(win, mw - j) * merge**2) ** 2 for i in range(0, mh, win) for j in range(0, mw, win))
    full = len(v.fullatt_block_indexes)
    attention = 4 * d * (full * n * n + (v.depth - full) * window_sq)  # QK^T + AV, non-causal
    merged = d * merge**2
    merger = 2 * (n // merge**2) * (merged * merged + merged * v.out_hidden_size)
    return patch_embed + linear + attention + merger


def llm_flops(t, prompt_tokens, output_tokens):
    """(prefill, decode) FLOPs of the language model; the first output token comes from the prefill."""
    h, layers = t.hidden_size, t.num_hidden_layers
    kv = t.num_key_value_heads * (h // t.num_attention_heads)
    per_token = 2 * layers * (2 * h * h + 2 * h * kv + 3 * h * t.intermediate_size)  # q, o, k, v, SwiGLU MLP
    lm_head = 2 * h * t.vocab_size

    def attention(first, last):  # causal QK^T + AV for queries at 0-based positions first..last-1
        return 4 * layers * h * (last * (last + 1) - first * (first + 1)) // 2

    prefill = prompt_tokens * per_token + attention(0, prompt_tokens) + (lm_head if output_tokens else 0)
    steps = max(0, output_tokens - 1)
    decode = steps * (per_token + lm_head) + attention(prompt_tokens, prompt_tokens + steps)
    return prefill, decode


def image_grid(v, width, height, max_side=None):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    if max_side and max(width, height) > max_side:  # trt_edgellm's max_image_side downscale
        scale = max_side / max(width, height)
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
    factor = v.patch_size * v.spatial_merge_size
    h, w = smart_resize(height, width, factor=factor, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return h // v.patch_size, w // v.patch_size


def load_config(model_tag):
    from huggingface_hub import snapshot_download

    path = snapshot_download(f"Qwen/{model_tag}", allow_patterns=["*.json"], local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
    config = json.load(open(os.path.join(path, "config.json")))
    text = SimpleNamespace(**config.get("text_config", config))
    return SimpleNamespace(**config["vision_config"]), text, path


def task_docs(task_name):
    from lmms_eval.tasks import TaskManager, get_task_dict

    task = get_task_dict([task_name], TaskManager(), task_type="simple")[task_name]
    return task, task.eval_docs


def run_flops(run_dir):
    model_tag = os.path.relpath(run_dir).split(os.sep)[-4]
    vision_cfg, text_cfg, model_path = load_config(model_tag)
    info = open(os.path.join(run_dir, "run_info.txt")).read() if os.path.exists(os.path.join(run_dir, "run_info.txt")) else ""
    max_side = int(m.group(1)) if (m := re.search(r"max_image_side=(\d+)", info)) else None
    tokenizer = None

    tasks = {}
    for samples_path in sorted(glob.glob(os.path.join(run_dir, "lmms_eval", "*_samples_*.jsonl"))):
        task_name = re.sub(r"^.*?_samples_", "", os.path.basename(samples_path))[: -len(".jsonl")]
        task, docs = task_docs(task_name)
        rows, recounted = [], 0
        for line in open(samples_path):
            sample = json.loads(line)
            images = [im for im in task.doc_to_visual(docs[sample["doc_id"]]) or [] if hasattr(im, "size")]
            grids = [image_grid(vision_cfg, *im.size, max_side=max_side) for im in images]
            image_tokens = sum(gh * gw for gh, gw in grids) // vision_cfg.spatial_merge_size**2
            counts = (sample.get("token_counts") or [None])[0] or {}
            if counts.get("input_tokens") is not None and counts.get("output_tokens") is not None:
                prompt_tokens, output_tokens = counts["input_tokens"], counts["output_tokens"]
            else:
                if tokenizer is None:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                content = [{"type": "image"} for _ in images] + [{"type": "text", "text": sample["input"]}]
                # One <|image_pad|> per image in the template; the real prompt has image_tokens of them.
                prompt_tokens = len(tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)) - len(images) + image_tokens
                response = sample.get("resps", sample.get("filtered_resps", ""))
                response = response if isinstance(response, str) else json.dumps(response)
                output_tokens = len(tokenizer(response)["input_ids"]) + 1  # + <|im_end|>
                recounted += 1
            vision = sum(vision_flops(vision_cfg, gh, gw) for gh, gw in grids)
            prefill, decode = llm_flops(text_cfg, prompt_tokens, output_tokens)
            rows.append(
                {
                    "doc_id": sample["doc_id"],
                    "image_grids": grids,
                    "image_tokens": image_tokens,
                    "text_tokens": prompt_tokens - image_tokens,
                    "output_tokens": output_tokens,
                    "vision_flops": vision,
                    "prefill_flops": prefill,
                    "decode_flops": decode,
                    "total_flops": vision + prefill + decode,
                }
            )
        n = len(rows)
        keys = ("image_tokens", "text_tokens", "output_tokens", "vision_flops", "prefill_flops", "decode_flops", "total_flops")
        tasks[task_name] = {
            "n": n,
            "token_counts": "recorded" if not recounted else ("tokenizer" if recounted == n else f"tokenizer for {recounted} of {n}"),
            "mean": {k: sum(r[k] for r in rows) / max(1, n) for k in keys},
            "total": {k: sum(r[k] for r in rows) for k in keys},
            "samples": rows,
        }
        print(
            f"{run_dir} {task_name}: n={n} mean GFLOPs vision {tasks[task_name]['mean']['vision_flops'] / 1e9:.1f} "
            f"prefill {tasks[task_name]['mean']['prefill_flops'] / 1e9:.1f} decode {tasks[task_name]['mean']['decode_flops'] / 1e9:.1f} "
            f"total {tasks[task_name]['mean']['total_flops'] / 1e9:.1f}",
            flush=True,
        )

    out = {
        "model": model_tag,
        "method": "analytical, 2 FLOPs per multiply-accumulate; linear layers + attention matmuls (see jetson/flops.py)",
        "image_pixels": [MIN_PIXELS, MAX_PIXELS],
        "max_image_side": max_side,
        "tasks": tasks,
    }
    json.dump(out, open(os.path.join(run_dir, "flops.json"), "w"), indent=1)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for run_dir in sys.argv[1:]:
        run_flops(run_dir.rstrip("/"))


if __name__ == "__main__":
    main()
