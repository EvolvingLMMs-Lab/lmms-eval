"""
Usage:
python3 -m llava.model.consolidate --src ~/model_weights/llava-7b --dst ~/model_weights/llava-7b_consolidate
"""

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lmms_eval.models.video_chatgpt.model import *


def consolidate_ckpt(src_path, dst_path):
    print("Loading model")
    src_model = AutoModelForCausalLM.from_pretrained(src_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    src_tokenizer = AutoTokenizer.from_pretrained(src_path)
    src_model.save_pretrained(dst_path)
    src_tokenizer.save_pretrained(dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)

    args = parser.parse_args()

    consolidate_ckpt(args.src, args.dst)
