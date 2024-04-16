import os

AVAILABLE_MODELS = {"llava": "Llava", "llava_hf": "LlavaHf", "qwen_vl": "Qwen_VL", "fuyu": "Fuyu", "gpt4v": "GPT4V", "instructblip": "InstructBLIP", "minicpm_v": "MiniCPM_V", "llava_vid": "LlavaVid"}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError:
        pass

import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
