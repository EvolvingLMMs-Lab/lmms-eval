import importlib
import os
import hf_transfer
from loguru import logger
import sys

import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "llava": "Llava",
    "qwen_vl": "Qwen_VL",
    "fuyu": "Fuyu",
    "batch_gpt4": "BatchGPT4",
    "gpt4v": "GPT4V",
    "instructblip": "InstructBLIP",
    "minicpm_v": "MiniCPM_V",
    "llava_vid": "LlavaVid",
    "videoChatGPT": "VideoChatGPT",
    "llama_vid": "LLaMAVid",
    "video_llava": "VideoLLaVA",
    "xcomposer2_4KHD": "XComposer2_4KHD",
    "claude": "Claude",
    "qwen_vl_api": "Qwen_VL_API",
    "llava_sglang": "LlavaSglang",
    "idefics2": "Idefics2",
    "internvl": "InternVLChat",
    "internvl2": "InternVL2",
    "gemini_api": "GeminiAPI",
    "reka": "Reka",
    "from_log": "FromLog",
    "mplug_owl_video": "mplug_Owl",
    "phi3v": "Phi3v",
    "tinyllava": "TinyLlava",
    "llava_hf": "LlavaHf",
    "longva": "LongVA",
    "llava_hf": "LlavaHf",
    "longva": "LongVA",
    "vila": "VILA",
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError as e:
        logger.warning(f"Failed to import {model_class} from {model_name}: {e}")

if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            try:
                exec(f"from {plugin}.models.{model_name} import {model_class}")
            except ImportError:
                logger.warning(f"Failed to import {model_class} from {model_name}")
