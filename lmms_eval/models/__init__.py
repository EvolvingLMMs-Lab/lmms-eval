from loguru import logger
import sys

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
    "gemini_api": "GeminiAPI",
    "reka": "Reka",
    "from_log": "FromLog",
    "mplug_owl_video": "mplug_Owl",
    "phi3v": "Phi3v",
    "tinyllava": "TinyLlava",
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except ImportError as e:
        # logger.warning(f"Failed to import {model_class} from {model_name}: {e}")
        pass
