import importlib
import os
import sys
from typing import Literal

from loguru import logger

# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
# Configure logger with detailed format including file path, function name, and line number
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " "<level>{message}</level>"
logger.add(sys.stdout, level="WARNING", format=log_format)


AVAILABLE_SIMPLE_MODELS = {
    "aero": "Aero",
    "plm": "PerceptionLM",
    "aria": "Aria",
    "auroracap": "AuroraCap",
    "batch_gpt4": "BatchGPT4",
    "claude": "Claude",
    "cogvlm2": "CogVLM2",
    "from_log": "FromLog",
    "fuyu": "Fuyu",
    "gemini_api": "GeminiAPI",
    "gpt4o_audio": "GPT4OAudio",
    "gemma3": "Gemma3",
    "gpt4v": "GPT4V",
    "idefics2": "Idefics2",
    "instructblip": "InstructBLIP",
    "internvideo2": "InternVideo2",
    "internvl": "InternVLChat",
    "internvl2": "InternVL2",
    "llama_vid": "LLaMAVid",
    "llama_vision": "LlamaVision",
    "llava": "Llava",
    "llava_hf": "LlavaHf",
    "llava_onevision": "Llava_OneVision",
    "llava_onevision1_5": "Llava_OneVision1_5",
    "llava_onevision_moviechat": "Llava_OneVision_MovieChat",
    "llava_sglang": "LlavaSglang",
    "llava_vid": "LlavaVid",
    "longva": "LongVA",
    "mantis": "Mantis",
    "minicpm_v": "MiniCPM_V",
    "minimonkey": "MiniMonkey",
    "moviechat": "MovieChat",
    "mplug_owl_video": "mplug_Owl",
    "ola": "Ola",
    "openai_compatible": "OpenAICompatible",
    "oryx": "Oryx",
    "phi3v": "Phi3v",
    "phi4_multimodal": "Phi4",
    "qwen2_5_omni": "Qwen2_5_Omni",
    "qwen2_5_vl": "Qwen2_5_VL",
    "qwen2_5_vl_interleave": "Qwen2_5_VL_Interleave",
    "qwen2_audio": "Qwen2_Audio",
    "qwen2_vl": "Qwen2_VL",
    "qwen_vl": "Qwen_VL",
    "qwen_vl_api": "Qwen_VL_API",
    "reka": "Reka",
    "ross": "Ross",
    "slime": "Slime",
    "srt_api": "SRT_API",
    "tinyllava": "TinyLlava",
    "videoChatGPT": "VideoChatGPT",
    "videochat2": "VideoChat2",
    "videollama3": "VideoLLaMA3",
    "video_llava": "VideoLLaVA",
    "vila": "VILA",
    "vita": "VITA",
    "vllm": "VLLM",
    "xcomposer2_4KHD": "XComposer2_4KHD",
    "xcomposer2d5": "XComposer2D5",
    "egogpt": "EgoGPT",
    "internvideo2_5": "InternVideo2_5",
    "videochat_flash": "VideoChat_Flash",
    "whisper": "Whisper",
    "whisper_vllm": "WhisperVllm",
    "vora": "VoRA",
}

AVAILABLE_CHAT_TEMPLATE_MODELS = {
    "llava_hf": "LlavaHf",
    "qwen2_5_vl": "Qwen2_5_VL",
    "thyme": "Thyme",
    "openai_compatible": "OpenAICompatible",
    "vllm": "VLLM",
    "vllm_generate": "VLLMGenerate",
    "sglang": "Sglang",
    "huggingface": "Huggingface",
    "async_openai": "AsyncOpenAIChat",
    "longvila": "LongVila",
}


def get_model(model_name, force_simple: bool = False):
    if model_name not in AVAILABLE_SIMPLE_MODELS and model_name not in AVAILABLE_CHAT_TEMPLATE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    if model_name in AVAILABLE_CHAT_TEMPLATE_MODELS:
        model_type = "chat"
        AVAILABLE_MODELS = AVAILABLE_CHAT_TEMPLATE_MODELS
    else:
        model_type = "simple"
        AVAILABLE_MODELS = AVAILABLE_SIMPLE_MODELS

    # Override with force_simple if needed, but only if the model exists in AVAILABLE_SIMPLE_MODELS
    if force_simple and model_name in AVAILABLE_SIMPLE_MODELS:
        model_type = "simple"
        AVAILABLE_MODELS = AVAILABLE_SIMPLE_MODELS

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"lmms_eval.models.{model_type}.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise


if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        # For plugin users, this will be replaced by chat template model later
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            AVAILABLE_SIMPLE_MODELS[model_name] = f"{plugin}.models.{model_name}.{model_class}"
