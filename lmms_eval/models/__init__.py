from __future__ import annotations

import importlib
import os
import sys
import warnings
from typing import Literal

from loguru import logger

from lmms_eval.models.registry_v2 import ModelManifest, ModelRegistryV2

logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " "<level>{message}</level>"
logger.add(sys.stdout, level="WARNING", format=log_format)


AVAILABLE_SIMPLE_MODELS = {
    "aero": "Aero",
    "aria": "Aria",
    "audio_flamingo_3": "AudioFlamingo3",
    "glm4v": "GLM4V",
    "auroracap": "AuroraCap",
    "bagel": "Bagel",
    "bagel_umm": "BagelUMM",
    "baichuan_omni": "BaichuanOmni",
    "batch_gpt4": "BatchGPT4",
    "claude": "Claude",
    "cogvlm2": "CogVLM2",
    "cambrians": "CambrianS",
    "dummy_video_reader": "DummyVideoReader",
    "egogpt": "EgoGPT",
    "from_log": "FromLog",
    "fuyu": "Fuyu",
    "gemini_api": "GeminiAPI",
    "gpt4o_audio": "GPT4OAudio",
    "gemma3": "Gemma3",
    "gpt4v": "GPT4V",
    "idefics2": "Idefics2",
    "instructblip": "InstructBLIP",
    "internvideo2_5": "InternVideo2_5",
    "internvideo2": "InternVideo2",
    "internvl": "InternVLChat",
    "internvl2": "InternVL2",
    "internvl3": "InternVL3",
    "internvl3_5": "InternVL3_5",
    "kimi_audio": "KimiAudio",
    "llama_vid": "LLaMAVid",
    "llama_vision": "LlamaVision",
    "llama4_scout": "Llama4Scout",
    "llava_hf": "LlavaHf",
    "llava_onevision": "Llava_OneVision",
    "llava_onevision1_5": "Llava_OneVision1_5",
    "llava_onevision_moviechat": "Llava_OneVision_MovieChat",
    "llava_sglang": "LlavaSglang",
    "llava_vid": "LlavaVid",
    "llava": "Llava",
    "longva": "LongVA",
    "mantis": "Mantis",
    "minicpm_o": "MiniCPM_O",
    "minicpm_v": "MiniCPM_V",
    "minimonkey": "MiniMonkey",
    "moviechat": "MovieChat",
    "mplug_owl_video": "mplug_Owl",
    "ola": "Ola",
    "omnivinci": "OmniVinci",
    "openai": "OpenAICompatible",
    "oryx": "Oryx",
    "phi3v": "Phi3v",
    "phi4_multimodal": "Phi4",
    "plm": "PerceptionLM",
    "qwen_vl_api": "Qwen_VL_API",
    "qwen_vl": "Qwen_VL",
    "qwen2_5_omni": "Qwen2_5_Omni",
    "qwen2_5_vl": "Qwen2_5_VL",
    "qwen2_audio": "Qwen2_Audio",
    "qwen2_vl": "Qwen2_VL",
    "qwen3_omni": "Qwen3_Omni",
    "qwen3_vl": "Qwen3_VL",
    "reka": "Reka",
    "ross": "Ross",
    "sam3": "SAM3",
    "slime": "Slime",
    "srt_api": "SRT_API",
    "tinyllava": "TinyLlava",
    "uni_moe_2_omni": "UniMoE2Omni",
    "videoChatGPT": "VideoChatGPT",
    "video_llava": "VideoLLaVA",
    "video_salmonn_2": "VideoSALMONN2",
    "videochat2": "VideoChat2",
    "videollama3": "VideoLLaMA3",
    "videochat_flash": "VideoChat_Flash",
    "vila": "VILA",
    "vita": "VITA",
    "vllm": "VLLM",
    "vora": "VoRA",
    "whisper_vllm": "WhisperVllm",
    "whisper": "Whisper",
    "whisper_tt": "WhisperTT",
    "xcomposer2_4KHD": "XComposer2_4KHD",
    "xcomposer2d5": "XComposer2D5",
}

AVAILABLE_CHAT_TEMPLATE_MODELS = {
    "bagel_lmms_engine": "BagelLmmsEngine",
    "internvl_hf": "InternVLHf",
    "llava_hf": "LlavaHf",
    "nanovlm": "NanoVLM",
    "phi4_multimodal": "Phi4",
    "qwen3_vl": "Qwen3_VL",
    "qwen2_5_vl": "Qwen2_5_VL",
    "thyme": "Thyme",
    "openai": "OpenAICompatible",
    "vllm": "VLLM",
    "vllm_generate": "VLLMGenerate",
    "sglang": "Sglang",
    "huggingface": "Huggingface",
    "async_openai": "AsyncOpenAIChat",
    "async_hf_model": "AsyncHFModel",
    "longvila": "LongVila",
    "llava_onevision1_5": "Llava_OneVision1_5",
}

MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "openai": ("openai_compatible", "openai_compatible_chat"),
    "async_openai": ("async_openai_compatible_chat", "async_openai_compatible"),
    "async_hf_model": ("async_hf",),
}


def _build_class_path(
    model_name: str,
    model_type: Literal["simple", "chat"],
    class_name: str,
) -> str:
    if "." in class_name:
        return class_name
    return f"lmms_eval.models.{model_type}.{model_name}.{class_name}"


def _build_builtin_manifests() -> list[ModelManifest]:
    model_ids = sorted(
        set(AVAILABLE_SIMPLE_MODELS) | set(AVAILABLE_CHAT_TEMPLATE_MODELS),
    )
    manifests: list[ModelManifest] = []
    for model_id in model_ids:
        simple_class = AVAILABLE_SIMPLE_MODELS.get(model_id)
        chat_class = AVAILABLE_CHAT_TEMPLATE_MODELS.get(model_id)
        aliases = MODEL_ALIASES.get(model_id, ())
        manifests.append(
            ModelManifest(
                model_id=model_id,
                simple_class_path=(_build_class_path(model_id, "simple", simple_class) if simple_class else None),
                chat_class_path=(_build_class_path(model_id, "chat", chat_class) if chat_class else None),
                aliases=aliases,
            ),
        )
    return manifests


def _merge_legacy_plugin_models(registry: ModelRegistryV2) -> None:
    plugins = os.environ.get("LMMS_EVAL_PLUGINS")
    if not plugins:
        return

    warnings.warn(
        "LMMS_EVAL_PLUGINS is deprecated. Prefer Python entry-points group " "'lmms_eval.models' for plugin model registration.",
        DeprecationWarning,
        stacklevel=2,
    )
    for plugin in plugins.split(","):
        module = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(module, "AVAILABLE_MODELS").items():
            class_path = f"{plugin}.models.{model_name}.{model_class}"
            AVAILABLE_SIMPLE_MODELS[model_name] = class_path
            registry.register_manifest(
                ModelManifest(
                    model_id=model_name,
                    simple_class_path=class_path,
                ),
                overwrite=True,
            )


def _initialize_model_registry() -> ModelRegistryV2:
    registry = ModelRegistryV2()
    for manifest in _build_builtin_manifests():
        registry.register_manifest(manifest)

    _merge_legacy_plugin_models(registry)
    try:
        registry.load_entrypoint_manifests(overwrite=True)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to load model entry-point manifests: {exc}")

    return registry


MODEL_REGISTRY_V2 = _initialize_model_registry()


def _build_available_models_preferred() -> dict[str, str]:
    model_map: dict[str, str] = {}
    for model_id in MODEL_REGISTRY_V2.list_canonical_model_ids():
        manifest = MODEL_REGISTRY_V2.get_manifest(model_id)
        class_path = manifest.chat_class_path or manifest.simple_class_path
        if class_path:
            model_map[model_id] = class_path.rsplit(".", 1)[-1]
    return model_map


AVAILABLE_MODELS = _build_available_models_preferred()


def list_available_models(include_aliases: bool = False) -> list[str]:
    """List available model names.

    Args:
        include_aliases: If True, include aliases exposed by manifests.
    """

    if include_aliases:
        return MODEL_REGISTRY_V2.list_model_names()
    return MODEL_REGISTRY_V2.list_canonical_model_ids()


def get_model_manifest(model_name: str) -> ModelManifest:
    """Return resolved model manifest for a canonical id or alias."""

    return MODEL_REGISTRY_V2.get_manifest(model_name)


def get_model(model_name: str, force_simple: bool = False) -> type:
    try:
        return MODEL_REGISTRY_V2.get_model_class(model_name, force_simple=force_simple)
    except Exception as exc:
        logger.error(f"Failed to import model from '{model_name}': {exc}")
        raise
