from __future__ import annotations

import importlib
import os
import sys
import warnings
from collections.abc import Callable, Mapping
from typing import Literal

from loguru import logger

from lmms_eval.models.registry_v2 import (
    ModelManifest,
    ModelRegistryV2,
    PluginLoadFailure,
)

logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " "<level>{message}</level>"
logger.add(sys.stdout, level="WARNING", format=log_format)


_BUILTIN_SIMPLE_MODELS = {
    "aero": "Aero",
    "generation_api": "GenerationApi",
    "cosmos_wm": "CosmosWorldModel",
    "wan2_1_t2i": "Wan2_1_T2I",
    "wan2_1_t2v": "Wan2_1_T2V",
    "wan2_2": "Wan2_2",
    "wan2_2_t2v": "Wan2_2_T2V",
    "ltx_video": "LTXVideo",
    "magi1_wm": "Magi1WorldModel",
    "aria": "Aria",
    "audio_flamingo_3": "AudioFlamingo3",
    "glm4v": "GLM4V",
    "auroracap": "AuroraCap",
    "bagel": "Bagel",
    "bagel_umm": "BagelUMM",
    "bagel_unig2u": "BagelUniG2U",
    "baichuan_omni": "BaichuanOmni",
    "batch_gpt4": "BatchGPT4",
    "claude": "Claude",
    "cogvlm2": "CogVLM2",
    "cambrians": "CambrianS",
    "cambrians_vsc": "CambriansVSC",
    "cambrians_vsc_streaming": "CambriansVSCStreaming",
    "cambrians_vsr": "CambriansVSR",
    "dummy": "Dummy",
    "egogpt": "EgoGPT",
    "from_log": "FromLog",
    "fuyu": "Fuyu",
    "gemini": "Gemini",
    "gpt4o_audio": "GPT4OAudio",
    "gemma3": "Gemma3",
    "gpt4v": "GPT4V",
    "idefics2": "Idefics2",
    "illume_plus": "ILLUMEPlus",
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
    "litellm": "LiteLLMCompatible",
    "longva": "LongVA",
    "mantis": "Mantis",
    "minicpm_o": "MiniCPM_O",
    "minicpm_v": "MiniCPM_V",
    "minimonkey": "MiniMonkey",
    "mmada": "MMaDA",
    "moviechat": "MovieChat",
    "mplug_owl_video": "mplug_Owl",
    "ola": "Ola",
    "omnivinci": "OmniVinci",
    "openai": "OpenAICompatible",
    "oryx": "Oryx",
    "ovis_u1": "OvisU1",
    "penguinvl": "PenguinVL",
    "phi3v": "Phi3v",
    "phi4_multimodal": "Phi4",
    "plm": "PerceptionLM",
    "qwen_vl_api": "Qwen_VL_API",
    "qwen_vl": "Qwen_VL",
    "qwen2_5_omni": "Qwen2_5_Omni",
    "qwen2_5_vl": "Qwen2_5_VL",
    "qwen2_audio": "Qwen2_Audio",
    "qwen2_vl": "Qwen2_VL",
    "qwen_image_edit": "QwenImageEdit",
    "qwen3_omni": "Qwen3_Omni",
    "qwen3_vl": "Qwen3_VL",
    "qwen3_5": "Qwen3_5",
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

_BUILTIN_CHAT_TEMPLATE_MODELS = {
    "gemini": "Gemini",
    "aero_realtime_vllm": "AeroRealtimeVLLM",
    "bagel_lmms_engine": "BagelLmmsEngine",
    "fastvideo": "FastVideo",
    "internvl_hf": "InternVLHf",
    "llava_hf": "LlavaHf",
    "nanovlm": "NanoVLM",
    "neo_ov": "NeoOV",
    "phi4_multimodal": "Phi4",
    "qwen3_vl": "Qwen3_VL",
    "qwen3_5": "Qwen3_5",
    "qwen2_5_vl": "Qwen2_5_VL",
    "qwen2_5_omni": "Qwen2_5_Omni",
    "qwen3_omni": "Qwen3_Omni",
    "omnivinci": "OmniVinci",
    "baichuan_omni": "BaichuanOmni",
    "minicpm_o": "MiniCPM_O",
    "thyme": "Thyme",
    "openai": "OpenAICompatible",
    "vllm": "VLLM",
    "vllm_generate": "VLLMGenerate",
    "sglang": "Sglang",
    "huggingface": "Huggingface",
    "litellm": "LiteLLMCompatible",
    "async_openai": "AsyncOpenAIChat",
    "async_hf_model": "AsyncHFModel",
    "longvila": "LongVila",
    "llava_onevision1_5": "Llava_OneVision1_5",
    "llava_onevision2": "Llava_OneVision2",
    "lfm2_5_vl": "LFM2_5_VL",
}

_BUILTIN_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "aero_realtime_vllm": ("aero_realtime_vllm_chat",),
    "gemini": ("gemini_api",),
    "dummy": ("dummy_video_reader",),
    "openai": ("openai_compatible", "openai_compatible_chat"),
    "async_openai": ("async_openai_compatible_chat", "async_openai_compatible"),
    "async_hf_model": ("async_hf",),
    "litellm": ("litellm_chat", "litellm_compatible"),
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
        set(_BUILTIN_SIMPLE_MODELS) | set(_BUILTIN_CHAT_TEMPLATE_MODELS),
    )
    manifests: list[ModelManifest] = []
    for model_id in model_ids:
        simple_class = _BUILTIN_SIMPLE_MODELS.get(model_id)
        chat_class = _BUILTIN_CHAT_TEMPLATE_MODELS.get(model_id)
        aliases = _BUILTIN_MODEL_ALIASES.get(model_id, ())
        manifests.append(
            ModelManifest(
                model_id=model_id,
                simple_class_path=(_build_class_path(model_id, "simple", simple_class) if simple_class else None),
                chat_class_path=(_build_class_path(model_id, "chat", chat_class) if chat_class else None),
                aliases=aliases,
            ),
        )
    return manifests


def _load_legacy_plugin_models(registry: ModelRegistryV2, plugins: str | None) -> tuple[PluginLoadFailure, ...]:
    failures: list[PluginLoadFailure] = []
    for plugin in (item.strip() for item in (plugins or "").split(",")):
        if not plugin:
            continue
        try:
            module = importlib.import_module(f"{plugin}.models")
            available_models = getattr(module, "AVAILABLE_MODELS")
            if not isinstance(available_models, Mapping):
                raise TypeError("AVAILABLE_MODELS must be a mapping")
            entries = tuple(available_models.items())
            if any(not isinstance(value, str) or not value.strip() for pair in entries for value in pair):
                raise TypeError("AVAILABLE_MODELS must contain non-empty string pairs")
            registry.register_manifests(
                (
                    ModelManifest(
                        model_id=model_name,
                        simple_class_path=f"{plugin}.models.{model_name}.{model_class}",
                    )
                    for model_name, model_class in entries
                ),
                overwrite=True,
            )
        except Exception as exc:
            failures.append(PluginLoadFailure(f"legacy:{plugin}", type(exc).__name__, str(exc)))
    return tuple(failures)


def _initialize_model_registry() -> ModelRegistryV2:
    registry = ModelRegistryV2()
    registry.register_manifests(_build_builtin_manifests())

    plugins = os.environ.get("LMMS_EVAL_PLUGINS")
    if plugins:
        warnings.warn(
            "LMMS_EVAL_PLUGINS is deprecated. Prefer Python entry-points group " "'lmms_eval.models' for plugin model registration.",
            DeprecationWarning,
            stacklevel=2,
        )
    failures = (*_load_legacy_plugin_models(registry, plugins), *registry.load_entrypoint_manifests(overwrite=True))
    for failure in failures:
        logger.warning(f"Failed to load model plugin {failure.source}: {failure.error_type}: {failure.message}")

    return registry


MODEL_REGISTRY_V2 = _initialize_model_registry()


class _RegistryView(Mapping[str, object]):
    def __init__(
        self,
        registry: ModelRegistryV2,
        project: Callable[[list[ModelManifest]], dict[str, object]],
    ) -> None:
        self._registry = registry
        self._project = project

    def _snapshot(self) -> dict[str, object]:
        return self._project(self._registry.list_manifests())

    def __getitem__(self, key: str) -> object:
        return self._snapshot()[key]

    def __iter__(self):
        return iter(self._snapshot())

    def __len__(self) -> int:
        return len(self._snapshot())


def _build_legacy_views(registry: ModelRegistryV2):
    def project_lane(
        manifests: list[ModelManifest],
        model_type: Literal["simple", "chat"],
    ) -> dict[str, object]:
        projected: dict[str, object] = {}
        for manifest in manifests:
            class_path = getattr(manifest, f"{model_type}_class_path")
            if class_path is None:
                continue
            prefix = f"lmms_eval.models.{model_type}.{manifest.model_id}."
            projected[manifest.model_id] = class_path.rsplit(".", 1)[-1] if class_path.startswith(prefix) else class_path
        return projected

    def project_aliases(manifests: list[ModelManifest]) -> dict[str, object]:
        projected: dict[str, object] = {}
        for manifest in manifests:
            aliases = tuple(alias for alias in manifest.aliases if registry.resolve(alias).model_id == manifest.model_id)
            if aliases:
                projected[manifest.model_id] = aliases
        return projected

    def project_preferred(manifests: list[ModelManifest]) -> dict[str, object]:
        return {manifest.model_id: (manifest.chat_class_path or manifest.simple_class_path).rsplit(".", 1)[-1] for manifest in manifests}

    return (
        _RegistryView(registry, lambda manifests: project_lane(manifests, "simple")),
        _RegistryView(registry, lambda manifests: project_lane(manifests, "chat")),
        _RegistryView(registry, project_aliases),
        _RegistryView(registry, project_preferred),
    )


AVAILABLE_SIMPLE_MODELS, AVAILABLE_CHAT_TEMPLATE_MODELS, MODEL_ALIASES, AVAILABLE_MODELS = _build_legacy_views(MODEL_REGISTRY_V2)


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
