from __future__ import annotations

from lmms_eval.models.registry_v2 import ModelManifest

CORE_MODEL_MANIFESTS: tuple[ModelManifest, ...] = (
    ModelManifest(
        model_id="vllm",
        simple_class_path="lmms_eval.models.simple.vllm.VLLM",
        chat_class_path="lmms_eval.models.chat.vllm.VLLM",
        aliases=("vllm_chat",),
    ),
    ModelManifest(
        model_id="sglang",
        chat_class_path="lmms_eval.models.chat.sglang.Sglang",
        aliases=("sglang_runtime",),
    ),
    ModelManifest(
        model_id="openai_compatible",
        simple_class_path="lmms_eval.models.simple.openai_compatible.OpenAICompatible",
        chat_class_path="lmms_eval.models.chat.openai_compatible.OpenAICompatible",
        aliases=("openai_compatible_chat",),
    ),
)
