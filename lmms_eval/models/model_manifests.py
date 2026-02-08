from __future__ import annotations

from lmms_eval.models.registry_v2 import ModelManifest

# Hand-curated manifests for high-traffic integrations. These augment built-in
# manifests with aliases and capability metadata while preserving legacy ids.
CORE_MODEL_MANIFESTS: tuple[ModelManifest, ...] = (
    ModelManifest(
        model_id="vllm",
        simple_class_path="lmms_eval.models.simple.vllm.VLLM",
        chat_class_path="lmms_eval.models.chat.vllm.VLLM",
        aliases=("vllm_chat",),
        source="core",
        capabilities=("generate_until", "loglikelihood", "chat", "multimodal"),
        description="vLLM runtime backend",
    ),
    ModelManifest(
        model_id="sglang",
        chat_class_path="lmms_eval.models.chat.sglang.Sglang",
        aliases=("sglang_runtime",),
        source="core",
        capabilities=("generate_until", "chat", "multimodal", "tool_calling"),
        description="SGLang runtime backend",
    ),
    ModelManifest(
        model_id="openai_compatible",
        simple_class_path="lmms_eval.models.simple.openai_compatible.OpenAICompatible",
        chat_class_path="lmms_eval.models.chat.openai_compatible.OpenAICompatible",
        aliases=("openai_compatible_chat",),
        source="core",
        capabilities=("generate_until", "chat", "api"),
        description="OpenAI-compatible HTTP backend",
    ),
)
