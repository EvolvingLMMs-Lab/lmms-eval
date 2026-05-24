from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.vllm import VLLM


@register_model("apertus_1p5_vllm")
class Apertus1p5VLLM(VLLM):
    """Apertus 1.5 vLLM chat wrapper with chat-template-safe tokenization."""

    _chat_add_special_tokens = False
