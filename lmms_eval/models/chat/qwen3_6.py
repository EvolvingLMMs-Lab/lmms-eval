from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen3_vl import Qwen3_VL as Qwen3_VL_Chat
from lmms_eval.models.simple.qwen3_6 import Qwen3_6 as Qwen3_6_Simple


@register_model("qwen3_6_chat")
class Qwen3_6(Qwen3_VL_Chat, Qwen3_6_Simple):
    """Qwen3.6 chat adapter with structured multimodal message support."""

    pass
