from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen3_vl import Qwen3_VL as Qwen3_VL_Chat
from lmms_eval.models.simple.qwen3_8 import Qwen3_8 as Qwen3_8_Simple


@register_model("qwen3_8_chat")
class Qwen3_8(Qwen3_VL_Chat, Qwen3_8_Simple):
    """Qwen3.8 dense chat adapter with structured multimodal message support."""

    pass
