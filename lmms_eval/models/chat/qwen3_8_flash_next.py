from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen3_vl import Qwen3_VL as Qwen3_VL_Chat
from lmms_eval.models.simple.qwen3_8_flash_next import Qwen3_8_FlashNext as Qwen3_8_FlashNext_Simple


@register_model("qwen3_8_flash_next_chat")
class Qwen3_8_FlashNext(Qwen3_VL_Chat, Qwen3_8_FlashNext_Simple):
    """Qwen3.8-Flash-Next chat adapter with structured multimodal message support."""

    pass
