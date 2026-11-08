from typing import Any, Optional

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_5 import Qwen3_5


@register_model("qwen3_8")
class Qwen3_8(Qwen3_5):
    """Qwen3.8 dense models, which use the Qwen3.5 multimodal architecture.

    https://huggingface.co/Qwen/Qwen3.8-27B
    For Qwen3.8-Flash-Next (``qwen4_exp``), use ``qwen3_8_flash_next``.
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.8-27B",
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 128 * 32 * 32,
        total_pixels: int = 224 * 1024 * 32 * 32,
        max_num_frames: int = 768,
        max_frames: Optional[int] = None,
        enable_thinking: Optional[bool] = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            max_num_frames=max_num_frames,
            max_frames=max_frames,
            enable_thinking=enable_thinking,
            **kwargs,
        )
