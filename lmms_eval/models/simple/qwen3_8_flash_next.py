from typing import Any, Optional

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_5 import Qwen3_5


@register_model("qwen3_8_flash_next")
class Qwen3_8_FlashNext(Qwen3_5):
    """Qwen3.8-Flash-Next, a sparse MoE model with an n-gram embedding.

    It previews the Qwen4 architecture (``qwen4_exp``) and loads with
    ``Qwen4ExpForConditionalGeneration``. It shares the Qwen3.5 vision processor.
    https://huggingface.co/Qwen/Qwen3.8-Flash-Next
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.8-Flash-Next",
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
