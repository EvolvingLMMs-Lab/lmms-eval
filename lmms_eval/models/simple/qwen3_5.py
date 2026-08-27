from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL


@register_model("qwen3_5")
class Qwen3_5(Qwen3_VL):
    """
    Qwen3.5 model -- thin wrapper over Qwen3_VL with Qwen3.5-specific defaults.
    https://huggingface.co/Qwen/Qwen3.5-4B
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 128 * 32 * 32,
        total_pixels: int = 224 * 1024 * 32 * 32,
        max_num_frames: int = 768,
        max_frames: Optional[int] = None,
        enable_thinking: Optional[bool] = True,
        **kwargs,
    ):
        # Accept max_frames as backward-compat alias for max_num_frames
        if max_frames is not None:
            max_num_frames = max_frames
        super().__init__(
            pretrained=pretrained,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            max_num_frames=max_num_frames,
            enable_thinking=enable_thinking,
            **kwargs,
        )
