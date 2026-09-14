"""GRT Qwen2.5-VL wrapper; stock qwen2_5_vl behavior is unchanged."""

from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.grt.qwen2_5_vl import Qwen2_5_VL
from lmms_eval.models.model_utils.grt.runtime import require_grt_runtime


@register_model("grt_qwen2_5_vl")
class GRTQwen2_5VL(Qwen2_5_VL):
    """Qwen2.5-VL with explicit patch-projection reuse controls."""

    def __init__(self, *args, **kwargs) -> None:
        require_grt_runtime()
        super().__init__(*args, **kwargs)
