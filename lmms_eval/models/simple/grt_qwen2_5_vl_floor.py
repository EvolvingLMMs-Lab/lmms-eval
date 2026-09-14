"""Prompt-routed GRT Qwen2.5-VL with a fail-closed recomputation floor."""

from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.grt.qwen_dual_route import Qwen2_5_VL_DualRouteFloor
from lmms_eval.models.model_utils.grt.runtime import require_grt_runtime


@register_model("grt_qwen2_5_vl_floor")
class GRTQwen2_5VLFloor(Qwen2_5_VL_DualRouteFloor):
    """Route public prompt text to a threshold, never document answers or IDs."""

    def __init__(self, *args, **kwargs) -> None:
        require_grt_runtime()
        super().__init__(*args, **kwargs)
