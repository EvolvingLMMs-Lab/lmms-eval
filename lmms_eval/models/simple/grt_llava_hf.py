"""GRT LLaVA-HF wrapper; stock llava_hf behavior is unchanged."""

from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.grt.llava_hf import LlavaHf
from lmms_eval.models.model_utils.grt.runtime import require_grt_runtime


@register_model("grt_llava_hf")
class GRTLlavaHf(LlavaHf):
    """LLaVA-OneVision with explicit patch-projection reuse controls."""

    def __init__(self, *args, **kwargs) -> None:
        require_grt_runtime()
        super().__init__(*args, **kwargs)
