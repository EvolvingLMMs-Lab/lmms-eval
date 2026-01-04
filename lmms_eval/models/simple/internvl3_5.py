from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3 import InternVL3


@register_model("internvl3_5")
class InternVL3_5(InternVL3):
    """InternVL3.5 model wrapper.

    Uses the same implementation as InternVL3 since both share identical interfaces.
    Default pretrained model is set to InternVL3_5-8B.
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B",
        **kwargs,
    ):
        super().__init__(pretrained=pretrained, **kwargs)
