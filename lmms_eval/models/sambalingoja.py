import base64
import logging
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import List, Optional, Tuple, Union

import requests as url_requests
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.hfmodel import HFModel


@register_model("sambalingoja")
class SambaLingoJA(HFModel):
    def __init__(
        self,
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__(pretrained="sambanovasystems/SambaLingo-Japanese-Chat", batch_size=batch_size)
