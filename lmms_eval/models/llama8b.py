from io import BytesIO
from copy import deepcopy
import os
import base64
from typing import List, Tuple, Union
from tqdm import tqdm
import requests as url_requests
import time
import logging
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils
from typing import List, Optional, Union
from lmms_eval.models.hfmodel import HFModel


@register_model("llama8b")
class LLaMa8b(HFModel):
    def __init__(
        self,
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__(pretrained="/import/snvm-sc-podscratch4/jonathanl/generic_checkpoints/llama_3/Meta-Llama-3-8B-Instruct", batch_size=batch_size)
